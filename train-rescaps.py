from __future__ import print_function
import argparse
import os
import io
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from dataset import smallNORB
from model import rescapsules
from loss import CapsuleLoss
from utils import AverageMeter, exp_lr_decay, log_reconstruction_sample, accuracy, snapshot


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Matrix-Capsules-EM')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--test-intvl', type=int, default=1, metavar='N',
                    help='test intvl (default: 1)')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=3e-3, metavar='LR',
                    help='learning rate (default: 0.01)') # Hinton: 3e-3, according to openreview
parser.add_argument('--weight-decay', type=float, default=2e-7, metavar='WD',
                    help='weight decay (default: 0)') # Hinton: 2e-7, according to openreview
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--em-iters', type=int, default=2, metavar='N',
                    help='iterations of EM Routing')
parser.add_argument('--add-decoder', default=False,
                    help='adds a reconstruction network')
parser.add_argument('--alpha', default=1.0, type=float,
                  help='Regularization coefficient to scale down the reconstruction loss (default: 0.05)')
parser.add_argument('--snapshot-folder', type=str, default='./snapshots/rescapsnet-stl-1', metavar='SF',
                    help='where to store the snapshots')
parser.add_argument('--logdir', type=str, default='./runs', metavar='LD',
                    help='where tensorboard will write the logs')
parser.add_argument('--data-folder', type=str, default='./data', metavar='DF',
                    help='where to store the datasets')
parser.add_argument('--dataset', type=str, default='stl10', metavar='D',
                    help='dataset for training(stl10,smallNORB)')


def get_setting(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    path = os.path.join(args.data_folder, args.dataset)
    if args.dataset == 'stl10':
        num_class = 10
        img_dim = (96,96,3)
        train_loader = torch.utils.data.DataLoader(
            datasets.STL10(path, split='test', folds=None, download=True, 
                           transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.STL10(path, split='train', folds=None, download=True, 
                           transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'smallNORB':
        num_class = 5
        img_dim = (48,48,1)
        train_loader = torch.utils.data.DataLoader(
            smallNORB(path, train=True, download=True,
                      transform=transforms.Compose([
                          transforms.Resize(48),
#                           transforms.RandomCrop(32),
                          transforms.ColorJitter(brightness=32./255, contrast=0.5),
                          transforms.ToTensor()
                      ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            smallNORB(path, train=False,
                      transform=transforms.Compose([
                          transforms.Resize(48),
#                           transforms.CenterCrop(32),
                          transforms.ToTensor()
                      ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise NameError('Undefined dataset {}'.format(args.dataset))
    return num_class, img_dim, train_loader, test_loader
    

def train(train_loader, model, criterion, optimizer, epoch, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    train_len = len(train_loader)
    total_score = 0
    end = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        act_output, reconstructions = model(data)
        r = (1.*batch_idx + (epoch-1)*train_len) / (args.epochs*train_len)
        act_loss, reconstruction_loss, total_loss = criterion(act_output, target, data, reconstructions, r)
        
        score = accuracy(act_output, target)[0].item()
        total_score += score
        
        global_step = (batch_idx+1) + (epoch - 1) * len(train_loader) 
        # change the learning rate exponentially
        exp_lr_decay(optimizer = optimizer, global_step = global_step)
        
        total_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.log_interval == 0:
            # Logging
            train_writer.add_scalar('Loss/Spread', act_loss.item(), global_step)
            if args.add_decoder == True:
                train_writer.add_scalar('Loss/Reconstruction', reconstruction_loss.item(), global_step)
                log_reconstruction_sample(train_writer, data, reconstructions, global_step)
            train_writer.add_scalar('Loss/Total', total_loss.item(), global_step)
            train_writer.add_scalar('Metrics/Accuracy', score, global_step)
            
            # Console output
            print('Train Epoch: {}\t[{}/{} ({:.0f}%)]\t'
                  'Loss: {:.6f}\tScore: {:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader),
                  total_loss.item(), score,
                  batch_time=batch_time, data_time=data_time))
    
    total_score /= train_len
    return total_score

    
def test(test_loader, model, criterion, step, device):
    model.eval()
    test_len = len(test_loader)
    rand = random.randint(1, test_len)
    
    outputs, targets = [], []
    START_FLAG = True
    with torch.no_grad():
        test_batch_time = AverageMeter()
        test_end = time.time()
        
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output, reconstructions = model(data)
            if START_FLAG:
                outputs = output
                targets = target
                START_FLAG = False
            else:
                outputs = torch.cat([outputs, output])
                targets = torch.cat([targets, target])
                
            test_batch_time.update(time.time() - test_end)
            test_end = time.time()
                
            if batch_idx == rand and args.add_decoder == True:
                log_reconstruction_sample(test_writer, data, reconstructions, step)

    act_loss, rec_loss, tot_loss = criterion(outputs, targets, None, None)
    test_loss = act_loss.item()
    score = accuracy(outputs, targets)[0].item()
    
    # Logging
    test_writer.add_scalar('Loss/Spread', test_loss, step)
    test_writer.add_scalar('Metrics/Accuracy', score, step)
    
    print('\nTest set: Average loss: {:.6f}, Score: {:.6f} Average Test Time {:.3f} \n'.format(
        test_loss, score, test_batch_time.avg))
    return test_loss, score


def main():
    global args, train_writer, test_writer
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")
    
    # tensorboard logging
    train_writer = SummaryWriter(comment='train')
    test_writer = SummaryWriter(comment='test')
    
    # dataset
    num_class, img_dim, train_loader, test_loader = get_setting(args)

    # model
#     A, B, C, D = 64, 8, 16, 16
    A, B, C, D = 32, 32, 32, 32
    model = rescapsules(A=A, B=B, C=C, D=D, E=num_class,
                     iters=args.em_iters, add_decoder=args.add_decoder, img_dim=img_dim).to(device)

    print("Number of trainable parameters: {}".format(sum(param.numel() for param in model.parameters())))
    criterion = CapsuleLoss(alpha=args.alpha, mode='spread', num_class=num_class, add_decoder=args.add_decoder)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss, best_score = test(test_loader, model, criterion, 0, device)
    for epoch in range(1, args.epochs + 1):
        score = train(train_loader, model, criterion, optimizer, epoch, device)
        
        if epoch % args.test_intvl == 0:
            test_loss, test_score = test(test_loader, model, criterion, epoch*len(train_loader), device)
            if test_loss < best_loss or test_score > best_score:
                snapshot(model, args.snapshot_folder, epoch)
            best_loss = min(best_loss, test_loss)
            best_score = max(best_score, test_score)
    print('best test score: {:.6f}'.format(best_score))
    
    train_writer.close()
    test_writer.close()

    # save end model
    snapshot(model, args.snapshot_folder, 'end_{}'.format(args.epochs))

    
if __name__ == '__main__':
    main()

