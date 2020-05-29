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

from model import capsules
from loss import CapsuleLoss
from datasets.multimnist import *
from utils import AverageMeter, exp_lr_decay, log_reconstruction_sample, log_heatmap, calc_metrics, snapshot


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Matrix-Capsules-EM')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--test-intvl', type=int, default=1, metavar='N',
                    help='test intvl (default: 1)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
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
                  help='Regularization coefficient to scale down the reconstruction loss (default: 0.0005)')
parser.add_argument('--snapshot-folder', type=str, default='./snapshots', metavar='SF',
                    help='where to store the snapshots')
parser.add_argument('--logdir', type=str, default='./runs', metavar='LD',
                    help='where tensorboard will write the logs')
parser.add_argument('--data-folder', type=str, default='./data', metavar='DF',
                    help='where to store the datasets')
parser.add_argument('--dataset', type=str, default='multimnist', metavar='D',
                    help='dataset for training(multimnist)')


def get_setting(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    path = os.path.join(args.data_folder, args.dataset)
    if args.dataset == 'multimnist':
        num_class = 10
        train_loader = torch.utils.data.DataLoader(
            MultiMNIST('./data/double_mnist_seed_123/double_mnist_seed_123_image_size_64_64/', mode='train',
                       transform=transforms.Compose(
                           [transforms.RandomCrop(62),
                            transforms.Resize((48,48)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.0501,), (0.2010,))]
                       )),
                       batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            MultiMNIST('./data/double_mnist_seed_123/double_mnist_seed_123_image_size_64_64/', mode='val',
                       transform=transforms.Compose(
                           [transforms.RandomCrop(62),
                            transforms.Resize((48,48)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.0501,), (0.2010,))]
                       )),
                       batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        raise NameError('Undefined dataset {}'.format(args.dataset))
    return num_class, train_loader, test_loader


def train(train_loader, model, criterion, optimizer, epoch, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    train_len = len(train_loader)
    scores = {
        'acc': 0,
        'rec': 0,
        'f1': 0,
        'hamm': 0,
        'emr': 0,
    }
    end = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        act_output, reconstructions = model(data)
        act_loss, reconstruction_loss, total_loss = criterion(act_output, target, data, reconstructions)
        
        acc, rec, f1, hamm, emr = calc_metrics(act_output, target)
        scores.update(acc=scores['acc']+acc,
                      rec=scores['rec']+rec,
                      f1=scores['f1']+f1,
                      hamm=scores['hamm']+hamm,
                      emr=scores['emr']+emr)
        
        global_step = (batch_idx+1) + (epoch - 1) * len(train_loader) 
        # change the learning rate exponentially
        exp_lr_decay(optimizer = optimizer, global_step = global_step)
        
        total_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.log_interval == 0:
            # Logging
            train_writer.add_scalar('Loss/BCE', act_loss.item(), global_step)
            if args.add_decoder:
                train_writer.add_scalar('Loss/Reconstruction', reconstruction_loss.item(), global_step)
                log_reconstruction_sample(train_writer, data, reconstructions, global_step)
                train_writer.add_scalar('Loss/Total', total_loss.item(), global_step)
            train_writer.add_scalar('Metrics/Precision', acc, global_step)
            train_writer.add_scalar('Metrics/Recall', rec, global_step)
            train_writer.add_scalar('Metrics/F1-score', f1, global_step)
            train_writer.add_scalar('Metrics/HammingScore', hamm, global_step)
            train_writer.add_scalar('Metrics/ExactMatchRatio', emr, global_step)
            
            # Console output
            print('Train Epoch: {}\t[{}/{} ({:.0f}%)]\t'
                  'Loss: {:.6f}\tF1-Score: {:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader),
                  total_loss.item(), f1,
                  batch_time=batch_time, data_time=data_time))
    
    scores.update(acc=scores['acc']/train_len,
                  rec=scores['rec']/train_len,
                  f1=scores['f1']/train_len,
                  hamm=scores['hamm']/train_len,
                  emr=scores['emr']/train_len)
    return scores

    
def test(test_loader, model, criterion, step, device):
    model.eval()
    test_len = len(test_loader)
    rand = random.randint(1, test_len)
    
    outputs, targets = [], []
    START_FLAG = True
    with torch.no_grad():
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
                
            if batch_idx == rand and args.add_decoder:
                log_reconstruction_sample(test_writer, data, reconstructions, step)

    act_loss, rec_loss, tot_loss = criterion(outputs, targets, None, None)
    test_loss = act_loss.item()
    acc, rec, f1, hamm, emr = calc_metrics(outputs, targets)
    
    # Logging
    test_writer.add_scalar('Loss/BCE', test_loss, step)
    test_writer.add_scalar('Metrics/Precision', acc, step)
    test_writer.add_scalar('Metrics/Recall', rec, step)
    test_writer.add_scalar('Metrics/F1-score', f1, step)
    test_writer.add_scalar('Metrics/HammingScore', hamm, step)
    test_writer.add_scalar('Metrics/ExactMatchRatio', emr, step)
    log_heatmap(outputs, targets, step)
    
    print('\nTest set: Average loss: {:.6f}, F1-Score: {:.6f}, ExactMatch: {:.6f} \n'.format(
        test_loss, f1, emr))
    return test_loss, f1


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
    num_class, train_loader, test_loader = get_setting(args)

    # model
#     A, B, C, D = 64, 8, 16, 16
    A, B, C, D = 32, 32, 32, 32
    model = capsules(A=A, B=B, C=C, D=D, E=num_class,
                     iters=args.em_iters, add_decoder=args.add_decoder).to(device)

    print("Number of trainable parameters: {}".format(sum(param.numel() for param in model.parameters())))
    criterion = CapsuleLoss(alpha=args.alpha, mode='bce', add_decoder=args.add_decoder)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss, best_score = test(test_loader, model, criterion, 0, device)
    for epoch in range(1, args.epochs + 1):
        scores = train(train_loader, model, criterion, optimizer, epoch, device)
        
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

