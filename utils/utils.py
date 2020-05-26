import torch
import os


def exp_lr_decay(optimizer, global_step, init_lr = 3e-3, decay_steps = 20000,
                                         decay_rate = 0.96, lr_clip = 3e-3 ,staircase=False):
    ''' decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)  '''

    if staircase:
         lr = (init_lr * decay_rate**(global_step // decay_steps)) 
    else:
         lr = (init_lr * decay_rate**(global_step / decay_steps)) 

    for param_group in optimizer.param_groups:
         param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def log_reconstruction_sample(writer, originals, reconstructions, step):
    '''
        Writes reconstructions to TensorBoard.
    '''
    b = originals.shape[0]
    writer.add_images('Reconstructions', torch.cat((originals, reconstructions.view(b,1,28,28)), dim=0), step, dataformats='NCHW')

    
def snapshot(model, folder, epoch):
    path = os.path.join(folder, 'model_{}.pth'.format(epoch))
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    print('saving model to {}'.format(path))
    torch.save(model.state_dict(), path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

