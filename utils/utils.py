import torch
import os
import io
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as skm


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


def calc_metrics(output, target, threshold=0.5):
    '''
        Computes Precision, Recall, F1-score, hamilton-score & Exact Match Ratio based on threshold.
    '''
    outputs_np = output.cpu().detach().numpy() > threshold
    targets_np = target.cpu().detach().numpy()
    
    acc, rec, f1, _ = skm.precision_recall_fscore_support(targets_np, outputs_np, average='weighted')
    hamm = 1 - skm.hamming_loss(targets_np, outputs_np)
    emr = skm.accuracy_score(targets_np, outputs_np)
    
    return acc, rec, f1, hamm, emr


def log_heatmap(output, target, step):
    '''
        Calculates metrics and writes a heatmap for all thresholds in range 0.1,0.2,...0.9 to TensorBoard.
    '''
    thresholds = np.arange(0.1, 1.0, 0.1)
    res = { t: calc_metrics(output, target, threshold=t) for t in thresholds }
        
    t_data = pd.DataFrame.from_dict(res, orient='index', columns=['Precision','Recall','F1-Score','Hamming','ExactMatch'])

    plt.figure(figsize=(10,10))
    ax = sns.heatmap(data=t_data, cmap='Blues', annot=True, yticklabels=t_data.index.values.round(1))
    ax.invert_yaxis()
    plt.title('Scores on different range of thresholds for step {}'.format(step))

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image = transforms.ToTensor()(image)
    test_writer.add_image('Heatmap', image, step)
    
    
def log_reconstruction_sample(writer, originals, reconstructions, step):
    '''
        Writes reconstructions to TensorBoard.
    '''
    b = originals.shape[0]
    writer.add_images('Reconstructions', torch.cat((originals, reconstructions.view(originals.shape)), dim=0), step, dataformats='NCHW') # this changes to reconstructions.view(b,1,48,48)) for multiMNIST

    
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

