import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


class Spread(_Loss):

    def __init__(self, m_min=0.2, m_max=0.9, num_class=10):
        super(Spread, self).__init__()
        self.m_min = m_min
        self.m_max = m_max
        self.num_class = num_class

    def forward(self, x, target, r):
        b, E = x.shape
        assert E == self.num_class
        margin = self.m_min + (self.m_max - self.m_min)*r

        at = torch.cuda.FloatTensor(b).fill_(0)
        for i, lb in enumerate(target):
            at[i] = x[i][lb]
        at = at.view(b, 1).repeat(1, E)

        zeros = x.new_zeros(x.shape)
        loss = torch.max(margin - (at - x), zeros)
        loss = loss**2
        loss = loss.sum() / b - margin**2

        return loss


class CapsuleLoss(_Loss):
    def __init__(self, alpha, mode='bce', num_class=10, add_decoder=True):
        super(CapsuleLoss, self).__init__()
        self.alpha = alpha
        if mode == 'bce':
            self.criterion = nn.BCELoss()
        elif mode == 'spread':
            self.criterion = Spread(num_class=num_class)
        self.mode = mode
        self.add_decoder = add_decoder

    def forward(self, outputs, labels, images, reconstructions, r=1):
        if self.mode == 'bce':
            act_loss = self.criterion(outputs, labels)
        elif self.mode == 'spread':
            act_loss = self.criterion(outputs, labels, r)

        reconstruction_loss = 0
        if self.add_decoder and reconstructions is not None:
            assert torch.numel(images) == torch.numel(reconstructions), "Reconstruction dimensions do not fit input."
            reconstruction_loss = torch.mean((reconstructions.view(images.shape) - images) ** 2)

        total_loss = act_loss + self.alpha * reconstruction_loss
        return act_loss, reconstruction_loss, total_loss