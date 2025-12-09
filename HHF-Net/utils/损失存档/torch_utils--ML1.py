import torch
import torch.nn as nn

class My_SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0):
        super(My_SmoothL1Loss, self).__init__()
        self.beta = beta

    def forward(self, input, target):
        diff = torch.abs(input - target)
        loss = torch.where(diff < self.beta,
                           0.5 * diff ** 2 / self.beta,
                           diff - 0.5 * self.beta)
        return loss.mean()

