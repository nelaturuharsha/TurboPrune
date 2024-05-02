from torch.nn import init
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import math

import numpy as np


class TrackActReLU(nn.Module):
    def __init__(self):
        super(TrackActReLU, self).__init__()
        self.collect_preact = True
        self.avg_preacts = None

    def forward(self, preact):
        print('preact: ', preact.shape)
        if self.collect_preact:
            # Take the mean of the activation over the batch dimension
            print('preacts mean: ', preact.mean(0).shape)
            self.avg_preacts = preact.mean(0).detach()

        act = F.relu(preact)
        print('returning activation')
        return act


