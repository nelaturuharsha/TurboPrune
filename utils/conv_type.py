import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvMask(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.activation = torch.relu
        self.register_buffer('mask', torch.ones_like(self.weight))
            
    def forward(self, x):
        
        sparseWeight = self.mask.to(self.weight.device) * self.weight
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x
    
    def set_er_mask(self, p):
        self.mask = torch.zeros_like(self.weight).bernoulli_(p)

    def getSparsity(self, f=torch.sigmoid):
        sparseWeight = self.mask.to(self.weight.device) * self.weight
        temp = sparseWeight.detach().cpu()
        temp[temp!=0] = 1
        return (100 - temp.mean().item()*100), temp.numel(), 0

class LinearMask(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', torch.ones_like(self.weight))
            
    def forward(self, x):
        
        sparseWeight = self.mask.to(self.weight.device) * self.weight
        x = F.linear(x, sparseWeight)
        return x
    
    def set_er_mask(self, p):
        self.mask = torch.zeros_like(self.weight).bernoulli_(p)

    def getSparsity(self, f=torch.sigmoid):
        sparseWeight = self.mask.to(self.weight.device) * self.weight
        temp = sparseWeight.detach().cpu()
        temp[temp!=0] = 1
        return (100 - temp.mean().item()*100), temp.numel(), 0





