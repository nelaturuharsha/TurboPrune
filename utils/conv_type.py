import torch
import torch.nn as nn
import torch.nn.functional as F

from fastargs import get_current_config
from fastargs.decorators import param


get_current_config()

class ConvMask(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

class Conv1dMask(nn.Conv1d):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            bias=bias
        )

        self.register_buffer('mask', torch.ones_like(self.weight))
            
    def forward(self, x):
        x = x.unsqueeze(-1) 
        sparseWeight = self.mask.to(self.weight.device) * self.weight
        x = F.conv1d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        x = x.squeeze(-1)
        return x
    
    def set_er_mask(self, p):
        self.mask = torch.zeros_like(self.weight).bernoulli_(p)

    def getSparsity(self, f=torch.sigmoid):
        sparseWeight = self.mask.to(self.weight.device) * self.weight
        temp = sparseWeight.detach().cpu()
        temp[temp != 0] = 1
        return (100 - temp.mean().item() * 100), temp.numel(), 0


@param('model_params.conv_type')
def replace_layers(conv_type, model):
    layers_to_replace = []

    conv_layer_of_type = globals().get(conv_type)

    for name, layer in model.named_modules():
        if 'downsample' in name:
            continue
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            layers_to_replace.append((name, layer))

    for name, layer in layers_to_replace:
        parts = name.split('.')
        parent_module = model
        for part in parts[:-1]:
            parent_module = getattr(parent_module, part)

        if isinstance(layer, nn.Linear):
            in_features = layer.in_features
            out_features = layer.out_features
            bias = layer.bias is not None
        
            conv_layer = Conv1dMask(in_features, out_features, bias)

            setattr(parent_module, parts[-1], conv_layer)
        if isinstance(layer, nn.Conv2d):
            conv_mask_layer = conv_layer_of_type(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                bias=layer.bias is not None
            )
            setattr(parent_module, parts[-1], conv_mask_layer)
    print(model)
    return model

        


