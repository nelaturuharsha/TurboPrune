import math

import torch
import torch.nn as nn

import utils.conv_type
import utils.custom_activation
import utils.bn_type

import fastargs
from fastargs import get_current_config
from fastargs.decorators import param

get_current_config()

class Builder(object):
    def __init__(self, conv_layer, bn_layer, first_layer=None):
        self.config = get_current_config()
        self.conv_layer = conv_layer
        self.bn_layer = bn_layer
        self.first_layer = first_layer or conv_layer

    def conv(self, kernel_size, in_planes, out_planes, stride=1, first_layer=False):
        conv_layer = self.first_layer if first_layer else self.conv_layer

        if first_layer:
            print(f"==> Building first layer with {self.config['model_params.first_layer_type']}")

        if kernel_size == 3:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
        elif kernel_size == 1:
            conv = conv_layer(
                in_planes, out_planes, kernel_size=1, stride=stride, bias=False
            )
        elif kernel_size == 5:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=5,
                stride=stride,
                padding=2,
                bias=False,
            )
        elif kernel_size == 7:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=7,
                stride=stride,
                padding=3,
                bias=False,
            )
        else:
            return None

        self._init_conv(conv=conv)

        return conv

    def conv2d(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        return self.conv_layer(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

    def conv3x3(self, in_planes, out_planes, stride=1, first_layer=False):
        """3x3 convolution with padding"""
        c = self.conv(3, in_planes, out_planes, stride=stride, first_layer=first_layer)
        return c

    def conv1x1(self, in_planes, out_planes, stride=1, first_layer=False):
        """1x1 convolution with padding"""
        c = self.conv(1, in_planes, out_planes, stride=stride, first_layer=first_layer)
        return c

    def conv7x7(self, in_planes, out_planes, stride=1, first_layer=False):
        """7x7 convolution with padding"""
        c = self.conv(7, in_planes, out_planes, stride=stride, first_layer=first_layer)
        return c

    def conv5x5(self, in_planes, out_planes, stride=1, first_layer=False):
        """5x5 convolution with padding"""
        c = self.conv(5, in_planes, out_planes, stride=stride, first_layer=first_layer)
        return c

    def batchnorm(self, planes, last_bn=False, first_layer=False):
        return self.bn_layer(planes)
    
    @param('model_params.nonlinearity')
    def activation(self, nonlinearity):
        if nonlinearity == "relu":
            return (lambda: nn.ReLU())()
        elif nonlinearity == "leaky_relu":
            return (lambda: nn.LeakyReLU(negative_slope=0.01, inplace=True))()
        elif nonlinearity == "trackact-relu":
            return (lambda: utils.custom_activation.TrackActReLU())()
        else:
            raise ValueError(f"{nonlinearity} is not an initialization option!")

    @param('model_params.init')
    @param('model_params.scale_fan')
    @param('model_params.nonlinearity')
    @param('prune_params.prune_rate')
    @param('model_params.mode')
    def _init_conv(self, init, scale_fan, nonlinearity, prune_rate, mode, conv):
        if init == "signed_constant":

            fan = nn.init._calculate_correct_fan(conv.weight, mode)
            if scale_fan:
                fan = fan * (1 - prune_rate)
            gain = nn.init.calculate_gain(nonlinearity)
            std = gain / math.sqrt(fan)
            conv.weight.data = conv.weight.data.sign() * std

        elif init == "unsigned_constant":

            fan = nn.init._calculate_correct_fan(conv.weight, mode)
            if scale_fan:
                fan = fan * (1 - prune_rate)

            gain = nn.init.calculate_gain()
            std = gain / math.sqrt(fan)
            conv.weight.data = torch.ones_like(conv.weight.data) * std

        elif init == "kaiming_normal":

            if scale_fan:
                fan = nn.init._calculate_correct_fan(conv.weight, mode)
                fan = fan * (1 - prune_rate)
                gain = nn.init.calculate_gain(nonlinearity)
                std = gain / math.sqrt(fan)
                with torch.no_grad():
                    conv.weight.data.normal_(0, std)
            else:
                if nonlinearity == 'trackact-relu':    
                    nn.init.kaiming_normal_(
                        conv.weight, mode=mode, nonlinearity='relu'
                    )
                else:
                    nn.init.kaiming_normal_(
                    conv.weight, mode=mode, nonlinearity=nonlinearity
                    )

        elif init == "standard":
            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))
        else:
            raise ValueError(f"{init} is not an initialization option!")

@param('model_params.conv_type')
@param('model_params.bn_type')
@param('model_params.first_layer_type')
def get_builder(conv_type, bn_type, first_layer_type):
    print("==> Conv Type: {}".format(conv_type))
    print("==> BN Type: {}".format(bn_type))

    conv_layer = getattr(utils.conv_type, conv_type)
    bn_layer = getattr(utils.bn_type, bn_type)
    first_layer_type = None
    
    if first_layer_type is not None:
        print('popular')
        first_layer = getattr(utils.conv_type, first_layer_type)
        print(f"==> First Layer Type {first_layer_type}")
    else:
        first_layer = None

    builder = Builder(conv_layer=conv_layer, bn_layer=bn_layer, first_layer=first_layer)

    return builder
