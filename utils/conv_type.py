import torch
import torch.nn as nn
import torch.nn.functional as F

from fastargs import get_current_config
from fastargs.decorators import param

get_current_config()


class ConvMask(nn.Conv2d):
    """
    Conv2d layer which inherits from the original PyTorch Conv2d layer with an additionally initialized mask parameter.
    This mask is applied during the forward pass to the weights of the layer.

    Args:
        **kwargs: Keyword arguments for nn.Conv2d.
    """

    def __init__(self, **kwargs: any) -> None:
        super().__init__(**kwargs)
        self.register_buffer("mask", torch.ones_like(self.weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with masked weights."""
        sparseWeight = self.mask.to(self.weight.device) * self.weight
        return F.conv2d(
            x,
            sparseWeight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def set_er_mask(self, p: float) -> None:
        """
        Method for setting the mask while using random pruning.

        Args:
            p (float): Probability for Bernoulli distribution.
        """
        self.mask = torch.zeros_like(self.weight).bernoulli_(p)


class LinearMask(nn.Linear):
    """
    Linear layer which inherits from the original PyTorch Linear layer with an additionally initialized mask parameter.
    This mask is applied during the forward pass to the weights of the layer.

    Args:
        **kwargs: Keyword arguments for nn.Linear.
    """

    def __init__(self, **kwargs: any) -> None:
        super().__init__(**kwargs)
        self.register_buffer("mask", torch.ones_like(self.weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method for setting the mask while using random pruning.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        sparseWeight = self.mask.to(self.weight.device) * self.weight
        return F.linear(x, sparseWeight, self.bias)

    def set_er_mask(self, p: float) -> None:
        """
        Meth setting the mask using random pruning.

        Args:
            p (float): Probability for Bernoulli distribution.
        """
        self.mask = torch.zeros_like(self.weight).bernoulli_(p)


class Conv1dMask(nn.Conv1d):
    """
    Conv1d layer which inherits from the original PyTorch Conv1d layer with an additionally initialized mask parameter.
    This mask is applied during the forward pass to the weights of the layer.
    Used for replacing linear layers with an equivalent 1D convolutional layer.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): If True, adds a learnable bias to the output. Default: True.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            bias=bias,
        )
        self.register_buffer("mask", torch.ones_like(self.weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with masked weights.
        """
        x = x.unsqueeze(-1)
        sparseWeight = self.mask.to(self.weight.device) * self.weight
        x = F.conv1d(
            x,
            sparseWeight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return x.squeeze(-1)

    def set_er_mask(self, p: float) -> None:
        """
        Enables setting the mask random pruning.

        Args:
            p (float): Probability for Bernoulli distribution.
        """
        self.mask = torch.zeros_like(self.weight).bernoulli_(p)

@param('model_params.conv_type')
def replace_layers(conv_type: str, model: nn.Module) -> nn.Module:
    """
    Replaces nn.Linear and nn.Conv2d layers in the model with corresponding masked layers.
    Skips layers which are part of the shortcut connections.

    Args:
        conv_type (str): The type of masked layer to use (e.g., 'ConvMask').
        model (nn.Module): The model in which to replace layers.

    Returns:
        nn.Module: The model with replaced layers.
    """
    layers_to_replace = []

    conv_layer_of_type = globals().get(conv_type)

    for name, layer in model.named_modules():
        if "downsample" in name:
            continue
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            layers_to_replace.append((name, layer))

    for name, layer in layers_to_replace:
        parts = name.split(".")
        parent_module = model
        for part in parts[:-1]:
            parent_module = getattr(parent_module, part)

        if isinstance(layer, nn.Linear):
            in_features = layer.in_features
            out_features = layer.out_features
            bias = layer.bias is not None

            conv_layer = Conv1dMask(in_features, out_features, bias)
            setattr(parent_module, parts[-1], conv_layer)
        elif isinstance(layer, nn.Conv2d):
            conv_mask_layer = conv_layer_of_type(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                bias=layer.bias is not None,
            )
            setattr(parent_module, parts[-1], conv_mask_layer)

    return model
