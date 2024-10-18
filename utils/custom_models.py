import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, List, Dict
from torchvision import models
from rich.console import Console
from rich.table import Table

from fastargs.decorators import param

from utils.mask_layers import *
from utils.deit import *

class PruneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None

    def prepare(self, dataset_name: str):
        raise NotImplementedError("Subclasses must implement prepare method")

    def forward(self, x):
        return self.model(x)

    def print_layer_sparsity(self):
        console = Console()
        table = Table(title="Layer-wise Sparsity")
        table.add_column("Layer", style="cyan")
        table.add_column("Sparsity (%)", style="magenta")

        total_params = 0
        total_zeros = 0

        for name, module in self.model.named_modules():
            if isinstance(module, (ConvMask, Conv1dMask, LinearMask)):
                params = module.mask.numel()
                zeros = (module.mask == 0).sum().item()
                sparsity = (zeros / params) * 100

                table.add_row(name, f"{sparsity:.2f}")

                total_params += params
                total_zeros += zeros

        console.print(table)

    def get_overall_sparsity(self) -> float:
        total_params = 0
        total_zeros = 0

        for module in self.model.modules():
            if isinstance(module, (ConvMask, Conv1dMask, LinearMask)):
                params = module.mask.numel()
                zeros = (module.mask == 0).sum().item()
                total_params += params
                total_zeros += zeros

        return (total_zeros / total_params) * 100 if total_params > 0 else 0
    

    def replace_layers(model: nn.Module, layer_types_map: Dict[Type[nn.Module], Type[nn.Module]]):
        """
        Replaces layers in the model based on the provided layer_types_map.

        Args:
            model: The model whose layers will be replaced.
            layer_types_map: A dictionary where keys are the original layer types (e.g., nn.Linear) and values
                            are the replacement layer types (e.g., Conv1dMask).
        """
        def replace_layer(module: nn.Module):
            for name, child in module.named_children():
                for old_type, new_type in layer_types_map.items():
                    if isinstance(child, old_type):
                        if old_type == nn.Linear and new_type == Conv1dMask:
                            new_layer = Conv1dMask(
                                in_features=child.in_features,
                                out_features=child.out_features,
                                bias=child.bias is not None
                            )
                        elif old_type == nn.Conv2d and issubclass(new_type, nn.Conv2d):
                            new_layer = new_type(
                                in_channels=child.in_channels,
                                out_channels=child.out_channels,
                                kernel_size=child.kernel_size,
                                stride=child.stride,
                                padding=child.padding,
                                bias=child.bias is not None
                            )
                        else:
                            new_layer = new_type(
                                in_features=child.in_features,
                                out_features=child.out_features,
                                bias=child.bias is not None
                            )   
                        
                        setattr(module, name, new_layer)
                        break
                else:
                    replace_layer(child)

        replace_layer(model)

    def load_model(self, load_path):
        self.model.load_state_dict(torch.load(load_path))


class TorchVisionModel(PruneModel):
    @param('model_params.model_name')
    @param('model_params.mask_layer_type')
    def __init__(self, model_name: str,mask_layer_type: str):
        super(TorchVisionModel, self).__init__()
        self.model_name = model_name
        self.mask_layer_type = mask_layer_type
        self.prepare()

    @param('dataset.dataset_name')
    def prepare(self, dataset_name: str):
        if hasattr(models, self.model_name):
            print(f"Using {self.model_name} from torchvision.models")
            self.model = getattr(models, self.model_name)(weights=None)
        else:
            raise ValueError(f"Model {self.model_name} not found in torchvision.models.")
        self.model = self.model.to(memory_format=torch.channels_last)
        if dataset_name.lower() in ['cifar10', 'cifar100']:
            self._prepare_for_cifar(dataset_name.lower())
        
        self._replace_layers()

    def _prepare_for_cifar(self, dataset_name: str):
        num_classes = 10 if dataset_name == 'cifar10' else 100
        
        if self.model_name.startswith('resnet'):
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.model.maxpool = nn.Identity()
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif self.model_name.startswith('vgg'):
            self.model.features[0] = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            if hasattr(self.model, 'classifier'):
                if isinstance(self.model.classifier, nn.Sequential):
                    in_features = self.model.classifier[-1].in_features
                    self.model.classifier[-1] = nn.Linear(in_features, num_classes)
                else:
                    in_features = self.model.classifier.in_features
                    self.model.classifier = nn.Linear(in_features, num_classes)


    def _replace_layers(self):
        conv_layer_of_type = globals().get(self.mask_layer_type)
        layer_types_map = {
            nn.Linear: Conv1dMask,
            nn.Conv2d: conv_layer_of_type
        }
        self.replace_layers(layer_types_map)

class CustomModel(PruneModel):
    @param('model_params.model_name')
    @param('model_params.mask_layer_type')
    def __init__(self, model_name: str, mask_layer_type: str):
        super(CustomModel, self).__init__()
        self.model_name = model_name
        self.mask_layer_type =mask_layer_type
        self.prepare()

    def prepare(self):
        if self.model_name in globals():
            self.model = globals()[self.model_name]()
        else:
            raise ValueError(f"Model {self.model_name} not found in torchvision.models or in the custom_models definition.")
        self.model = self.model.to(memory_format=torch.channels_last)
        
        self._replace_layers()

    def _replace_layers(self):
        layer_types_map = {
            nn.Linear: LinearMask,
        }
        self.replace_layers(layer_types_map)





class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock used in a ResNet.
    The flow goes x (input) -> BN -> ReLU -> Conv instead of x (input) -> Conv -> BN -> ReLU.

    Args:
        in_planes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int, optional): Stride size. Default is 1.
    """

    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the PreActBlock."""
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    """Pre-activation ResNet model.

    Args:
        block (Type[PreActBlock]): Block type for the pre-activation ResNet.
        num_blocks (List[int]): List containing the number of blocks per layer.
        num_classes (int, optional): Number of output classes. Default is 10.
    """

    def __init__(
        self, block: Type[PreActBlock], num_blocks: List[int], num_classes: int = 10
    ) -> None:
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(
        self, block: Type[PreActBlock], planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        """Create a layer for the PreActResNet.

        Args:
            block (Type[PreActBlock]): Block type for the layer.
            planes (int): Number of planes.
            num_blocks (int): Number of blocks.
            stride (int): Stride size.

        Returns:
            Residual block with the desired configuration.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the PreActResNet."""
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



