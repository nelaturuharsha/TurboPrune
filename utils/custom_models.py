import os
from typing import Type, Dict


import torch
import torch.nn as nn
from torchvision import models

from utils.mask_layers import *
from utils.deit import *

from rich.console import Console
from rich.table import Table

from omegaconf import DictConfig


class PruneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None

    def prepare(self, cfg: DictConfig):
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

    def replace_layers(
        model: nn.Module, layer_types_map: Dict[Type[nn.Module], Type[nn.Module]]
    ):
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
                                bias=child.bias is not None,
                            )
                        elif old_type == nn.Conv2d and issubclass(new_type, nn.Conv2d):
                            new_layer = new_type(
                                in_channels=child.in_channels,
                                out_channels=child.out_channels,
                                kernel_size=child.kernel_size,
                                stride=child.stride,
                                padding=child.padding,
                                bias=child.bias is not None,
                            )
                        else:
                            new_layer = new_type(
                                in_features=child.in_features,
                                out_features=child.out_features,
                                bias=child.bias is not None,
                            )

                        setattr(module, name, new_layer)
                        break
                else:
                    replace_layer(child)

        replace_layer(model)

    def load_model(self, load_path):
        self.model.load_state_dict(torch.load(load_path))

    def reset_weights(self, cfg: DictConfig, expt_dir: str) -> None:
        """Reset (or don't) the weights to a given checkpoint based on the provided training type.

        Args:
            config: Training configuration
            expt_dir: Directory of the experiment.
        """
        training_type = cfg.pruning_params.training_type
        if training_type == "imp":
            print("Rewinding to init")
            checkpoint_file = "model_init.pt"
        elif training_type == "wr":
            print("Rewinding to warmup init (Epoch 10)")
            checkpoint_file = "model_rewind.pt"
        else:
            print(
                "Probably LRR, nothing to do -- or if PaI, we aren't touching it in any case."
            )
            return

        original_dict = torch.load(
            os.path.join(expt_dir, "checkpoints", checkpoint_file)
        )
        current_state_dict = self.model.state_dict()

        for name, param in original_dict.items():
            if (
                name in current_state_dict
                and current_state_dict[name].shape == param.shape
                and not name.endswith("mask")
            ):
                print(f"Restoring weights for {name}")
                current_state_dict[name].copy_(param)

        self.model.load_state_dict(current_state_dict)

    def reset_masks(self):
        for module in self.model.modules():
            if isinstance(module, (ConvMask, Conv1dMask, LinearMask)):
                module.mask.fill_(1)

    def load_only_masks(self, load_path: str):
        original_dict = torch.load(load_path)
        current_state_dict = self.model.state_dict()

        for name, param in original_dict.items():
            if (
                name in current_state_dict
                and current_state_dict[name].shape == param.shape
                and name.endswith("mask")
            ):
                print(f"Restoring masks for {name}")
                current_state_dict[name].copy_(param)

        self.model.load_state_dict(current_state_dict)


class TorchVisionModel(PruneModel):
    def __init__(self, cfg: DictConfig):
        super(TorchVisionModel, self).__init__()
        self.model_name = cfg.model_params.model_name
        self.mask_layer_type = cfg.model_params.mask_layer_type
        self.prepare(cfg)

    def prepare(self, cfg: DictConfig):
        if hasattr(models, self.model_name):
            print(f"Using {self.model_name} from torchvision.models")
            self.model = getattr(models, self.model_name)(weights=None)
        else:
            raise ValueError(
                f"Model {self.model_name} not found in torchvision.models."
            )
        self.model = self.model.to(memory_format=torch.channels_last)
        dataset_name = cfg.experiment_params.dataset_name.lower()
        if dataset_name in ["cifar10", "cifar100"]:
            self._prepare_for_cifar(dataset_name)

        self._replace_layers()
        self.model = self.model.to(memory_format=torch.channels_last)

    def _prepare_for_cifar(self, dataset_name: str):
        num_classes = 10 if dataset_name == "cifar10" else 100

        if self.model_name.startswith("resnet"):
            self.model.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.model.maxpool = nn.Identity()
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif self.model_name.startswith("vgg"):
            self.model.features[0] = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            if hasattr(self.model, "classifier"):
                if isinstance(self.model.classifier, nn.Sequential):
                    in_features = self.model.classifier[-1].in_features
                    self.model.classifier[-1] = nn.Linear(in_features, num_classes)
                else:
                    in_features = self.model.classifier.in_features
                    self.model.classifier = nn.Linear(in_features, num_classes)

    def _replace_layers(self):
        conv_layer_of_type = globals().get(self.mask_layer_type)
        layer_types_map = {nn.Linear: Conv1dMask, nn.Conv2d: conv_layer_of_type}
        self.replace_layers(layer_types_map)


class CustomModel(PruneModel):
    def __init__(self, cfg: DictConfig):
        super(CustomModel, self).__init__()
        self.model_name = cfg.model_params.model_name
        self.mask_layer_type = cfg.model_params.mask_layer_type
        self.prepare(cfg)

    def prepare(self):
        if self.model_name in globals():
            self.model = globals()[self.model_name]()
        else:
            raise ValueError(
                f"Model {self.model_name} not found in torchvision.models or in the custom_models definition."
            )
        self.model = self.model.to(memory_format=torch.channels_last)

        self._replace_layers()

    def _replace_layers(self):
        layer_types_map = {
            nn.Linear: LinearMask,
        }
        self.replace_layers(layer_types_map)
