## pythonic imports
import numpy as np
import os

## torch
import torch
import torch.nn as nn

## torchvision
import torchvision.models as models
from torch.amp import autocast

## file-based imports
from utils.conv_type import ConvMask, Conv1dMask, LinearMask, replace_layers, replace_vit_layers
from utils.harness_params import get_current_params
from utils.custom_models import PreActResNet, PreActBlock
from utils.dataset import imagenet, CIFARLoader

# import deit model
from utils.deit import deit_tiny_patch16_224, deit_small_patch16_224 

## fastargs
from fastargs import get_current_config
from fastargs.decorators import param
from typing import Optional, Dict, Any

import airbench

get_current_params()

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224 / 256


def get_sparsity(model: nn.Module) -> float:
    """Calculate the sparsity of the model.

    Args:
        model (nn.Module): The model to calculate sparsity for.

    Returns:
        float: The sparsity of the model.
    """
    nz = 0
    total = 0
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, Conv1dMask, LinearMask)):
            nz += m.mask.sum()
            total += m.mask.numel()

    return nz / total


class PruningStuff:
    """Class to handle pruning-related functionalities."""

    def __init__(self, model: Optional[nn.Module] = None) -> None:
        """Initialize the PruningStuff class.

        Args:
            model (nn.Module, optional): The model to prune. Default is None.
                                         If no model is provided, then one is automatically set based on configuration.
        """
        self.this_device = "cuda:0"
        self.config = get_current_config()

        if self.config['dataset.dataset_name'] == 'ImageNet':
            self.loaders = imagenet(distributed=False, this_device='cuda:0')
            self.train_loader = self.loaders.train_loader
        elif 'CIFAR' in self.config['dataset.dataset_name']:
            #self.train_loader = airbench.CifarLoader(path='./cifar10', batch_size=512, train=True, aug={'flip' : True, 'translate' : 2}, altflip=True)
            self.loaders = CIFARLoader(distributed=False)
            self.train_loader = self.loaders.train_loader
        if model is None:
            self.model = self.acquire_model()
        else:
            self.model = model
        self.criterion = nn.CrossEntropyLoss()

    @param("model_params.model_name")
    @param("dataset.dataset_name")
    def acquire_model(
        self, model_name: str, dataset_name: str
    ) -> nn.Module:
        """Acquire the model based on the provided parameters.

        Args:
            model_name (str): Name of the model.
            dataset_name (str): Name of the dataset.

        Returns:
            nn.Module: The acquired model.
        """
        if dataset_name == 'CIFAR10':
            num_classes = 10
        elif dataset_name == 'CIFAR100':
            num_classes = 100
        elif dataset_name == 'ImageNet':
            num_classes = 1000
        
        if model_name == "preresnet":
            model = PreActResNet(block=PreActBlock, num_blocks=[2, 2, 2, 2])
        
        elif "ImageNet" in dataset_name and "deit-small" in model_name:
            model = deit_small_patch16_224()
        elif "ImageNet" in dataset_name and "deit-tiny" in model_name:
            print('Initializing a DeiT')
            model = deit_tiny_patch16_224()
        else:
            model = getattr(models, model_name)(num_classes=num_classes)

        if "CIFAR" in dataset_name and "resnet" in model_name:
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            model.maxpool = nn.Identity()
        if "CIFAR" in dataset_name and "vgg11" in model_name:
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model.classifier = nn.Sequential(
                nn.Linear(512 * 1 * 1, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )

        # Replacing the layers.
        if "deit" in model_name:
            replace_vit_layers(model=model)
        else:
            replace_layers(model=model)
        
        # Load from a fixed init wherever possible, this should be seed specific
        # TODO
        # if 'resnet18' in model_name:
        #     model.load_state_dict(torch.load(os.path.join(self.config['experiment_params.base_dir'], 'base_resnet18_ckpt.pt')))
        # if ('resnet50' in model_name) and ('ImageNet' in dataset_name):
        #     model.load_state_dict(torch.load(os.path.join(self.config['experiment_params.base_dir'], 'imagenet_resnet50_ckpt.pt'))) 
        # elif ('resnet50' in model_name) and ('CIFAR100' in dataset_name):
        #     model.load_state_dict(torch.load(os.path.join(self.config['experiment_params.base_dir'], 'cifar100_resnet50_ckpt.pt')))


        model.to(self.this_device)
        return model

    @param("prune_params.er_method")
    @param("prune_params.er_init")
    def prune_at_initialization(
        self, er_method: str, er_init: float) -> nn.Module:
        """Prune the model at initialization.

        Args:
            er_method (str): Method of pruning.
            er_init (float): Initial sparsity target.

        Returns:
            nn.Module: The pruned model.
        """
        self.model = self.acquire_model()
        total = 0
        nonzero = 0
        for n, m in self.model.named_modules():
            if isinstance(m, (Conv1dMask, ConvMask, LinearMask)):
                total += m.mask.numel()
                nonzero += m.mask.sum()
        
        print(f'density is {(total / nonzero) * 100:3f}')
                
        print('prior to pruning at init')
        er_method_name = f"prune_{er_method}"
        pruner_method = globals().get(er_method_name)
        if er_method in {"synflow", "snip"}:
            self.model = pruner_method(self.model, self.train_loader, er_init)
        else:
            pruner_method(self.model, er_init)

        print('We just pruned at init, woohoo!')


    @param("prune_params.prune_method")
    def level_pruner(self, prune_method: str, density: float) -> None:
        """Prune the model at a specific density level.

        Args:
            prune_method (str): Method of pruning.
            density (float): Desired density after pruning.
        """
        print("---" * 20)
        print(f"Density before pruning: {get_sparsity(self.model)}")
        print("---" * 20)

        prune_method_name = f"prune_{prune_method}"
        pruner_method = globals().get(prune_method_name)
        if prune_method in {"synflow", "snip"}:
            self.model = pruner_method(self.model, self.train_loader, density)
        else:
            pruner_method(self.model, density)

        print("---" * 20)
        print(f"Density after pruning: {get_sparsity(self.model)}")
        print("---" * 20)

    def load_from_ckpt(self, path: str) -> None:
        """Load the model from a checkpoint.

        Args:
            path (str): Path to the checkpoint.
        """
        self.model.load_state_dict(torch.load(path))


def prune_mag(model: nn.Module, density: float) -> nn.Module:
    """Magnitude-based pruning of the model.

    Args:
        model (nn.Module): The model to prune.
        density (float): Desired density after pruning.

    Returns:
        nn.Module: The pruned model.
    """
    score_list = {}


    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, Conv1dMask, LinearMask)):
            score_list[n] = (m.mask.to(m.weight.device) * m.weight).detach().abs_()

    global_scores = torch.cat([torch.flatten(v) for v in score_list.values()])
    k = int((1 - density) * global_scores.numel())
    threshold, _ = torch.kthvalue(global_scores, k)

    if not k < 1:
        total_num = 0
        total_den = 0

        for n, m in model.named_modules():
            if isinstance(m, (ConvMask, Conv1dMask, LinearMask)):
                score = score_list[n].to(m.weight.device)
                zero = torch.tensor([0.0]).to(m.weight.device)
                one = torch.tensor([1.0]).to(m.weight.device)
                m.mask = torch.where(score <= threshold, zero, one)
                total_num += (m.mask == 1).sum()
                total_den += m.mask.numel()
    print(
        "Overall model density after magnitude pruning at current iteration = ",
        (total_num / total_den).item(),
    )
    return model


def prune_random_erk(model: nn.Module, density: float) -> nn.Module:
    """Random ERK-based pruning of the model.

    Args:
        model (nn.Module): The model to prune.
        density (float): Desired density after pruning.

    Returns:
        nn.Module: The pruned model.
    """
    sparsity_list = []
    num_params_list = []
    total_params = 0
    score_list = {}

    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, Conv1dMask, LinearMask)):
            score_list[n] = (
                (
                    m.mask.to(m.weight.device)
                    * torch.randn_like(m.weight).to(m.weight.device)
                )
                .detach()
                .abs_()
            )
            sparsity_list.append(torch.tensor(m.weight.shape).sum() / m.weight.numel())
            num_params_list.append(m.weight.numel())
            total_params += m.weight.numel()

    num_params_kept = (
        torch.tensor(sparsity_list) * torch.tensor(num_params_list)
    ).sum()
    num_params_to_keep = total_params * density
    C = num_params_to_keep / num_params_kept
    print("Factor: ", C)
    sparsity_list = [torch.clamp(C * s, 0, 1) for s in sparsity_list]

    total_num = 0
    total_den = 0
    cnt = 0
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, Conv1dMask, LinearMask)):
            global_scores = torch.flatten(score_list[n])
            k = int((1 - sparsity_list[cnt]) * global_scores.numel())
            if k == 0:
                threshold = 0
            else:
                threshold, _ = torch.kthvalue(global_scores, k)
            print("Layer", n, " params ", k, global_scores.numel())

            score = score_list[n].to(m.weight.device)
            zero = torch.tensor([0.0]).to(m.weight.device)
            one = torch.tensor([1.0]).to(m.weight.device)
            m.mask = torch.where(score <= threshold, zero, one)
            total_num += (m.mask == 1).sum()
            total_den += m.mask.numel()
            cnt += 1

    print(
        "Overall model density after random global (ERK) pruning at current iteration = ",
        total_num / total_den,
    )
    return model


def prune_snip(model: nn.Module, trainloader: Any, density: float) -> nn.Module:
    """SNIP method for pruning of the model.

    Args:
        model (nn.Module): The model to prune.
        trainloader (Any): The training data loader.
        density (float): Desired density after pruning.

    Returns:
        nn.Module: The pruned model.
    """
    criterion = nn.CrossEntropyLoss()
    for i, (images, target) in enumerate(trainloader):
        images = images.to(torch.device("cuda"))
        target = target.to(torch.device("cuda")).long()
        with autocast(dtype=torch.bfloat16, device_type='cuda'):
            model.zero_grad()
            output = model(images)
            criterion(output, target).backward()
        break

    score_list = {}
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, Conv1dMask, LinearMask)):
            score_list[n] = (
                (m.weight.grad * m.weight * m.mask.to(m.weight.device)).detach().abs_()
            )

    global_scores = torch.cat([torch.flatten(v) for v in score_list.values()])
    k = int((1 - density) * global_scores.numel())
    threshold, _ = torch.kthvalue(global_scores, k)

    if not k < 1:
        total_num = 0
        total_den = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask, Conv1dMask, LinearMask)):
                score = score_list[n].to(m.weight.device)
                zero = torch.tensor([0.0]).to(m.weight.device)
                one = torch.tensor([1.0]).to(m.weight.device)
                m.mask = torch.where(score <= threshold, zero, one)
                total_num += (m.mask == 1).sum()
                total_den += m.mask.numel()

    print(
        "Overall model density after snip pruning at current iteration = ",
        total_num / total_den,
    )
    return model


def prune_synflow(model: nn.Module, trainloader: Any, density: float) -> nn.Module:
    """SynFlow method pruning of the model.

    Args:
        model (nn.Module): The model to prune.
        trainloader (Any): The training data loader.
        density (float): Desired density after pruning.

    Returns:
        nn.Module: The pruned model.
    """

    @torch.no_grad()
    def linearize(model: nn.Module) -> Dict[str, torch.Tensor]:
        """Linearize the model by taking the absolute value of its parameters.

        Args:
            model (nn.Module): The model to linearize.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of parameter signs.
        """
        signs = {}
        for name, param in model.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(model: nn.Module, signs: Dict[str, torch.Tensor]) -> None:
        """Restore the signs of the model parameters.

        Args:
            model (nn.Module): The model to restore.
            signs (Dict[str, torch.Tensor]): Dictionary of parameter signs.
        """
        for n, param in model.state_dict().items():
            param.mul_(signs[n])

    signs = linearize(model)

    for i, (images, target) in enumerate(trainloader):
        images = images.to(torch.device("cuda"))
        target = target.to(torch.device("cuda")).long()
        input_dim = list(images[0, :].shape)
        input = torch.ones([1] + input_dim).to("cuda")
        with autocast(dtype=torch.bfloat16, device_type='cuda'):
            output = model(input)
            torch.sum(output).backward()
        break

    score_list = {}
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, Conv1dMask, LinearMask)):
            score_list[n] = (
                (m.mask.to(m.weight.device) * m.weight.grad * m.weight).detach().abs_()
            )

    model.zero_grad()
    nonlinearize(model, signs)

    global_scores = torch.cat([torch.flatten(v) for v in score_list.values()])
    k = int((1 - density) * global_scores.numel())
    threshold, _ = torch.kthvalue(global_scores, k)

    if not k < 1:
        total_num = 0
        total_den = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask, Conv1dMask, LinearMask)):
                score = score_list[n].to(m.weight.device)
                zero = torch.tensor([0.0]).to(m.weight.device)
                one = torch.tensor([1.0]).to(m.weight.device)
                m.mask = torch.where(score <= threshold, zero, one)
                total_num += (m.mask == 1).sum()
                total_den += m.mask.numel()

    print(
        "Overall model density after synflow pruning at current iteration = ",
        total_num / total_den,
    )
    return model


def prune_random_balanced(model: nn.Module, density: float) -> nn.Module:
    """Random balanced pruning of the model.

    Args:
        model (nn.Module): The model to prune.
        density (float): Desired density after pruning.

    Returns:
        nn.Module: The pruned model.
    """
    total_params = 0
    l = 0
    sparsity_list = []
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, Conv1dMask, LinearMask)):
            total_params += m.weight.numel()
            l += 1
    L = l
    X = density * total_params / l
    score_list = {}
    l = 0
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, Conv1dMask, LinearMask)):
            score_list[n] = (
                (
                    m.mask.to(m.weight.device)
                    * torch.randn_like(m.weight).to(m.weight.device)
                )
                .detach()
                .abs_()
            )

            if X / m.weight.numel() < 1.0:
                sparsity_list.append(X / m.weight.numel())
            else:
                sparsity_list.append(1)
                # correction for taking care of exact sparsity
                diff = X - m.mask.numel()
                X = X + diff / (L - l)
            l += 1

    total_num = 0
    total_den = 0
    cnt = 0
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, Conv1dMask, LinearMask)):
            global_scores = torch.flatten(score_list[n])
            k = int((1 - sparsity_list[cnt]) * global_scores.numel())
            if k == 0:
                threshold = 0
            else:
                threshold, _ = torch.kthvalue(global_scores, k)
            print("Layer", n, " params ", k, global_scores.numel())

            score = score_list[n].to(m.weight.device)
            zero = torch.tensor([0.0]).to(m.weight.device)
            one = torch.tensor([1.0]).to(m.weight.device)
            m.mask = torch.where(score <= threshold, zero, one)
            total_num += (m.mask == 1).sum()
            total_den += m.mask.numel()
            cnt += 1

    print(
        "Overall model density after random global (balanced) pruning at current iteration = ",
        total_num / total_den,
    )
    return model


def prune_er_erk(model: nn.Module, er_sparse_init: float) -> None:
    """ERK-based pruning at initialization.

    Args:
        model (nn.Module): The model to prune.
        er_sparse_init (float): Initial sparsity target.
    """
    sparsity_list = []
    num_params_list = []
    total_params = 0
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, Conv1dMask, LinearMask)):
            sparsity_list.append(torch.tensor(m.weight.shape).sum() / m.weight.numel())
            num_params_list.append(m.weight.numel())
            total_params += m.weight.numel()

    num_params_kept = (
        torch.tensor(sparsity_list) * torch.tensor(num_params_list)
    ).sum()
    num_params_to_keep = total_params * er_sparse_init
    C = num_params_to_keep / num_params_kept
    sparsity_list = [torch.clamp(C * s, 0, 1) for s in sparsity_list]
    l = 0
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, Conv1dMask, LinearMask)):
            m.set_er_mask(sparsity_list[l])
            l += 1
    print(sparsity_list)


def prune_er_balanced(model: nn.Module, er_sparse_init: float) -> None:
    """ER-balanced pruning at initialization.

    Args:
        model (nn.Module): The model to prune.
        er_sparse_init (float): Initial sparsity target.
    """
    total_params = 0
    l = 0
    sparsity_list = []
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, Conv1dMask, LinearMask)):
            total_params += m.weight.numel()
            l += 1
    L = l
    X = er_sparse_init * total_params / l
    l = 0
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, Conv1dMask, LinearMask)):
            if X / m.weight.numel() < 1.0:
                sparsity_list.append(X / m.weight.numel())
            else:
                sparsity_list.append(1)
                # correction for taking care of exact sparsity
                diff = X - m.mask.numel()
                X = X + diff / (L - l)
            l += 1

    l = 0
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, Conv1dMask, LinearMask)):
            m.set_er_mask(sparsity_list[l])
            l += 1
    print(sparsity_list)
