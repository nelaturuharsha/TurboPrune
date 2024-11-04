## pythonic imports
import numpy as np

## torch
import torch
import torch.nn as nn

## torchvision
from torch.amp import autocast
import torch.distributed as dist

## file-based imports
from utils.harness_params import get_current_params
from utils.custom_models import TorchVisionModel, CustomModel
from utils.dataset import imagenet, AirbenchLoaders
from utils.mask_layers import ConvMask, Conv1dMask, LinearMask

## fastargs
from fastargs import get_current_config
from fastargs.decorators import param
from typing import Optional, Dict, Any

from rich.console import Console
from rich.columns import Columns
from rich.panel import Panel


get_current_params()

class PruningStuff:
    """Class to handle pruning-related functionalities."""
    
    def __init__(self, model: Optional[nn.Module] = None) -> None:
        """Initialize the PruningStuff class.

        Args:
            model (nn.Module, optional): The model to prune. Default is None.
                                         If no model is provided, then one is automatically set based on configuration.
        """
        self.console = Console()
        self.this_device = "cuda:0"
        self.config = get_current_config()
        self.dataset_name = self.config['dataset.dataset_name'].lower()
        self.distributed = self.config['dist_params.distributed'] and torch.cuda.device_count() > 1 and not self.dataset_name.startswith("cifar")
        if self.distributed:
            self.rank = dist.get_rank()

        if self.dataset_name.startswith('cifar'):
            self.loaders = AirbenchLoaders()
        else:
            self.batch_size = self.config['dataset.batch_size']
            self.loaders = imagenet(distributed=False, this_device='cuda:0', batch_size=self.batch_size*torch.cuda.device_count())
        
        self.train_loader = self.loaders.train_loader

        if model is None:
            if self.distributed and self.rank == 0:
                self.console.print(Panel("[bold dark_orange] Starting from scratch [/bold dark_orange]", border_style="dark_orange", expand=False))
            else:
                self.console.print(Panel("[bold dark_orange] Starting from scratch [/bold dark_orange]", border_style="dark_orange", expand=False))
            self.model = self.acquire_model()
        else:
            if self.distributed and self.rank == 0:
                self.console.print(Panel("[bold plum1]Using provided model.[/bold plum1]", border_style="plum1", expand=False))
            else:
                self.console.print(Panel("[bold plum1]Using provided model.[/bold plum1]", border_style="plum1", expand=False))
            self.model = model
        
    def acquire_model(self) -> nn.Module:
        """Acquire the model based on the provided parameters.
        Returns:
            nn.Module: The acquired model.
        """
        try:
            model = TorchVisionModel()
            if self.distributed and self.rank == 0:
                self.console.print("[bold turquoise]Using Torchvision Model :)[/bold turquoise]")
        except:
            model = CustomModel()
            if self.distributed and self.rank == 0:
                self.console.print("[bold turquoise]Using Custom Model :D[/bold turquoise]")
        model = model.to(self.this_device)
        return model

    @param("prune_params.prune_method")
    def prune_the_model(self, prune_method: str, target_density: float) -> None:
        """Prune the model using the specified method and density.

        Args:
            prune_method (str): Method of pruning.
            target_density (float): Desired density after pruning.
        """
        init_sparsity = self.model.get_overall_sparsity()

        prune_method_name = f"prune_{prune_method}"
        pruner_method = globals().get(prune_method_name)
        
        if pruner_method is None:
            self.console.print(f"[bold red]Error: Unknown pruning method '{prune_method}'[/bold red]")
            return

        if prune_method in {"synflow", "snip"}:
            self.model = pruner_method(self.model, self.train_loader, target_density)
        else:
            pruner_method(self.model, target_density)

        final_sparsity = self.model.get_overall_sparsity()

        initial = Panel(f"[magenta]{init_sparsity:.4f}[/magenta]", title="Initial Sparsity")
        final = Panel(f"[cyan]{final_sparsity:.4f}[/cyan]", title="Final Sparsity")
        self.console.print(Columns([initial, final]))
        self.console.print(f"[bold green]Pruning completed using {prune_method} method![/bold green]")

    def load_from_ckpt(self, path: str) -> None:
        """Load the model from a checkpoint.

        Args:
            path (str): Path to the checkpoint.
        """
        self.console.print(f"[bold blue]Loading from {path}[/bold blue]")
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
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask, Conv1dMask, LinearMask)):
                score = score_list[n].to(m.weight.device)
                zero = torch.tensor([0.0]).to(m.weight.device)
                one = torch.tensor([1.0]).to(m.weight.device)
                m.mask = torch.where(score <= threshold, zero, one)

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
            cnt += 1

    return model

@param('experiment_params.training_precision')
def get_dtype_amp(training_precision):
    dtype_map = {
        'bfloat16': (torch.bfloat16, True),
        'float16': (torch.float16, True),
        'float32': (torch.float32, False)
    }
    return dtype_map.get(training_precision, (torch.float32, False))

def prune_snip(model: nn.Module, trainloader: Any, density: float) -> nn.Module:
    """SNIP method for pruning of the model.

    Args:
        model (nn.Module): The model to prune.
        trainloader (Any): The training data loader.
        density (float): Desired density after pruning.

    Returns:
        nn.Module: The pruned model.
    """

    precision, use_amp = get_dtype_amp()
    criterion = nn.CrossEntropyLoss()
    for i, (images, target) in enumerate(trainloader):
        images = images.to(torch.device("cuda"))
        target = target.to(torch.device("cuda")).long()
        with autocast('cuda', dtype=precision, enabled = use_amp):
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
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask, Conv1dMask, LinearMask)):
                score = score_list[n].to(m.weight.device)
                zero = torch.tensor([0.0]).to(m.weight.device)
                one = torch.tensor([1.0]).to(m.weight.device)
                m.mask = torch.where(score <= threshold, zero, one)

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

    precision, use_amp = get_dtype_amp()
    signs = linearize(model)

    for i, (images, target) in enumerate(trainloader):
        images = images.to(torch.device("cuda"))
        target = target.to(torch.device("cuda")).long()
        input_dim = list(images[0, :].shape)
        input = torch.ones([1] + input_dim).to("cuda")
        with autocast('cuda', dtype=precision, enabled = use_amp):
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
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask, Conv1dMask, LinearMask)):
                score = score_list[n].to(m.weight.device)
                zero = torch.tensor([0.0]).to(m.weight.device)
                one = torch.tensor([1.0]).to(m.weight.device)
                m.mask = torch.where(score <= threshold, zero, one)

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

            cnt += 1

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

