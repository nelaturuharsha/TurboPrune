import torch
import os
import prettytable
from typing import Any, Dict, Optional, Tuple 

from utils.mask_layers import ConvMask, Conv1dMask, LinearMask
from datetime import datetime
import uuid

import numpy as np
import random
import yaml
import pandas as pd

from fastargs.decorators import param
from fastargs import get_current_config

from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from rich.layout import Layout

import wandb

from argparse import ArgumentParser

def reset_weights(
    expt_dir: str, model: torch.nn.Module, training_type: str
) -> torch.nn.Module:
    """Reset (or don't) the weight to a given checkpoint based on the provided training type.

    Args:
        expt_dir (str): Directory of the experiment.
        model (torch.nn.Module): The model to reset.
        training_type (str): Type of training ('imp', 'wr', or 'lrr').

    Returns:
        torch.nn.Module: The model with reset weights.
    """
    if training_type == "imp":
        print('Rewinding to init')
        original_dict = torch.load(
            os.path.join(expt_dir, "checkpoints", "model_init.pt")
        )
    elif training_type == "wr":
        print('I gotchu, rewinding to warmup init (Epoch 10)')
        original_dict = torch.load(
            os.path.join(expt_dir, "checkpoints", "model_rewind.pt")
        )
    else:
        print("probably LRR, aint nothing to do -- or if PaI, we aren't touching it any case.")
        return model

    original_weights = dict(
        filter(lambda v: v[0].endswith((".weight", ".bias")), original_dict.items())
    )
    model_dict = model.state_dict()
    model_dict.update(original_weights)
    model.load_state_dict(model_dict)

    return model


def reset_optimizer(
    expt_dir: str, optimizer: torch.optim.Optimizer, training_type: str
) -> torch.optim.Optimizer:
    """Reset the optimizer state based on the provided training type.

    Args:
        expt_dir (str): Directory of the experiment.
        optimizer (torch.optim.Optimizer): The optimizer to reset.
        training_type (str): Type of training ('imp', 'wr', or 'lrr').

    Returns:
        torch.optim.Optimizer: The reset optimizer.
    """
    if training_type in {"imp", "lrr"}:
        optimizer.load_state_dict(
            torch.load(os.path.join(expt_dir, "artifacts", "optimizer_init.pt"))
        )
    elif training_type == "wr":
        optimizer.load_state_dict(
            torch.load(os.path.join(expt_dir, "artifacts", "optimizer_rewind.pt"))
        )

    return optimizer


def reset_only_weights(expt_dir: str, ckpt_name: str, model: torch.nn.Module) -> None:
    """Reset only the weights of the model from a specified checkpoint.

    Args:
        expt_dir (str): Directory of the experiment.
        ckpt_name (str): Checkpoint name.
        model (torch.nn.Module): The model to reset.
    """
    original_dict = torch.load(os.path.join(expt_dir, "checkpoints", ckpt_name))
    original_weights = dict(
        filter(lambda v: v[0].endswith((".weight", ".bias")), original_dict.items())
    )
    model_dict = model.state_dict()
    model_dict.update(original_weights)
    model.load_state_dict(model_dict)


def reset_only_masks(expt_dir: str, ckpt_name: str, model: torch.nn.Module) -> None:
    """Reset only the masks of the model from a specified checkpoint.

    Args:
        expt_dir (str): Directory of the experiment.
        ckpt_name (str): Checkpoint name.
        model (torch.nn.Module): The model to reset.
    """
    original_dict = torch.load(os.path.join(expt_dir, "checkpoints", ckpt_name))
    original_weights = dict(
        filter(lambda v: v[0].endswith(".mask"), original_dict.items())
    )
    model_dict = model.state_dict()
    model_dict.update(original_weights)
    model.load_state_dict(model_dict)


def compute_sparsity(tensor: torch.Tensor) -> Tuple[float, int, int]:
    """Compute the sparsity of a given tensor. Sparsity = number of elements which are 0 in the mask.

    Args:
        tensor (torch.Tensor): The tensor to compute sparsity for.

    Returns:
        tuple: Sparsity, number of non-zero elements, and total elements.
    """
    remaining = tensor.sum().item()
    total = tensor.numel()
    sparsity = 1.0 - (remaining / total)
    return sparsity, remaining, total


def print_sparsity_info(model: torch.nn.Module, verbose: bool = True) -> float:
    """Print and return the sparsity information of the model.

    Args:
        model (torch.nn.Module): The model to check.
        verbose (bool, optional): Whether to print detailed sparsity info of each layer. Default is True.

    Returns:
        float: Overall sparsity of the model.
    """
    my_table = prettytable.PrettyTable()
    my_table.field_names = ["Layer Name", "Layer Sparsity", "Density", "Non-zero/Total"]
    total_params = 0
    total_params_kept = 0
    for name, layer in model.named_modules():
        if isinstance(layer, (ConvMask, Conv1dMask, LinearMask)):
            weight_mask = layer.mask
            sparsity, remaining, total = compute_sparsity(weight_mask)
            my_table.add_row([name, sparsity, 1 - sparsity, f"{remaining}/{total}"])
            total_params += total
            total_params_kept += remaining
    overall_sparsity = 1 - (total_params_kept / total_params)
    
    if verbose:
        print(my_table)
        print("-----------")
        print(f"Overall Sparsity of All Layers: {overall_sparsity:.4f}")
        print("-----------")

    return overall_sparsity


@param("experiment_params.base_dir")
@param("experiment_params.resume_level")
@param("experiment_params.resume_expt_name")
def gen_expt_dir(
    base_dir: str, resume_level: int, resume_expt_name: Optional[str] = None
) -> str:
    """Create a new experiment directory and all the necessary subdirectories.
       If provided, instead of creating a new directory -- set the directory to the one provided.

    Args:
        base_dir (str): Base directory for experiments.
        resume_level (int): Level to resume from.
        resume_expt_name (str, optional): Name of the experiment to resume from. Default is None.

    Returns:
        str: Path to the experiment directory.
    """
    config = get_current_config()
    prefix = None
    prune_method = config['prune_params.prune_method']
    er_method = config['prune_params.er_method']

    common_prefix = f"{config['dataset.dataset_name']}_{config['model_params.model_name']}_{config['experiment_params.training_type']}_seed_{config['experiment_params.seed']}_budget_{config['cyclic_training.total_epoch_budget']}_cycles_{config['cyclic_training.num_cycles']}_strat_{config['cyclic_training.strategy']}"

    if prune_method != 'just dont':
        prefix = f"{common_prefix}_{prune_method}_rate_{config['prune_params.prune_rate']}"
    elif er_method != 'just dont':
        prefix = f"{common_prefix}_{er_method}_rate_{1-config['prune_params.er_init']}"
    else:
        prefix = f"{common_prefix}_{prune_method}_{config['prune_params.prune_rate']}_{er_method}_rate_{1-config['prune_params.er_init']}"
    
    if resume_level != 0 and resume_expt_name:
        expt_dir = os.path.join(base_dir, resume_expt_name)
        print(f"Resuming from Level -- {resume_level}")
    elif resume_level == 0 and resume_expt_name is None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:6]
        unique_name = f"{prefix}_{current_time}_{unique_id}"
        expt_dir = os.path.join(base_dir, unique_name)
        print(f"Creating this Folder {expt_dir}:)")
    else:
        raise AssertionError(
            "Either start from scratch, or provide a path to the checkpoint :)"
        )

    if not os.path.exists(expt_dir):
        os.makedirs(expt_dir)
        os.makedirs(f"{expt_dir}/checkpoints")
        os.makedirs(f"{expt_dir}/metrics")
        os.makedirs(f"{expt_dir}/metrics/epochwise_metrics")
        os.makedirs(f"{expt_dir}/artifacts/")
        os.makedirs(f"{expt_dir}/extended_checkpoints")

    return prefix, expt_dir


@param("experiment_params.seed")
def set_seed(seed: int, is_deterministic: bool = False) -> None:
    """Set the random seed for reproducibility.

    Args:
        seed (int): Seed value.
        is_deterministic (bool, optional): Whether to set deterministic behavior. Default is False.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    if is_deterministic:
        print("This ran")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

@param("prune_params.prune_method")
@param("prune_params.target_sparsity")
@param("prune_params.prune_rate")
def generate_densities(prune_method: str, target_sparsity: float, prune_rate: float, current_sparsity: float) -> list[float]:
    """Generate a list of densities for pruning. The density is calculated as
       (1 - prune_rate) ^ i multiplied by current_sparsity until target_sparsity is reached.
    Args:
        prune_method (str): Method of pruning.
        target_sparsity (float): The target sparsity to reach.
        current_sparsity (float): The current density (1 - current_sparsity).
        prune_rate (float): Rate of pruning.
    Returns:
        list[float]: List of densities until target sparsity is reached.
    """
    densities = []
    current_density = 1 - current_sparsity
    target_density = 1 - target_sparsity
    while current_density > target_density:
        densities.append(current_density)
        current_density *= (1 - prune_rate)
    if current_density <= target_density:
        densities.append(current_density)
    return densities

def save_config(expt_dir: str, config: Any) -> None:
    """Save the experiment configuration to a YAML file in the experiment directory.

    Args:
        expt_dir (str): Directory of the experiment.
        config (Any): Configuration to save.
    """
    nested_dict: Dict[str, Dict[str, Any]] = {}
    for (outer_key, inner_key), value in config.content.items():
        if outer_key not in nested_dict:
            nested_dict[outer_key] = {}
        nested_dict[outer_key][inner_key] = value

    with open(os.path.join(expt_dir, "expt_config.yaml"), "w") as file:
        yaml.dump(nested_dict, file, default_flow_style=False)


@param('cyclic_training.min_epochs')
@param('cyclic_training.max_epochs')
@param('cyclic_training.num_cycles')
@param('cyclic_training.strategy')
def generate_epoch_schedule(min_epochs, max_epochs, num_cycles, strategy):
    """
    Generates a schedule of epochs per cycle based on the given strategy.
    
    Parameters:
    - min_epochs (int): Minimum number of epochs.
    - max_epochs (int): Maximum number of epochs.
    - num_cycles (int): Number of cycles.
    - strategy (str): Strategy for epoch scheduling. Options are:
                      'linear_decrease', 'linear_increase', 'exponential_decrease',
                      'exponential_increase', 'cyclic_peak', 'alternating', 'plateau'.
                      
    Returns:
    - List[int]: A list of epochs for each cycle.
    """
    
    if strategy == 'linear_decrease':
        step = (max_epochs - min_epochs) / (num_cycles - 1)
        epochs = [int(max_epochs - step * i) for i in range(num_cycles)]

    elif strategy == 'linear_increase':
        step = (max_epochs - min_epochs) / (num_cycles - 1)
        epochs = [int(min_epochs + step * i) for i in range(num_cycles)]

    elif strategy == 'exponential_decrease':
        factor = (min_epochs / max_epochs) ** (1 / (num_cycles - 1))
        epochs = [int(max_epochs * (factor ** i)) for i in range(num_cycles)]

    elif strategy == 'exponential_increase':
        factor = (max_epochs / min_epochs) ** (1 / (num_cycles - 1))
        epochs = [int(min_epochs * (factor ** i)) for i in range(num_cycles)]

    elif strategy == 'cyclic_peak':
        mid_point = num_cycles // 2
        increase_step = (max_epochs - min_epochs) / mid_point
        decrease_step = (max_epochs - min_epochs) / (num_cycles - mid_point - 1)
        epochs = [int(min_epochs + increase_step * i) for i in range(mid_point)]
        epochs += [int(max_epochs - decrease_step * (i - mid_point)) for i in range(mid_point, num_cycles)]

    elif strategy == 'alternating':
        high = max_epochs
        low = min_epochs
        epochs = [high if i % 2 == 0 else low for i in range(num_cycles)]

    elif strategy == 'plateau':
        increase_cycles = num_cycles // 2
        plateau_cycles = num_cycles - increase_cycles
        increase_step = (max_epochs - min_epochs) / increase_cycles
        epochs = [int(min_epochs + increase_step * i) for i in range(increase_cycles)]
        epochs += [max_epochs for _ in range(plateau_cycles)]

    else:
        epochs = [min_epochs for _ in range(num_cycles)]

    return epochs

@param('cyclic_training.total_epoch_budget')
@param('cyclic_training.num_cycles')
@param('cyclic_training.strategy')
def generate_budgeted_schedule(total_epoch_budget, num_cycles, strategy):
    """
    Generates a schedule of epochs per cycle based on the given strategy and total epoch budget.
    
    Parameters:
    - total_epoch_budget (int): Total number of epochs to be distributed across cycles.
    - num_cycles (int): Number of cycles.
    - strategy (str): Strategy for epoch scheduling. Options are:
                      'linear_decrease', 'linear_increase', 'exponential_decrease',
                      'exponential_increase', 'cyclic_peak', 'alternating', 'plateau'.
                      
    Returns:
    - List[int]: A list of epochs for each cycle.
    """
    
    if strategy == 'linear_decrease':
        step = total_epoch_budget / (num_cycles * (num_cycles + 1) / 2)
        epochs = [int(step * (num_cycles - i)) for i in range(num_cycles)]

    elif strategy == 'linear_increase':
        step = total_epoch_budget / (num_cycles * (num_cycles + 1) / 2)
        epochs = [int(step * (i + 1)) for i in range(num_cycles)]

    elif strategy == 'exponential_decrease':
        factor = 0.5 ** (1 / (num_cycles - 1))
        total_factor = sum(factor ** i for i in range(num_cycles))
        epochs = [int(total_epoch_budget * (factor ** i) / total_factor) for i in range(num_cycles)]

    elif strategy == 'exponential_increase':
        factor = 2 ** (1 / (num_cycles - 1))
        total_factor = sum(factor ** i for i in range(num_cycles))
        epochs = [int(total_epoch_budget * (factor ** i) / total_factor) for i in range(num_cycles)]

    elif strategy == 'cyclic_peak':
        mid_point = num_cycles // 2
        increase_step = total_epoch_budget / (mid_point * (mid_point + 1) / 2)
        decrease_step = total_epoch_budget / ((num_cycles - mid_point) * (num_cycles - mid_point + 1) / 2)
        epochs = [int(increase_step * (i + 1)) for i in range(mid_point)]
        epochs += [int(decrease_step * (num_cycles - i)) for i in range(mid_point, num_cycles)]

    elif strategy == 'alternating':
        high = total_epoch_budget // (num_cycles // 2 + num_cycles % 2)
        low = total_epoch_budget // (2 * (num_cycles // 2 + num_cycles % 2))
        epochs = [high if i % 2 == 0 else low for i in range(num_cycles)]

    elif strategy == 'plateau':
        increase_cycles = num_cycles // 2
        plateau_cycles = num_cycles - increase_cycles
        increase_step = total_epoch_budget / (increase_cycles * (increase_cycles + 1) / 2)
        epochs = [int(increase_step * (i + 1)) for i in range(increase_cycles)]
        epochs += [total_epoch_budget // num_cycles for _ in range(plateau_cycles)]

    else:
        epochs = [total_epoch_budget // num_cycles for _ in range(num_cycles)]

    current_total = sum(epochs)
    if current_total > total_epoch_budget:
        scaling_factor = total_epoch_budget / current_total
        epochs = [int(epoch * scaling_factor) for epoch in epochs]

        current_total = sum(epochs)
        excess = current_total - total_epoch_budget
        
        if excess > 0:
            reduction_per_epoch = excess // len(epochs)
            remainder = excess % len(epochs)
            
            epochs = [epoch - reduction_per_epoch for epoch in epochs]
            
            for i in range(remainder):
                epochs[i] -= 1
    
 
    return epochs

def save_metrics_and_update_summary(console, model, expt_dir, prefix, level, level_metrics, num_cycles, epoch_schedule):
    metrics_path = os.path.join(expt_dir, "metrics", f"level_{level}_metrics.csv")
    pd.DataFrame(level_metrics).to_csv(metrics_path, index=False)
    console.log(f"[bold cyan]Saved metrics for level {level}[/bold cyan]")

    summary_path = os.path.join(expt_dir, "metrics", f"{prefix}_overall_summary.csv")
    sparsity = model.get_overall_sparsity()
    new_data = {
        "level": [level],
        "sparsity": [round(sparsity, 4)],
        "num_cycles": [num_cycles],
        "max_test_acc": [round(max(level_metrics["test_acc"]), 4)],
        "final_test_acc": [round(level_metrics["test_acc"][-1], 4)],
        "epoch_schedule": ['-'.join(map(str, epoch_schedule))]
    }
    df = pd.DataFrame(new_data)
    if os.path.exists(summary_path):
        df.to_csv(summary_path, mode='a', header=False, index=False)
    else:
        df.to_csv(summary_path, index=False)
    console.log(f"[bold cyan]Updated overall summary for level {level}[/bold cyan]")

    wandb.log(  {
        "sparsity": new_data["sparsity"][0],
        "max_test_acc": new_data["max_test_acc"][0]
    })

def display_training_info(cycle_info, training_info, optimizer_info):
    
    console = Console()


    hardware_tree = Tree("Training Harness Configuration")
    for key, value in training_info.items():
        hardware_tree.add(f"[bold cyan]{key.capitalize()}:[/bold cyan] [yellow]{value}[/yellow]")
    hardware_panel = Panel(hardware_tree, title="Training Configuration", border_style="cyan")

    schedule_tree = Tree("Training Schedule")
    schedule_tree.add(f"[bold cyan]Number of Cycles:[/bold cyan] [yellow]{cycle_info['Number of Cycles']}[/yellow]")
    schedule_tree.add(f"[bold cyan]Epochs per Cycle:[/bold cyan] [yellow]{', '.join(map(str, cycle_info['Epochs per Cycle']))}[/yellow]")
    schedule_tree.add(f"[bold cyan]Total Training Length:[/bold cyan] [yellow]{cycle_info['Total Training Length']}[/yellow]")
    schedule_panel = Panel(schedule_tree, title="Overall Training Schedule", border_style="cyan")

    cycle_tree = Tree(f"Training Cycle {cycle_info['Training Cycle']}")
    cycle_tree.add(f"[bold cyan]Epochs this cycle:[/bold cyan] [yellow]{cycle_info['Epochs this cycle']}[/yellow]")
    cycle_tree.add(f"[bold cyan]Total epochs so far:[/bold cyan] [yellow]{cycle_info['Total epochs so far']}[/yellow]")
    cycle_tree.add(f"[bold cyan]Current Sparsity:[/bold cyan] [yellow]{cycle_info['Current Sparsity']}[/yellow]")
    cycle_panel = Panel(cycle_tree, title="Current Cycle Information", border_style="cyan")

    optimizer_tree = Tree("Optimizer Configuration")
    for key, value in optimizer_info.items():
        optimizer_tree.add(f"[bold cyan]{key}:[/bold cyan] [yellow]{value}[/yellow]")
    optimizer_panel = Panel(optimizer_tree, title="Optimizer Details", border_style="cyan")

    experiment_config = Layout()

    experiment_config.split(
        Layout(hardware_panel, name="hardware", ratio=1),
        Layout(name="bottom_row", ratio=3)
    )

    experiment_config["bottom_row"].split_row(
        Layout(schedule_panel, name="schedule"),
        Layout(cycle_panel, name="cycle"),
        Layout(optimizer_panel, name="optimizer")
    )

    console.print("\n") 
    console.print(experiment_config)
    console.print("\n") 

def save_model(model, save_path, distributed: bool):
    if hasattr(model, '_orig_mod') and not distributed:
        torch.save(model._orig_mod.model.state_dict(), save_path) 
    elif hasattr(model, '_orig_mod') and distributed:
        torch.save(model._orig_mod.module.state_dict(), save_path)
    elif distributed:
        torch.save(model.module.model.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)