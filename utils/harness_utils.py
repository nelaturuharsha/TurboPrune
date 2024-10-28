import os
import uuid
from datetime import datetime
import prettytable
import yaml
from typing import Any, Dict, Optional, Tuple 

import numpy as np
import random
import pandas as pd

import torch

from fastargs.decorators import param
from fastargs import get_current_config

from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from rich.layout import Layout

import wandb

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

@param("experiment_params.base_dir")
@param("experiment_params.resume_level")
@param("experiment_params.resume_expt_name")
def gen_expt_dir(
    base_dir: str, resume_level: int, resume_expt_name: Optional[str] = None
) -> Tuple[str, str]:
    """Create a new experiment directory and all the necessary subdirectories.
       If provided, instead of creating a new directory -- set the directory to the one provided.

    Args:
        base_dir (str): Base directory for experiments.
        resume_level (int): Level to resume from.
        resume_expt_name (str, optional): Name of the experiment to resume from. Default is None.

    Returns:
        Tuple[str, str]: Prefix and path to the experiment directory.
    """
    config = get_current_config()
    
    # Create prefix using f-string with config values
    prefix = (
        f"{config['dataset.dataset_name']}"
        f"_{config['model_params.model_name']}"
        f"_{config['experiment_params.training_type']}"
        f"_{config['prune_params.prune_method']}"
        f"_{config['prune_params.target_sparsity']}"
        f"_seed_{config['experiment_params.seed']}"
        f"_budget_{config['cyclic_training.epochs_per_level']}"
        f"_cycles_{config['cyclic_training.num_cycles']}"
        f"_strat_{config['cyclic_training.strategy']}"
        f"_lr_{config['optimizer.lr_start']}-{config['optimizer.lr_peak']}-{config['optimizer.lr_end']}"
        f"_skipwarmup_{config['optimizer.skip_warmup']}"
    )
    
    if resume_level != 0 and resume_expt_name:
        expt_dir = os.path.join(base_dir, resume_expt_name)
        print(f"Resuming from Level -- {resume_level}")
    elif resume_level == 0 and resume_expt_name is None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:6]
        unique_name = f"{prefix}_{current_time}_{unique_id}"
        expt_dir = os.path.join(base_dir, unique_name)
        print(f"Creating this Folder {expt_dir} :)")
    else:
        raise AssertionError(
            "Either start from scratch, or provide a path to the checkpoint :)"
        )

    if not os.path.exists(expt_dir):
        os.makedirs(expt_dir)
        for subdir in ['checkpoints', 'metrics', 'metrics/epochwise_metrics', 'artifacts']:
            os.makedirs(os.path.join(expt_dir, subdir))

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

@param("prune_params.at_init")
@param("prune_params.prune_method")
@param("prune_params.target_sparsity")
@param("prune_params.prune_rate")
def generate_densities(at_init: str, prune_method: str, target_sparsity: float, prune_rate: float, current_sparsity: float) -> list[float]:
    """Generate a list of densities for pruning. The density is calculated as
       (1 - prune_rate) ^ i multiplied by current_sparsity until target_sparsity is reached.
    Args:
        at_init (bool): Whether to generate densities at initialization.
        prune_method (str): Method of pruning.
        target_sparsity (float): The target sparsity to reach.
        current_sparsity (float): The current density (1 - current_sparsity).
        prune_rate (float): Rate of pruning.
    Returns:
        list[float]: List of densities until target sparsity is reached.
    """
    at_init = True if at_init == 'true' else False
    if prune_method in ['mag', 'random_erk', 'random_balanced']:
        if at_init:
            return [1-target_sparsity]
        else:
            densities = []
            current_density = 1 - current_sparsity
            target_density = 1 - target_sparsity
            while current_density > target_density:
                densities.append(current_density)
                current_density *= (1 - prune_rate)
            if current_density <= target_density:
                densities.append(current_density)
            return densities
    elif prune_method in  ['er_erk', 'er_balanced', 'synflow', 'snip']:
        assert at_init, "This pruning method can only be used for pruning at initialization."
        return [1 - target_sparsity]
    elif prune_method == 'just dont':
        return [1.0]
    else:
        raise ValueError(f"Unknown pruning method: {prune_method}")

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

@param('cyclic_training.epochs_per_level')
@param('cyclic_training.num_cycles')
@param('cyclic_training.strategy')
def generate_budgeted_schedule(epochs_per_level, num_cycles, strategy):
    """
    Generates a schedule of epochs per cycle based on the given strategy and total epoch budget.
    
    Parameters:
    - epochs_per_level (int): Total number of epochs to be distributed across cycles.
    - num_cycles (int): Number of cycles.
    - strategy (str): Strategy for epoch scheduling. Options are:
                      'linear_decrease', 'linear_increase', 'exponential_decrease',
                      'exponential_increase', 'cyclic_peak', 'alternating', 'plateau'.
                      
    Returns:
    - List[int]: A list of epochs for each cycle.
    """
    
    if strategy == 'linear_decrease':
        step = epochs_per_level / (num_cycles * (num_cycles + 1) / 2)
        epochs = [int(step * (num_cycles - i)) for i in range(num_cycles)]

    elif strategy == 'linear_increase':
        step = epochs_per_level / (num_cycles * (num_cycles + 1) / 2)
        epochs = [int(step * (i + 1)) for i in range(num_cycles)]

    elif strategy == 'exponential_decrease':
        factor = 0.5 ** (1 / (num_cycles - 1))
        total_factor = sum(factor ** i for i in range(num_cycles))
        epochs = [int(epochs_per_level * (factor ** i) / total_factor) for i in range(num_cycles)]

    elif strategy == 'exponential_increase':
        factor = 2 ** (1 / (num_cycles - 1))
        total_factor = sum(factor ** i for i in range(num_cycles))
        epochs = [int(epochs_per_level * (factor ** i) / total_factor) for i in range(num_cycles)]

    elif strategy == 'cyclic_peak':
        mid_point = num_cycles // 2
        increase_step = epochs_per_level / (mid_point * (mid_point + 1) / 2)
        decrease_step = epochs_per_level / ((num_cycles - mid_point) * (num_cycles - mid_point + 1) / 2)
        epochs = [int(increase_step * (i + 1)) for i in range(mid_point)]
        epochs += [int(decrease_step * (num_cycles - i)) for i in range(mid_point, num_cycles)]

    elif strategy == 'alternating':
        high = epochs_per_level // (num_cycles // 2 + num_cycles % 2)
        low = epochs_per_level // (2 * (num_cycles // 2 + num_cycles % 2))
        epochs = [high if i % 2 == 0 else low for i in range(num_cycles)]

    elif strategy == 'plateau':
        increase_cycles = num_cycles // 2
        plateau_cycles = num_cycles - increase_cycles
        increase_step = epochs_per_level / (increase_cycles * (increase_cycles + 1) / 2)
        epochs = [int(increase_step * (i + 1)) for i in range(increase_cycles)]
        epochs += [epochs_per_level // num_cycles for _ in range(plateau_cycles)]

    else:
        epochs = [epochs_per_level // num_cycles for _ in range(num_cycles)]

    current_total = sum(epochs)
    if current_total > epochs_per_level:
        scaling_factor = epochs_per_level / current_total
        epochs = [int(epoch * scaling_factor) for epoch in epochs]

        current_total = sum(epochs)
        excess = current_total - epochs_per_level
        
        if excess > 0:
            reduction_per_epoch = excess // len(epochs)
            remainder = excess % len(epochs)
            
            epochs = [epoch - reduction_per_epoch for epoch in epochs]
            
            for i in range(remainder):
                epochs[i] -= 1
    
    return epochs

def save_metrics_and_update_summary(console, model, expt_dir, prefix, level, level_metrics, num_cycles, epoch_schedule):
    metrics_path = os.path.join(expt_dir, 'metrics', 'epochwise_metrics', f"level_{level}_metrics.csv")
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

from rich.text import Text
from rich.console import Group

def display_training_info(cycle_info, training_info, optimizer_info):
    console = Console()

    def create_wrapped_tree(title, info_dict):
        tree = Tree(title)
        for key, value in info_dict.items():
            if 'expt_dir' in key:
                value = os.path.basename(value)
            text = Text.from_markup(f"[bold cyan]{key.capitalize()}:[/bold cyan] ")
            text.append(str(value), style="yellow")
            tree.add(Group(text))
        return tree

    hardware_tree = create_wrapped_tree("Training Harness Configuration", training_info)
    hardware_panel = Panel(hardware_tree, title="Training Configuration", border_style="cyan")

    schedule_tree = Tree("Training Schedule")
    schedule_tree.add(Group(Text.from_markup(f"[bold cyan]Number of Cycles:[/bold cyan] [yellow]{cycle_info['Number of Cycles']}[/yellow]")))
    epochs_text = Text.from_markup(f"[bold cyan]Epochs per Cycle:[/bold cyan] ")
    epochs_text.append(', '.join(map(str, cycle_info['Epochs per Cycle'])), style="yellow")
    schedule_tree.add(Group(epochs_text))
    schedule_tree.add(Group(Text.from_markup(f"[bold cyan]Total Training Length:[/bold cyan] [yellow]{cycle_info['Total Training Length']}[/yellow]")))
    schedule_panel = Panel(schedule_tree, title="Overall Training Schedule", border_style="cyan")

    cycle_tree = Tree(f"Training Cycle {cycle_info['Training Cycle']}")
    cycle_tree.add(Group(Text.from_markup(f"[bold cyan]Epochs this cycle:[/bold cyan] [yellow]{cycle_info['Epochs this cycle']}[/yellow]")))
    cycle_tree.add(Group(Text.from_markup(f"[bold cyan]Total epochs so far:[/bold cyan] [yellow]{cycle_info['Total epochs so far']}[/yellow]")))
    cycle_tree.add(Group(Text.from_markup(f"[bold cyan]Current Sparsity:[/bold cyan] [yellow]{cycle_info['Current Sparsity']}[/yellow]")))
    cycle_panel = Panel(cycle_tree, title="Current Cycle Information", border_style="cyan")

    optimizer_tree = create_wrapped_tree("Optimizer Configuration", optimizer_info)
    optimizer_panel = Panel(optimizer_tree, title="Optimizer Details", border_style="cyan")

    experiment_config = Layout()
    experiment_config.split(
        Layout(hardware_panel, name="hardware", ratio=1),
        Layout(schedule_panel, name="schedule", ratio=1),
        Layout(cycle_panel, name="cycle", ratio=1),
        Layout(optimizer_panel, name="optimizer", ratio=1),
    )
    experiment_config.update(Panel(experiment_config, title="Experiment Configuration", border_style="bold magenta"))

    console.print("\n") 
    console.print(experiment_config)
    console.print("\n") 

def save_model(model, save_path, distributed: bool):
    if distributed and hasattr(model, '_orig_mod'):
        model_to_save = model._orig_mod.module.model
    elif distributed:
        model_to_save = model.module.model
    elif hasattr(model, '_orig_mod'):
        model_to_save = model._orig_mod.model
    else:
        model_to_save = model.model

    torch.save(model_to_save.state_dict(), save_path)
    print(f"Model saved to {save_path}")
