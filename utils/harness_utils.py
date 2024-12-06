import os
import uuid
from datetime import datetime
import yaml
from typing import Any, Dict, Optional, Tuple

import numpy as np
import random
import pandas as pd

import torch

from omegaconf import DictConfig

from rich.console import Console, Group
from rich.panel import Panel
from rich.tree import Tree
from rich.layout import Layout
from rich.text import Text

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


def gen_expt_dir(cfg: DictConfig) -> Tuple[str, str]:
    """Create a new experiment directory and all the necessary subdirectories.
       If provided, instead of creating a new directory -- set the directory to the one provided.

    Args:
        cfg (DictConfig): Hydra config object.

    Returns:
        Tuple[str, str]: Prefix and path to the experiment directory.
    """
    base_dir = cfg.experiment_params.base_dir

    # Create prefix using f-string with config values
    prefix = (
        f"{cfg.experiment_params.dataset_name}"
        f"_model_{cfg.model_params.model_name}"
        f"_trainingtype_{cfg.pruning_params.training_type}"
        f"_prunemethod_{cfg.pruning_params.prune_method}"
        f"_target_{cfg.pruning_params.target_sparsity:.2f}"
        f"_seed_{cfg.experiment_params.seed}"
        f"_budget_{cfg.experiment_params.epochs_per_level}epochs"
        f"_cycles_{cfg.cyclic_training.num_cycles}"
        f"_strat_{cfg.cyclic_training.strategy}"
        f"_lr_{cfg.optimizer_params.lr:.3f}"
        f"_mom_{cfg.optimizer_params.momentum:.1f}"
        f"_wd_{cfg.optimizer_params.weight_decay:.4f}"
        f"_sched_{cfg.optimizer_params.scheduler_type}"
    )

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    unique_name = f"{prefix}__{unique_id}__{current_time}"
    expt_dir = os.path.join(base_dir, unique_name)
    print(f"Creating this Folder {expt_dir} :)")

    os.makedirs(expt_dir, exist_ok=True)
    for subdir in ["checkpoints", "metrics", "metrics/level_wise_metrics", "artifacts"]:
        os.makedirs(os.path.join(expt_dir, subdir), exist_ok=True)

    return prefix, expt_dir


def set_seed(cfg: DictConfig, is_deterministic: bool = False) -> None:
    """Set the random seed for reproducibility.

    Args:
        cfg (DictConfig): Hydra config object.
        is_deterministic (bool, optional): Whether to set deterministic behavior. Default is False.
    """
    seed = cfg.experiment_params.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if is_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def generate_densities(cfg: DictConfig, current_sparsity: float) -> list[float]:
    """Generate a list of densities for pruning. The density is calculated as
       (1 - prune_rate) ^ i multiplied by current_sparsity until target_sparsity is reached.
    Args:
        cfg (DictConfig): Hydra config object.
        current_sparsity (float): The current density (1 - current_sparsity).
    Returns:
        list[float]: List of densities until target sparsity is reached.
    """
    prune_method = cfg.pruning_params.prune_method
    target_sparsity = cfg.pruning_params.target_sparsity
    prune_rate = cfg.pruning_params.prune_rate

    if prune_method in ["mag", "random_erk", "random_balanced"]:
        densities = []
        current_density = 1 - current_sparsity
        target_density = 1 - target_sparsity
        while current_density > target_density:
            densities.append(current_density)
            current_density *= 1 - prune_rate
        if current_density <= target_density:
            densities.append(current_density)
        return densities
    elif prune_method in ["er_erk", "er_balanced", "synflow", "snip"]:
        return [1 - target_sparsity]
    elif prune_method == "just dont":
        return [1.0]
    else:
        raise ValueError(f"Unknown pruning method: {prune_method}")


def save_config(expt_dir: str, cfg: DictConfig) -> None:
    """Save the experiment configuration to a YAML file in the experiment directory.

    Args:
        expt_dir (str): Directory of the experiment.
        config (DictConfig): Configuration to save.
    """
    with open(os.path.join(expt_dir, "expt_config.yaml"), "w") as file:
        yaml.dump(cfg, file, default_flow_style=False)


def generate_cyclical_schedule(cfg: DictConfig):
    """
    Generates a schedule of epochs per cycle based on the given strategy and total epoch budget.

    Parameters:
    - cfg (DictConfig): Hydra config object.
    Returns:
    - List[int]: A list of epochs for each cycle.
    """
    epochs_per_level = cfg.experiment_params.epochs_per_level
    num_cycles = cfg.cyclic_training.num_cycles
    strategy = cfg.cyclic_training.strategy

    if num_cycles > 1:
        if strategy == "linear_decrease":
            step = epochs_per_level / (num_cycles * (num_cycles + 1) / 2)
            epochs = [int(step * (num_cycles - i)) for i in range(num_cycles)]

        elif strategy == "linear_increase":
            step = epochs_per_level / (num_cycles * (num_cycles + 1) / 2)
            epochs = [int(step * (i + 1)) for i in range(num_cycles)]

        elif strategy == "exponential_decrease":
            factor = 0.5 ** (1 / (num_cycles - 1))
            total_factor = sum(factor**i for i in range(num_cycles))
            epochs = [
                int(epochs_per_level * (factor**i) / total_factor)
                for i in range(num_cycles)
            ]

        elif strategy == "exponential_increase":
            factor = 2 ** (1 / (num_cycles - 1))
            total_factor = sum(factor**i for i in range(num_cycles))
            epochs = [
                int(epochs_per_level * (factor**i) / total_factor)
                for i in range(num_cycles)
            ]

        elif strategy == "cyclic_peak":
            mid_point = num_cycles // 2
            increase_step = epochs_per_level / (mid_point * (mid_point + 1) / 2)
            decrease_step = epochs_per_level / (
                (num_cycles - mid_point) * (num_cycles - mid_point + 1) / 2
            )
            epochs = [int(increase_step * (i + 1)) for i in range(mid_point)]
            epochs += [
                int(decrease_step * (num_cycles - i))
                for i in range(mid_point, num_cycles)
            ]

        elif strategy == "alternating":
            high = epochs_per_level // (num_cycles // 2 + num_cycles % 2)
            low = epochs_per_level // (2 * (num_cycles // 2 + num_cycles % 2))
            epochs = [high if i % 2 == 0 else low for i in range(num_cycles)]

        elif strategy == "plateau":
            increase_cycles = num_cycles // 2
            plateau_cycles = num_cycles - increase_cycles
            increase_step = epochs_per_level / (
                increase_cycles * (increase_cycles + 1) / 2
            )
            epochs = [int(increase_step * (i + 1)) for i in range(increase_cycles)]
            epochs += [epochs_per_level // num_cycles for _ in range(plateau_cycles)]

        elif strategy == "constant":
            epochs = [epochs_per_level // num_cycles for _ in range(num_cycles)]
    else:
        epochs = [epochs_per_level]

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


def save_metrics_and_update_summary(
    console, model, expt_dir, prefix, level, level_metrics, num_cycles, epoch_schedule
):
    metrics_path = os.path.join(
        expt_dir, "metrics", "epochwise_metrics", f"level_{level}_metrics.csv"
    )
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
        "epoch_schedule": ["-".join(map(str, epoch_schedule))],
    }
    df = pd.DataFrame(new_data)
    if os.path.exists(summary_path):
        df.to_csv(summary_path, mode="a", header=False, index=False)
    else:
        df.to_csv(summary_path, index=False)
    console.log(f"[bold cyan]Updated overall summary for level {level}[/bold cyan]")

    wandb.log(
        {
            "sparsity": new_data["sparsity"][0],
            "max_test_acc": new_data["max_test_acc"][0],
        }
    )


def display_training_info(cycle_info, training_info, optimizer_info):
    console = Console()

    def create_wrapped_tree(title, info_dict):
        tree = Tree(title)
        for key, value in info_dict.items():
            if "expt_dir" in key:
                value = os.path.basename(value)
            text = Text.from_markup(f"[bold cyan]{key.capitalize()}:[/bold cyan] ")
            text.append(str(value), style="yellow")
            tree.add(Group(text))
        return tree

    hardware_tree = create_wrapped_tree("Training Harness Configuration", training_info)
    hardware_panel = Panel(
        hardware_tree, title="Training Configuration", border_style="cyan"
    )

    schedule_tree = Tree("Training Schedule")
    schedule_tree.add(
        Group(
            Text.from_markup(
                f"[bold cyan]Number of Cycles:[/bold cyan] [yellow]{cycle_info['Number of Cycles']}[/yellow]"
            )
        )
    )
    epochs_text = Text.from_markup(f"[bold cyan]Epochs per Cycle:[/bold cyan] ")
    epochs_text.append(
        ", ".join(map(str, cycle_info["Epochs per Cycle"])), style="yellow"
    )
    schedule_tree.add(Group(epochs_text))
    schedule_tree.add(
        Group(
            Text.from_markup(
                f"[bold cyan]Total Training Length:[/bold cyan] [yellow]{cycle_info['Total Training Length']}[/yellow]"
            )
        )
    )
    schedule_panel = Panel(
        schedule_tree, title="Overall Training Schedule", border_style="cyan"
    )

    cycle_tree = Tree(f"Training Cycle {cycle_info['Training Cycle']}")
    cycle_tree.add(
        Group(
            Text.from_markup(
                f"[bold cyan]Epochs this cycle:[/bold cyan] [yellow]{cycle_info['Epochs this cycle']}[/yellow]"
            )
        )
    )
    cycle_tree.add(
        Group(
            Text.from_markup(
                f"[bold cyan]Total epochs so far:[/bold cyan] [yellow]{cycle_info['Total epochs so far']}[/yellow]"
            )
        )
    )
    cycle_tree.add(
        Group(
            Text.from_markup(
                f"[bold cyan]Current Sparsity:[/bold cyan] [yellow]{cycle_info['Current Sparsity']}[/yellow]"
            )
        )
    )
    cycle_panel = Panel(
        cycle_tree, title="Current Cycle Information", border_style="cyan"
    )

    optimizer_tree = create_wrapped_tree("Optimizer Configuration", optimizer_info)
    optimizer_panel = Panel(
        optimizer_tree, title="Optimizer Details", border_style="cyan"
    )

    experiment_config = Layout()
    experiment_config.split(
        Layout(hardware_panel, name="hardware", ratio=1),
        Layout(schedule_panel, name="schedule", ratio=1),
        Layout(cycle_panel, name="cycle", ratio=1),
        Layout(optimizer_panel, name="optimizer", ratio=1),
    )
    experiment_config.update(
        Panel(
            experiment_config,
            title="Experiment Configuration",
            border_style="bold magenta",
        )
    )


def save_model(model, save_path, distributed: bool):
    if distributed and hasattr(model, "_orig_mod"):
        model_to_save = model._orig_mod.module.model
    elif distributed:
        model_to_save = model.module.model
    elif hasattr(model, "_orig_mod"):
        model_to_save = model._orig_mod.model
    else:
        model_to_save = model.model

    torch.save(model_to_save.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def resume_experiment(cfg: DictConfig, expt_dir: str):
    resume_level = cfg.experiment_params.resume_experiment_stuff.resume_level
    resume_expt_name = cfg.experiment_params.resume_experiment_stuff.resume_expt_name
    training_type = cfg.pruning_params.training_type

    if resume_level != 0 and resume_expt_name:
        expt_dir = os.path.join(expt_dir, resume_expt_name)
        prefix = os.path.basename(expt_dir)
        print(f"Resuming from Level -- {resume_level}")

        if training_type in ["imp", "wr", "lrr"]:
            checkpoint_path = os.path.join(
                expt_dir, "checkpoints", f"model_level_{resume_level-1}.pt"
            )
            assert os.path.exists(
                checkpoint_path
            ), f"Previous level checkpoint not found at {checkpoint_path}"

    return prefix, expt_dir
