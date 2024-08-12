import torch
import os
import prettytable
from utils.conv_type import ConvMask, Conv1dMask
from datetime import datetime
import uuid

import numpy as np
import random
import yaml

from fastargs.decorators import param
from typing import Any, Dict, Optional, Tuple 


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
        original_dict = torch.load(
            os.path.join(expt_dir, "checkpoints", "model_init.pt")
        )
    elif training_type == "wr":
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
        if isinstance(layer, (ConvMask, Conv1dMask)):
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
    if resume_level != 0 and resume_expt_name:
        expt_dir = os.path.join(base_dir, resume_expt_name)
        print(f"Resuming from Level -- {resume_level}")
    elif resume_level == 0 and resume_expt_name is None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:6]
        unique_name = f"experiment_{current_time}_{unique_id}"
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

    return expt_dir


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
@param("prune_params.num_levels")
@param("prune_params.prune_rate")
def generate_densities(prune_method: str, num_levels: int, prune_rate: float
) -> list[float]:
    """Generate a list of densities for pruning. The density is calculated as (1 - prune_rate) ^ i where i is the sparsity level.
       For example, if prune_rate = 0.2 and the num_levels = 5, the densities will be [1.0, 0.8, 0.64, 0.512, 0.4096].
    Args:
        prune_method (str): Method of pruning.
        num_levels (int): Number of pruning levels.
        prune_rate (float): Rate of pruning.

    Returns:
        list[float]: List of densities for each level.
    """
    
    densities = [(1 - prune_rate) ** i for i in range(num_levels)]
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
