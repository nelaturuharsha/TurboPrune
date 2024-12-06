import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
from torch.optim.optimizer import Optimizer


def get_multistep_warmup_scheduler(cfg: DictConfig, optimizer: Optimizer):
    """Creates a PyTorch SequentialLR scheduler combining warmup and multistep decay.
    Default schedule assumes total training epochs of 150.

    Args:
        cfg (DictConfig): OmegaConf config containing optimizer parameters
        optimizer (Optimizer): PyTorch optimizer

    Returns:
        torch.optim.lr_scheduler.SequentialLR: Sequential scheduler combining warmup and decay
    """
    warmup_steps = (
        cfg.optimizer_params.warmup_fraction * cfg.experiment_params.epochs_per_level
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )

    main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60, 120], gamma=0.1
    )

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps],
    )


def ImageNetLRDropsWarmup(
    cfg: DictConfig, optimizer: Optimizer
) -> torch.optim.lr_scheduler.SequentialLR:
    """Learning rate scheduler with warmup for ImageNet using SequentialLR.
       Combines linear warmup with linear decay.
       Assumes that the total number of epochs is 90.

    Args:
        cfg (DictConfig): OmegaConf config containing optimizer parameters
        optimizer (Optimizer): PyTorch optimizer

    Returns:
        torch.optim.lr_scheduler.SequentialLR: Sequential scheduler combining warmup and decay
    """

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=10
    )

    main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[40, 70], gamma=0.1
    )

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[10]
    )


def step_trapezoidal(it, lr, num_iterations, warmup_iters, warmdown_iters):
    # 1) linear warmup for warmup_iters steps
    assert it <= num_iterations
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    # 2) constant lr for a while
    elif it < num_iterations - warmdown_iters:
        return 1
    # 3) linear warmdown
    else:
        decay_ratio = (num_iterations - it) / warmdown_iters
        return decay_ratio


def TriangularSchedule(
    cfg: DictConfig,
    optimizer: Optimizer,
    steps_per_epoch: int,
    epochs_per_level: int = None,
):
    """Triangular learning rate schedule. Best performance with CIFAR10.
    credits: https://x.com/kellerjordan0/status/1776701859669172398

    Args:
        cfg (DictConfig): OmegaConf config containing optimizer parameters
        optimizer (Optimizer): PyTorch optimizer
        steps_per_epoch (int): Number of steps per epoch.
        epochs_so_far (int): Number of epochs so far.
    Returns:
        torch.optim.lr_scheduler.LambdaLR: Lambda learning rate scheduler.
    """
    epochs_per_level = (
        epochs_per_level
        if epochs_per_level is not None
        else cfg.experiment_params.epochs_per_level
    )
    total_train_steps = epochs_per_level * steps_per_epoch
    print(f"Epochs per level: {epochs_per_level}, steps per epoch: {steps_per_epoch}, total train steps: {total_train_steps}")


    lr_schedule = np.interp(
        np.arange(1 + total_train_steps),
        [
            0,
            int(cfg.optimizer_params.warmup_fraction * total_train_steps),
            total_train_steps,
        ],
        [0.2, 1, 0],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)

    return scheduler


def TrapezoidalSchedule(
    cfg: DictConfig,
    optimizer: Optimizer,
    steps_per_epoch: int,
    epochs_per_level: int = None,
):
    epochs_per_level = (
        epochs_per_level
        if epochs_per_level is not None
        else cfg.experiment_params.epochs_per_level
    )
    total_train_steps = epochs_per_level * steps_per_epoch
    lr_schedule = [
        step_trapezoidal(
            it,
            cfg.optimizer_params.lr,
            total_train_steps,
            warmup_iters=cfg.optimizer_params.trapezoidal_scheduler_stuff.warmup_steps,
            warmdown_iters=cfg.optimizer_params.trapezoidal_scheduler_stuff.cooldown_steps,
        )
        for it in range(1 + total_train_steps)
    ]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
    return scheduler
