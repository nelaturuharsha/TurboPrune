import numpy as np
import fastargs
from fastargs import get_current_config
from fastargs.decorators import param
import math
import torch
from torch.optim.optimizer import Optimizer

get_current_config()


def _warmup_lr(base_lr: float, warmup_length: int, epoch: int) -> float:
    """Calculate the learning rate during warmup.

    Args:
        base_lr (float): Base learning rate.
        warmup_length (int): Number of warmup epochs.
        epoch (int): Current epoch.

    Returns:
        float: Adjusted learning rate for the current epoch.
    """
    return base_lr * (epoch + 1) / warmup_length


class LRScheduler:
    """Base class for learning rate schedulers defined here.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        last_epoch (int, optional): The index of last epoch. Default is -1.
    """

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        self.optimizer = optimizer
        self.last_epoch = last_epoch

    def step(self) -> None:
        """Update the learning rate."""
        self.last_epoch += 1
        new_lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

    def get_lr(self) -> float:
        """Compute the learning rate for the current epoch.

        Returns:
            float: Learning rate for the current epoch.
        """
        raise NotImplementedError


class CosineLRWarmup(LRScheduler):
    """Cosine learning rate scheduler with warmup.

    Args:
        epochs_per_level (int): Number of epochs per level.
        warmup_steps (int): Number of warmup epochs.
        lr (float): Initial learning rate.
        lr_min (float): Minimum learning rate.
        optimizer (Optimizer): Wrapped optimizer.
        last_epoch (int, optional): The index of last epoch. Default is -1.
    """

    @param("experiment_params.epochs_per_level")
    @param("optimizer.warmup_steps")
    @param("optimizer.lr")
    @param("optimizer.lr_min")
    def __init__(
        self,
        epochs_per_level: int,
        warmup_steps: int,
        lr: float,
        lr_min: float,
        optimizer: Optimizer,
        last_epoch: int = -1,
    ) -> None:
        super().__init__(optimizer, last_epoch)
        self.epochs_per_level = epochs_per_level
        self.warmup_steps = warmup_steps
        self.lr = lr
        self.lr_min = lr_min

    def get_lr(self) -> float:
        """Compute the learning rate for the current epoch.

        Returns:
            float: Learning rate for the current epoch.
        """
        if self.last_epoch < self.warmup_steps:
            return _warmup_lr(self.lr, self.warmup_steps, self.last_epoch)
        else:
            adjusted_epoch = self.last_epoch - self.warmup_steps
            total_adjusted_epochs = self.epochs_per_level - self.warmup_steps
            return self.lr_min + 0.5 * (
                1 + np.cos(np.pi * adjusted_epoch / total_adjusted_epochs)
            ) * (self.lr - self.lr_min)


class MultiStepLRWarmup(LRScheduler):
    """Step learning rate scheduler with warmup, for CIFAR datasets.
       Default schedule assumes total training epochs of 150.

    Args:
        lr (float): Initial learning rate.
        warmup_steps (int): Number of warmup epochs.
        optimizer (Optimizer): Wrapped optimizer.
        last_epoch (int, optional): The index of last epoch. Default is -1.
    """

    @param("optimizer.lr")
    @param("optimizer.warmup_steps")
    def __init__(
        self, lr: float, warmup_steps: int, optimizer: Optimizer, last_epoch: int = -1
    ) -> None:
        super().__init__(optimizer, last_epoch)
        self.lr = lr
        self.warmup_steps = warmup_steps

    def get_lr(self) -> float:
        """Compute the learning rate for the current epoch.

        Returns:
            float: Learning rate for the current epoch.
        """
        if self.last_epoch < self.warmup_steps:
            return _warmup_lr(self.lr, self.warmup_steps, self.last_epoch)
        else:
            return self.lr * (0.1 ** ((self.last_epoch - self.warmup_steps) // 60))

class ImageNetLRDropsWarmup(LRScheduler):
    """Step learning rate scheduler with warmup for ImageNet.
       Assumes that the total number of epochs is 90.

    Args:
        lr (float): Initial learning rate.
        optimizer (Optimizer): Wrapped optimizer.
        last_epoch (int, optional): The index of last epoch. Default is -1.
    """

    @param("optimizer.lr")
    def __init__(self, lr: float, optimizer: Optimizer, last_epoch: int = -1) -> None:
        super().__init__(optimizer, last_epoch)
        self.lr = lr

    def get_lr(self) -> float:
        """Compute the learning rate for the current epoch.

        Returns:
            float: Learning rate for the current epoch.
        """
        if self.last_epoch < 10:
            return self.lr * 0.1
        elif 10 <= self.last_epoch < 40:
            return self.lr
        elif 40 <= self.last_epoch < 70:
            return 0.1 * self.lr
        else:
            return (0.1**2) * self.lr




def step_trapezoidal(it, lr, num_iterations, warmup_iters, warmdown_iters):
    # 1) linear warmup for warmup_iters steps
    assert it <= num_iterations
    if it < warmup_iters:
        return (it+1) / warmup_iters
    # 2) constant lr for a while
    elif it < num_iterations - warmdown_iters:
        return 1
    # 3) linear warmdown
    else:
        decay_ratio = (num_iterations - it) / warmdown_iters
        return decay_ratio

@param("experiment_params.epochs_per_level")
@param('optimizer.lr')
def TriangularSchedule(epochs_per_level, lr, optimizer, steps_per_epoch):
    total_train_steps = epochs_per_level * steps_per_epoch
    lr_schedule = np.interp(np.arange(1+total_train_steps), [0, int(lr * total_train_steps), total_train_steps], [lr, 1, 0]) 
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
    return scheduler

@param("experiment_params.epochs_per_level")
@param("optimizer.warmup_steps")
@param("optimizer.cooldown_steps")
@param("optimizer.lr")
def TrapezoidalSchedule(epochs_per_level, warmup_steps, cooldown_steps, lr, optimizer, steps_per_epoch):
    total_train_steps = epochs_per_level * steps_per_epoch
    lr_schedule = [step_trapezoidal(it, lr, total_train_steps, warmup_iters=warmup_steps, warmdown_iters=cooldown_steps) for it in range(1+total_train_steps)]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
    return scheduler
