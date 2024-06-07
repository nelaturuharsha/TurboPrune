import numpy as np
import fastargs
from fastargs import get_current_config
from fastargs.decorators import param
import math

import torch

get_current_config()

# Helper function for warmup
def _warmup_lr(base_lr, warmup_length, epoch):
    return base_lr * (epoch + 1) / warmup_length

# Base class for all LR schedulers
class LRScheduler:
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch

    def step(self):
        self.last_epoch += 1
        new_lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

    def get_lr(self):
        raise NotImplementedError

class CosineLRWarmup(LRScheduler):
    @param('experiment_params.epochs_per_level')
    @param('optimizer.warmup_epochs')
    @param('optimizer.lr')
    @param('optimizer.lr_min')
    def __init__(self, epochs_per_level, warmup_epochs, lr, lr_min, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch)
        self.epochs_per_level = epochs_per_level
        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.lr_min = lr_min

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return _warmup_lr(self.lr, self.warmup_epochs, self.last_epoch)
        else:
            # Adjust the epoch count so that the cosine annealing phase starts right after the warmup
            adjusted_epoch = self.last_epoch - self.warmup_epochs
            total_adjusted_epochs = self.epochs_per_level - self.warmup_epochs
            return self.lr_min + 0.5 * (1 + np.cos(np.pi * adjusted_epoch / total_adjusted_epochs)) * (self.lr - self.lr_min)

# MultiStep LR with Warmup Scheduler
class MultiStepLRWarmup(LRScheduler):
    @param('optimizer.lr')
    @param('optimizer.warmup_epochs')
    def __init__(self, lr, warmup_epochs, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch)
        self.lr = lr
        self.warmup_epochs = warmup_epochs

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return _warmup_lr(self.lr, self.warmup_epochs, self.last_epoch)
        else:
            return self.lr * (0.1 ** ((self.last_epoch - self.warmup_epochs) // 60))

# ImageNet LR Drops with Warmup Scheduler
class ImageNetLRDropsWarmup(LRScheduler):
    @param('optimizer.lr')
    def __init__(self,  lr, optimizer,last_epoch=-1):
        super().__init__(optimizer, last_epoch)
        self.lr = lr

    def get_lr(self):
        if self.last_epoch < 10:
            return self.lr * 0.1
        elif 10 <= self.last_epoch < 40:
            return self.lr
        elif 40 <= self.last_epoch < 70:
            return 0.1 * self.lr
        else:
            return (0.1 ** 2) * self.lr

class CustomLRScheduler:
    def __init__(self, optimizer, warmup_length):
        self.optimizer = optimizer
        self.warmup_length = warmup_length
        self.epoch = 0

    def get_step_lr(self, epoch, lr):
        if epoch < self.warmup_length:
            calc_lr = lr * (epoch + 1) / self.warmup_length
        elif (epoch >= 10 and epoch < 60):
            calc_lr = lr
        elif (epoch >= 60) and (epoch < 120):
            calc_lr = lr / 10
        else:
            calc_lr = lr / 100

        return calc_lr

    def step(self):
        self.epoch += 1
        lr = self.get_step_lr(self.epoch, lr=0.1 * math.sqrt(4))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

@param('experiment_params.epochs_per_level')
def TriangularSchedule(optimizer, epochs_per_level, steps_per_epoch):
    total_train_steps = steps_per_epoch * epochs_per_level
    lr_schedule = np.interp(np.arange(1+total_train_steps),
        [0, int(0.2 * total_train_steps), total_train_steps],
        [0.2, 1, 0]) # Triangular learning rate schedule
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)

    return scheduler