## pythonic imports
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from prettytable import PrettyTable
import tqdm
import itertools

## torch
import torch
import torch.nn as nn
from torch import optim
from torch.cuda.amp import autocast

## file-based imports
import utils.schedulers as schedulers
import utils.pruning_utils as pruning_utils
from utils.harness_utils import *
from utils.metric_utils import *

## fastargs
from fastargs import get_current_config
from fastargs.decorators import param

from utils.dataset import imagenet

import wandb
wandb.require("core")

import schedulefree
from typing import List, Tuple

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224 / 256

from utils.airbench_loader import CifarLoader

torch._dynamo.config.guard_nn_modules=True

class Harness:
    """Harness class to handle training and evaluation.

    Args:
        gpu_id (int): current rank of the process while training using DDP.
        expt_dir (str): Experiment directory to save artifacts/checkpoints/metrics to.
        model (Optional[nn.Module], optional): The model to train.
    """

    def __init__(self, gpu_id: int, expt_dir: Tuple[str, str], model: nn.Module) -> None:
        self.config = get_current_config()
        self.gpu_id = gpu_id
        self.this_device = f"cuda:{self.gpu_id}"
        self.criterion = nn.CrossEntropyLoss()

        if "CIFAR" not in self.config["dataset.dataset_name"]:
            self.loaders = imagenet(distributed=False, this_device=self.this_device)
            self.train_loader = self.loaders.train_loader
            self.test_loader = self.loaders.test_loader
        else:
            self.train_loader = CifarLoader(path='./cifar10', batch_size=512, train=True, aug={'flip' : True, 'translate' : 2}, altflip=True, dataset=self.config['dataset.dataset_name'])
            self.test_loader = CifarLoader(path='./cifar10', batch_size=512, train=False, dataset=self.config['dataset.dataset_name'])

        self.model = model.to(self.this_device)
        #self.model = torch.compile(self.model, mode='reduce-overhead')
        self.prefix, self.expt_dir = expt_dir

        self.epoch_counter = 0
    
    @param("optimizer.lr")
    @param("optimizer.momentum")
    @param("optimizer.weight_decay")
    @param("optimizer.scheduler_type")
    def create_optimizers(
        self, lr: float, momentum: float, weight_decay: float, scheduler_type: str, epochs_per_level: int
    ) -> None:
        """Instantiate the optimizer and learning rate scheduler.

        Args:
            lr (float): Initial learning rate.
            momentum (float): Momentum for SGD.
            weight_decay (float): Weight decay for optimizer.
            scheduler_type (str): Type of scheduler.
        """
        if scheduler_type =='ScheduleFree':
            self.optimizer = schedulefree.SGDScheduleFree(
                self.model.parameters(),
                warmup_steps=self.config['optimizer.warmup_steps'],
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
            self.scheduler = None
            print('using schedule free')
        
        else:
            self.optimizer = optim.SGD(
                    self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
                )
            if scheduler_type != 'OneCycleLR':
                scheduler = getattr(schedulers, scheduler_type)            
                
                if scheduler_type == 'TriangularSchedule':
                    self.scheduler = scheduler(optimizer=self.optimizer, steps_per_epoch=len(self.train_loader), epochs_per_level=epochs_per_level)
                elif scheduler_type == 'TrapezoidalSchedule':
                    self.scheduler = scheduler(optimizer=self.optimizer, steps_per_epoch=len(self.train_loader),
                                                warmup_steps=len(self.train_loader) * self.config['optimizer.warmup_steps'],
                                                cooldown_steps=len(self.train_loader) * self.config['optimizer.cooldown_steps'])
                elif scheduler_type == 'MultiStepLRWarmup':
                    self.scheduler = scheduler(optimizer=self.optimizer)
            else:
                self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                              max_lr=lr,
                                                              epochs=epochs_per_level,
                                                              steps_per_epoch=len(self.train_loader))
                

    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train the model for one epoch.

        Args:
            epoch (int): Current epoch.
        Returns:
            (float, float): Training loss and accuracy.
        """        
        model = self.model
        model.train()
        if self.scheduler is None:
            self.optimizer.train()
        train_loss = 0
        correct = 0
        total = 0
        tepoch = tqdm.tqdm(self.train_loader, unit="batch", desc=f"Epoch {epoch}")

        for inputs, targets in tepoch:
            if "CIFAR" in self.config["dataset.dataset_name"]:
                inputs, targets = inputs.to(self.this_device), targets.to(
                    self.this_device
                )

            self.optimizer.zero_grad()
            with autocast(dtype=torch.bfloat16, device_type='cuda'):
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (self.scheduler is not None) and (self.config['optimizer.scheduler_type'] != 'MultiStepLRWarmup'):
                #print('Running Trapezoidal/Triangular/OneCycle')
                self.scheduler.step()
        if self.config["optimizer.scheduler_type"] == 'MultiStepLRWarmup':
            #print('Step LR')
            self.scheduler.step()
        
        train_loss /= len(self.train_loader)
        accuracy = 100.0 * (correct / total)
        return train_loss, accuracy

    def test(self) -> Tuple[float, float]:
        """Evaluate the model on the test set.

        Returns:
            (float, float): Test loss and accuracy.
        """
        model = self.model

        if self.scheduler is None:
            model.train()
            self.optimizer.eval()
            for batch in itertools.islice(self.train_loader, 50): 
                with autocast(dtype=torch.bfloat16, device_type='cuda'):
                    model(batch[0].cuda())

        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        tloader = tqdm.tqdm(self.test_loader, desc="Testing")
        with torch.no_grad():
            for inputs, targets in tloader:
                if "CIFAR" in self.config["dataset.dataset_name"]:
                    inputs, targets = inputs.to(self.this_device), targets.to(
                        self.this_device
                    )
                with autocast(dtype=torch.bfloat16, device_type='cuda'):
                    outputs = model(inputs)
                    loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_loss /= len(self.test_loader)
        accuracy = 100.0 * correct / total

        return test_loss, accuracy

    @param("experiment_params.training_type")
    @param("cyclic_training.num_cycles")
    def train_one_level(self, training_type: str, num_cycles: int, level: int, single_cycle_length = None) -> None:
        metrics = {k: [] for k in ["cycle_epoch", "train_loss", "test_loss", "train_acc", "test_acc"]}
        if single_cycle_length == None:
            epoch_schedule = generate_budgeted_schedule()
        else:
            epoch_schedule = [single_cycle_length]
        print(f'Epoch Schedule: {epoch_schedule}')

        for cycle in range(num_cycles):
            print(f'{"#" * 50}\nTraining Cycle {cycle} for {epoch_schedule[cycle]} epochs.\n{"#" * 50}')
            self.create_optimizers(epochs_per_level=epoch_schedule[cycle])
            
            if cycle == 0 and level == 0:
                torch.save(self.optimizer.state_dict(), os.path.join(self.expt_dir, "artifacts", "optimizer_init.pt"))
            elif level != 0:
                self.optimizer = reset_optimizer(self.expt_dir, self.optimizer, self.config["experiment_params.training_type"])

            for epoch in range(epoch_schedule[cycle]):
                self.epoch_counter += 1
                train_loss, train_acc = self.train_one_epoch(epoch)
                test_loss, test_acc = self.test()
                
                if self.gpu_id == 0:
                    self._log_metrics(cycle, epoch, train_loss, test_loss, train_acc, test_acc, metrics)
                    self._save_rewind_checkpoint(level, training_type, epoch)
            if self.gpu_id == 0:
                self._save_cycle_data(cycle, metrics)
        if self.gpu_id == 0:
            self._save_level_data(level, metrics)


    def _log_metrics(self, cycle, epoch, train_loss, test_loss, train_acc, test_acc, metrics):
        for k, v in zip(metrics.keys(), [f'{cycle}_{epoch}', train_loss, test_loss, train_acc, test_acc]):
            metrics[k].append(v)
        wandb.log({
            "train_loss": train_loss, "test_loss": test_loss,
            "train_acc": train_acc, "test_acc": test_acc,
            "epoch": self.epoch_counter
        })

    def _save_rewind_checkpoint(self, level, training_type, epoch):
        if level == 0 and training_type == "wr" and epoch == 10:
            for name, obj in [("model", self.model), ("optimizer", self.optimizer)]:
                torch.save(obj.state_dict(), os.path.join(self.expt_dir, f"{name}_rewind.pt"))

    def _save_cycle_data(self, cycle, metrics):
        torch.save(self.model.state_dict(), os.path.join(self.expt_dir, "checkpoints", f"model_cycle_{cycle}.pt"))
        pd.DataFrame(metrics).to_csv(os.path.join(self.expt_dir, "metrics", f"cycle_{cycle}_metrics.csv"))
        self._update_summary(cycle, metrics)

    def _update_summary(self, cycle, metrics):
        summary_path = os.path.join(self.expt_dir, "metrics", f"{self.prefix}_summary.csv")
        sparsity = print_sparsity_info(self.model, verbose=False)
        new_data = {
            "level": [cycle], "sparsity": [round(sparsity, 4)],
            "final_test_acc": [round(metrics["test_acc"][-1], 4)],
            "max_test_acc": [round(max(metrics["test_acc"]), 4)],
            "training_schedule": ['-'.join(map(str, generate_budgeted_schedule()))]
        }
        if not os.path.exists(summary_path):
            pd.DataFrame(new_data).to_csv(summary_path, index=False)
        else:
            pd.concat([pd.read_csv(summary_path), pd.DataFrame(new_data)], ignore_index=True).to_csv(summary_path, index=False)

    def _save_level_data(self, level, metrics):
        pd.DataFrame(metrics).to_csv(os.path.join(self.expt_dir, "metrics", "epochwise_metrics", f"level_{level}_metrics.csv"))
        sparsity = print_sparsity_info(self.model, verbose=False)
        wandb.log({
            "sparsity": round(sparsity, 4),
            "max_test_acc": round(max(metrics["test_acc"]), 4)
        })

def main(model: nn.Module, level: int, expt_dir: str) -> None:
    """Main function for distributed training.

    Args:
        rank (int): Rank of the current process.
        model (nn.Module): The model to train.
        level (int): Current sparsity level.
        expt_dir (str): Experiment directory to save artifacts/checkpoints/metrics to.
    """
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode="stderr")
    set_seed()

    harness = Harness(model=model, expt_dir=expt_dir, gpu_id=0)

    if (level == 0):
        torch.save(
            harness.model.state_dict(),
            os.path.join(expt_dir[1], "checkpoints", "model_init.pt"),
        )

    harness.train_one_level(level=level)

    ckpt = os.path.join(expt_dir[1], "checkpoints", f"model_level_{level}.pt")
    torch.save(harness.model.state_dict(), ckpt)

def initialize_config():
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode="stderr")
    config.summary()
    return config

if __name__ == "__main__":
    config = initialize_config()
    wandb.login(key='9a942da6eaf97ac7a754c2b1c1a3c0436f0d0df2')
    wandb.init(project=config['wandb_params.project_name'])
    world_size = torch.cuda.device_count()
    print(f"Training on {world_size} GPUs")
    perturbation_table = PrettyTable()
    parser = ArgumentParser()

    prune_harness = pruning_utils.PruningStuff()

    #### STEP ONE: Prune at init #### 
    prune_harness.prune_at_initialization()
    prefix, expt_dir = gen_expt_dir()

    packaged = (prefix, expt_dir)
    save_config(expt_dir=expt_dir, config=config)
    
    #### STEP TWO: Cyclic Training ####
    print_sparsity_info(prune_harness.model, verbose=False)
    main(model = prune_harness.model, level=0, expt_dir=packaged)

    #### STEP THREE: Magnitude Pruning to Target ####
    prune_harness.level_pruner(density=float(config['prune_params.prune_rate']))

    level = 100

    harness = Harness(model=prune_harness.model, expt_dir=packaged, gpu_id=0)

    harness.train_one_level(num_cycles=1, level=level, single_cycle_length=90)
    ckpt = os.path.join(expt_dir, "checkpoints", f"model_level_{level}.pt")
    torch.save(harness.model.state_dict(), ckpt)

    wandb.finish()
