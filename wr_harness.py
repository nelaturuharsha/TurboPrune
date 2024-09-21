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
from utils.dataset import imagenet

## fastargs
from fastargs import get_current_config
from fastargs.decorators import param

import wandb

import schedulefree
from typing import List, Tuple

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224 / 256

from utils.airbench_loader import CifarLoader


from rich.console import Console
from rich.table import Table

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
                self.scheduler.step()
                
        if self.config["optimizer.scheduler_type"] == 'MultiStepLRWarmup':
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
    def train_one_level(self, training_type: str, num_cycles: int, level: int) -> None:
        """Train the model for one full level. This can be thought of as one full training run."""
        level_metrics = {
            "cycle_epoch": [],
            "train_loss": [],
            "test_loss": [],
            "train_acc": [],
            "test_acc": [],
        }
        summary_metrics = {
            "level": [],
            "sparsity": [],
            "max_test_acc": [],
            "final_test_acc": [],
            "training_schedule": []
        }

        # Initialize Rich table
        table = Table(title=f"Training Metrics for Level {level}")
        table.add_column("Cycle_Epoch", justify="center", style="cyan")
        table.add_column("Train Loss", justify="right", style="green")
        table.add_column("Test Loss", justify="right", style="red")
        table.add_column("Train Acc", justify="right", style="green")
        table.add_column("Test Acc", justify="right", style="red")

        console = Console()

        epoch_schedule = generate_budgeted_schedule()
        print(f'Epoch Schedule: {epoch_schedule}')

        for cycle in range(num_cycles):
            print(f'{"#" * 50}\nTraining Cycle {cycle} for {epoch_schedule[cycle]} epochs.\n{"#" * 50}')
            self.create_optimizers(epochs_per_level=epoch_schedule[cycle])
            
            if cycle == 0 and level == 0:
                torch.save(self.optimizer.state_dict(), os.path.join(self.expt_dir, "artifacts", "optimizer_init.pt"))
            if level != 0:
                self.optimizer = reset_optimizer(expt_dir=self.expt_dir, optimizer=self.optimizer, training_type=self.config["experiment_params.training_type"])
            
            for epoch in range(epoch_schedule[cycle]):
                self.epoch_counter += 1
                train_loss, train_acc = self.train_one_epoch(epoch)
                test_loss, test_acc = self.test()

                cycle_epoch = f'{cycle}_{epoch}'
                self._log_metrics(cycle_epoch, train_loss, test_loss, train_acc, test_acc, level_metrics)

                # Update Rich table
                table.add_row(
                    cycle_epoch,
                    f"{train_loss:.4f}",
                    f"{test_loss:.4f}",
                    f"{train_acc:.2f}%",
                    f"{test_acc:.2f}%"
                )
                console.clear()
                console.print(table)

            torch.save(self.model.state_dict(), os.path.join(self.expt_dir, "extended_checkpoints", f"extended_level_{level}_cycle_{cycle}.pt"))
            self._update_summary_metrics(level, summary_metrics, level_metrics, epoch_schedule)

        self._save_level_metrics(level, level_metrics)

    def _log_metrics(self, cycle_epoch, train_loss, test_loss, train_acc, test_acc, level_metrics):
        level_metrics['cycle_epoch'].append(cycle_epoch)
        level_metrics['train_loss'].append(train_loss)
        level_metrics['test_loss'].append(test_loss)
        level_metrics['train_acc'].append(train_acc)
        level_metrics['test_acc'].append(test_acc)
        
        wandb.log({
            "train_loss": train_loss, "test_loss": test_loss,
            "train_acc": train_acc, "test_acc": test_acc,
            "epoch": self.epoch_counter
        })

    def _save_level_metrics(self, level, level_metrics):
        pd.DataFrame(level_metrics).to_csv(
            os.path.join(
                self.expt_dir,
                "metrics",
                "epochwise_metrics",
                f"extended_level_{level}_metrics.csv",
            )
        )

    def _update_summary_metrics(self, level, summary_metrics, level_metrics, epoch_schedule):
        sparsity = print_sparsity_info(self.model, verbose=False)
        summary_metrics["level"].append(level)
        summary_metrics["sparsity"].append(round(sparsity, 4))
        summary_metrics["final_test_acc"].append(round(level_metrics["test_acc"][-1], 4))
        summary_metrics["max_test_acc"].append(round(max(level_metrics["test_acc"]), 4))
        epoch_schedule_str = '-'.join(map(str, epoch_schedule))
        summary_metrics['training_schedule'].append(epoch_schedule_str)

        wandb.log({
            "sparsity": round(sparsity, 4),
            "max_test_acc": round(max(level_metrics["test_acc"]), 4)
        })

        summary_path = os.path.join(self.expt_dir, "metrics", f"extended_level_{level}_{self.prefix}_summary.csv")
        self._save_or_update_summary(summary_path, summary_metrics)

    def _save_or_update_summary(self, summary_path, summary_metrics):
        if not os.path.exists(summary_path):
            pd.DataFrame(summary_metrics).to_csv(summary_path, index=False)
        else:
            pre_df = pd.read_csv(summary_path)
            new_df = pd.DataFrame(summary_metrics)
            updated_df = pd.concat([pre_df, new_df], ignore_index=True)
            updated_df.to_csv(summary_path, index=False)

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
    for resume_level in [6, 8]:
        config = initialize_config()

        wandb.init(project=config['wandb_params.project_name'])
        world_size = torch.cuda.device_count()
        from rich.console import Console
        from rich.panel import Panel
        from rich.columns import Columns

        level = resume_level

        prefix, expt_dir = gen_expt_dir()

        ## create a new folder for extended checkpoints
        expt_dir_extended = os.path.join(expt_dir, "extended_checkpoints")
        os.makedirs(expt_dir_extended, exist_ok=True)

        packaged = (prefix, expt_dir)
        save_config(expt_dir=expt_dir, config=config)

        console = Console()
        prune_harness = pruning_utils.PruningStuff()

        prune_harness.load_from_ckpt(
            os.path.join(expt_dir, "checkpoints", f"model_level_{level}.pt")
        )
        prune_harness.model = reset_weights(
            expt_dir=expt_dir,
            model=prune_harness.model,
            training_type=config["experiment_params.training_type"],
        )

        sparsity = print_sparsity_info(prune_harness.model, verbose=False)

        gpu_info = Panel(f"Training on {world_size} GPUs", title="GPU Info", border_style="blue")
        pruning_info = Panel(f"Model at level: {level}", title="Pruning Info", border_style="green")
        sparsity_info = Panel(f"Sparsity at the current level: {sparsity:.2%}", title="Sparsity Info", border_style="yellow")

        console.print(Columns([gpu_info, pruning_info, sparsity_info]))


        main(model = prune_harness.model, level=level, expt_dir=packaged)
        
        print(f"Training level {level} complete, moving on to {level+1}") 

        wandb.finish()
