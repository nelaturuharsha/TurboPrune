## pythonic imports
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from prettytable import PrettyTable
import tqdm
from typing import Tuple

## torch
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim
import torch.multiprocessing as mp
from torch.amp import autocast

## file-based imports
import utils.schedulers as schedulers
import utils.pruning_utils as pruning_utils
from utils.harness_utils import *
from utils.dataset import CIFARLoader, imagenet

## fastargs
from fastargs import get_current_config
from fastargs.decorators import param
import schedulefree

import wandb

from rich.console import Console
from rich.table import Table

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

wandb.login(key='9a942da6eaf97ac7a754c2b1c1a3c0436f0d0df2')

class Harness:
    """Harness class to handle training and evaluation.

    Args:
        gpu_id (int): current rank of the process while training using DDP.
        expt_dir (str): Experiment directory to save artifacts/checkpoints/metrics to.
        model (Optional[nn.Module], optional): The model to trai.
    """

    def __init__(self, gpu_id: int, expt_dir: str, model: nn.Module) -> None:
        self.config = get_current_config()
        self.gpu_id = gpu_id
        self.this_device = f"cuda:{self.gpu_id}"
        self.criterion = nn.CrossEntropyLoss()

        if "CIFAR" not in self.config["dataset.dataset_name"]:
            self.loaders = imagenet(this_device=self.this_device, distributed=True)
        else:
            self.loaders = CIFARLoader(distributed=True)

        self.train_loader = self.loaders.train_loader
        self.test_loader = self.loaders.test_loader

        model = model.to(self.this_device)
        self.model = DDP(model, device_ids=[self.gpu_id])
        self.prefix, self.expt_dir = expt_dir

        self.precision, self.use_amp = self.get_dtype_amp()

        self.epoch_counter = 0

    @param('experiment_params.training_precision')
    def get_dtype_amp(self, training_precision):
        dtype_map = {
            'bfloat16': (torch.bfloat16, True),
            'float16': (torch.float16, True),
            'float32': (torch.float32, False)
        }
        return dtype_map.get(training_precision, (torch.float32, False))
    
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
        if "CIFAR" in self.config["dataset.dataset_name"]:
            self.loaders.train_sampler.set_epoch(epoch)
        model = self.model
        model.train()
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
            with autocast('cuda',dtype=self.precision, enabled = self.use_amp):
                outputs = model(inputs.contiguous())
                loss = self.criterion(outputs, targets)
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
                with autocast('cuda', dtype=self.precision, enabled = self.use_amp):
                    outputs = model(inputs.contiguous())
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

        level_metrics = {'cycle': [], 'epoch': [], 'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
        summary_metrics = {'level': [], 'sparsity': [], 'final_test_acc': [], 'max_test_acc': [], 'training_schedule': []}

        # Initialize Rich table
        table = Table(title=f"Training Metrics for Level {level}")
        table.add_column("Cycle", justify="center", style="cyan")
        table.add_column("Epoch", justify="center", style="magenta")
        table.add_column("Train Loss", justify="right", style="green")
        table.add_column("Test Loss", justify="right", style="red")
        table.add_column("Train Acc", justify="right", style="green")
        table.add_column("Test Acc", justify="right", style="red")

        console = Console()

        if single_cycle_length == None:
            epoch_schedule = generate_budgeted_schedule()
        else:
            epoch_schedule = [single_cycle_length]
            num_cycles = 1
        if self.gpu_id == 0:
            print(f'Epoch Schedule: {epoch_schedule}')

        for cycle in range(num_cycles):
            if self.gpu_id == 0:
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

                metrics_tensor = torch.tensor([train_loss, train_acc, test_loss, test_acc], device=self.this_device)
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)

                if self.gpu_id == 0:
                    train_loss, train_acc, test_loss, test_acc = metrics_tensor.tolist()

                    self._log_metrics(cycle, epoch, train_loss, test_loss, train_acc, test_acc, level_metrics)
                    if level == 0 and training_type == "wr" and epoch == 10:
                        for name, obj in [("model", self.model), ("optimizer", self.optimizer)]:
                            torch.save(obj.state_dict(), os.path.join(self.expt_dir, f"{name}_rewind.pt"))

                    # Update Rich table
                    table.add_row(
                        str(cycle),
                        str(epoch),
                        f"{train_loss:.4f}",
                        f"{test_loss:.4f}",
                        f"{train_acc:.2f}%",
                        f"{test_acc:.2f}%"
                    )
                    console.clear()
                    console.print(table)

            if self.gpu_id == 0:
                torch.save(self.model.state_dict(), os.path.join(self.expt_dir, "checkpoints", f"model_cycle_{cycle}.pt"))
                metrics_path = os.path.join(self.expt_dir, "metrics", f"level_{level}_metrics.csv")
                df = pd.DataFrame(level_metrics)
                
                if os.path.exists(metrics_path):
                    df.to_csv(metrics_path, mode='a', header=False, index=False)
                else:
                    df.to_csv(metrics_path, index=False)
                
                print(f"Saved metrics for level {level}, cycle {cycle}")

        if self.gpu_id == 0:
            wandb.log({
                "sparsity": round(print_sparsity_info(self.model, verbose=False), 4),
                "max_test_acc": round(max(level_metrics["test_acc"]), 4)
            })

            summary_path = os.path.join(self.expt_dir, "metrics", f"{self.prefix}_summary.csv")
            sparsity = print_sparsity_info(self.model, verbose=False)
            new_data = {
                "level": [level],
                "cycle": [cycle],
                "sparsity": [round(sparsity, 4)],
                "final_test_acc": [round(level_metrics["test_acc"][-1], 4)],
                "max_test_acc": [round(max(level_metrics["test_acc"]), 4)],
                "training_schedule": ['-'.join(map(str, epoch_schedule))]
            }
            df = pd.DataFrame(new_data)
            if os.path.exists(summary_path):
                df.to_csv(summary_path, mode='a', header=False, index=False)
            else:
                df.to_csv(summary_path, index=False)
             

    def _log_metrics(self, cycle, epoch, train_loss, test_loss, train_acc, test_acc, level_metrics):
        level_metrics['cycle'].append(cycle)
        level_metrics['train_loss'].append(train_loss)
        level_metrics['test_loss'].append(test_loss)
        level_metrics['train_acc'].append(train_acc)
        level_metrics['test_acc'].append(test_acc)
        
        wandb.log({
            "train_loss": train_loss, "test_loss": test_loss,
            "train_acc": train_acc, "test_acc": test_acc,
            "epoch": self.epoch_counter,
            "cycle": cycle
        })


@param("dist_params.address")
@param("dist_params.port")
def setup_distributed(address: str, port: int, gpu_id: int) -> None:
    """Setup distributed training environment.

    Args:
        address (str): Master address for distributed training.
        port (int): Master port for distributed training.
        gpu_id (int): current rank/gpu.
    """
    os.environ["MASTER_ADDR"] = address
    os.environ["MASTER_PORT"] = str(port)
    world_size = torch.cuda.device_count()
    dist.init_process_group("nccl", rank=gpu_id, world_size=world_size)
    torch.cuda.set_device(gpu_id)


def main(rank: int, model: nn.Module, level: int, expt_dir: str, run_id, single_cycle_length: int = None) -> None:
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

    setup_distributed(gpu_id=rank)


    if (rank == 0):
        wandb.init(id=run_id, resume='allow')
    harness = Harness(model=model, expt_dir=expt_dir, gpu_id=rank)

    if (level == 0) and (rank == 0):
        torch.save(
            harness.model.module.state_dict(),
            os.path.join(expt_dir[1], "checkpoints", "model_init.pt"),
        )

    harness.train_one_level(level=level, single_cycle_length=single_cycle_length)

    if rank == 0:
        ckpt = os.path.join(expt_dir[1], "checkpoints", f"model_level_{level}.pt")
        torch.save(harness.model.module.state_dict(), ckpt)

    dist.destroy_process_group()


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
    
    world_size = torch.cuda.device_count()
    print(f"Training on {world_size} GPUs")
    perturbation_table = PrettyTable()
    parser = ArgumentParser()

    prune_harness = pruning_utils.PruningStuff()

    wandb.init()
    run_id = wandb.run.id

    resume_level = config["experiment_params.resume_level"]
    prefix, expt_dir = gen_expt_dir()
    packaged = (prefix, expt_dir)
    save_config(expt_dir=expt_dir, config=config)

    is_iterative = (config['prune_params.er_method'] == 'just dont') and (config['prune_params.prune_method'] != 'just dont')

    if is_iterative:
        densities = generate_densities(current_sparsity=print_sparsity_info(prune_harness.model, verbose=False))
    else:
        densities = [config['prune_params.er_init']]

    for level in range(resume_level, len(densities)):
        if is_iterative:
            if level != 0:
                logging.info(f"Pruning Model at level: {level}")
                prune_harness.load_from_ckpt(
                    os.path.join(expt_dir, "checkpoints", f"model_level_{level-1}.pt")
                )
                prune_harness.level_pruner(density=densities[level])
                prune_harness.model = reset_weights(
                    expt_dir=expt_dir,
                    model=prune_harness.model,
                    training_type=config["experiment_params.training_type"],
                )

                sparsity = print_sparsity_info(prune_harness.model, verbose=False)
                logging.info(f"Model sparsity after pruning: {sparsity:.4f}")
        else:
            logging.info('Pruning at initialization')
            prune_harness.prune_at_initialization(er_init=densities[level])
            sparsity = print_sparsity_info(prune_harness.model, verbose=False)
            logging.info(f"Model sparsity after pruning at init: {sparsity:.4f}")
        mp.spawn(main, args=(prune_harness.model, level, packaged, run_id),
            nprocs=world_size,
            join=True,
        )
        print(f"Training level {level} complete, moving on to {level+1}")
    wandb.finish()
