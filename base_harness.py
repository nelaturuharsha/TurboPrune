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
from torch.cuda.amp import autocast

## file-based imports
import utils.schedulers as schedulers
import utils.pruning_utils as pruning_utils
from utils.harness_utils import *
from utils.dataset import CIFARLoader, imagenet

## fastargs
from fastargs import get_current_config
from fastargs.decorators import param
import schedulefree


class Harness:
    """Harness class to handle training and evaluation.

    Args:
        gpu_id (int): current rank of the process while training using DDP.
        expt_dir (str): Experiment directory to save artifacts/checkpoints/metrics to.
        model (Optional[nn.Module], optional): The model to train.
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
        self.create_optimizers()
        self.expt_dir = expt_dir

        self.precision, self.use_amp = self.get_dtype_amp()

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
    @param("experiment_params.epochs_per_level")
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
            with autocast(dtype=self.precision, enabled = self.use_amp):
                outputs = model(inputs.contiguous())
                loss = self.criterion(outputs, targets)
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
                with autocast(dtype=self.precision, enabled = self.use_amp):
                    outputs = model(inputs.contiguous())
                    loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_loss /= len(self.test_loader)
        accuracy = 100.0 * correct / total

        return test_loss, accuracy

    @param("experiment_params.epochs_per_level")
    @param("experiment_params.training_type")
    def train_one_level(
        self, epochs_per_level: int, training_type: str, level: int
    ) -> None:
        """Train the model for one full level. This can thought of as one full training run.

        Args:
            epochs_per_level (int): Total number of epochs to train for at each level.
            training_type (str): Type of training can be {'wr', 'lrr' or 'imp}.
            level (int): Current sparsity level.
        """
        new_table = PrettyTable()
        new_table.field_names = [
            "Epoch",
            "Train Loss",
            "Test Loss",
            "Train Acc",
            "Test Acc",
        ]
        sparsity_level_df = {
            "epoch": [],
            "train_acc": [],
            "test_acc": [],
            "train_loss": [],
            "test_loss": [],
        }
        data_df = {
            "level": [],
            "sparsity": [],
            "max_test_acc": [],
            "final_test_acc": [],
        }

        for epoch in range(epochs_per_level):
            train_loss, train_acc = self.train_one_epoch(epoch)
            test_loss, test_acc = self.test()

            metrics = [torch.tensor(val).to(self.model.device) for val in [train_loss, train_acc, test_loss, test_acc]]

            for metric in metrics:
                dist.reduce(metric, dst=0, op=dist.ReduceOp.AVG)

            if self.gpu_id == 0:
                tr_l, tr_a, te_l, te_a = [metric.item() for metric in metrics]
                new_table.add_row([epoch, tr_l, te_l, tr_a, te_a])
                print(new_table)
                sparsity_level_df["epoch"].append(round(epoch, 4))
                sparsity_level_df["train_loss"].append(round(tr_l, 4))
                sparsity_level_df["test_loss"].append(round(te_l, 4))
                sparsity_level_df["train_acc"].append(round(tr_a, 4))
                sparsity_level_df["test_acc"].append(round(te_a, 4))
                save_matching = (
                    (level == 0) and (training_type == "wr") and (epoch == 9)
                )
                if save_matching and (self.gpu_id == 0):
                    torch.save(
                        self.model.module.state_dict(),
                        os.path.join(self.expt_dir, "checkpoints", "model_rewind.pt"),
                    )
                    torch.save(
                        self.optimizer.state_dict(),
                        os.path.join(self.expt_dir, "artifacts", "optimizer_rewind.pt"),
                    )

        if self.gpu_id == 0:
            pd.DataFrame(sparsity_level_df).to_csv(
                os.path.join(
                    self.expt_dir,
                    "metrics",
                    "epochwise_metrics",
                    f"level_{level}_metrics.csv",
                )
            )
            sparsity = print_sparsity_info(self.model, verbose=False)
            data_df["level"].append(level)
            data_df["sparsity"].append(round(sparsity, 4))
            data_df["final_test_acc"].append(round(te_a, 4))
            data_df["max_test_acc"].append(round(max(sparsity_level_df["test_acc"]), 4))
            summary_path = os.path.join(self.expt_dir, "metrics", "summary.csv")

            if not os.path.exists(summary_path):
                pd.DataFrame(data_df).to_csv(summary_path, index=False)
            else:
                pre_df = pd.read_csv(summary_path)
                new_df = pd.DataFrame(data_df)
                updated_df = pd.concat([pre_df, new_df], ignore_index=True)
                updated_df.to_csv(summary_path, index=False)


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


def main(rank: int, model: nn.Module, level: int, expt_dir: str) -> None:
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

    harness = Harness(model=model, expt_dir=expt_dir, gpu_id=rank)

    if level != 0:
        harness.optimizer = reset_optimizer(
            expt_dir=expt_dir,
            optimizer=harness.optimizer,
            training_type=config["experiment_params.training_type"],
        )

    if (level == 0) and (rank == 0):
        torch.save(
            harness.optimizer.state_dict(),
            os.path.join(expt_dir, "artifacts", "optimizer_init.pt"),
        )
        torch.save(
            harness.model.module.state_dict(),
            os.path.join(expt_dir, "checkpoints", "model_init.pt"),
        )

    harness.train_one_level(level=level)

    if rank == 0:
        ckpt = os.path.join(expt_dir, "checkpoints", f"model_level_{level}.pt")
        torch.save(harness.model.module.state_dict(), ckpt)

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Training on {world_size} GPUs")

    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)

    config.validate(mode="stderr")
    config.summary()

    prune_harness = pruning_utils.PruningStuff()

    # if you provide resume level and resume experiment directory, it will pick up from where it stopped automatically
    resume_level = config["experiment_params.resume_level"]
    expt_dir = gen_expt_dir()
    save_config(expt_dir=expt_dir, config=config)
    densities = generate_densities()

    for level in range(resume_level, len(densities)):
        print_sparsity_info(prune_harness.model, verbose=False)
        if level != 0:
            print(f"Pruning Model at level: {level}")
            prune_harness.load_from_ckpt(
                os.path.join(expt_dir, "checkpoints", f"model_level_{level-1}.pt")
            )
            prune_harness.level_pruner(density=densities[level])
            prune_harness.model = reset_weights(
                expt_dir=expt_dir,
                model=prune_harness.model,
                training_type=config["experiment_params.training_type"],
            )

            print_sparsity_info(prune_harness.model, verbose=False)

        mp.spawn(
            main,
            args=(prune_harness.model, level, expt_dir),
            nprocs=world_size,
            join=True,
        )
        print(f"Training level {level} complete, moving on to {level+1}")