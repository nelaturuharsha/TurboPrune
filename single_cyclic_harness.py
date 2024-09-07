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

import wandb

##ffcv
try:
    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import (
        ToTensor,
        ToDevice,
        Squeeze,
        NormalizeImage,
        RandomHorizontalFlip,
        ToTorchImage,
    )
    from ffcv.fields.decoders import (
        RandomResizedCropRGBImageDecoder,
        CenterCropRGBImageDecoder,
    )
    from ffcv.fields.basics import IntDecoder
except:
    pass

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
            raise ValueError('Not Implemented')
        else:
            self.train_loader = CifarLoader(path='./cifar10', batch_size=512, train=True, aug={'flip' : True, 'translate' : 2}, altflip=True, dataset=self.config['dataset.dataset_name'])
            self.test_loader = CifarLoader(path='./cifar10', batch_size=512, train=False, dataset=self.config['dataset.dataset_name'])

        self.model = model.to(self.this_device)
        #self.model = torch.compile(self.model, mode='reduce-overhead')
        self.create_optimizers(epochs_per_level=self.config['experiment_params.epochs_per_level'])
        
        self.prefix, self.expt_dir = expt_dir

        self.epoch_counter = 0

    @param("dataset.batch_size")
    @param("dataset.num_workers")
    @param("dataset.data_root")
    def create_train_loader(
        self,
        batch_size: int,
        num_workers: int,
        data_root: str,
        distributed: bool = True,
    ) -> Any:
        """Create the train dataloader.

        Args:
            batch_size (int): Batch size for data loading.
            num_workers (int): Number of workers for data loading.
            data_root (str): Root directory for data.
            distributed (bool, optional): Whether to use distributed data loading. Default is True.

        Returns:
            Any: Train dataloader.
        """
        train_image_pipeline = [
            RandomResizedCropRGBImageDecoder((224, 224)),
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(torch.device(self.this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(self.this_device), non_blocking=True),
        ]

        train_loader = Loader(
            os.path.join(data_root, "train_500_0.50_90.beton"),
            batch_size=batch_size,
            num_workers=num_workers,
            order=OrderOption.RANDOM,
            os_cache=True,
            drop_last=True,
            pipelines={"image": train_image_pipeline, "label": label_pipeline},
            distributed=distributed,
        )

        return train_loader

    @param("dataset.batch_size")
    @param("dataset.num_workers")
    @param("dataset.data_root")
    def create_test_loader(
        self,
        batch_size: int,
        num_workers: int,
        data_root: str,
        distributed: bool = True,
    ) -> Any:
        """Create the test dataloader.

        Args:
            batch_size (int): Batch size for data loading.
            num_workers (int): Number of workers for data loading.
            data_root (str): Root directory for data.
            distributed (bool, optional): Whether to use distributed data loading. Default is True.

        Returns:
            Any: Test dataloader.
        """
        val_image_pipeline = [
            CenterCropRGBImageDecoder((256, 256), ratio=DEFAULT_CROP_RATIO),
            ToTensor(),
            ToDevice(torch.device(self.this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(self.this_device), non_blocking=True),
        ]

        val_loader = Loader(
            os.path.join(data_root, "val_500_0.50_90.beton"),
            batch_size=batch_size,
            num_workers=num_workers,
            order=OrderOption.SEQUENTIAL,
            drop_last=False,
            pipelines={"image": val_image_pipeline, "label": label_pipeline},
            distributed=distributed,
        )
        return val_loader

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

    @param("experiment_params.epochs_per_level")
    @param("experiment_params.training_type")
    @param("cyclic_training.num_cycles")
    def train_one_level(
        self, epochs_per_level: int, training_type: str, num_cycles: int, level: int) -> None:
        """Train the model for one full level. This can thought of as one full training run.

        Args:
            epochs_per_level (int): Total number of epochs to train for at each level.
            training_type (str): Type of training can be {'wr', 'lrr' or 'imp}.
            level (int): Current sparsity level.
        """
        new_table = PrettyTable()
        new_table.field_names = [
            "Cycle_Epoch",
            "Train Loss",
            "Test Loss",
            "Train Acc",
            "Test Acc",
        ]
        sparsity_level_df = {
            "cycle_epoch": [],
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
            #"hessian_trace" : [],
            "training_schedule" : [] }
        

        #epoch_schedule = generate_epoch_schedule()
        epoch_schedule = generate_budgeted_schedule()
        print(f'Epoch Schedule: {epoch_schedule}')
        for cycle in range(num_cycles):
            print('#' * 50)
            print(f'Training Cycle {cycle} for {epoch_schedule[cycle]} epochs.')
            print('#' * 50)
            self.create_optimizers(epochs_per_level=epoch_schedule[cycle])
            for epoch in range(epoch_schedule[cycle]):
                self.epoch_counter += 1
                train_loss, train_acc = self.train_one_epoch(epoch)
                test_loss, test_acc = self.test()

                if self.gpu_id == 0:
                    new_table.add_row([f'{cycle}_{epoch}', train_loss, test_loss, train_acc, test_acc])
                    print(new_table)
                    sparsity_level_df[f"cycle_epoch"].append(f'{cycle}_{epoch}')
                    sparsity_level_df["train_loss"].append(train_loss)
                    sparsity_level_df["test_loss"].append(test_loss)
                    sparsity_level_df["train_acc"].append(train_acc)
                    sparsity_level_df["test_acc"].append(test_acc)

                    wandb.log({
                        "train_loss": train_loss,
                        "test_loss": test_loss,
                        "train_acc": train_acc,
                        "test_acc": test_acc,
                        "epoch": self.epoch_counter
                    })

                    save_matching = (
                        (level == 0) and (training_type == "wr") and (epoch == 9)
                    )
                    if save_matching and (self.gpu_id == 0):
                        torch.save(
                            self.model.state_dict(),
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
            data_df["final_test_acc"].append(round(test_acc, 4))
            data_df["max_test_acc"].append(round(max(sparsity_level_df["test_acc"]), 4))
            #data_df['hessian_trace'].append(hessian_trace(self.train_loader, model=self.model))
            epoch_schedule_str = '-'.join(map(str, epoch_schedule))
            data_df['training_schedule'].append(epoch_schedule_str)
            wandb.log({
                "sparsity": round(sparsity, 4),
                "max_test_acc": round(max(sparsity_level_df["test_acc"]), 4)
            })


            summary_path = os.path.join(self.expt_dir, "metrics", f"{self.prefix}_summary.csv")

            if not os.path.exists(summary_path):
                pd.DataFrame(data_df).to_csv(summary_path, index=False)
            else:
                pre_df = pd.read_csv(summary_path)
                new_df = pd.DataFrame(data_df)
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

    if level != 0:
        harness.optimizer = reset_optimizer(
            expt_dir=expt_dir[1],
            optimizer=harness.optimizer,
            training_type=config["experiment_params.training_type"],
        )

    if (level == 0):
        torch.save(
            harness.optimizer.state_dict(),
            os.path.join(expt_dir[1], "artifacts", "optimizer_init.pt"),
        )
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

    wandb.init(project=config['wandb_params.project_name'])
    world_size = torch.cuda.device_count()
    print(f"Training on {world_size} GPUs")
    perturbation_table = PrettyTable()
    parser = ArgumentParser()

    prune_harness = pruning_utils.PruningStuff()

    resume_level = config["experiment_params.resume_level"]
    prefix, expt_dir = gen_expt_dir()
    print(expt_dir)
    packaged = (prefix, expt_dir)
    save_config(expt_dir=expt_dir, config=config)

    is_iterative = (config['prune_params.er_method'] == 'just dont') and (config['prune_params.prune_method'] != 'just dont')
    
    metric_path = os.path.join(expt_dir, 'metrics')
    perturbation_dict_path = os.path.join(metric_path, 'perturbation.csv')
    pert_stats = {'level'         : [],
                  'pre_sparsity'  : [],
                  'post_sparsity' : [],
                  'pre_test_acc'  : [],
                  'post_test_acc' : [],
                  'perturbation'  : []}
    
    #densities = generate_densities()
    densities = [config['prune_params.er_init']]
    for level in range(resume_level, len(densities)):
        print_sparsity_info(prune_harness.model, verbose=False)
        if is_iterative:
            if level != 0:
                perturbation_level_dict = compute_perturbation(
                                                            model = prune_harness.model,
                                                            density=densities[level],
                                                            perturbation_table=perturbation_table,
                                                            level=level) 
                pert_stats['level'].append(perturbation_level_dict['level'])
                pert_stats['pre_sparsity'].append(perturbation_level_dict['pre_sparsity'])
                pert_stats['post_sparsity'].append(perturbation_level_dict['post_sparsity'])
                pert_stats['pre_test_acc'].append(perturbation_level_dict['pre_test_acc'])
                pert_stats['post_test_acc'].append(perturbation_level_dict['post_test_acc'])
                pert_stats['perturbation'].append(perturbation_level_dict['perturbation'])
                
                pd.DataFrame(pert_stats).to_csv(perturbation_dict_path, index=False)
                
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
        else:
            print('we out here pruning at init')
            prune_harness.prune_at_initialization(er_init=densities[level])

            print_sparsity_info(prune_harness.model, verbose=False)
        main(model = prune_harness.model, level=level, expt_dir=packaged)
        if (level != 0) and (is_iterative):
            linear_mode = LinearModeConnectivity(expt_path=expt_dir, model=prune_harness.model)
            linear_mode.gen_linear_mode(level1=level-1, level2=level)
        print(f"Training level {level} complete, moving on to {level+1}") 

    wandb.finish()
