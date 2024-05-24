## pythonic imports
import os
import random
import time
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from prettytable import PrettyTable
import tqdm
from copy import deepcopy

## torch
import torch
import torch.nn as nn
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim
import torch.multiprocessing as mp
import torchvision
## torchvision

## file-based imports
import models
import pruning_utils
from dataset_utils import FFCVImageNet
import schedulers
from pruning_utils import *

## fastargs
import fastargs
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf
from torch.cuda.amp import GradScaler, autocast


def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12346'
    init_process_group('nccl', rank=rank, world_size=world_size)

class Harness:
    def __init__(self, gpu_id, expt_dir, model=None):
        self.config = get_current_config()

        self.gpu_id = gpu_id
        this_device = f'cuda:{self.gpu_id}'
        self.device = torch.device(this_device)
        self.world_size = dist.get_world_size()

        dls = FFCVImageNet(distributed=True, this_device=this_device)
        self.train_loader, self.test_loader = dls.train_loader, dls.val_loader
        self.model = model
        self.model = self.model.to(self.device)

        self.model = DDP(self.model, device_ids=[self.gpu_id])

        self.create_optimizers()
        self.criterion = nn.CrossEntropyLoss()
        self.data_df = {}
        self.data_df['sparsity'] = []
        self.data_df['test_acc'] = []
        self.data_df['max_test_acc'] = []
        self.expt_dir = expt_dir 
        self.scaler = GradScaler()


    @param('optimizer.lr')
    @param('optimizer.momentum')
    @param('optimizer.weight_decay')
    @param('optimizer.scheduler_type')
    @param('dataset.batch_size')
    def create_optimizers(self, lr, momentum, weight_decay, scheduler_type, batch_size):
        self.optimizer = optim.SGD(self.model.parameters(),
                                lr=lr,
                                momentum=momentum,
                                 weight_decay=weight_decay)

        scheduler = getattr(schedulers, scheduler_type)
        self.scheduler = scheduler(optimizer=self.optimizer)
    
    def train_one_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        tepoch = tqdm.tqdm(self.train_loader, unit='batch', desc=f'Epoch {epoch}')
        for inputs, targets in tepoch:
            self.optimizer.zero_grad()
            with autocast(dtype=torch.float16):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        train_loss /= (len(self.train_loader))
        accuracy = 100. * (correct / total)
        
        return train_loss, accuracy

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0

        # Initialize tqdm only on rank 0
        tloader = tqdm.tqdm(self.test_loader, leave=False, desc='Testing')
        with torch.no_grad():
            for inputs, targets in tloader:
                with autocast(dtype=torch.float16):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_loss /= (len(self.test_loader))
        accuracy = 100. * correct / total

        return test_loss, accuracy

  
    @param('experiment_params.epochs_per_level')
    @param('experiment_params.training_type')
    def train_one_level(self, epochs_per_level, training_type, threshold):
        new_table = PrettyTable()
        new_table.field_names = ["Epoch", "Train Loss", "Test Loss", "Train Acc", "Test Acc"]
        sparsity_level_df = {
            'train_acc': [],
            'test_acc': [],
            'train_loss': [],
            'test_loss': []
        }

        dist.barrier()
        
        for epoch in range(epochs_per_level):
            train_loss, train_acc = self.train_one_epoch(epoch)
            test_loss, test_acc = self.test()
            train_loss_tensor = torch.tensor(train_loss).to(self.model.device)
            train_acc_tensor = torch.tensor(train_acc).to(self.model.device)
            test_loss_tensor = torch.tensor(test_loss).to(self.model.device)
            test_acc_tensor = torch.tensor(test_acc).to(self.model.device)

            
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_acc_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_acc_tensor, op=dist.ReduceOp.SUM)

            train_loss_tensor /= self.world_size
            train_acc_tensor /= self.world_size
            test_loss_tensor /= self.world_size
            test_acc_tensor /= self.world_size

            if self.gpu_id == 0:
                new_table.add_row([epoch, train_loss_tensor.item(), test_loss_tensor.item(), train_acc_tensor.item(), test_acc_tensor.item()])
                print(new_table)

                sparsity_level_df['train_acc'].append(train_acc_tensor.item())
                sparsity_level_df['test_acc'].append(test_acc)
                sparsity_level_df['train_loss'].append(train_loss_tensor.item())
                sparsity_level_df['test_loss'].append(test_loss)

                save_matching = (threshold == 1.0) and (training_type == 'wr') and (epoch == 9)
                if save_matching and self.gpu_id == 0:
                    checkpoint_path = f"{self.expt_dir}/checkpoints/model_{self.config['optimizer.warmup_epochs']}.pt"
                    torch.save(self.model.module.state_dict(), checkpoint_path)

        if self.gpu_id == 0:
            pd.DataFrame(sparsity_level_df).to_csv(f'{self.expt_dir}/metrics/epochwise_metrics/level_{threshold}_metrics.csv')
            sparsity = self.compute_sparsity(self.model)
            self.data_df['sparsity'].append(sparsity)
            self.data_df['text_acc'].append(test_acc_tensor.item())
            self.data_df['max_test_acc'].append(max(sparsity_level_df['test_acc']))

            #torch.save(self.model.module.state_dict(), f'{self.expt_dir}/checkpoints/model_{threshold}_epoch_final.pt')

        dist.barrier()

    def reset_weights(self, init=False):
        if init:
            original_dict = torch.load(f'{self.expt_dir}/checkpoints/model_init.pt')
        else:
            original_dict = torch.load(f"{self.expt_dir}/checkpoints/model_{self.config['optimizer.warmup_epochs']}.pt")
        original_weights = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), original_dict.items()))
        model_dict = self.model.state_dict()

        model_dict.update(original_weights)
        self.model.load_state_dict(model_dict)

        optimizer.load_state_dict(torch.load(f'{self.expt_dir}/artifacts/optimizer.pt'))

    def reset_only_weights(self, ckpt_name):
        original_dict = torch.load(f"{self.expt_dir}/checkpoints/{ckpt_name}.pt")
        original_weights = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), original_dict.items()))
        model_dict = self.model.state_dict()

        model_dict.update(original_weights)
        self.model.load_state_dict(model_dict)

    def reset_only_masks(self, ckpt_name):
        original_dict = torch.load(f"{self.expt_dir}/checkpoints/{ckpt_name}.pt")
        original_weights = dict(filter(lambda v: (v[0].endswith(('.mask'))), original_dict.items()))
        model_dict = self.model.state_dict()

        model_dict.update(original_weights)
        self.model.load_state_dict(model_dict)
            
    
def main(rank, model, world_size, threshold, level):
    expt_dir = './experiments/'
    ddp_setup(rank, world_size)
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)

    config.validate(mode='stderr')

    harness = Harness(gpu_id=rank, model=model, expt_dir=expt_dir)
    if rank == 0:
        torch.save(harness.optimizer.state_dict(), f'{expt_dir}/artifacts/optimizer.pt')

    harness.train_one_level(threshold=threshold)

    if rank == 0:
        torch.save(harness.model.module.state_dict(), f'{expt_dir}/checkpoints/model_level_{i}.pt')
    
        if config['experiment_params.training_type'] == 'imp':
            harness.reset_weights(init=True)
        elif config['experiment_params.training_type'] == 'wr':
            harness.reset_weights()

        harness.data_df.to_csv(f'{expt_dir}/metrics/summary.csv')
        torch.save(harness.model.state_dict(), f'{expt_dir}/checkpoints/model_level_{level}.pt') 

    destroy_process_group()


if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)

    config.validate(mode='stderr')
    config.summary()

    #garbage_harness = pruning_utils.PruningStuff(model=torchvision.models.resnet18())
    #init_model = garbage_harness.model.cpu()
    init_model = torchvision.models.resnet50().to(memory_format=torch.channels_last)
    
    prune_rate = 0.2
    num_levels = 30
    thresholds = [(1 - prune_rate) ** level for level in range(num_levels)]
    expt_dir = './experiments/'

    if not os.path.exists(expt_dir):
        os.makedirs(expt_dir)
        os.makedirs(f'{expt_dir}/checkpoints')
        os.makedirs(f'{expt_dir}/metrics')
        os.makedirs(f'{expt_dir}/metrics/epochwise_metrics')
        os.makedirs(f'{expt_dir}/artifacts/')

    torch.save(init_model.state_dict(), f'{expt_dir}/checkpoints/random_init_start.pt')
    
    world_size = torch.cuda.device_count()
    for i, threshold in enumerate(thresholds):
        mp.spawn(main, args=(init_model, world_size, threshold, i), nprocs=world_size, join=True)
        
        if threshold != 1.0:
            prune_harness = PrunerStuff().load_from_ckpt(f'{expt_dir}/checkpoints/model_level_{i-1}.pt')
            prune_harnes.level_pruner(density=threshold)
        



    

