## pythonic imports
import os
import random
import time
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from prettytable import PrettyTable
import tqdm

## torch
import torch
import torch.nn as nn
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim
## torchvision

## file-based imports
import models
import pruners
from dataset_utils import FFCVImageNet
import schedulers

## fastargs
import fastargs
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf
from torch.cuda.amp import GradScaler, autocast



Section('model_params', 'model details').params(
    pretrained=Param(int, 'is pre-trained? (1/0)', default=0),
    model_name=Param(str, 'model_choice', default='ResNet50', required=True),
    first_layer_type=Param(str, 'layer type'),
    conv_type=Param(And(str, OneOf(['ConvMask'])), required=True),
    bn_type=Param(And(str, OneOf(['LearnedBatchNorm'])), required=True),
    init=Param(And(str, OneOf(['kaiming_normal'])), required=True),
    nonlinearity=Param(And(str, OneOf(['relu', 'leaky_relu'])), required=True),
    first_layer_dense=Param(bool, 'is first layer dense?', default=False),
    last_layer_dense=Param(bool, 'last layer dense?', default=False),
    mode=Param(And(str, OneOf(['fan_in'])), required=True),
    scale_fan=Param(bool, 'use scale fan', required=True))

Section('dataset', 'dataset configuration').params(
    num_classes=Param(And(int, OneOf([1000])), 'number of classes',required=True),
    batch_size=Param(int, 'batch size', default=512),
    num_workers=Param(int, 'num_workers', default=8))

Section('prune_params', 'pruning configuration').params(
    prune_rate=Param(float, 'pruning percentage',required=True),
    er_init=Param(float, 'sparse init percentage/target', required=True),
    er_method=Param(And(OneOf(['er_erk', 'er_balanced', 'synflow', 'snip', 'just dont'])), required=True),
    prune_method=Param(And(OneOf(['random_erk', 'random_balanced', 'synflow', 'snip', 'mag']))))

Section('experiment_params', 'parameters to train model').params(
    total_epochs=Param(int, 'number of epochs per level', required=True),
    num_levels=Param(int, 'number of pruning levels', required=True),
    training_type=Param(And(str, OneOf(['imp', 'wr', 'lrr'])), required=True))


Section('dataset', 'data related stuff').params(
    dataset_name=Param(str, 'Name of dataset', required=True),
    batch_size=Param(int, 'batch size', default=512),
    num_workers=Param(int, 'num_workers', default=8))

Section('optimizer', 'data related stuff').params(
    lr=Param(float, 'Name of dataset', required=True),
    num_workers=Param(int, 'num_workers', default=8),
    momentum=Param(float, 'momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=1e-4),
    warmup_epochs=Param(int, 'warmup length', default=10),
    scheduler_type=Param(And(str, OneOf(['MultiStepLRWarmup', 'ImageNetLRDropsWarmup', 'CosineLRWarmup'])), required=True),
    lr_min=Param(float, 'minimum learning rate for cosine', default=0.01))


class Harness:
    def __init__(self):
        self.ddp_setup()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f'cuda:{self.rank}')

        self.config = get_current_config()

        self.model = self.acquire_model()
        
        self.model.to(self.device)
        

        self.model = DDP(self.model, device_ids=[self.rank])
        dls = FFCVImageNet()
        self.train_loader, self.test_loader = dls.train_loader, dls.val_loader
        self.create_optimizers()
        self.criterion = nn.NLLLoss()
        self.data_df = {}
        self.data_df['sparsity'] = []
        self.data_df['test_acc'] = []
        self.data_df['max_test_acc'] = []
        self.expt_dir = './experiments/'
        self.scaler = GradScaler()

    def ddp_setup(self):
        init_process_group(backend='nccl')
    
    def cleanup(self):
        dist.destroy_process_group()

    @param("model_params.model_name")
    @param("dataset.num_classes")
    def acquire_model(self, model_name, num_classes):
        model_cls = getattr(models, model_name)
        model = model_cls(num_classes=num_classes)
        return model

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
       
    @param('prune_params.er_method')
    @param('prune_params.er_init')
    def prune_at_initialization(self, er_method, er_init):
        if dist.get_rank() == 0:
            er_method_name = f'prune_{er_method}'
            pruner_method = getattr(pruners, er_method_name)

            if er_method in {'synflow', 'snip'}:
                self.model = pruner_method(self.model, self.train_loader, er_init)
            else:
                pruner_method(self.model, er_init)

            self.broadcast_model()


    def broadcast_model(self):
        # Ensure all processes reach this point before proceeding
        dist.barrier()
        
        # Broadcast parameters of the model
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)
        
        # Broadcast buffers of the model (e.g., running statistics in BatchNorm)
        for buffer in self.model.buffers():
            dist.broadcast(buffer.data, src=0)

        # Ensure all processes finish broadcasting before proceeding
        dist.barrier()

        # Verify that parameters are identical across all devices
        local_params = [param.data for param in self.model.parameters()]
        all_params = [torch.zeros_like(param) for param in local_params]

        # Gather parameters from all processes
        dist.all_gather(all_params, local_params)

        # Compare parameters across all processes
        for i in range(dist.get_world_size()):
            if i != dist.get_rank():
                for local_param, param in zip(local_params, all_params[i]):
                    if not torch.allclose(local_param, param):
                        raise ValueError("Parameters on device {} are not identical!".format(dist.get_rank()))

        print("Parameters on all devices are identical.")


    @param('prune_params.prune_method')
    def level_pruner(self, prune_method, density):
        if dist.get_rank() == 0:
            prune_method_name = f'prune_{prune_method}'
            pruner_method = getattr(pruners, prune_method_name)
            if prune_method in {'synflow', 'snip'}:
                self.model = pruner_method(self.model, self.train_loader, density)
            else:
                self.model = pruner_method(self.model, density)

            self.broadcast_model()
    
    def train_one_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0

        tepoch = tqdm.tqdm(self.train_loader, unit='batch', desc=f'Epoch {epoch}')
        for inputs, targets in tepoch:
            if self.rank == 0:
                tepoch.set_description(f'Epoch: {epoch}')
            
            inputs, targets = inputs.to(self.device), targets.to(self.device)
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
                inputs, targets = inputs.to(self.device), targets.to(self.device)
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

  
    @param('experiment_params.total_epochs')
    @param('experiment_params.training_type')
    def train_one_level(self, total_epochs, training_type, threshold, rank, world_size):
        new_table = PrettyTable()
        new_table.field_names = ["Epoch", "Train Loss", "Test Loss", "Train Acc", "Test Acc"]
        sparsity_level_df = {
            'train_acc': [],
            'test_acc': [],
            'train_loss': [],
            'test_loss': []
        }

        dist.barrier()
        
        for epoch in range(total_epochs):
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

            train_loss_tensor /= world_size
            train_acc_tensor /= world_size
            test_loss_tensor /= world_size
            test_acc_tensor /= world_size

            if rank == 0:

                new_table.add_row([epoch, train_loss_tensor.item(), test_loss_tensor.item(), train_acc_tensor.item(), test_acc_tensor.item()])
                print(new_table)

                sparsity_level_df['train_acc'].append(train_acc_tensor.item())
                sparsity_level_df['test_acc'].append(test_acc)
                sparsity_level_df['train_loss'].append(train_loss_tensor.item())
                sparsity_level_df['test_loss'].append(test_loss)

                save_matching = (threshold == 1.0) and (training_type == 'wr') and (epoch == 9)
                if save_matching and rank == 0:
                    checkpoint_path = f'{self.expt_dir}/checkpoints/model_{threshold}_epoch_{epoch}.pt'
                    torch.save(self.model.module.state_dict(), checkpoint_path)

        if rank == 0:
            pd.DataFrame(sparsity_level_df).to_csv(f'{self.expt_dir}/metrics/epochwise_metrics/level_{threshold}_metrics.csv')
            sparsity = self.compute_sparsity(self.model)
            self.data_df['sparsity'].append(sparsity)
            self.data_df['text_acc'].append(test_acc_tensor.item())
            self.data_df['max_test_acc'].append(max(sparsity_level_df['test_acc']))

            torch.save(self.model.module.state_dict(), f'{self.expt_dir}/checkpoints/model_{threshold}_epoch_final.pt')

        dist.barrier()

if __name__ == '__main__':

    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    
    

    harness = Harness()
    prune_rate = 0.2
    num_levels = 30
    thresholds = [(1 - prune_rate) ** level for level in range(num_levels)]
    expt_dir = './experiments/'
    
    rank = dist.get_rank()
    if rank == 0:
        config.summary()


    if config['prune_params.er_method'] != 'just dont':
        thresholds = [1.0 for level in range(30)]
    #harness.model = torch.compile(harness.model, mode='reduce-overhead')

    if rank == 0:
        if not os.path.exists(expt_dir):
            os.makedirs(expt_dir)
            os.makedirs(f'{expt_dir}/checkpoints')
            os.makedirs(f'{expt_dir}/metrics')
            os.makedirs(f'{expt_dir}/metrics/epochwise_metrics')
            os.makedirs(f'{expt_dir}/artifacts/')

        torch.save(harness.model.module.state_dict(), f'{expt_dir}/checkpoints/random_init_start.pt')
        torch.save(harness.optimizer.state_dict(), f'{expt_dir}/artifacts/optimizer.pt')
        
    dist.barrier()

    if config['prune_params.er_method'] != 'just dont':
        harness.prune_at_initialization()

    dist.barrier()

    for i, threshold in enumerate(thresholds):
        harness.create_optimizers()
        if threshold != 1.0 and rank == 0:
            harness.level_pruner(threshold)

        harness.train_one_level(threshold=threshold, rank=rank, world_size=2)

        if rank == 0: 
            torch.save(harness.model.module.state_dict(), f'{expt_dir}/checkpoints/model_level_{i}.pt')
            
            if config['experiment_params.training_type'] == 'imp':
                harness.reset_weights(init=True)
            elif config['experiment_params.training_type'] == 'wr':
                harness.reset_weights()

        if rank == 0:
            harness.data_df.to_csv(f'{expt_dir}/metrics/summary.csv')

    dist.destroy_process_group()
