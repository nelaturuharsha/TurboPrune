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
## torchvision

## file-based imports
import models
import pruners
from dataset_utils import FFCVImageNet
import schedulers
from garbage import *

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

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    init_process_group('nccl', rank=rank, world_size=world_size)

class Harness:
    def __init__(self, gpu_id, model=None):
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{self.gpu_id}')
        dls = FFCVImageNet(distributed=True)
        self.train_loader, self.test_loader = dls.train_loader, dls.val_loader

        self.config = get_current_config()

        print(self.gpu_id, self.device)

        self.model = model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        

        self.create_optimizers()
        self.criterion = nn.CrossEntropyLoss()
        self.data_df = {}
        self.data_df['sparsity'] = []
        self.data_df['test_acc'] = []
        self.data_df['max_test_acc'] = []
        self.expt_dir = './experiments/'
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
        print('prog 1')
        tepoch = tqdm.tqdm(self.train_loader, unit='batch', desc=f'Epoch {epoch}')
        print('prog 2')
        for inputs, targets in tepoch:            
            print('prog 3')
            #inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            print('prog 4')
            #with autocast(dtype=torch.float16):
            outputs = self.model(inputs)
            print('prog 5')
            loss = self.criterion(outputs, targets)
            print('prog 6')
            #self.scaler.scale(loss).backward()
            #self.scaler.step(self.optimizer)
            #self.scaler.update()
            #self.scheduler.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            #print(100. * (correc))
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

  
    @param('experiment_params.total_epochs')
    @param('experiment_params.training_type')
    def train_one_level(self, total_epochs, training_type, threshold):
        new_table = PrettyTable()
        new_table.field_names = ["Epoch", "Train Loss", "Test Loss", "Train Acc", "Test Acc"]
        sparsity_level_df = {
            'train_acc': [],
            'test_acc': [],
            'train_loss': [],
            'test_loss': []
        }

        #dist.barrier()
        
        for epoch in range(total_epochs):
            train_loss, train_acc = self.train_one_epoch(epoch)
            test_loss, test_acc = self.test()
            train_loss_tensor = torch.tensor(train_loss).to(self.model.device)
            train_acc_tensor = torch.tensor(train_acc).to(self.model.device)
            test_loss_tensor = torch.tensor(test_loss).to(self.model.device)
            test_acc_tensor = torch.tensor(test_acc).to(self.model.device)

            '''
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_acc_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_acc_tensor, op=dist.ReduceOp.SUM)

            train_loss_tensor /= world_size
            train_acc_tensor /= world_size
            test_loss_tensor /= world_size
            test_acc_tensor /= world_size

            if self.gpu_id == 0:

                new_table.add_row([epoch, train_loss_tensor.item(), test_loss_tensor.item(), train_acc_tensor.item(), test_acc_tensor.item()])
                print(new_table)

                sparsity_level_df['train_acc'].append(train_acc_tensor.item())
                sparsity_level_df['test_acc'].append(test_acc)
                sparsity_level_df['train_loss'].append(train_loss_tensor.item())
                sparsity_level_df['test_loss'].append(test_loss)

                save_matching = (threshold == 1.0) and (training_type == 'wr') and (epoch == 9)
                if save_matching and self.gpu_id == 0:
                    checkpoint_path = f'{self.expt_dir}/checkpoints/model_{threshold}_epoch_{epoch}.pt'
                    torch.save(self.model.module.state_dict(), checkpoint_path)

        if self.gpu_id == 0:
            pd.DataFrame(sparsity_level_df).to_csv(f'{self.expt_dir}/metrics/epochwise_metrics/level_{threshold}_metrics.csv')
            sparsity = self.compute_sparsity(self.model)
            self.data_df['sparsity'].append(sparsity)
            self.data_df['text_acc'].append(test_acc_tensor.item())
            self.data_df['max_test_acc'].append(max(sparsity_level_df['test_acc']))

            torch.save(self.model.module.state_dict(), f'{self.expt_dir}/checkpoints/model_{threshold}_epoch_final.pt')

        dist.barrier()
            '''
    
def main(rank, model, world_size, threshold):
    ddp_setup(rank, world_size)
    print('rank', rank)
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)

    config.validate(mode='stderr')
    if rank == 0:
        config.summary()

    harness = Harness(gpu_id=rank, model=model)
    harness.train_one_level(threshold=threshold)
    destroy_process_group()


if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)

    config.validate(mode='stderr')
    config.summary()

    garbage_harness = PruneStuff()
    init_model = deepcopy(garbage_harness.model.cpu())
    del garbage_harness
    
    prune_rate = 0.2
    num_levels = 30
    thresholds = [(1 - prune_rate) ** level for level in range(num_levels)]
    expt_dir = './experiments/'

    world_size = torch.cuda.device_count()
    threshold = 0.105
    #for i, threshold in enumerate(thresholds):
    mp.spawn(main, args=(init_model, world_size, threshold), nprocs=world_size, join=True)

    '''torch.save(harness.model.module.state_dict(), f'{expt_dir}/checkpoints/model_level_{i}.pt')
    
    if config['experiment_params.training_type'] == 'imp':
        harness.reset_weights(init=True)
    elif config['experiment_params.training_type'] == 'wr':
        harness.reset_weights()

    harness.data_df.to_csv(f'{expt_dir}/metrics/summary.csv')'''

