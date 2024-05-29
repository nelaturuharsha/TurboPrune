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
import math

## torch
import torch
import torch.nn as nn
from torch import optim
import torchvision
## torchvision

## file-based imports
import schedulers
import pruning_utils
import models 
## fastargs
from fastargs import get_current_config
from fastargs.decorators import param
from torch.cuda.amp import autocast

##ffcv
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.decoders import RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

#os.environ['TORCH_COMPILE_DEBUG'] = '1'

DEFAULT_CROP_RATIO = 224/256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255

torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class Harness:
    def __init__(self, gpu_id, expt_dir, model=None):
        self.config = get_current_config()

        self.gpu_id = gpu_id
        this_device = f'cuda:{self.gpu_id}'
        self.device = torch.device(this_device)

        self.train_loader, self.test_loader = self.create_train_loader(this_device=this_device), self.create_test_loader(this_device=this_device)

        self.model = model.to(self.device)
        self.model = torch.compile(self.model, mode='max-autotune')
        self.create_optimizers()

        self.criterion = nn.CrossEntropyLoss()
        self.data_df = {}
        self.data_df['sparsity'] = []
        self.data_df['test_acc'] = []
        self.data_df['max_test_acc'] = []
        self.expt_dir = expt_dir 

    @param('dataset.batch_size')
    @param('dataset.num_workers')
    def create_train_loader(self, batch_size, num_workers, this_device, distributed=True):
        data_root = '/home/c02hane/CISPA-projects/ffcv_imagenet-2023/'
        train_image_pipeline = [RandomResizedCropRGBImageDecoder((224, 224)),
                            RandomHorizontalFlip(),
                            ToTensor(),
                            ToDevice(torch.device(this_device), non_blocking=True),
                            ToTorchImage(),
                            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)]

        label_pipeline = [IntDecoder(),
                            ToTensor(),
                            Squeeze(),
                            ToDevice(torch.device(this_device), non_blocking=True)]


        train_loader = Loader(data_root + 'train_500_0.50_90.beton', 
                              batch_size  = batch_size,
                              num_workers = num_workers,
                              order       = OrderOption.RANDOM,
                              os_cache    = True,
                              drop_last   = True,
                              pipelines   = { 'image' : train_image_pipeline,
                                              'label' : label_pipeline},
                              )
        
        return train_loader

    @param('dataset.batch_size')
    @param('dataset.num_workers')
    def create_test_loader(self, batch_size, num_workers, this_device, distributed=True):
        data_root = '/home/c02hane/CISPA-projects/ffcv_imagenet-2023/' 
        val_image_pipeline = [CenterCropRGBImageDecoder((256, 256), ratio=DEFAULT_CROP_RATIO),
                              ToTensor(),
                              ToDevice(torch.device(this_device), non_blocking=True),
                              ToTorchImage(),
                              NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)]

        label_pipeline = [IntDecoder(),
                            ToTensor(),
                            Squeeze(),
                            ToDevice(torch.device(this_device), non_blocking=True)]

        val_loader = Loader(data_root + 'val_500_0.50_90.beton',
                            batch_size  = batch_size,
                            num_workers = num_workers,
                            order       = OrderOption.SEQUENTIAL,
                            drop_last   = True,
                            pipelines   = { 'image' : val_image_pipeline,
                                            'label' : label_pipeline},
                            )
        return val_loader


    @param('optimizer.lr')
    @param('optimizer.momentum')
    @param('optimizer.weight_decay')
    @param('optimizer.scheduler_type')
    def create_optimizers(self, lr, momentum, weight_decay, scheduler_type):
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=momentum, weight_decay=weight_decay)
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
            with autocast(dtype=torch.bfloat16):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss /= (len(self.train_loader))
        accuracy = 100. * (correct / total)
        self.scheduler.step()

        return train_loss, accuracy

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0

        tloader = tqdm.tqdm(self.test_loader, desc='Testing')
        with torch.no_grad():
            for inputs, targets in tloader:
                with autocast(dtype=torch.bfloat16):
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

        for epoch in range(epochs_per_level):
            train_loss, train_acc = self.train_one_epoch(epoch)
            test_loss, test_acc = self.test()

            if self.gpu_id == 0:
                tr_l, te_l, tr_a, te_a = train_loss, test_loss, train_acc, test_acc
                new_table.add_row([epoch, tr_l, te_l, tr_a, te_a])
                print(new_table)

                sparsity_level_df['train_loss'].append(tr_l)
                sparsity_level_df['test_loss'].append(te_l)
                sparsity_level_df['train_acc'].append(tr_a)
                sparsity_level_df['test_acc'].append(te_a)
            

def main():
    expt_dir = '/home/c02hane/CISPA-projects/neuron_pruning-2024/lottery-ticket-harness/experiments'
    threshold = 0.2

    model = models.ResNet18()

    harness = Harness(gpu_id=0, model=model, expt_dir=expt_dir)
    torch.save(harness.optimizer.state_dict(), f'{expt_dir}/artifacts/optimizer.pt')
    harness.train_one_level(threshold=threshold)


if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)

    config.validate(mode='stderr')
    config.summary()


    expt_dir = '/home/c02hane/CISPA-projects/neuron_pruning-2024/lottery-ticket-harness/experiments'
    if not os.path.exists(expt_dir):
        os.makedirs(expt_dir)
        os.makedirs(f'{expt_dir}/checkpoints')
        os.makedirs(f'{expt_dir}/metrics')
        os.makedirs(f'{expt_dir}/metrics/epochwise_metrics')
        os.makedirs(f'{expt_dir}/artifacts/')
    main()