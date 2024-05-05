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
import torch.multiprocessing as mp
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
 
class PruneStuff:
    def __init__(self):
        self.device = torch.device('cuda')
        self.config = get_current_config()

        dls = FFCVImageNet(distributed=False)
        self.train_loader, self.test_loader = dls.train_loader, dls.val_loader

        self.model = self.acquire_model()
        self.criterion = nn.CrossEntropyLoss()

    @param("model_params.model_name")
    @param("dataset.num_classes")
    def acquire_model(self, model_name, num_classes):
        model_cls = getattr(models, model_name)
        model = model_cls(num_classes=num_classes)
        model.to(self.device)
        if self.config['prune_params.er_method'] != 'just dont':
            model = self.prune_at_initialization(model=model)
        return model

    @param('prune_params.er_method')
    @param('prune_params.er_init')
    def prune_at_initialization(self, model, er_method, er_init):
        print(er_init)
        er_method_name = f'prune_{er_method}'
        pruner_method = getattr(pruners, er_method_name)

        if er_method in {'synflow', 'snip'}:
            model = pruner_method(model, self.train_loader, er_init)
        else:
            pruner_method(model, er_init)
        
        return model

if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)

    config.validate(mode='stderr')
    config.summary()
    o = PruneStuff()
    model = o.model
    for i, j in o.train_loader:
        model(i)
