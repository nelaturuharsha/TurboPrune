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
 
class PruneStuff:
    def __init__(self, model=None):
        self.device = torch.device('cuda')
        self.config = get_current_config()

        dls = FFCVImageNet(distributed=False)
        self.train_loader, self.test_loader = dls.train_loader, dls.val_loader

        if model is None:
            self.model = self.acquire_model()
        else:
            self.model = model
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
        er_method_name = f'prune_{er_method}'
        pruner_method = getattr(pruners, er_method_name)

        if er_method in {'synflow', 'snip'}:
            model = pruner_method(model, self.train_loader, er_init)
        else:
            pruner_method(model, er_init)
        
        return model

    @param('prune_params.prune_method')
    def level_pruner(self, model, prune_method, density):
        prune_method_name = f'prune_{prune_method}'
        pruner_method = getattr(pruners, prune_method_name)
        if prune_method in {'synflow', 'snip'}:
            self.model = pruner_method(self.model, self.train_loader, density)
        else:
            pruner_method(self.model, density)
