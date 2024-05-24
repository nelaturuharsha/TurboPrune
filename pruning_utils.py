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
from dataset_utils import FFCVImageNet
import schedulers
from utils.conv_type import ConvMask
from harness_params import *


## fastargs
import fastargs
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

get_current_params()

def get_sparsity(model):
        # finds the current density of the model and returns the density scalar value
        nz = 0
        total = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                nz += m.mask.sum()
                total += m.mask.numel()
        
        return nz / total
 
class PruningStuff:
    def __init__(self, model=None):
        self.device = torch.device('cuda')
        this_device = 'cuda:0'
        self.config = get_current_config()

        dls = FFCVImageNet(distributed=False, this_device=this_device)
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
        print(f'Sparsity before pruning: {get_sparsity(self.model)}')
        prune_method_name = f'prune_{prune_method}'
        pruner_method = getattr(pruners, prune_method_name)
        if prune_method in {'synflow', 'snip'}:
            self.model = pruner_method(self.model, self.train_loader, density)
        else:
            pruner_method(self.model, density)
        print(f'Sparsity after pruning: {get_sparsity(self.model)}')


import torch
import torch.nn as nn
from utils.conv_type import ConvMask, LinearMask

def prune_mag(model, density):
    score_list = {}
    for n, m in model.named_modules():
        # torch.cat([torch.flatten(v) for v in self.scores.values()])
        if isinstance(m, (ConvMask, LinearMask)):
            score_list[n] = (m.mask.to(m.weight.device) * m.weight).detach().abs_()

    global_scores = torch.cat([torch.flatten(v) for v in score_list.values()])
    k = int((1 - density) * global_scores.numel())
    threshold, _ = torch.kthvalue(global_scores, k)

    if not k < 1:
        total_num = 0
        total_den = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask, LinearMask)):
                score = score_list[n].to(m.weight.device)
                zero = torch.tensor([0.]).to(m.weight.device)
                one = torch.tensor([1.]).to(m.weight.device)
                m.mask = torch.where(score <= threshold, zero, one)
                total_num += (m.mask == 1).sum()
                total_den += m.mask.numel()

    print('Overall model density after magnitude pruning at current iteration = ', total_num / total_den)
    return model

def prune_random_erk(model, density):

    sparsity_list = []
    num_params_list = []
    total_params = 0
    score_list = {}

    for n, m in model.named_modules():
        if isinstance(m, (ConvMask)):
            score_list[n] = (m.mask.to(m.weight.device) * torch.randn_like(m.weight).to(m.weight.device)).detach().abs_()

            sparsity_list.append(torch.tensor(m.weight.shape).sum() / m.weight.numel())
            num_params_list.append(m.weight.numel())
            total_params += m.weight.numel()
    
    num_params_kept = (torch.tensor(sparsity_list) * torch.tensor(num_params_list)).sum()
    num_params_to_keep = total_params * density
    C = num_params_to_keep / num_params_kept
    print('Factor: ', C)
    sparsity_list = [torch.clamp(C*s, 0, 1) for s in sparsity_list]

    total_num = 0
    total_den = 0
    cnt = 0
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask)):
            global_scores = torch.flatten(score_list[n])
            k = int((1 - sparsity_list[cnt]) * global_scores.numel())
            if k == 0:
                threshold = 0
            else: 
                threshold, _ = torch.kthvalue(global_scores, k)
            print('Layer', n, ' params ', k, global_scores.numel())

            score = score_list[n].to(m.weight.device)
            zero = torch.tensor([0.]).to(m.weight.device)
            one = torch.tensor([1.]).to(m.weight.device)
            m.mask = torch.where(score <= threshold, zero, one)
            total_num += (m.mask == 1).sum()
            total_den += m.mask.numel()
            cnt += 1

    print('Overall model density after random global (ERK) pruning at current iteration = ', total_num / total_den)
    return model

def prune_snip(model, trainloader, density):
    criterion = nn.CrossEntropyLoss()
    for i, (images, target) in enumerate(trainloader):
        images = images.to(torch.device('cuda'))
        target = target.to(torch.device('cuda')).long()
        model.zero_grad()
        output = model(images)
        criterion(output, target).backward()
        break
    
    score_list = {}
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask)):
            score_list[n] = (m.weight.grad * m.weight * m.mask.to(m.weight.device)).detach().abs_()
    
    global_scores = torch.cat([torch.flatten(v) for v in score_list.values()])
    k = int((1 - density) * global_scores.numel())
    threshold, _ = torch.kthvalue(global_scores, k)

    if not k < 1:
        total_num = 0
        total_den = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                score = score_list[n].to(m.weight.device)
                zero = torch.tensor([0.]).to(m.weight.device)
                one = torch.tensor([1.]).to(m.weight.device)
                m.mask = torch.where(score <= threshold, zero, one)
                total_num += (m.mask == 1).sum()
                total_den += m.mask.numel()

    print('Overall model density after snip pruning at current iteration = ', total_num / total_den)
    return model


def prune_synflow(model, trainloader, density):

    @torch.no_grad()
    def linearize(model):
        # model.double()
        signs = {}
        for name, param in model.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(model, signs):
        # model.float()
        for n, param in model.state_dict().items():
            param.mul_(signs[n])
    
    signs = linearize(model)

    # (data, _) = next(iter(trainloader))
    for i, (images, target) in enumerate(trainloader):
        images = images.to(torch.device('cuda'))
        target = target.to(torch.device('cuda')).long()
        input_dim = list(images[0,:].shape)
        input = torch.ones([1] + input_dim).to('cuda')#, dtype=torch.float64).to(device)
        output = model(input)
        torch.sum(output).backward()
        break
    
    score_list = {}
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask)):
            score_list[n] = (m.mask.to(m.weight.device) * m.weight.grad * m.weight).detach().abs_()
    
    model.zero_grad()

    nonlinearize(model, signs)

    global_scores = torch.cat([torch.flatten(v) for v in score_list.values()])
    k = int((1 - density) * global_scores.numel())
    threshold, _ = torch.kthvalue(global_scores, k)

    if not k < 1:
        total_num = 0
        total_den = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                score = score_list[n].to(m.weight.device)
                zero = torch.tensor([0.]).to(m.weight.device)
                one = torch.tensor([1.]).to(m.weight.device)
                m.mask = torch.where(score <= threshold, zero, one)
                total_num += (m.mask == 1).sum()
                total_den += m.mask.numel()

    print('Overall model density after synflow pruning at current iteration = ', total_num / total_den)
    return model

def prune_random_balanced(model, density):

        total_params = 0
        l = 0
        sparsity_list = []
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask, LinearMask)):
                total_params += m.weight.numel()
                l += 1
        L = l
        X = density * total_params / l
        score_list = {}
        l = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask, LinearMask)):
                score_list[n] = (m.mask.to(m.weight.device) * torch.randn_like(m.weight).to(m.weight.device)).detach().abs_()

                if X / m.weight.numel() < 1.0:
                    sparsity_list.append(X / m.weight.numel())
                else: 
                    sparsity_list.append(1)
                    # correction for taking care of exact sparsity
                    diff = X - m.mask.numel()
                    X = X + diff / (L - l)
                l += 1

        total_num = 0
        total_den = 0
        cnt = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask, LinearMask)):
                global_scores = torch.flatten(score_list[n])
                k = int((1 - sparsity_list[cnt]) * global_scores.numel())
                if k == 0:
                    threshold = 0
                else: 
                    threshold, _ = torch.kthvalue(global_scores, k)
                print('Layer', n, ' params ', k, global_scores.numel())

                score = score_list[n].to(m.weight.device)
                zero = torch.tensor([0.]).to(m.weight.device)
                one = torch.tensor([1.]).to(m.weight.device)
                m.mask = torch.where(score <= threshold, zero, one)
                total_num += (m.mask == 1).sum()
                total_den += m.mask.numel()
                cnt += 1

        print('Overall model density after random global (balanced) pruning at current iteration = ', total_num / total_den)
        return model

def prune_er_erk(model, er_sparse_init):
    sparsity_list = []
    num_params_list = []
    total_params = 0
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, LinearMask)):
            sparsity_list.append(torch.tensor(m.weight.shape).sum() / m.weight.numel())
            num_params_list.append(m.weight.numel())
            total_params += m.weight.numel()
    
    num_params_kept = (torch.tensor(sparsity_list) * torch.tensor(num_params_list)).sum()
    num_params_to_keep = total_params * er_sparse_init
    C = num_params_to_keep / num_params_kept
    sparsity_list = [torch.clamp(C*s, 0, 1) for s in sparsity_list]
    l = 0
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, LinearMask)):
            m.set_er_mask(sparsity_list[l])
            l += 1
    print(sparsity_list)

def prune_er_balanced(model, er_sparse_init):
    total_params = 0
    l = 0
    sparsity_list = []
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, LinearMask)):
            total_params += m.weight.numel()
            l += 1
    L = l
    X = er_sparse_init * total_params / l
    l = 0
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, LinearMask)):
            if X / m.weight.numel() < 1.0:
                sparsity_list.append(X / m.weight.numel())
            else: 
                sparsity_list.append(1)
                # correction for taking care of exact sparsity
                diff = X - m.mask.numel()
                X = X + diff / (L - l)
            l += 1

    l = 0
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, LinearMask)):
            m.set_er_mask(sparsity_list[l])
            l += 1
    print(sparsity_list)

