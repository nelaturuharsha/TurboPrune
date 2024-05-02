import os
import pathlib
import random
import shutil
import time
import json
import numpy as np
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import save_checkpoint, get_lr, LabelSmoothing
from utils.schedulers import get_policy, cosine_lr, assign_learning_rate, warmup_lr, constant_lr, multistep_lr
from utils.conv_type import STRConv, STRConvER, ConvER, ConvMask
from utils.conv_type import sparseFunction

from trainer import train, validate, get_preds, hessian_trace



class CompensatePrune:
    def __init__(self, model):
        # set the current teacher
        self.curr_teacher = model
    
    def step(self, student):
        
        student_weight = []
        teacher_weight = []
        student_mask = []
        teacher_mask = []
        student_bn = []
        teacher_bn = []

        for (n, m), (_, ref) in zip(student.named_modules(), self.curr_teacher.named_modules()):
            
            if isinstance(m, ConvMask) and not ('downsample' in n):
                student_weight.append(m.weight.detach().clone())
                student_mask.append(m.mask)
                teacher_weight.append(ref.weight.detach().clone() * ref.mask)
                teacher_mask.append(ref.mask)

            if isinstance(m, nn.BatchNorm2d) and not ('downsample' in n):
                student_bn.append([m.weight.detach().clone(), m.bias.detach().clone(), m.running_mean, m.running_var])
                teacher_bn.append([ref.weight.detach().clone(), ref.bias.detach().clone(), ref.running_mean, ref.running_var])
        
        # now optimize layerwise
        L = len(student_weight)
        print(len(student_weight), len(student_bn), len(teacher_weight), len(teacher_bn))
        for i in range(L-1):
            if i < L-2:
                l1t = teacher_weight[i]
                l2t = teacher_weight[i+1]
                l1t_bn = teacher_bn[i]
                l2t_bn = teacher_bn[i+1]
                l1t_bn_gamma = l1t_bn[0]
                l1t_bn_beta = l1t_bn[1]
                l2t_bn_gamma = l2t_bn[0]
                l2t_bn_beta = l2t_bn[1]

                m2t = teacher_mask[i+1]
                m1t = teacher_mask[i]

                m1s = student_mask[i]
                m2s = student_mask[i+1]


                l1s_bn = student_bn[i]
                l1s_bn_gamma = nn.Parameter(l1s_bn[0], requires_grad=True)
                l1s_bn_beta = nn.Parameter(l1s_bn[1], requires_grad=True)
                l2s_bn = student_bn[i+1]
                l2s_bn_gamma = nn.Parameter(l2s_bn[0], requires_grad=True)
                l2s_bn_beta = nn.Parameter(l2s_bn[1], requires_grad=True)

                l1s = nn.Parameter(student_weight[i], requires_grad=True)
                l2s = nn.Parameter(student_weight[i+1], requires_grad=True)

                optimizer = torch.optim.LBFGS([l1s, l2s, l1s_bn_gamma, l1s_bn_beta, l2s_bn_gamma, l2s_bn_beta], lr=1, max_iter=100)
                criterion = nn.MSELoss()
                print(l1s.shape, l1t.shape, l1s_bn_gamma.shape, l1t_bn_gamma.shape)
                print('L1S value before opt: ', (m1s * l1s).norm().detach().item(), (m2t * l2s).norm().detach().item())
                for iters in range(300):
                    def closure():
            
                        optimizer.zero_grad()
                        x = torch.FloatTensor(256, l1t.shape[1], 32, 32).uniform_(-2, 2).to(l1t.device)
                        student_act = F.conv2d(x, l1s * m1s, stride=1, padding=1)
                        student_act = F.relu(l1s_bn_gamma[None, :, None, None] * ((student_act - l1s_bn[2][None, :, None, None]) /l1s_bn[3][None, :, None, None]) + l1s_bn_beta[None, :, None, None])
                        student_act = F.conv2d(student_act, l2s * m2t, stride=1, padding=1)
                        student_act = l2s_bn_gamma[None, :, None, None] * ((student_act - l2s_bn[2][None, :, None, None]) /l2s_bn[3][None, :, None, None]) + l2s_bn_beta[None, :, None, None]

                        target_act = F.conv2d(x, m1t * l1t, stride=1, padding=1)
                        target_act = F.relu(l1t_bn_gamma[None, :, None, None] * ((target_act - l1t_bn[2][None, :, None, None]) /l1t_bn[3][None, :, None, None]) + l1t_bn_beta[None, :, None, None])
                        target_act = F.conv2d(target_act, m2t * l2t, stride=1, padding=1)
                        target_act = l2t_bn_gamma[None, :, None, None] * ((target_act - l2t_bn[2][None, :, None, None]) /l2t_bn[3][None, :, None, None]) + l2t_bn_beta[None, :, None, None]

                        # loss = ((target_act - student_act)**2).sum()
                        loss = criterion(target_act, student_act)
                        loss.backward()
                        return loss

                    optimizer.step(closure)
                    if iters % 50 == 49:
                        optimizer.step(closure)
                        loss = closure()
                        print('Loss = ', loss)

                print('Completed LBFGS optimization for layer: ', i)
                print('L1S value after opt: ', l1s.norm().detach().item(), l2s.norm().detach().item())
                
                student_weight[i] = l1s
                student_weight[i+1] = l2s
                student_bn[i] = [l1s_bn_gamma, l1s_bn_beta, l1s_bn[2], l1s_bn[3]]
                student_bn[i+1] = [l2s_bn_gamma, l2s_bn_beta, l2s_bn[2], l2s_bn[3]]

            #########
            else: 
                l1t = teacher_weight[i]
                l2t = teacher_weight[i+1]
                l1t_bn = teacher_bn[i]
                
                l1t_bn_gamma = l1t_bn[0]
                l1t_bn_beta = l1t_bn[1]
                

                m2t = teacher_mask[i+1]
                m1t = teacher_mask[i]

                m1s = student_mask[i]
                m2s = student_mask[i+1]


                l1s_bn = student_bn[i]
                l1s_bn_gamma = nn.Parameter(l1s_bn[0], requires_grad=True)
                l1s_bn_beta = nn.Parameter(l1s_bn[1], requires_grad=True)
                

                l1s = nn.Parameter(student_weight[i], requires_grad=True)
                l2s = nn.Parameter(student_weight[i+1], requires_grad=True)

                optimizer = torch.optim.LBFGS([l1s, l2s, l1s_bn_gamma, l1s_bn_beta], max_iter=100)
                criterion = nn.MSELoss()
                print(l1s.shape, l1t.shape, l1s_bn_gamma.shape, l1t_bn_gamma.shape)
                print('L1S value before opt for the last layer without BN: ', (m1s * l1s).norm().detach().item(), (m2t * l2s).norm().detach().item())
                for iters in range(200):
                    def closure():
            
                        optimizer.zero_grad()
                        x = torch.FloatTensor(256, l1t.shape[1], 32, 32).uniform_(-3, 3).to(l1t.device)
                        student_act = F.conv2d(x, l1s * m1s, stride=1, padding=1)
                        student_act = F.relu(l1s_bn_gamma[None, :, None, None] * ((student_act - l1s_bn[2][None, :, None, None]) /l1s_bn[3][None, :, None, None]) + l1s_bn_beta[None, :, None, None])
                        student_act = F.conv2d(student_act, l2s * m2t, stride=1, padding=1)

                        target_act = F.conv2d(x, l1t * m1t, stride=1, padding=1)
                        target_act = F.relu(l1t_bn_gamma[None, :, None, None] * ((target_act - l1t_bn[2][None, :, None, None]) /l1t_bn[3][None, :, None, None]) + l1t_bn_beta[None, :, None, None])
                        target_act = F.conv2d(target_act, l2t * m2t, stride=1, padding=1)

                        # loss = ((target_act - student_act)**2).sum()
                        loss = criterion(target_act, student_act)
                        loss.backward()
                        return loss

                    if iters % 50 == 49:
                        optimizer.step(closure)
                        loss = closure()
                        print('Loss = ', loss)

                print('Completed LBFGS optimization for layer: ', i)
                print('L1S value after opt: ', l1s.norm().detach().item(), l2s.norm().detach().item())
                
                student_weight[i] = l1s
                student_weight[i+1] = l2s
                student_bn[i] = [l1s_bn_gamma, l1s_bn_beta, l1s_bn[2], l1s_bn[3]]
                
        self.curr_teacher = student
        cnt = 0
        for (n, m), (_, ref) in zip(student.named_modules(), self.curr_teacher.named_modules()):
            if isinstance(m, ConvMask) and not ('downsample' in n):
                m.weight = student_weight[cnt]
                cnt += 1
        cnt = 0
        for (n, m), (_, ref) in zip(student.named_modules(), self.curr_teacher.named_modules()):
            if isinstance(m, nn.BatchNorm2d) and not ('downsample' in n):
                m.weight = student_bn[cnt][0]
                m.bias = student_bn[cnt][1]
                cnt += 1
                
        return student
