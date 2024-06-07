## pythonic imports
import numpy as np
import os

## torch
import torch
import torch.nn as nn
## torchvision

## file-based imports
import torchvision.models as models
from utils.conv_type import * 
from utils.harness_params import *
from utils.dataset import CIFARLoader
from utils.custom_models import *
## fastargs
from fastargs import get_current_config
from fastargs.decorators import param

## ffcv
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.decoders import RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

get_current_params()

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

def get_sparsity(model):
        # finds the current density of the model and returns the density scalar value
        nz = 0
        total = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask, Conv1dMask)):
                nz += m.mask.sum()
                total += m.mask.numel()
        
        return nz / total
 
class PruningStuff:
    def __init__(self, model=None):
        self.this_device = 'cuda:0'
        self.config = get_current_config()

        self.train_loader = self.create_train_loader()
        if model is None:
            self.model = self.acquire_model()
        else:
            self.model = model
        self.criterion = nn.CrossEntropyLoss()

    @param('dataset.batch_size')
    @param('dataset.num_workers')
    @param('dataset.data_root')
    @param('dataset.dataset_name')
    def create_train_loader(self, batch_size, num_workers, data_root, dataset_name):
        if not 'CIFAR' in dataset_name:
            train_image_pipeline = [RandomResizedCropRGBImageDecoder((224, 224)),
                                RandomHorizontalFlip(),
                                ToTensor(),
                                ToDevice(torch.device('cuda:0'), non_blocking=True),
                                ToTorchImage(),
                                NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)]

            label_pipeline = [IntDecoder(),
                                ToTensor(),
                                Squeeze(),
                                ToDevice(torch.device('cuda:0'), non_blocking=True)]


            train_loader = Loader(os.path.join(data_root, 'train_500_0.50_90.beton'),
                                batch_size  = batch_size,
                                num_workers = num_workers,
                                order       = OrderOption.RANDOM,
                                os_cache    = True,
                                drop_last   = True,
                                pipelines   = { 'image' : train_image_pipeline,
                                                'label' : label_pipeline},
                                )
        else:
            loader = CIFARLoader(distributed=False)
            train_loader = loader.train_loader

        return train_loader

    @param("model_params.model_name")
    @param("dataset.num_classes")
    @param("dataset.dataset_name")
    def acquire_model(self, model_name, num_classes, dataset_name):
        if model_name == 'preresnet':
            model = PreActResNet(block=PreActBlock, num_blocks=[2, 2, 2, 2])
        else:
            model = getattr(models, model_name)(num_classes=num_classes)

        if ('CIFAR' in dataset_name) and ('resnet' in model_name):
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            model.maxpool = nn.Identity()
        if ('CIFAR' in dataset_name) and ('vgg11' in model_name):
            model.avgpool = self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
            model.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes))

        
        replace_layers(model=model)

        model.to(self.this_device)
        if self.config['prune_params.er_method'] != 'just dont':
            model = self.prune_at_initialization(model=model)
        return model

    @param('prune_params.er_method')
    @param('prune_params.er_init')
    def prune_at_initialization(self, er_method, er_init, model):
        er_method_name = f'prune_{er_method}'
        pruner_method = globals().get(er_method_name)
        if er_method in {'synflow', 'snip'}:
            model = pruner_method(model, self.train_loader, er_init)
        else:
            pruner_method(model, er_init)
        
        return model
        
    @param('prune_params.prune_method')
    def level_pruner(self, prune_method, density):
        print('---' * 20)
        print(f'Density before pruning: {get_sparsity(self.model)}')
        print('---' * 20)

        prune_method_name = f'prune_{prune_method}'
        pruner_method = globals().get(prune_method_name)
        if prune_method in {'synflow', 'snip'}:
            self.model = pruner_method(self.model, self.train_loader, density)
        else:
            pruner_method(self.model, density)

        print('---' * 20)
        print(f'Density after pruning: {get_sparsity(self.model)}')
        print('---' * 20)
    def load_from_ckpt(self, path):
        self.model.load_state_dict(torch.load(path))


def prune_mag(model, density):
    score_list = {}
    for n, m in model.named_modules():

        if isinstance(m, (ConvMask, Conv1dMask)):
            score_list[n] = (m.mask.to(m.weight.device) * m.weight).detach().abs_()

    global_scores = torch.cat([torch.flatten(v) for v in score_list.values()])
    k = int((1 - density) * global_scores.numel())
    threshold, _ = torch.kthvalue(global_scores, k)

    if not k < 1:
        total_num = 0
        total_den = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask, Conv1dMask)):
                score = score_list[n].to(m.weight.device)
                zero = torch.tensor([0.]).to(m.weight.device)
                one = torch.tensor([1.]).to(m.weight.device)
                m.mask = torch.where(score <= threshold, zero, one)
                total_num += (m.mask == 1).sum()
                total_den += m.mask.numel()
    print('Overall model density after magnitude pruning at current iteration = ', (total_num / total_den).item())
    return model

def prune_random_erk(model, density):
    sparsity_list = []
    num_params_list = []
    total_params = 0
    score_list = {}

    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, Conv1dMask)):
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
        if isinstance(m, (ConvMask, Conv1dMask)):
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
        if isinstance(m, (ConvMask, Conv1dMask)):
            score_list[n] = (m.weight.grad * m.weight * m.mask.to(m.weight.device)).detach().abs_()
    
    global_scores = torch.cat([torch.flatten(v) for v in score_list.values()])
    k = int((1 - density) * global_scores.numel())
    threshold, _ = torch.kthvalue(global_scores, k)

    if not k < 1:
        total_num = 0
        total_den = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask, Conv1dMask)):
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
        if isinstance(m, (ConvMask, Conv1dMask)):
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
            if isinstance(m, (ConvMask, Conv1dMask)):
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
            if isinstance(m, (ConvMask, Conv1dMask)):
                total_params += m.weight.numel()
                l += 1
        L = l
        X = density * total_params / l
        score_list = {}
        l = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask, Conv1dMask)):
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
            if isinstance(m, (ConvMask, Conv1dMask)):
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
        if isinstance(m, (ConvMask, Conv1dMask)):
            sparsity_list.append(torch.tensor(m.weight.shape).sum() / m.weight.numel())
            num_params_list.append(m.weight.numel())
            total_params += m.weight.numel()
    
    num_params_kept = (torch.tensor(sparsity_list) * torch.tensor(num_params_list)).sum()
    num_params_to_keep = total_params * er_sparse_init
    C = num_params_to_keep / num_params_kept
    sparsity_list = [torch.clamp(C*s, 0, 1) for s in sparsity_list]
    l = 0
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, Conv1dMask)):
            m.set_er_mask(sparsity_list[l])
            l += 1
    print(sparsity_list)

def prune_er_balanced(model, er_sparse_init):
    total_params = 0
    l = 0
    sparsity_list = []
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, Conv1dMask)):
            total_params += m.weight.numel()
            l += 1
    L = l
    X = er_sparse_init * total_params / l
    l = 0
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, Conv1dMask)):
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
        if isinstance(m, (ConvMask, Conv1dMask)):
            m.set_er_mask(sparsity_list[l])
            l += 1
    print(sparsity_list)