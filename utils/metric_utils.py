import numpy as np
import pandas as pd
import tqdm
import copy

import torch
import torch.nn as nn
from torch.amp import autocast

from utils.pruning_utils import PruningStuff
from fastargs import get_current_config
from utils.harness_utils import *

from utils.airbench_loader import CifarLoader
from pyhessian import hessian

get_current_config()

def test(model):
    """Evaluate the model on the test set.

    Returns:
        (float, float): Test loss and accuracy.
    """
    config = get_current_config()
    this_device = 'cuda'
    test_loader = CifarLoader(path='./cifar10', batch_size=512, train=False, dataset=config['dataset.dataset_name'])

    model.to(this_device)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    tloader = tqdm.tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for inputs, targets in tloader:
            inputs, targets = inputs.to(this_device), targets.to(this_device)
            with autocast(dtype=torch.bfloat16, device_type=this_device):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / total

    return test_loss, accuracy

@param('prune_params.er_method')
@param('prune_params.prune_method')
def compute_perturbation(er_method, prune_method, model, density, perturbation_table, level):
    is_iterative = (prune_method != 'just dont') and (er_method == 'just dont') 
    if is_iterative:
        pruning_harness = PruningStuff(model=model)
    else:
        pruning_harness = PruningStuff()

    pre_sparsity = print_sparsity_info(pruning_harness.model, verbose=False)
    pre_test_acc = test(pruning_harness.model)[1]

    if is_iterative:
        pruning_harness.level_pruner(density=density)
    else:
        pruning_harness.prune_at_initialization(er_init=density)

    
    post_sparsity = print_sparsity_info(pruning_harness.model, verbose=False)
    
    post_test_acc = test(pruning_harness.model)[1]

    delta = post_test_acc - pre_test_acc

    perturbation_table.field_names = ["Level", "Pre-Sparsity", "Post-Sparsity", "Accuracy (pre-pruning)", "Accuracy (post-pruning)", "Perturbation"]
    perturbation_table.add_row([level, pre_sparsity, post_sparsity, pre_test_acc, post_test_acc, delta])
    
    print(perturbation_table)

    return {'level': level, 
            'pre_sparsity'  : pre_sparsity, 
            'post_sparsity' : post_sparsity,
            'pre_test_acc'  : pre_test_acc,
            'post_test_acc' : post_test_acc,
            'perturbation'  : delta} 


class LinearModeConnectivity:
    def __init__(self, expt_path : str, model : nn.Module):
        super(LinearModeConnectivity, self).__init__()

        self.config = get_current_config()

        self.this_device = 'cuda'
        self.train_loader = CifarLoader(path='./cifar10', batch_size=512, train=True, aug={'flip' : True, 'translate' : 2}, altflip=True, dataset=self.config['dataset.dataset_name'])
        self.test_loader = CifarLoader(path='./cifar10', batch_size=512, train=False, dataset=self.config['dataset.dataset_name'])
        
        self.alphas = np.arange(0, 1, 0.1)
        self.criterion = nn.CrossEntropyLoss()

        self.expt_path = expt_path
        self.model = model
        self.model.to(self.this_device)

        self.save_path = os.path.join(self.expt_path, 'metrics', 'linear_mode')

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
    def gen_masked_dict(self, level):
        model = copy.deepcopy(self.model)
        model.load_state_dict(torch.load(os.path.join(self.expt_path, 'checkpoints', f'model_level_{level}.pt')))

        for m in model.modules():
            if isinstance(m, (ConvMask, Conv1dMask, LinearMask)):
                m.weight.data *= m.mask
        
        return model.state_dict()

    def test(self) -> Tuple[float, float]:
        """Evaluate the model on the test set.

        Returns:
            (float, float): Test loss and accuracy.
        """
        model = self.model

        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        tloader = tqdm.tqdm(self.test_loader, desc="Testing")
        with torch.no_grad():
            for inputs, targets in tloader:
                inputs, targets = inputs.to(self.this_device), targets.to(self.this_device)
                with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
                    outputs = model(inputs)
                    loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_loss /= len(self.test_loader)
        accuracy = 100.0 * correct / total

        return test_loss, accuracy
    
    def train(self):
        """Train the model for one epoch.

        Returns:
            (float, float): Training loss and accuracy.
        """
        model = self.model
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        tepoch = tqdm.tqdm(self.train_loader, unit="batch")

        for inputs, targets in tepoch:
            inputs, targets = inputs.to(self.this_device), targets.to(self.this_device)
            with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss /= len(self.train_loader)
        accuracy = 100.0 * (correct / total)
        return train_loss, accuracy
        
    
    def gen_linear_mode(self, level1, level2):
        
        data_dict = {'alpha' : [],
                     'train_loss' : [],
                     'test_loss' : [],
                     'train_acc' : [],
                     'test_acc' : []
                     }
        print(f'Computing Linear mode for: {level1} and {level2}')        
        m1_params = self.gen_masked_dict(level=level1)
        m2_params = self.gen_masked_dict(level=level2)

        for alpha in self.alphas:
            param_dict = {}
            for n in m1_params.keys():
                param_dict[n] = alpha * (m1_params[n]) + (1 - alpha) * (m2_params[n])
            
            model = copy.deepcopy(self.model)
            model.load_state_dict(param_dict)

            train_loss, train_acc = self.train()
            test_loss, test_acc = self.test()

            data_dict['alpha'].append(alpha)
            data_dict['train_loss'].append(train_loss)
            data_dict['test_loss'].append(test_loss)
            data_dict['train_acc'].append(train_acc)
            data_dict['test_acc'].append(test_acc)

        save_path = os.path.join(self.save_path, f'linear_mode_{level1}_{level2}.csv')
        pd.DataFrame(data_dict).to_csv(save_path, index=False)
        
def hessian_trace(train_loader, model):
    config = get_current_config()
    train_loader = CifarLoader(path='./cifar10', batch_size=1000, train=True, aug={'flip' : True, 'translate' : 2}, altflip=True, drop_last=True, dataset=config['dataset.dataset_name'])

    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        break
    with autocast(dtype=torch.bfloat16, device_type='cuda'):
        images = images.cuda()

        target = target.cuda().long()

        hessian_comp = hessian(model, criterion, data=(images, target), cuda=True)
        trace = hessian_comp.trace()
    trace = np.array(trace).mean()
    del hessian_comp
    print('trace of hessian: ', trace)
    return trace
