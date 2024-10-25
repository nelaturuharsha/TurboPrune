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
from utils.mask_layers import ConvMask, Conv1dMask, LinearMask

from utils.dataset import AirbenchLoaders
from pyhessian import hessian


from rich.table import Table
from rich.console import Console
import pandas as pd
import os


def test(model):
    """Evaluate the model on the test set.

    Returns:
        (float, float): Test loss and accuracy.
    """
    config = get_current_config()
    this_device = 'cuda'
    loaders = AirbenchLoaders()
    test_loader = loaders.test_loader

    model.to(this_device)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    tloader = tqdm.tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for inputs, targets in tloader:
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


def compute_lr_perturbation(level, metric_list, expt_dir, prefix):
    console = Console()
    perturbation_tree = Tree("Perturbation Metrics")

    for i in range(len(metric_list) - 1):
        current_metric = metric_list[i]
        next_metric = metric_list[i + 1]

        current_cycle = current_metric['cycle_num']
        next_cycle = next_metric['cycle_num']
        # Extract relevant data
        _, current_epoch = current_metric['epochs_iter']
        next_epoch, _ = next_metric['epochs_iter']

        current_print_epoch = current_metric[f'epoch_{current_epoch}_current_epoch']
        next_print_epoch = next_metric[f'epoch_{next_epoch}_current_epoch']

        current_acc = current_metric[f'epoch_{current_epoch}_test_acc']
        next_acc = next_metric[f'epoch_{next_epoch}_test_acc']

        current_max_eig = current_metric[f'epoch_{current_epoch}_max_eig']
        next_max_eig = next_metric[f'epoch_{next_epoch}_max_eig']

        delta_acc = next_acc - current_acc
        delta_max_eig = next_max_eig - current_max_eig

        data = {
            'level': level,
            'prev_cycle': current_cycle,
            'next_cycle': next_cycle,
            'prev_epoch': current_print_epoch,
            'next_epoch': next_print_epoch,
            'prev_acc': f'{current_acc:.4f}',
            'next_acc': f'{next_acc:.4f}',
            'prev_max_eig': f'{current_max_eig:.4f}',
            'next_max_eig': f'{next_max_eig:.4f}',
            'delta_acc': f'{delta_acc:.4f}',
            'delta_max_eig': f'{delta_max_eig:.4f}'
        }

        # Write to CSV
        csv_path = os.path.join(expt_dir, 'metrics', f'{prefix}_perturbation_metrics.csv')
        pd.DataFrame([data]).to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

        # Add to tree for printing
        cycle_branch = perturbation_tree.add(f"Level {level}, Cycle {current_cycle} -> Cycle {next_cycle}", style="cyan")
        cycle_branch.add(f"Epochs: {current_print_epoch} -> {next_print_epoch}", style="magenta")
        cycle_branch.add(f"Accuracy: {current_acc:.2f} -> {next_acc:.2f} (Δ: {delta_acc:.2f})", style="yellow")
        cycle_branch.add(f"Max Eigenvalue: {current_max_eig:.2f} -> {next_max_eig:.2f} (Δ: {delta_max_eig:.2f})", style="green")

    # Print the tree
    console.print(Panel(perturbation_tree, title="Perturbation Metrics", border_style="cyan"))


def compute_level(epoch, level, cycle, model, density, is_iterative, packaged):
    config = get_current_config()

    densities = generate_densities(current_sparsity=0.0 if is_iterative else [config['prune_params.er_init']])

    expt_dir, prefix = packaged

    model_in_question = copy.deepcopy(model)
    print('Computing hessian max eigenvalue')
    hessian_max_eig = hessian_max_eigenvalue(model_in_question)

    if is_iterative:
        pruning_harness = PruningStuff(model=model_in_question)
    else:
        pruning_harness = PruningStuff()

    pre_sparsity = pruning_harness.model.get_overall_sparsity()
    pre_test_acc = test(pruning_harness.model)[1]

    pruning_harness.prune_the_model(prune_method=config['prune_params.prune_method'], density=densities[level+1])

    post_sparsity = pruning_harness.model.get_overall_sparsity()
    
    post_test_acc = test(pruning_harness.model)[1]

    delta = post_test_acc - pre_test_acc

    console = Console()
    perturbation_table = Table(title="Perturbation Metrics")
    perturbation_table.add_column("Level", style="cyan")
    perturbation_table.add_column("Pre-Sparsity", style="magenta")
    perturbation_table.add_column("Post-Sparsity", style="green")
    perturbation_table.add_column("Accuracy (pre-pruning)", style="red")
    perturbation_table.add_column("Accuracy (post-pruning)", style="blue")
    perturbation_table.add_column("Perturbation", style="yellow")

    perturbation_table.add_row(str(level), f"{pre_sparsity:.4f}", f"{post_sparsity:.4f}", 
                               f"{pre_test_acc:.2f}", f"{post_test_acc:.2f}", f"{delta:.2f}")
    console.print('\n')
    console.print(perturbation_table)
    console.print('\n')

    # Convert to DataFrame and append to CSV
    df = pd.DataFrame([{
        'level': level, 
        'epoch': epoch,
        'cycle': cycle,
        'pre_sparsity': pre_sparsity, 
        'post_sparsity': post_sparsity,
        'pre_test_acc': pre_test_acc,
        'post_test_acc': post_test_acc,
        'perturbation': delta,
        'hessian_max_eig': hessian_max_eig
    }])

    csv_path = os.path.join(expt_dir, 'metrics', f'{prefix}_{epoch}_{cycle}_{level}_perturbation_metrics.csv')
    df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

    
class LinearModeConnectivity:
    def __init__(self, expt_path : str, model : nn.Module):
        super(LinearModeConnectivity, self).__init__()

        self.config = get_current_config()

        self.this_device = 'cuda'
        loaders = AirbenchLoaders()
        self.train_loader = loaders.train_loader
        self.test_loader = loaders.test_loader
        
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
        
def hessian_trace(model):
    config = get_current_config()
    loaders = AirbenchLoaders(batch_size=1024)
    train_loader = loaders.train_loader

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

def hessian_max_eigenvalue(model):
    loaders = AirbenchLoaders(batch_size=1024)
    train_loader = loaders.train_loader

    criterion = nn.CrossEntropyLoss()
    model.train()
    model.cuda()
    
    for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        break
    
    images = images.cuda()
    target = target.cuda()

    hessian_comp = hessian(model, criterion, data=(images, target), cuda=True)
    
    # Compute top eigenvalue
    top_eigenvalues, _ = hessian_comp.eigenvalues()
    max_eigenvalue = top_eigenvalues[0]
    
    del hessian_comp
    print('max eigenvalue of hessian: ', max_eigenvalue)
    return max_eigenvalue
