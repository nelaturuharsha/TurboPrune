import torch
import os
import prettytable
from utils.conv_type import ConvMask, Conv1dMask
from datetime import datetime
import uuid

import numpy as np
import random
import yaml


from utils.pruning_utils import PruningStuff
from utils.dataset import CIFARLoader

from fastargs import get_current_config
from fastargs.decorators import param

def reset_weights(expt_dir, model, training_type):
    if training_type == 'imp':
        original_dict = torch.load(os.path.join(expt_dir, 'checkpoints', 'model_init.pt'))
    elif training_type == 'wr':
        original_dict = torch.load(os.path.join(expt_dir, 'checkpoints', 'model_rewind.pt'))
    else:
        print('probably LRR, aint nothing to do')
        return model    
    original_weights = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), original_dict.items()))
    model_dict = model.state_dict()

    model_dict.update(original_weights)
    model.load_state_dict(model_dict)

    return model

def reset_optimizer(expt_dir, optimizer, training_type):
    if training_type in {'imp', 'lrr'}:
        optimizer.load_state_dict(torch.load(os.path.join(expt_dir, 'artifacts', 'optimizer_init.pt')))
    elif training_type == 'wr':
        optimizer.load_state_dict(torch.load(os.path.join(expt_dir, 'artifacts', 'optimizer_rewind.pt')))
    
    return optimizer
    

def reset_only_weights(expt_dir, ckpt_name, model):
    original_dict = torch.load(os.path.join(expt_dir, 'checkpoints', ckpt_name))
    original_weights = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), original_dict.items()))
    model_dict = model.state_dict()

    model_dict.update(original_weights)
    model.load_state_dict(model_dict)

def reset_only_masks(expt_dir, ckpt_name, model):
    original_dict = torch.load(os.path.join(expt_dir, 'checkpoints', ckpt_name))
    original_weights = dict(filter(lambda v: (v[0].endswith(('.mask'))), original_dict.items()))
    model_dict = model.state_dict()

    model_dict.update(original_weights)
    model.load_state_dict(model_dict)

def compute_sparsity(tensor):
    remaining = tensor.sum()
    total = tensor.numel()
    sparsity = 1.0 - (remaining / total)
    return sparsity, remaining, total

def print_sparsity_info(model, verbose=True):
        my_table = prettytable.PrettyTable()
        my_table.field_names = ["Layer Name", "Layer Sparsity", "Density", "Non-zero/Total"]
        total_params = 0
        total_params_kept = 0

        for (name, layer) in model.named_modules():
             if isinstance(layer, (ConvMask, Conv1dMask)):
                weight_mask = layer.mask
                sparsity, remaining, total = compute_sparsity(weight_mask)
                my_table.add_row([name, sparsity.item(), 1-sparsity.item(), f'{remaining}/{total}'])
                total_params += total
                total_params_kept += remaining
        
        overall_sparsity = 1-(total_params_kept/total_params)

        if verbose:
            print(my_table)
            print('-----------')
            print(f"Overall Sparsity of All Layers: {overall_sparsity:.4f}")
            print('-----------')
        
        return overall_sparsity

@param('experiment_params.base_dir')
@param('experiment_params.resume_level')
@param('experiment_params.resume_expt_name')
def gen_expt_dir(base_dir, resume_level, resume_expt_name):
    if resume_level != 0 and resume_expt_name:
        expt_dir = os.path.join(base_dir, resume_expt_name)

        print(f'Resuming from Level -- {resume_level}')
    elif resume_level == 0 and resume_expt_name is None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:6]
        
        unique_name = f"experiment_{current_time}_{unique_id}"
        expt_dir = os.path.join(base_dir, unique_name)
    else:
        raise AssertionError('Either start from scratch, or provide a path to the checkpoint :)')

    if not os.path.exists(expt_dir):
        os.makedirs(expt_dir)
        os.makedirs(f'{expt_dir}/checkpoints')
        os.makedirs(f'{expt_dir}/metrics')
        os.makedirs(f'{expt_dir}/metrics/epochwise_metrics')
        os.makedirs(f'{expt_dir}/artifacts/')
    
    return expt_dir

## seed everything
@param('experiment_params.seed')
def set_seed(seed : int, is_deterministic=False) -> None:

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set

    if is_deterministic:
        print("This ran")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

@param('prune_params.prune_method')
@param('prune_params.num_levels')
@param('prune_params.prune_rate')
def generate_densities(prune_method, num_levels, prune_rate):
    densities = [(1-prune_rate)**i for i in range(num_levels)]

    return densities

def save_config(expt_dir, config):
    nested_dict = {}
    for (outer_key, inner_key), value in config.content.items():
        if outer_key not in nested_dict:
            nested_dict[outer_key] = {}
        nested_dict[outer_key][inner_key] = value
        with open(os.path.join(expt_dir, 'expt_config.yaml'),  'w') as file:
            yaml.dump(nested_dict, file, default_flow_style=False)