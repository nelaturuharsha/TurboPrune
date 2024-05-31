import torch
import os
import prettytable
from utils.conv_type import ConvMask
from datetime import datetime
import uuid

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
    if training_type == 'imp':
        optimizer.load_state_dict(torch.load(os.path.join(expt_dir, 'artifacts', 'optimizer_init.pt')))
    elif training_type == 'wr':
        optimizer.load_state_dict(torch.load(os.path.join(expt_dir, 'artifacts', 'optimizer_rewind.pt')))
    else:
        print('probably LRR, aint nothing to do')
        return optimizer
    
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
            if isinstance(layer, ConvMask):
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


def create_experiment_dir_name(expt_type):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    unique_name = f"experiment_{current_time}_{unique_id}"

    if expt_type == 'cispa':
        expt_dir = f"/home/c02hane/CISPA-projects/neuron_pruning-2024/lottery-ticket-harness/{unique_name}"
    else:
        expt_dir = f"./{unique_name}"

    return expt_dir