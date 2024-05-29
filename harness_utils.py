import torch
import os
import prettytable
from utils.conv_type import ConvMask

def reset_weights(expt_dir, model, training_type=False):
    if training_type == 'imp':
        original_dict = torch.load(os.path.join(expt_dir, 'checkpoints', 'model_init.pt'))
    elif training_type == 'wr':
        original_dict = torch.load(os.path.join(expt_dir, 'checkpoints', 'model_rewind.pt'))
    
    original_weights = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), original_dict.items()))
    model_dict = model.state_dict()

    model_dict.update(original_weights)
    model.load_state_dict(model_dict)

def reset_optimizer(expt_dir, optimizer):
    optimizer.load_state_dict(torch.load(os.path.join(expt_dir, 'artifacts', 'optimizer.pt')))

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

def print_sparsity_info(model):
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

        print(my_table)

        overall_sparsity = 1-(total_params_kept/total_params)

        print('-----------')
        print(f"Overall Sparsity of All Layers: {overall_sparsity:.4f}")
        print('-----------')

        return overall_sparsity