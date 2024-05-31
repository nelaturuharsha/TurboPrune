## pythonic imports
import os
import random
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from prettytable import PrettyTable
import tqdm
from copy import deepcopy

## torch
import torch
import torch.nn as nn
from torch.distributed import init_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim
import torch.multiprocessing as mp
from torch.cuda.amp import autocast

## torchvision
import torchvision

## file-based imports
import schedulers
import pruning_utils
from harness_utils import *

## fastargs
from fastargs import get_current_config
from fastargs.decorators import param

##ffcv
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.decoders import RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder


DEFAULT_CROP_RATIO = 224/256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255

torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

## seed everything
def set_seed(seed: int = 42, is_deterministic=False) -> None:

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

set_seed(1234)

class Harness:
    def __init__(self, gpu_id, expt_dir, model=None):
        self.config = get_current_config()
        self.gpu_id = gpu_id
        self.this_device = f'cuda:{self.gpu_id}'

        self.train_loader, self.test_loader = self.create_train_loader(), self.create_test_loader()

        model = model.to(self.this_device)
        self.model = DDP(model, device_ids=[self.gpu_id])

        self.create_optimizers()

        self.expt_dir = expt_dir 

    @param('dataset.batch_size')
    @param('dataset.num_workers')
    @param('dataset.data_root')
    def create_train_loader(self, batch_size, num_workers, data_root, distributed=True):
        train_image_pipeline = [RandomResizedCropRGBImageDecoder((224, 224)),
                            RandomHorizontalFlip(),
                            ToTensor(),
                            ToDevice(torch.device(self.this_device), non_blocking=True),
                            ToTorchImage(),
                            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)]

        label_pipeline = [IntDecoder(),
                            ToTensor(),
                            Squeeze(),
                            ToDevice(torch.device(self.this_device), non_blocking=True)]


        train_loader = Loader(os.path.join(data_root, 'train_500_0.50_90.beton'), 
                              batch_size  = batch_size,
                              num_workers = num_workers,
                              order       = OrderOption.RANDOM,
                              os_cache    = True,
                              drop_last   = True,
                              pipelines   = { 'image' : train_image_pipeline,
                                              'label' : label_pipeline},
                              distributed = distributed,
                              )
        
        return train_loader

    @param('dataset.batch_size')
    @param('dataset.num_workers')
    @param('dataset.data_root')
    def create_test_loader(self, batch_size, num_workers, data_root, distributed=True):
        val_image_pipeline = [CenterCropRGBImageDecoder((256, 256), ratio=DEFAULT_CROP_RATIO),
                              ToTensor(),
                              ToDevice(torch.device(self.this_device), non_blocking=True),
                              ToTorchImage(),
                              NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)]

        label_pipeline = [IntDecoder(),
                            ToTensor(),
                            Squeeze(),
                            ToDevice(torch.device(self.this_device), non_blocking=True)]

        val_loader = Loader(os.path.join(data_root, 'val_500_0.50_90.beton'),
                            batch_size  = batch_size,
                            num_workers = num_workers,
                            order       = OrderOption.SEQUENTIAL,
                            drop_last   = False,
                            pipelines   = { 'image' : val_image_pipeline,
                                            'label' : label_pipeline},
                            distributed = distributed,
                            )
        return val_loader

    @param('optimizer.lr')
    @param('optimizer.momentum')
    @param('optimizer.weight_decay')
    @param('optimizer.scheduler_type')
    def create_optimizers(self, lr, momentum, weight_decay, scheduler_type):
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = getattr(schedulers, scheduler_type)
        self.scheduler = scheduler(optimizer=self.optimizer)
        self.criterion = nn.CrossEntropyLoss()

    def train_one_epoch(self, epoch):
        model = self.model
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        tepoch = tqdm.tqdm(self.train_loader, unit='batch', desc=f'Epoch {epoch}')
        for inputs, targets in tepoch:
            self.optimizer.zero_grad()
            with autocast(dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        self.scheduler.step()
        train_loss /= (len(self.train_loader))
        accuracy = 100. * (correct / total)
        return train_loss, accuracy

    def test(self):
        model = self.model
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        tloader = tqdm.tqdm(self.test_loader, desc='Testing')
        with torch.no_grad():
            for inputs, targets in tloader:
                with autocast(dtype=torch.bfloat16):
                    outputs = model(inputs)
                    loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_loss /= (len(self.test_loader))
        accuracy = 100. * correct / total

        return test_loss, accuracy

    @param('experiment_params.epochs_per_level')
    @param('experiment_params.training_type')
    def train_one_level(self, epochs_per_level, training_type, level):
        new_table = PrettyTable()
        new_table.field_names = ["Epoch", "Train Loss", "Test Loss", "Train Acc", "Test Acc"]
        sparsity_level_df = {
            'epoch' : [],
            'train_acc': [],
            'test_acc': [],
            'train_loss': [],
            'test_loss': []
        }
        data_df = {'level' : [], 'sparsity' : [], 'max_test_acc' : [], 'final_test_acc' : []}


        for epoch in range(epochs_per_level):
            train_loss, train_acc = self.train_one_epoch(epoch)
            test_loss, test_acc = self.test()

            train_loss_tensor = torch.tensor(train_loss).to(self.model.device)
            train_acc_tensor = torch.tensor(train_acc).to(self.model.device)
            test_loss_tensor = torch.tensor(test_loss).to(self.model.device)
            test_acc_tensor = torch.tensor(test_acc).to(self.model.device)

            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_acc_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_acc_tensor, op=dist.ReduceOp.SUM)

            train_loss_tensor /= dist.get_world_size()
            train_acc_tensor /= dist.get_world_size()
            test_loss_tensor /= dist.get_world_size()
            test_acc_tensor /= dist.get_world_size()

            if self.gpu_id == 0:
                tr_l, te_l, tr_a, te_a = train_loss_tensor.item(), test_loss_tensor.item(), train_acc_tensor.item(), test_acc_tensor.item()
                new_table.add_row([epoch, tr_l, te_l, tr_a, te_a])
                print(new_table)
                sparsity_level_df['epoch'].append(epoch)
                sparsity_level_df['train_loss'].append(tr_l)
                sparsity_level_df['test_loss'].append(te_l)
                sparsity_level_df['train_acc'].append(tr_a)
                sparsity_level_df['test_acc'].append(te_a)

                save_matching = (level == 0) and (training_type == 'wr') and (epoch == 9)
                if save_matching and self.gpu_id == 0:
                    torch.save(self.model.module.state_dict(), os.path.join(self.expt_dir, 'checkpoints', 'model_rewind.pt'))
                    torch.save(self.optimizer.state_dict(), os.path.join(self.expt_dir, 'checkpoints', 'optimizer_rewind.pt'))

        if self.gpu_id == 0:
            pd.DataFrame(sparsity_level_df).to_csv(os.path.join(self.expt_dir, 'metrics', 'epochwise_metrics', f'level_{level}_metrics.csv'))
            sparsity = print_sparsity_info(self.model, verbose=False)
            data_df['level'].append(level)
            data_df['sparsity'].append(round(sparsity.item(), 4))
            data_df['final_test_acc'].append(round(te_a, 4))
            data_df['max_test_acc'].append(round(max(sparsity_level_df['test_acc']), 4))
            summary_path = os.path.join(self.expt_dir, 'metrics', 'summary.csv')

            if not os.path.exists(summary_path):
                pd.DataFrame(data_df).to_csv(summary_path, index=False)
            else:
                pre_df = pd.read_csv(summary_path)
                new_df = pd.DataFrame(data_df)
                updated_df = pd.concat([pre_df, new_df], ignore_index=True)
                updated_df.to_csv(summary_path, index=False)


@param('dist.address')
@param('dist.port')
def setup_distributed(address, port, gpu_id):

    os.environ['MASTER_ADDR'] = address
    os.environ['MASTER_PORT'] = port
    world_size = torch.cuda.device_count()
    init_process_group('nccl', rank=gpu_id, world_size=world_size)
    torch.cuda.set_device(gpu_id)

def main(rank, model, level, expt_dir):
    setup_distributed(gpu_id=rank)
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    global data_df

    harness = Harness(model=model, expt_dir=expt_dir, gpu_id=rank)

    if level != 0:
        harness.optimizer = reset_optimizer(expt_dir=expt_dir, optimizer=harness.optimizer, training_type=config['experiment_params.training_type'])
    
    if (level == 0) and (rank == 0):
        torch.save(harness.optimizer.state_dict(), os.path.join(expt_dir, 'artifacts', 'optimizer_init.pt'))
        torch.save(harness.model.module.state_dict(), os.path.join(expt_dir, 'checkpoints', 'model_init.pt'))

    harness.train_one_level(level=level)

    if rank == 0:
        ckpt = os.path.join(expt_dir, 'checkpoints', f'model_level_{level}.pt')
        torch.save(harness.model.module.state_dict(), ckpt)
    
    dist.destroy_process_group()

if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)

    config.validate(mode='stderr')
    config.summary()

    prune_harness = pruning_utils.PruningStuff()

    num_levels = config['experiment_params.num_levels']
    thresholds = [(1 - config['prune_params.prune_rate']) ** (level) for level in range(num_levels)]

    expt_dir = create_experiment_dir_name(config['experiment_params.expt_setup']) 

    if not os.path.exists(expt_dir):
        os.makedirs(expt_dir)
        os.makedirs(f'{expt_dir}/checkpoints')
        os.makedirs(f'{expt_dir}/metrics')
        os.makedirs(f'{expt_dir}/metrics/epochwise_metrics')
        os.makedirs(f'{expt_dir}/artifacts/')
    
    world_size = torch.cuda.device_count()
    print(f'Training on {world_size} GPUs')
    print_sparsity_info(prune_harness.model)

    for level in range(len(thresholds)):
        if level != 0:
            print(f'Pruning Model at level: {level}')
            prune_harness.load_from_ckpt(os.path.join(expt_dir, 'checkpoints', f'model_level_{level-1}.pt'))
            prune_harness.model = reset_weights(expt_dir=expt_dir, model=prune_harness.model, training_type=config['experiment_params.training_type'])
            
            prune_harness.level_pruner(density=thresholds[level])
            print_sparsity_info(prune_harness.model)
        
        mp.spawn(main, args=(prune_harness.model, level, expt_dir), nprocs=world_size, join=True)
        print(f'Training level {level} complete, moving on to {level+1}')
        


    

