## pythonic imports
import os
from typing import Tuple
from argparse import ArgumentParser

## rich stuff
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

## torch
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim
from torch.amp import autocast
from torchmetrics import Accuracy
import torch._dynamo

## fastargs
from fastargs import get_current_config
from fastargs.decorators import param
import schedulefree

import wandb

## file-based imports
import utils.schedulers as schedulers
from utils.pruning_utils import *
from utils.harness_utils import *
from utils.dataset import AirbenchLoaders, imagenet
from utils.distributed_utils import broadcast_object, setup_distributed
from utils.custom_models import TorchVisionModel, CustomModel
from utils.metric_utils import *

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch._dynamo.config.guard_nn_modules=True

class Harness:
    """Harness class to handle training and evaluation.

    Args:
        gpu_id (int): current rank of the process while training using DDP.
        expt_dir (str): Experiment directory to save artifacts/checkpoints/metrics to.
        model (Optional[nn.Module], optional): The model to train.
    """
    @param("dataset.dataset_name")
    @param("experiment_params.use_compile")
    @param("dist_params.distributed")
    def __init__(self, dataset_name: str, use_compile: bool, distributed: bool, gpu_id: int, expt_dir: str, model: nn.Module = None) -> None:
        self.config = get_current_config()
        self.dataset_name = dataset_name.lower()
        self.use_compile = True if use_compile == 'true' else False
        self.distributed = distributed and torch.cuda.device_count() > 1 and not self.dataset_name.startswith("cifar")
        self.num_classes = 1000 if self.dataset_name.startswith("imagenet") else 100 if self.dataset_name.startswith("cifar100") else 10
        self.epoch_counter, self.console = 0, Console() 
        self.gpu_id = gpu_id
    
        if self.dataset_name.startswith("cifar"):
            self.gpu_id = 0
        self.this_device = f"cuda:{self.gpu_id}"
        self.criterion = nn.CrossEntropyLoss()

        dist_sync = {"dist_sync_on_step": True} if self.distributed else {}
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes, **dist_sync).to(self.this_device)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes, **dist_sync).to(self.this_device)

        self.loaders = AirbenchLoaders() if self.dataset_name.startswith("cifar") else imagenet(this_device=self.this_device, distributed=self.distributed)
        
        self.train_loader = self.loaders.train_loader
        self.test_loader = self.loaders.test_loader
        if model is None:
            self.model = self.acquire_model()
        else:
            self.model = model
        self.model = torch.compile(self.model, mode='reduce-overhead') if self.use_compile else self.model
        self.model = DDP(self.model, device_ids=[self.gpu_id]) if self.distributed else self.model


        self.prefix, self.expt_dir = expt_dir
        self.precision, self.use_amp = self.get_dtype_amp()
        
        if self.gpu_id == 0:
            self.training_info = {'dataset' : self.dataset_name, 'use_compile' : self.use_compile, 'distributed' : self.distributed,
                            'precision' : self.precision, 'use_amp' : self.use_amp, 'expt_dir' : self.expt_dir}
            print(self.training_info)
    @param('experiment_params.training_precision')
    def get_dtype_amp(self, training_precision):
        dtype_map = {
            'bfloat16': (torch.bfloat16, True),
            'float16': (torch.float16, True),
            'float32': (torch.float32, False)
        }
        return dtype_map.get(training_precision, (torch.float32, False))
    
    def _log_metrics(self, cycle, train_loss, test_loss, train_acc, test_acc):
        wandb.log({
            "train_loss": train_loss, "test_loss": test_loss,
            "train_acc": train_acc, "test_acc": test_acc,
            "epoch": self.epoch_counter,
            "cycle": cycle
        })

    def acquire_model(self) -> nn.Module:
        """Acquire the model based on the provided parameters.
        Returns:
            nn.Module: The acquired model.
        """
        try:
            model = TorchVisionModel()
            if self.gpu_id == 0:
                self.console.print("[bold turquoise]Using Torchvision Model :)[/bold turquoise]")
        except:
            model = CustomModel()
            if self.gpu_id == 0:
                self.console.print("[bold turquoise]Using Custom Model :D[/bold turquoise]")
        model = model.to(self.this_device)
        return model
    
    @param("optimizer.lr")
    @param("optimizer.momentum")
    @param("optimizer.weight_decay") 
    @param("optimizer.scheduler_type")
    def create_optimizers(self, lr: float, momentum: float, weight_decay: float, scheduler_type: str, epochs_per_level: int) -> None:
        """Create optimizer and learning rate scheduler."""
        
        # Create base optimizer
        if scheduler_type == 'ScheduleFree':
            self.optimizer = schedulefree.SGDScheduleFree(
                self.model.parameters(),
                warmup_steps=self.config['optimizer.warmup_steps'],
                lr=lr, momentum=momentum, weight_decay=weight_decay
            )
            self.scheduler = None
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=lr, momentum=momentum, weight_decay=weight_decay
            )
            
            # Create scheduler
            if scheduler_type == 'OneCycleLR':
                self.scheduler = optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=lr,
                    epochs=epochs_per_level,
                    steps_per_epoch=len(self.train_loader)
                )
            else:
                scheduler_cls = getattr(schedulers, scheduler_type)
                scheduler_args = {
                    'optimizer': self.optimizer,
                    'steps_per_epoch': len(self.train_loader)
                }
                
                if scheduler_type == 'TriangularSchedule':
                    scheduler_args['epochs_per_level'] = epochs_per_level
                elif scheduler_type == 'TrapezoidalSchedule':
                    scheduler_args.update({
                        'warmup_steps': len(self.train_loader) * self.config['optimizer.trapezoidal_scheduler_stuff.warmup_steps'],
                        'cooldown_steps': len(self.train_loader) * self.config['optimizer.trapezoidal_scheduler_stuff.cooldown_steps']
                    })
                
                self.scheduler = scheduler_cls(**scheduler_args)

        # Log optimizer info on main process
        if self.gpu_id == 0:
            self.optimizer_info = {
                "type": type(self.optimizer).__name__,
                "lr": lr,
                "momentum": momentum, 
                "weight_decay": weight_decay,
                "scheduler": scheduler_type,
                "epochs_per_level": epochs_per_level
            }
    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train the model for one epoch.

        Args:
            epoch (int): Current epoch.
        Returns:
            (float, float): Training loss and accuracy.
        """
        self.model.train()
        train_loss = 0.0
        self.train_accuracy.reset()

        progress = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                            BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                            TextColumn("[cyan]{task.completed}/{task.total}"), TimeRemainingColumn(elapsed_when_finished=True))

        with progress:
            task = progress.add_task(f"[cyan]Epoch {epoch}", total=len(self.train_loader))

            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.this_device), targets.to(self.this_device)
                
                self.optimizer.zero_grad(set_to_none=True)
                with autocast('cuda', dtype=self.precision, enabled=self.use_amp):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                self.train_accuracy.update(outputs, targets)

                if self.scheduler is not None and self.config['optimizer.scheduler_type'] != 'MultiStepLRWarmup':
                    self.scheduler.step()
                    
                progress.update(task, advance=1)
                
        if self.config["optimizer.scheduler_type"] == 'MultiStepLRWarmup':
            self.scheduler.step()

        train_loss /= len(self.train_loader)
        if self.distributed:
            train_loss_tensor = torch.tensor(train_loss).to(self.this_device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
            train_loss = train_loss_tensor.item()
            del train_loss_tensor

        accuracy = self.train_accuracy.compute().item() * 100

        return train_loss, accuracy

    def test(self) -> Tuple[float, float]:
        """Evaluate the model on the test set.

        Returns:
            (float, float): Test loss and accuracy.
        """
        #model = self.model
        self.model.eval()
        test_loss = 0.0
        self.test_accuracy.reset()

        with torch.no_grad(), autocast('cuda', dtype=self.precision, enabled=self.use_amp):
            for inputs, targets in self.test_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                self.test_accuracy.update(outputs, targets)

        if self.distributed:
            test_loss_tensor = torch.tensor(test_loss).to(self.this_device)
            dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.AVG)
            test_loss = test_loss_tensor.item()
            del test_loss_tensor

        test_loss /= len(self.test_loader)
        accuracy = self.test_accuracy.compute().item() * 100

        return test_loss, accuracy
    
    @param("cyclic_training.num_cycles")
    @param("experiment_params.epochs_per_level")
    def train_one_level(self, num_cycles: int, epochs_per_level: int, level: int) -> None:
        
        use_cyclic_training = num_cycles > 1

        level_metrics = {'cycle': [], 'epoch': [], 'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
        
        epoch_schedule = generate_cyclical_schedule(epochs_per_level=epochs_per_level) if use_cyclic_training else [epochs_per_level]

        for cycle in range(num_cycles):
            if self.gpu_id == 0:
                model = self.model.module if self.distributed else self.model
                self.cycle_info = {
                    'Number of Cycles': num_cycles, 'Epochs per Cycle': epoch_schedule,
                    'Total Training Length': f"{sum(epoch_schedule)} epochs", "Training Cycle": f"{cycle + 1}/{num_cycles}",
                    "Epochs this cycle": f"{epoch_schedule[cycle]}", "Total epochs so far": f"{sum(epoch_schedule[:cycle+1])}/{sum(epoch_schedule)}",
                    "Current Sparsity": f"{model.get_overall_sparsity():.4f}"
                } 
            self.create_optimizers(epochs_per_level=epoch_schedule[cycle])
            
            if self.gpu_id == 0:
                display_training_info(cycle_info=self.cycle_info, training_info=self.training_info, optimizer_info=self.optimizer_info)
                if cycle == 0 and level == 0:
                    self.console.print(f"[bold cyan]Saving optimizer state for level {level}, cycle {cycle}[/bold cyan]")
                    torch.save(self.optimizer.state_dict(), os.path.join(self.expt_dir, "artifacts", "optimizer_init.pt"))
                elif level != 0:
                    self.optimizer.load_state_dict(torch.load(os.path.join(self.expt_dir, "artifacts", "optimizer_rewind.pt" if self.config['experiment_params.training_type'] == 'wr' else "optimizer_init.pt")))
                    self.console.print(f"[bold cyan]Loading optimizer state for level {level}, cycle {cycle}[/bold cyan]")

            for epoch in range(epoch_schedule[cycle]):
                self.epoch_counter += 1
                train_loss, train_acc = self.train_one_epoch(epoch)
                test_loss, test_acc = self.test()

                for key, value in [('cycle', cycle),
                                 ('epoch', self.epoch_counter),
                                 ('train_loss', train_loss),
                                 ('test_loss', test_loss),
                                 ('train_acc', train_acc),
                                 ('test_acc', test_acc)]:
                    level_metrics[key].append(value)

                if self.gpu_id == 0:
                    self._log_metrics(cycle, train_loss, test_loss, train_acc, test_acc)
                    
                    if not hasattr(self, 'metrics_table'):
                        self.metrics_table = Table(title=f"Training Metrics for Level {level}")
                        for col, style in [("Cycle","cyan"), ("Epoch","magenta"), 
                                         ("Train Loss","green"), ("Test Loss","red"),
                                         ("Train Acc","green"), ("Test Acc","red")]:
                            self.metrics_table.add_column(col, style=style)
                    
                    self.metrics_table.add_row(str(cycle), str(epoch), 
                                             f"{train_loss:.4f}", f"{test_loss:.4f}",
                                             f"{train_acc:.2f}%", f"{test_acc:.2f}%")
                    
                    self.console.rule(style="cyan")
                    self.console.print("\n", self.metrics_table, "\n")
                    
                    if level == 0 and epoch == 9:
                        save_model(self.model, os.path.join(self.expt_dir, "checkpoints", "model_rewind.pt"), 
                                 distributed=self.distributed)
                        torch.save(self.optimizer.state_dict(), 
                                 os.path.join(self.expt_dir, "artifacts", "optimizer_rewind.pt"))
            
            if self.gpu_id == 0:
                save_metrics_and_update_summary(self.console, model, self.expt_dir, self.prefix, level, level_metrics, num_cycles, epoch_schedule)
                save_model(self.model, os.path.join(self.expt_dir, "checkpoints", f"model_level_{level}_cycle_{cycle}.pt"), distributed=self.distributed)
                self.console.print(f"[bold cyan]Saved checkpoint for level {level}, cycle {cycle}[/bold cyan]")

def initialize_config(): 
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode="stderr")

    return config

def main():
    """Main function for distributed training."""
    config = initialize_config()
    use_distributed = (config['dist_params.distributed'] == 'true' 
                      and torch.cuda.device_count() > 1 
                      and not config['dataset.dataset_name'].startswith("cifar"))
    set_seed()
    if use_distributed:
        setup_distributed()
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank, world_size = 0, 1
    console = Console()

    if rank == 0:
        config.summary()
        console.print(f"[bold green]Training on {world_size} GPUs[/bold green]")
        wandb.init(project=config['wandb_params.project_name'])
        run_id = wandb.run.id
        wandb.run.tags += (config['dataset.dataset_name'].lower(),)
        prefix, expt_dir = (resume_experiment(config['experiment_params.base_dir'], config['experiment_params.resume_experiment_stuff.resume_level'],
                                          config['experiment_params.resume_experiment_stuff.resume_expt_name']) 
                         if config['experiment_params.resume_experiment'] == 'true' else gen_expt_dir())
        packaged = (prefix, expt_dir)
        save_config(expt_dir=expt_dir, config=config)
    else:
        run_id, packaged = None, None

    if use_distributed:
        run_id = broadcast_object(run_id)
        packaged = broadcast_object(packaged)

    harness = Harness(gpu_id=rank, expt_dir=packaged)
    model_in_question = harness.model.module if use_distributed else harness.model
    at_init = config['experiment_params.training_type'] == 'at_init'
    densities = generate_densities(current_sparsity=model_in_question.get_overall_sparsity())
    densities = densities[config['experiment_params.resume_experiment_stuff.resume_level']:] if config['experiment_params.resume_experiment'] == 'true' else densities
    
    # Pre-compute paths to avoid repeated string concatenation
    checkpoints_dir = os.path.join(packaged[1], "checkpoints")
    model_init_path = os.path.join(checkpoints_dir, "model_init.pt")
    
    for level, density in enumerate(densities):
        if rank == 0 :
            if at_init:
                console.print(f"[bold cyan]Pruning Model at initialization[/bold cyan]")
            elif level == 0:
                save_model(model_in_question, model_init_path, distributed=use_distributed)
                console.print(f'[bold cyan]Dense training homie![/bold cyan]')
            else:
                console.print(f"[bold cyan]Pruning Model at level: {level} to a target density of {density:.4f}[/bold cyan]")
                model_level_path = os.path.join(checkpoints_dir, f"model_level_{level-1}.pt")
                print(model_in_question)
                model_in_question.load_model(model_level_path)

            if level != 0 or at_init:
                prune_the_model(prune_method=config['prune_params.prune_method'], harness=harness, target_density=density)
                if not at_init:
                    model_in_question.reset_weights(training_type=config['experiment_params.training_type'], 
                                                    expt_dir=packaged[1])

        harness = Harness(model=model_in_question, expt_dir=packaged, gpu_id=rank)
 
        harness.train_one_level(level=level)
        
        if rank == 0:
            model_level_path = os.path.join(checkpoints_dir, f"model_level_{level}.pt")
            save_model(harness.model, model_level_path, distributed=use_distributed)
            console.print(f"[bold green]Training level {level} complete, moving on to {level+1}[/bold green]")

    if rank == 0:
        wandb.finish()

if __name__ == "__main__":
    main()
