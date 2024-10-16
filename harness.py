## pythonic imports
import os

from typing import Tuple

## torch
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim
from torch.amp import autocast
from torchmetrics import Accuracy

## file-based imports
import utils.schedulers as schedulers
import utils.pruning_utils as pruning_utils
from utils.harness_utils import *
from utils.dataset import AirbenchLoaders, imagenet
from utils.distributed_utils import broadcast_model, check_model_equality, broadcast_object, setup_distributed

## fastargs
from fastargs import get_current_config
from fastargs.decorators import param
import schedulefree

import wandb

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

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
    def __init__(self, dataset_name: str, use_compile: bool, distributed: bool, gpu_id: int, expt_dir: str, model: nn.Module) -> None:

        self.config = get_current_config()
        self.dataset_name = dataset_name.lower()
        self.use_compile = use_compile
        self.distributed = distributed and torch.cuda.device_count() > 1 and not self.dataset_name.startswith("cifar")
        self.num_classes = 1000 if self.dataset_name.startswith("imagenet") else 100 if self.dataset_name.startswith("cifar100") else 10
        self.gpu_id = gpu_id

        if self.dataset_name.startswith("cifar"):
            self.gpu_id = 0
        self.this_device = f"cuda:{self.gpu_id}"
        self.criterion = nn.CrossEntropyLoss()

        if self.distributed:
            self.train_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes, dist_sync_on_step=True).to(self.this_device)
            self.test_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes, dist_sync_on_step=True).to(self.this_device)
        else:
            self.train_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.this_device)
            self.test_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.this_device)

        self.loaders = AirbenchLoaders() if self.dataset_name.startswith("cifar") else imagenet(this_device=self.this_device, distributed=self.distributed)
        
        self.train_loader = self.loaders.train_loader
        self.test_loader = self.loaders.test_loader

        self.model = model.to(self.this_device)
        self.model = DDP(self.model, device_ids=[self.gpu_id]) if self.distributed else self.model
        self.model = torch.compile(self.model, mode='reduce-overhead') if self.use_compile else self.model

        self.prefix, self.expt_dir = expt_dir
        self.precision, self.use_amp = self.get_dtype_amp()
        self.epoch_counter, self.console = 0, Console()
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
    
    @param("optimizer.lr")
    @param("optimizer.momentum")
    @param("optimizer.weight_decay")
    @param("optimizer.scheduler_type")
    def create_optimizers(
        self, lr: float, momentum: float, weight_decay: float, scheduler_type: str, epochs_per_level: int
    ) -> None:
        """Instantiate the optimizer and learning rate scheduler.

        Args:
            lr (float): Initial learning rate.
            momentum (float): Momentum for SGD.
            weight_decay (float): Weight decay for optimizer.
            scheduler_type (str): Type of scheduler.
        """

        if scheduler_type =='ScheduleFree':
            self.optimizer = schedulefree.SGDScheduleFree(
                self.model.parameters(),
                warmup_steps=self.config['optimizer.warmup_steps'],
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
            self.scheduler = None
        
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            if scheduler_type != 'OneCycleLR':
                scheduler = getattr(schedulers, scheduler_type)            
                
                if scheduler_type == 'TriangularSchedule':
                    self.scheduler = scheduler(optimizer=self.optimizer, steps_per_epoch=len(self.train_loader), epochs_per_level=epochs_per_level)
                elif scheduler_type == 'TrapezoidalSchedule':
                    self.scheduler = scheduler(optimizer=self.optimizer, steps_per_epoch=len(self.train_loader),
                                                warmup_steps=len(self.train_loader) * self.config['optimizer.warmup_steps'],
                                                cooldown_steps=len(self.train_loader) * self.config['optimizer.cooldown_steps'])
                elif scheduler_type == 'MultiStepLRWarmup':
                    self.scheduler = scheduler(optimizer=self.optimizer)
            else:
                self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                              max_lr=lr,
                                                              epochs=epochs_per_level,
                                                              steps_per_epoch=len(self.train_loader))
                
        if self.gpu_id == 0:
            self.optimizer_info = {
                "Type": type(self.optimizer).__name__, "Learning rate": lr,
                "Momentum": momentum, "Weight decay": weight_decay,
                "Scheduler": scheduler_type, "Epochs per level": epochs_per_level,
                "Starting LR": lr
            }
    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train the model for one epoch.

        Args:
            epoch (int): Current epoch.
        Returns:
            (float, float): Training loss and accuracy.
        """

        model = self.model
        model.train()
        train_loss = torch.tensor(0.0).to(self.this_device)
        self.train_accuracy.reset()

        progress = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                            BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                            TextColumn("[cyan]{task.completed}/{task.total}"), TimeRemainingColumn(elapsed_when_finished=True))

        with progress:
            task = progress.add_task(f"[cyan]Epoch {epoch}", total=len(self.train_loader))

            for inputs, targets in self.train_loader:
                self.optimizer.zero_grad()
                with autocast('cuda', dtype=self.precision, enabled = self.use_amp):
                    outputs = model(inputs)
                    loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                self.train_accuracy.update(outputs, targets)

                if (self.scheduler is not None) and (self.config['optimizer.scheduler_type'] != 'MultiStepLRWarmup'):
                    self.scheduler.step()
                progress.update(task, advance=1)

        if self.config["optimizer.scheduler_type"] == 'MultiStepLRWarmup':
            self.scheduler.step()

        if self.distributed:
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        train_loss = train_loss.item() / len(self.train_loader)

        accuracy = self.train_accuracy.compute().item() * 100

        return train_loss, accuracy

    def test(self) -> Tuple[float, float]:
        """Evaluate the model on the test set.

        Returns:
            (float, float): Test loss and accuracy.
        """
        model = self.model
        model.eval()
        test_loss = torch.tensor(0.0).to(self.this_device)
        self.test_accuracy.reset()

        progress = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                            BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                            TextColumn("[cyan]{task.completed}/{task.total}"))
        
        with progress:
            task = progress.add_task(f"[cyan]Testing", total=len(self.test_loader))
            with torch.no_grad():
                for inputs, targets in self.test_loader:
                    with autocast('cuda', dtype=self.precision, enabled = self.use_amp):
                        outputs = model(inputs.contiguous())
                        loss = self.criterion(outputs, targets)

                    test_loss += loss.item()
                    self.test_accuracy.update(outputs, targets)
                    progress.update(task, advance=1)
        
        
        if self.distributed:
            dist.all_reduce(test_loss, op=dist.ReduceOp.AVG)
        test_loss = test_loss.item() / len(self.test_loader)
        accuracy = self.test_accuracy.compute().item() * 100

        return test_loss, accuracy
    
    @param("experiment_params.training_type")
    @param("cyclic_training.num_cycles")
    def train_one_level(self, training_type: str, num_cycles: int, level: int) -> None:
        level_metrics = {'cycle': [], 'epoch': [], 'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
        epoch_schedule = generate_budgeted_schedule()
            
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
                    self.console.print(f"[bold cyan]Loading optimizer state for level {level}, cycle {cycle}[/bold cyan]")
                    self.optimizer.load_state_dict(torch.load(os.path.join(self.expt_dir, "artifacts", "optimizer_init.pt", weights_only=True)))

            for epoch in range(epoch_schedule[cycle]):
                
                self.epoch_counter += 1
                train_loss, train_acc = self.train_one_epoch(epoch)
                test_loss, test_acc = self.test()

                for key, value in zip(['cycle', 'epoch', 'train_loss', 'test_loss', 'train_acc', 'test_acc'],
                                      [cycle, self.epoch_counter, train_loss, test_loss, train_acc, test_acc]):
                    level_metrics[key].append(value)

                if self.gpu_id == 0:
                    self._log_metrics(cycle, train_loss, test_loss, train_acc, test_acc)
                    if not hasattr(self, 'metrics_table'):
                        self.metrics_table = Table(title=f"Training Metrics for Level {level}")
                        self.metrics_table.add_column("Cycle", style="cyan")
                        self.metrics_table.add_column("Epoch", style="magenta")
                        self.metrics_table.add_column("Train Loss", style="green")
                        self.metrics_table.add_column("Test Loss", style="red")
                        self.metrics_table.add_column("Train Acc", style="green")
                        self.metrics_table.add_column("Test Acc", style="red")
                    
                    self.metrics_table.add_row(
                        str(cycle),
                        str(epoch),
                        f"{train_loss:.4f}",
                        f"{test_loss:.4f}",
                        f"{train_acc:.2f}%",
                        f"{test_acc:.2f}%"
                    )
                    
                    self.console.rule(style="cyan")
                    self.console.print("\n") 
                    self.console.print(self.metrics_table)
                    self.console.print("\n") 
                    if level == 0 and training_type == "wr" and epoch == 9:
                        save_model(self.model, os.path.join(self.expt_dir, "checkpoints", "model_rewind.pt"), distributed=self.distributed)
                        torch.save(self.optimizer.state_dict(), os.path.join(self.expt_dir, "artifacts", "optimizer_rewind.pt"))

            if self.gpu_id == 0:
                save_model(self.model, os.path.join(self.expt_dir, "checkpoints", f"model_level_{level}_cycle_{cycle}.pt"), distributed=self.distributed)
                self.console.print(f"[bold cyan]Saved checkpoint for level {level}, cycle {cycle}[/bold cyan]")

        if self.gpu_id == 0:
            model = self.model.module if self.distributed else self.model
            save_metrics_and_update_summary(self.console, model, self.expt_dir, self.prefix, level, level_metrics, num_cycles, epoch_schedule)

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
    use_distributed = config['dist_params.distributed']
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
        prefix, expt_dir = gen_expt_dir()
        packaged = (prefix, expt_dir)
        save_config(expt_dir=expt_dir, config=config)

    else:
        run_id, packaged = None, None

    if use_distributed:
        run_id = broadcast_object(run_id)
        packaged = broadcast_object(packaged)

    prune_harness = pruning_utils.PruningStuff()
    model = prune_harness.model
    
    if use_distributed:
        if rank == 0:
            state_dict = model.state_dict()
        else:
            state_dict = None
        
        state_dict = broadcast_object(state_dict)
        
        model.load_state_dict(state_dict)
        models_equal = check_model_equality(model)

    if use_distributed:
        if rank == 0:
            if models_equal:
                console.print("[bold green]Models are equal across all GPUs[/bold green]")
            else:
                console.print("[bold red]Models are not equal across all GPUs[/bold red]")

    prune_harness = pruning_utils.PruningStuff(model=model)

    resume_level = config["experiment_params.resume_level"]
    is_iterative = (config['prune_params.er_method'] == 'just dont') and (config['prune_params.prune_method'] != 'just dont')
    densities = generate_densities(current_sparsity=prune_harness.model.get_overall_sparsity()) if is_iterative else [config['prune_params.er_init']]
    for level in range(resume_level, len(densities)):
        if rank == 0:
            if is_iterative and level != 0:
                console.print(f"[bold cyan]Pruning Model at level: {level}[/bold cyan]")
                prune_harness.model.load_model(os.path.join(packaged[1], "checkpoints", f"model_level_{level-1}.pt"))
                prune_harness.level_pruner(density=densities[level])
                sparsity = prune_harness.model.get_overall_sparsity()
                panel = Panel(f"[bold green]Model sparsity after pruning: {sparsity:.4f}[/bold green]", title="Sparsity", border_style="green", expand=False)
                console.print(panel)
            elif not is_iterative:
                console.print('[bold cyan]Pruning at initialization[/bold cyan]')
                prune_harness.prune_at_initialization(er_init=densities[level])
        
        if use_distributed:
            broadcast_model(prune_harness.model)
        
        if use_distributed:
            models_equal = check_model_equality(prune_harness.model)
            if rank == 0:
                if models_equal:
                    console.print("[bold green]Models are equal across all GPUs[/bold green]")
                else:
                    console.print("[bold red]Models are not equal across all GPUs[/bold red]")
        harness = Harness(model=prune_harness.model, expt_dir=packaged, gpu_id=rank)

        if level == 0 and rank == 0:
            save_model(harness.model, os.path.join(packaged[1], "checkpoints", "model_init.pt"), distributed=use_distributed)

        harness.train_one_level(level=level)

        if rank == 0:
            save_model(harness.model, os.path.join(packaged[1], "checkpoints", f"model_level_{level}.pt"), distributed=use_distributed)
            console.print(f"[bold green]Training level {level} complete, moving on to {level+1}[/bold green]")

    if rank == 0:
        wandb.finish()

if __name__ == "__main__":
    main()