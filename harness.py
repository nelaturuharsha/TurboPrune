## Pythonic imports
from typing import Tuple, Dict, Optional, List

## Rich logging stuff
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
)

## PyTorch and related package imports
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch import optim
from torch.cuda.amp import autocast
from torchmetrics import Accuracy

## config
from omegaconf import DictConfig


from utils.pruning_utils import *
from utils.harness_utils import *
from utils.dataset import *
from utils.custom_models import *

import utils.schedulers as schedulers

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class BaseHarness:
    """Generic training harness that can be extended for different training paradigms.

    Args:
        cfg: OmegaConf DictConfig containing training parameters
        device: Device that this harness instance will run on.
        model: Model to train
        distributed: Whether to use distributed training
    """

    def __init__(
        self, cfg: DictConfig, device: str, model=None, distributed: bool = False
    ):
        self.cfg = cfg
        self.device = device
        self.distributed = distributed
        self.epoch_counter = 0
        self.console = Console()

        self.model = self._setup_model(model)
        self.criterion = self._setup_criterion()

        dist_sync = {"dist_sync_on_step": True} if self.distributed else {}
        self.train_accuracy = Accuracy(
            task="multiclass", num_classes=self.num_classes, **dist_sync
        ).to(self.device)
        self.test_accuracy = Accuracy(
            task="multiclass", num_classes=self.num_classes, **dist_sync
        ).to(self.device)

        self.train_loader, self.val_loader = self._setup_dataloaders()
        self.precision, self.use_amp = self._get_dtype_amp()

        self.training_info = {
            "dataset": self.dataset_name,
            "use_compile": self.use_compile,
            "distributed": self.distributed,
            "precision": self.precision,
            "use_amp": self.use_amp,
            "expt_dir": self.expt_dir,
        }

    def _setup_model(self, model):
        """Setup model and move to device."""
        if model is None:
            model = self._create_model()

        model = model.to(self.device)
        if self.distributed:
            model = DDP(model)
        return model

    def _create_model(self):
        """Create model based on config."""
        raise NotImplementedError("Implement in child class")

    def _setup_criterion(self):
        """Setup loss function."""
        return nn.CrossEntropyLoss()

    def _get_dtype_amp(self) -> Tuple[torch.dtype, bool]:
        """Get dtype and AMP settings based on config."""
        dtype_map = {
            "bfloat16": (torch.bfloat16, True),
            "float16": (torch.float16, True),
            "float32": (torch.float32, False),
        }
        return dtype_map.get(
            self.cfg.experiment_params.training_precision, (torch.float32, False)
        )

    def _setup_optimizer(self):
        """Setup optimizer based on config."""
        raise NotImplementedError("Implement in child class")

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        raise NotImplementedError("Implement in child class")

    def _setup_dataloaders(self):
        """Setup data loaders."""
        raise NotImplementedError("Implement in child class")

    def train_step(self, batch) -> Dict[str, float]:
        """Single training step with AMP support."""
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        self.optimizer.zero_grad()
        with autocast("cuda", dtype=self.precision, enabled=self.use_amp):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

        loss.backward()
        self.optimizer.step()

        self.train_accuracy.update(outputs, targets)

        return {"loss": loss.item()}

    def test_step(self, batch) -> Dict[str, float]:
        """Single test step with AMP support."""
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        with torch.no_grad(), autocast(
            "cuda", dtype=self.precision, enabled=self.use_amp
        ):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            self.test_accuracy.update(outputs, targets)

            return {"loss": loss.item()}

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = self.train_loader.num_batches

        if not self.distributed or dist.get_rank() == 0:
            progress_ctx = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            )
        else:
            from contextlib import nullcontext

            progress_ctx = nullcontext()

        with progress_ctx as progress:
            if not self.distributed or dist.get_rank() == 0:
                task = progress.add_task("[cyan]Training...", total=num_batches)

            for batch in self.train_loader:
                step_outputs = self.train_step(batch)
                total_loss += step_outputs["loss"]

                if (
                    self.scheduler is not None
                    and self.cfg.optimizer_params.scheduler_type
                    in ["OneCycleLR", "TriangularSchedule", "TrapezoidalSchedule"]
                ):
                    self.scheduler.step()

                if not self.distributed or dist.get_rank() == 0:
                    progress.advance(task)

        if self.cfg.optimizer_params.scheduler_type == "MultiStepLRWarmup":
            self.scheduler.step()

        avg_loss = total_loss / num_batches
        accuracy = self.train_accuracy.compute()

        if self.distributed:
            metrics = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
            avg_loss = metrics.item()
            del metrics

        self.train_accuracy.reset()

        return {"train_loss": avg_loss, "train_acc": accuracy.item() * 100}

    def test(self) -> Dict[str, float]:
        """Test model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = self.val_loader.num_batches

        if not self.distributed or dist.get_rank() == 0:
            progress_ctx = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            )
        else:
            from contextlib import nullcontext

            progress_ctx = nullcontext()

        with progress_ctx as progress:
            if not self.distributed or dist.get_rank() == 0:
                task = progress.add_task("[cyan]Testing...", total=num_batches)

            for batch in self.val_loader:
                step_outputs = self.test_step(batch)
                total_loss += step_outputs["loss"]

                if not self.distributed or dist.get_rank() == 0:
                    progress.advance(task)

        avg_loss = total_loss / num_batches
        accuracy = self.test_accuracy.compute()

        if self.distributed:
            metrics = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
            avg_loss = metrics.item()
            del metrics

        self.test_accuracy.reset()

        return {"test_loss": avg_loss, "test_acc": accuracy.item() * 100}

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics using rich table."""
        if not hasattr(self, "_metrics_table"):
            self._metrics_table = Table(
                show_header=True,
                header_style="bold magenta",
                title="Optimization Metrics",
            )
            metrics_order = [
                "cycle",
                "epoch",
                "train_loss",
                "train_acc",
                "test_loss",
                "test_acc",
            ]
            for metric in metrics_order:
                self._metrics_table.add_column(metric.replace("_", " ").title())

        metrics_order = [
            "cycle",
            "epoch",
            "train_loss",
            "train_acc",
            "test_loss",
            "test_acc",
        ]
        colors = {
            "cycle": "magenta",
            "epoch": "blue",
            "train_loss": "red",
            "train_acc": "green",
            "test_loss": "yellow",
            "test_acc": "cyan",
        }

        row = [
            f"[{colors[metric]}]{metrics[metric]:.4f}[/{colors[metric]}]"
            for metric in metrics_order
            if metric in metrics
        ]
        self._metrics_table.add_row(*row)

        # Create a standard pretty table
        from prettytable import PrettyTable

        summary_table = PrettyTable()
        summary_table.field_names = ["Metric", "Value"]
        summary_table.align["Metric"] = "l"  # Left align metric names
        summary_table.align["Value"] = "r"  # Right align values

        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                summary_table.add_row(
                    [metric.replace("_", " ").title(), f"{value:.4f}"]
                )

        self.console.print(self._metrics_table)
        print("\nCurrent Status:")
        print(summary_table)


class PruningHarness(BaseHarness):
    """Extends BaseHarness with specific functionality for cyclic training and pruning."""

    def __init__(
        self,
        cfg: DictConfig,
        gpu_id: int,
        expt_dir: str,
        model: Optional[nn.Module] = None,
    ):
        self.gpu_id = gpu_id
        self.dataset_name = cfg.dataset_params.dataset_name.lower()
        self.use_compile = cfg.model_params.use_compile
        self.num_classes = self._get_num_classes()
        self.this_device = f"cuda:{self.gpu_id}"
        self.prefix, self.expt_dir = expt_dir

        super().__init__(
            cfg=cfg,
            device=self.this_device,
            model=model,
            distributed=cfg.experiment_params.distributed
            and torch.cuda.device_count() > 1
            and not self.dataset_name.startswith("cifar"),
        )
        self.precision, self.use_amp = self._get_dtype_amp()

    def _setup_optimizer(self) -> None:
        """Create optimizer based on config."""
        lr = self.cfg.optimizer_params.lr
        momentum = self.cfg.optimizer_params.momentum
        weight_decay = self.cfg.optimizer_params.weight_decay
        scheduler_type = self.cfg.optimizer_params.scheduler_type

        if scheduler_type == "ScheduleFree":
            import schedulefree

            self.optimizer = schedulefree.SGDScheduleFree(
                self.model.parameters(),
                warmup_steps=self.cfg.optimizer_params.warmup_steps,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )

        if self.gpu_id == 0:
            self.optimizer_info = {
                "type": type(self.optimizer).__name__,
                "lr": lr,
                "momentum": momentum,
                "weight_decay": weight_decay,
            }

    def _setup_scheduler(self, epochs_per_level: int) -> None:
        """Create learning rate scheduler based on config."""
        scheduler_type = self.cfg.optimizer_params.scheduler_type
        print(f"Setting up scheduler of type {scheduler_type}")
        if scheduler_type == "ScheduleFree":
            self.scheduler = None
        elif scheduler_type == "OneCycleLR":
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.cfg.optimizer_params.lr,
                epochs=epochs_per_level,
                steps_per_epoch=self.train_loader.num_batches,
            )
        else:
            scheduler_cls = getattr(schedulers, scheduler_type)
            scheduler_args = {
                "cfg": self.cfg,
                "optimizer": self.optimizer,
                "steps_per_epoch": self.train_loader.num_batches,
            }

            if scheduler_type == "TriangularSchedule":
                scheduler_args["epochs_per_level"] = epochs_per_level
            elif scheduler_type == "TrapezoidalSchedule":
                scheduler_args.update(
                    {
                        "warmup_steps": len(self.train_loader)
                        * self.cfg.optimizer_params.trapezoidal_scheduler_stuff.warmup_steps,
                        "cooldown_steps": len(self.train_loader)
                        * self.cfg.optimizer_params.trapezoidal_scheduler_stuff.cooldown_steps,
                    }
                )

            self.scheduler = scheduler_cls(**scheduler_args)

    def _get_num_classes(self) -> int:
        if self.dataset_name.startswith("imagenet"):
            return 1000
        elif self.dataset_name.startswith("cifar100"):
            return 100
        return 10

    def _create_model(self) -> nn.Module:
        """Override base model creation with TorchVision/Custom model support."""
        try:
            model = TorchVisionModel(cfg=self.cfg)
            model_type = "TorchVision"
        except:
            model = CustomModel(config=self.config)
            model_type = "Custom"

        if self.gpu_id == 0:
            self.console.print(
                f"[bold turquoise]Using {model_type} Model :D[/bold turquoise]"
            )
        if self.use_compile:
            model = torch.compile(model, mode="reduce-overhead")
        return model

    def _setup_dataloaders(self):
        """Override dataloader setup with dataset-specific loaders."""
        if self.dataset_name.lower().startswith("cifar"):
            loaders = AirbenchLoaders(cfg=self.cfg)
        elif self.dataset_name.lower().startswith("imagenet"):
            imagenet_dataloader_type = self.cfg.dataset_params.dataloader_type
            if imagenet_dataloader_type == "webdataset":
                loaders = WebDatasetImageNet(cfg=self.cfg)
            elif imagenet_dataloader_type == "ffcv":
                loaders = FFCVImagenet(cfg=self.cfg, this_device=self.device)
            else:
                raise ValueError(f"Invalid dataloader type: {imagenet_dataloader_type}")
        return loaders.train_loader, loaders.test_loader

    def train_one_level(
        self, num_cycles: int, epochs_per_level: int, level: int
    ) -> None:
        """Train for one level with cyclic training support."""
        use_cyclic_training = num_cycles > 1
        level_metrics = {
            "cycle": [],
            "epoch": [],
            "train_loss": [],
            "test_loss": [],
            "train_acc": [],
            "test_acc": [],
            "max_test_acc": [],
            "sparsity": [],
        }

        epoch_schedule = (
            generate_cyclical_schedule(epochs_per_level=epochs_per_level)
            if use_cyclic_training
            else [epochs_per_level]
        )

        for cycle in range(num_cycles):
            if self.gpu_id == 0:
                model = self.model.module if self.distributed else self.model
                self.cycle_info = {
                    "Number of Cycles": num_cycles,
                    "Epochs per Cycle": epoch_schedule,
                    "Total Training Length": f"{sum(epoch_schedule)} epochs",
                    "Training Cycle": f"{cycle + 1}/{num_cycles}",
                    "Epochs this cycle": f"{epoch_schedule[cycle]}",
                    "Total epochs so far": f"{sum(epoch_schedule[:cycle+1])}/{sum(epoch_schedule)}",
                    "Current Sparsity": f"{model.get_overall_sparsity():.4f}",
                }

            self._setup_optimizer()
            self._setup_scheduler(epochs_per_level=epoch_schedule[cycle])
            if self.gpu_id == 0:
                display_training_info(
                    cycle_info=self.cycle_info,
                    training_info=self.training_info,
                    optimizer_info=self.optimizer_info,
                )

            if self.gpu_id == 0 and level == 0 and cycle == 0:
                save_model(
                    self.model,
                    os.path.join(self.expt_dir, "checkpoints", "model_rewind.pt"),
                    distributed=self.distributed,
                )
                torch.save(
                    self.optimizer.state_dict(),
                    os.path.join(self.expt_dir, "artifacts", "optimizer_rewind.pt"),
                )

            for epoch in range(epoch_schedule[cycle]):
                self.epoch_counter += 1

                if not self.distributed or dist.get_rank() == 0:
                    self.console.rule(
                        f"[bold blue]Current Epoch: {epoch + 1}/{epoch_schedule[cycle]}, Global Epoch: {self.epoch_counter}/{sum(epoch_schedule)}"
                    )

                train_metrics = self.train_epoch()
                test_metrics = self.test()

                metrics = {
                    "cycle": int(cycle),
                    "epoch": int(self.epoch_counter),
                    **train_metrics,
                    **test_metrics,
                }

                if not self.distributed or dist.get_rank() == 0:
                    self._log_metrics(metrics)

                if self.gpu_id == 0:
                    level_metrics["cycle"].append(cycle)
                    level_metrics["epoch"].append(self.epoch_counter)
                    level_metrics["train_loss"].append(metrics["train_loss"])
                    level_metrics["test_loss"].append(metrics["test_loss"])
                    level_metrics["train_acc"].append(metrics["train_acc"])
                    level_metrics["test_acc"].append(metrics["test_acc"])
                    level_metrics["max_test_acc"].append(max(level_metrics["test_acc"]))
                    level_metrics["sparsity"].append(model.get_overall_sparsity())

            if self.gpu_id == 0:
                df = pd.DataFrame(level_metrics)
                df.to_csv(
                    os.path.join(
                        self.expt_dir,
                        "metrics",
                        "level_wise_metrics",
                        f"level_{level}_metrics.csv",
                    ),
                    index=False,
                )

        if self.gpu_id == 0:
            model = self.model.module if self.distributed else self.model
            summary_data = {
                "Level": [level],
                "Sparsity": [model.get_overall_sparsity()],
                "Last_Test_Acc": [level_metrics["test_acc"][-1]],
                "Max_Test_Acc": [max(level_metrics["test_acc"])],
                "Schedule": [str(epoch_schedule)],
            }
            summary_df = pd.DataFrame(summary_data)

            summary_path = os.path.join(self.expt_dir, f"{self.prefix}_summary.csv")
            if os.path.exists(summary_path):
                summary_df.to_csv(summary_path, mode="a", header=False, index=False)
            else:
                summary_df.to_csv(summary_path, index=False)

    def _setup_cycle_info(
        self, cycle: int, num_cycles: int, epoch_schedule: List[int], level: int
    ):
        """Setup cycle information for logging."""
        model = self.model.module if self.distributed else self.model

        self.cycle_info = {
            "Number of Cycles": num_cycles,
            "Epochs per Cycle": epoch_schedule,
            "Total Training Length": f"{sum(epoch_schedule)} epochs",
            "Training Cycle": f"{cycle + 1}/{num_cycles}",
            "Epochs this cycle": f"{epoch_schedule[cycle]}",
            "Total epochs so far": f"{sum(epoch_schedule[:cycle+1])}/{sum(epoch_schedule)}",
            "Current Sparsity": f"{model.get_overall_sparsity():.4f}",
        }

        display_training_info(
            cycle_info=self.cycle_info,
            training_info=self.training_info,
            optimizer_info=self.optimizer_info,
        )

    def _log_metrics(self, metrics: Dict[str, float]):
        """Override metric logging with wandb support."""
        if self.gpu_id == 0:
            wandb.log({**metrics, "epoch": int(self.epoch_counter)})
            super()._log_metrics(metrics)
