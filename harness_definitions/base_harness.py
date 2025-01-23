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
from torch.amp import autocast
from torchmetrics import Accuracy

## config
from omegaconf import DictConfig

import wandb

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

        self.config_info = {
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
        with autocast(device_type=self.device, dtype=self.precision, enabled=self.use_amp):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

        loss.backward()
        self.optimizer.step()
        if self.gpu_id == 0:
            wandb.log({"learning_rate": self.optimizer.param_groups[0]["lr"]})

        self.train_accuracy.update(outputs, targets)

        return {"loss": loss.item()}

    def test_step(self, batch) -> Dict[str, float]:
        """Single test step with AMP support."""
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        with torch.no_grad(), autocast(device_type=self.device, dtype=self.precision, enabled=self.use_amp):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            self.test_accuracy.update(outputs, targets)

            return {"loss": loss.item()}

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

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
        num_batches = len(self.val_loader)

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

        from prettytable import PrettyTable

        summary_table = PrettyTable()
        summary_table.field_names = ["Metric", "Value"]
        summary_table.align["Metric"] = "l"
        summary_table.align["Value"] = "r"

        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                summary_table.add_row(
                    [metric.replace("_", " ").title(), f"{value:.4f}"]
                )

        self.console.print(self._metrics_table)
        print("\nCurrent Status:")
        print(summary_table)
