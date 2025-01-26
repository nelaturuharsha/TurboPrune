## Pythonic imports
from typing import Tuple, Dict, Optional, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import optim

## config
from omegaconf import DictConfig

## file imports
from utils.pruning_utils import *
from utils.harness_utils import *
from utils.dataset import *
from utils.custom_models import *
from harness_definitions.base_harness import BaseHarness

import utils.schedulers as schedulers

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class PruningHarness(BaseHarness):
    """Extends BaseHarness with specific functionality for training and pruning sparse networks."""

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
                "scheduler_type": scheduler_type,
                "initial_lr": lr,
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
                steps_per_epoch=len(self.train_loader),
            )
        else:
            scheduler_cls = getattr(schedulers, scheduler_type)
            scheduler_args = {
                "cfg": self.cfg,
                "optimizer": self.optimizer,
                "steps_per_epoch": len(self.train_loader),
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
                raise NotImplementedError("This is a WIP")
            elif imagenet_dataloader_type == "ffcv":
                loaders = FFCVImagenet(cfg=self.cfg, this_device=self.device)
            else:
                raise ValueError(f"Invalid dataloader type: {imagenet_dataloader_type}")
        return loaders.train_loader, loaders.test_loader

    def train_one_level(self, epochs_per_level: int, level: int) -> None:
        """Train for one sparsity level"""
        level_metrics = {
            "epoch": [],
            "train_loss": [],
            "test_loss": [],
            "train_acc": [],
            "test_acc": [],
            "max_test_acc": [],
            "sparsity": [],
        }

        if self.gpu_id == 0:
            model = self.model.module if self.distributed else self.model

        self._setup_optimizer()
        self._setup_scheduler(epochs_per_level=epochs_per_level)
        if self.gpu_id == 0:
            self.training_info = {
                "model_name": self.cfg.model_params.model_name,
                "Training Epochs": epochs_per_level,
                "Level": f"{level}",
                "Current Sparsity": f"{model.get_overall_sparsity():.4f} ",
                "Target Sparsity": f"{self.cfg.pruning_params.target_sparsity * 100:.4f}%",
            }
            display_training_info(
                config_info=self.config_info,
                optimizer_info=self.optimizer_info,
                training_info=self.training_info,
            )

        if self.gpu_id == 0 and level == 0:
            save_model(
                self.model,
                os.path.join(self.expt_dir, "checkpoints", "model_init.pt"),
                distributed=self.distributed,
            )
            torch.save(
                self.optimizer.state_dict(),
                os.path.join(self.expt_dir, "artifacts", "optimizer_init.pt"),
            )

        for epoch in range(epochs_per_level):
            self.epoch_counter += 1

            if not self.distributed or dist.get_rank() == 0:
                self.console.rule(
                    f"[bold blue]Current Epoch: {epoch + 1}/{epochs_per_level}"
                )

            train_metrics = self.train_epoch()
            test_metrics = self.test()

            save_rewind_checkpoint = getattr(self.cfg.pruning_params, 'rewind_epoch', None) == epoch and level == 0

            if self.gpu_id == 0 and save_rewind_checkpoint:
                save_model(
                    self.model,
                    os.path.join(self.expt_dir, "checkpoints", "model_rewind.pt"),
                    distributed=self.distributed,
                )
                torch.save(
                    self.optimizer.state_dict(),
                    os.path.join(self.expt_dir, "artifacts", "optimizer_rewind.pt"),
                )

            metrics = {
                "epoch": int(self.epoch_counter),
                **train_metrics,
                **test_metrics,
            }

            if not self.distributed or dist.get_rank() == 0:
                self._log_metrics(metrics)

            if self.gpu_id == 0:
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
            }
            summary_df = pd.DataFrame(summary_data)

            summary_path = os.path.join(self.expt_dir, f"{self.prefix}_summary.csv")
            if os.path.exists(summary_path):
                summary_df.to_csv(summary_path, mode="a", header=False, index=False)
            else:
                summary_df.to_csv(summary_path, index=False)

    def _log_metrics(self, metrics: Dict[str, float]):
        """Override metric logging with wandb support."""
        if self.gpu_id == 0:
            wandb.log({**metrics, "epoch": int(self.epoch_counter)})
            super()._log_metrics(metrics)
