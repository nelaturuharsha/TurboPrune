from dataclasses import dataclass
from typing import Literal, Optional
from omegaconf import MISSING


@dataclass
class DatasetConfig:
    dataset_name: Literal["CIFAR10", "CIFAR100", "ImageNet"] = MISSING
    data_root_dir: str = MISSING
    total_batch_size: int = 512
    num_workers: int = 16
    gpu_workers: Optional[int] = 4
    dataloader_type: Literal["torch", "ffcv", "webdataset"] = MISSING


@dataclass
class ModelConfig:
    model_name: str = MISSING
    mask_layer_type: Literal["ConvMask", "LinearMask"] = MISSING
    use_compile: bool = False


@dataclass
class PruneConfig:
    prune_rate: float = MISSING
    prune_method: Literal[
        "er_erk",
        "er_balanced",
        "random_erk",
        "random_balanced",
        "synflow",
        "snip",
        "mag",
        "just dont",
    ] = MISSING
    target_sparsity: float = MISSING
    training_type: Literal["imp", "wr", "lrr", "at_init"] = MISSING


@dataclass
class ResumeExperimentConfig:
    resume_level: int = MISSING
    resume_expt_name: str = MISSING


@dataclass
class ExperimentConfig:
    seed: int = 0
    base_dir: str = "./experiments"

    epochs_per_level: int = MISSING
    training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    distributed: bool = False
    resume_experiment: bool = False
    resume_experiment_stuff: Optional[ResumeExperimentConfig] = None
    wandb_project_name: str = "TurboPrune_runs"
    imagenet_dataloader_type: Literal["ffcv", "webdataset"] = "ffcv"


@dataclass
class OptimizerConfig:
    optimizer_name: Literal["SGD", "AdamW", "Shampoo", "Muon", "SOAP"] = MISSING
    lr: float = MISSING
    momentum: float = MISSING
    weight_decay: float = MISSING
    scheduler_type: Literal[
        "MultiStepLRWarmup",
        "ImageNetLRDropsWarmup",
        "TriangularSchedule",
        "ScheduleFree",
        "TrapezoidalSchedule",
        "OneCycleLR",
    ] = MISSING
    warmup_fraction: float = 0.2


@dataclass
class CyclicTrainingConfig:
    num_cycles: int = 1
    strategy: Literal[
        "linear_increase",
        "linear_decrease",
        "exponential_decrease",
        "exponential_increase",
        "cyclic_peak",
        "alternating",
        "plateau",
        "constant",
    ] = "constant"


@dataclass
class MainConfig:
    defaults: list[str] = MISSING
    dataset_params: DatasetConfig = MISSING
    model_params: ModelConfig = MISSING
    pruning_params: PruneConfig = MISSING
    experiment_params: ExperimentConfig = MISSING
    optimizer_params: OptimizerConfig = MISSING
    cyclic_training: CyclicTrainingConfig
