from fastargs import Param, Section
from fastargs.validation import And, OneOf

def get_current_params() -> None:
    """Define the various parameters and their constraints with fastargs."""
    Section("model_params", "model details").params(
        model_name=Param(str, "model_choice", default="resnet18", required=True),
        mask_layer_type=Param(And(str, OneOf(["ConvMask", "LinearMask"])), required=True),
    )

    Section("dataset", "dataset configuration").params(
        dataset_name=Param(
            And(str, OneOf(["CIFAR10", "CIFAR100", "ImageNet"])),
            "Name of dataset",
            required=True,
        ),
        batch_size=Param(int, "batch size", required=True),
        num_workers=Param(int, "num_workers", required=True),
        data_root_dir=Param(str, "path to betons", required=True),
    )

    Section("prune_params", "pruning configuration").params(
        prune_rate=Param(float, "percentage of parameters to remove per level or at initialization", required=True),
        prune_method=Param(
            And(
                str, OneOf(["er_erk", "er_balanced", "random_erk", "random_balanced", "synflow", "snip", "mag", "just dont"])
            ),
            required=True,
        ),
        target_sparsity=Param(float, 'target sparsity of the entire pruning cycle', required=True),
    )

    Section("experiment_params", "parameters to train model").params(
        seed=Param(int, "seed", default=0),
        base_dir=Param(str, "base directory", required=True, default="./experiments"),
        training_type=Param(And(str, OneOf(["imp", "wr", "lrr"])), required=True),
        resume_level=Param(
            int, "level to resume from -- 0 if starting afresh", default=0
        ),
        resume_expt_name=Param(str, "resume path"),
        training_precision = Param(And(str, OneOf(['bfloat16', 'float32'])), default='float32'),
        use_compile = Param(str, "use torch compile", default='false'),
        compute_metrics = Param(str, "compute metrics", default='true')
    )

    Section("optimizer", "data related stuff").params(
        lr=Param(float, "learning rate", required=True),
        momentum=Param(float, "momentum", required=True),
        weight_decay=Param(float, "weight decay", required=True),
        warmup_steps=Param(int, "warmup length"),
        cooldown_steps=Param(int, 'cooldown steps'),
        scheduler_type=Param(
            And(
                str,
                OneOf(
                    [
                        "MultiStepLRWarmup",
                        "ImageNetLRDropsWarmup",
                        "TriangularSchedule",
                        "ScheduleFree",
                        "TrapezoidalSchedule",
                        "OneCycleLR"
                    ]
                ),
            ),
            required=True,
        ),
        lr_start=Param(float, "start learning rate", required=True  ),
        lr_peak=Param(float, "peak learning rate", required=True),
        lr_end=Param(float, "end learning rate", required=True),
        skip_warmup=Param(str, "skip warmup", default='false'),
        use_single_scheduler_cycle=Param(str, "use single scheduler cycle", default='false')
    )

    Section("dist_params", "distributed parameters").params(
        distributed=Param(bool, "use distributed training", default=True))

    Section('cyclic_training', 'parameters for cyclic training').params(
        epochs_per_level=Param(int, 'maximum number of epochs each level is trained for', required=True),
        num_cycles=Param(int, "number of cycles used for cyclic training", default=1),
        strategy=Param(And(str, OneOf(['linear_increase', 'linear_decrease', 'exponential_decrease', 
                                      'exponential_increase', 'cyclic_peak', 'alternating', 'plateau', 'constant'])), default='constant'),
        total_training_budget=Param(int, "total training budget")
        )
    
    Section('wandb_params', 'parameters for wandb').params(
        project_name=Param(str, 'project', required=True)
    )