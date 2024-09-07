from fastargs import Param, Section
from fastargs.validation import And, OneOf

def get_current_params() -> None:
    """Define the various parameters and their constraints with fastargs."""
    Section("model_params", "model details").params(
        model_name=Param(str, "model_choice", default="resnet18", required=True),
        conv_type=Param(And(str, OneOf(["ConvMask"])), required=True),
    )

    Section("dataset", "dataset configuration").params(
        dataset_name=Param(
            And(str, OneOf(["CIFAR10", "CIFAR100", "ImageNet"])),
            "Name of dataset",
            required=True,
        ),
        batch_size=Param(int, "batch size", default=512),
        num_workers=Param(int, "num_workers", default=32),
        data_root=Param(str, "path to betons", required=True),
    )

    Section("prune_params", "pruning configuration").params(
        prune_rate=Param(float, "percentage of parameters to remove", default=0.2),
        er_init=Param(float, "sparse init percentage/target", default=0.2),
        er_method=Param(
            And(str, OneOf(["er_erk", "er_balanced", "synflow", "snip", "just dont"])),
            default="just dont",
        ),
        prune_method=Param(
            And(
                str, OneOf(["random_erk", "random_balanced", "synflow", "snip", "mag", "just dont"])
            ),
            default='mag',
        ),
    )

    Section("experiment_params", "parameters to train model").params(
        seed=Param(int, "seed", default=0),
        base_dir=Param(str, "base directory", required=True, default="./experiments"),
        epochs_per_level=Param(int, "number of epochs per level", required=True),
        training_type=Param(And(str, OneOf(["imp", "wr", "lrr"])), required=True),
        resume_level=Param(
            int, "level to resume from -- 0 if starting afresh", default=0
        ),
        resume_expt_name=Param(str, "resume path"),
        training_precision = Param(And(str, OneOf(['bfloat16', 'float32'])), default='float32')
    )

    Section("optimizer", "data related stuff").params(
        lr=Param(float, "learning rate", required=True),
        momentum=Param(float, "momentum", default=0.9),
        weight_decay=Param(float, "weight decay", default=1e-4),
        warmup_steps=Param(int, "warmup length", default=10),
        cooldown_steps=Param(int, 'cooldown steps', default=10),
        scheduler_type=Param(
            And(
                str,
                OneOf(
                    [
                        "MultiStepLRWarmup",
                        "ImageNetLRDropsWarmup",
                        #"CosineLRWarmup",
                        "TriangularSchedule",
                        "ScheduleFree",
                        "TrapezoidalSchedule",
                        "OneCycleLR"
                    ]
                ),
            ),
            required=True,
        ),
        lr_min=Param(float, "minimum learning rate for cosine", default=0.01),
    )

    Section("dist_params", "distributed parameters").params(
        distributed=Param(bool, "use distributed training", default=True),
        address=Param(str, "default address", default="localhost"),
        port=Param(int, "default port", default=12350),
    )

    Section('cyclic_training', 'parameters for cyclic training').params(
        total_epoch_budget=Param(int, 'maximum number of epochs the model is trained for across the entire training run', required=True),
        num_cycles=Param(int, "number of cycles used for cyclic training", default=1),
        strategy=Param(And(str, OneOf(['linear_increase', 'linear_decrease', 'exponential_decrease', 
                                      'exponential_increase', 'cyclic_peak', 'alternating', 'plateau', 'constant'])), default='constant'),
        )
    Section('wandb_params', 'parameters for wandb').params(
        project_name=Param(str, 'project', required=True)
    )
