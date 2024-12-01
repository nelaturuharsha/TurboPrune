from fastargs import Param, Section
from fastargs.validation import And, OneOf

def get_current_params() -> None:
    """Define the various parameters and their constraints with fastargs."""
    Section("model_params", "model details").params(
        model_name=Param(str, "model_choice", required=True),
        mask_layer_type=Param(And(str, OneOf(["ConvMask", "LinearMask"])), required=True),
    )

    Section("dataset", "dataset configuration").params(
        dataset_name=Param(
            And(str, OneOf(["CIFAR10", "CIFAR100", "ImageNet"])),
            "Name of dataset",
            required=True,
        ),
        batch_size=Param(int, "batch size", required=True),
        data_root_dir=Param(str, "path to betons", required=True),
        num_workers=Param(int, "num_workers", default=16),
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
        training_type=Param(And(str, OneOf(["imp", "wr", "lrr", "at_init"])), required=True),
        epochs_per_level=Param(int, 'maximum number of epochs each level is trained for', required=True),
        training_precision = Param(And(str, OneOf(['bfloat16', 'float32'])), default='bfloat16'),
        use_compile = Param(And(str, OneOf(['true', 'false'])), "use torch compile", default='false'),
        resume_experiment=Param(And(str, OneOf(['true', 'false'])), "resume experiment", default='false'),
    )

    Section('experiment_params.resume_experiment_stuff', 'parameters for resuming experiment').enable_if(lambda cfg: cfg['experiment_params.resume_experiment'] == 'true').params(
        resume_level=Param(int, "level to resume from -- 0 if starting afresh", required=True),
        resume_expt_name=Param(str, "resume experiment name", required=True)
    )

    Section("optimizer", "data related stuff").params(
        lr=Param(float, "learning rate", required=True),
        momentum=Param(float, "momentum", required=True),
        weight_decay=Param(float, "weight decay", required=True),
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
        warmup_fraction=Param(float, "fraction of epochs to warmup", default=0.2),
    )
    

    Section("dist_params", "distributed parameters").params(
        distributed=Param(And(str, OneOf(['true', 'false'])), "use distributed training", default='false'))

    Section('cyclic_training', 'parameters for cyclic training').params(
        num_cycles=Param(int, "number of cycles used for cyclic training", default=1),
        strategy=Param(And(str, OneOf(['linear_increase', 'linear_decrease', 'exponential_decrease', 
                                      'exponential_increase', 'cyclic_peak', 'alternating', 'plateau', 'constant'])), default='constant'),
    )
    
    Section('wandb_params', 'parameters for wandb').params(
        project_name=Param(str, 'project', required=False)
    )