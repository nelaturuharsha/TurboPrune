from fastargs import Param, Section
from fastargs.validation import And, OneOf

def get_current_params():
    Section('model_params', 'model details').params(
    model_name=Param(str, 'model_choice', default='resnet18', required=True),
    conv_type=Param(And(str, OneOf(['ConvMask'])), required=True))
    
Section('dataset', 'dataset configuration').params(
    dataset_name=Param(And(str, OneOf(['CIFAR10', 'CIFAR100', 'ImageNet'])),'Name of dataset', required=True),
    num_classes=Param(And(int, OneOf([10, 100, 1000])), 'number of classes',required=True),
    batch_size=Param(int, 'batch size', default=512),
    num_workers=Param(int, 'num_workers', default=8),
    data_root=Param(str, 'path to betons', required=True))

Section('prune_params', 'pruning configuration').params(
    prune_rate=Param(float, 'percentage of parameters to remove',required=True),
    er_init=Param(float, 'sparse init percentage/target', required=True),
    er_method=Param(And(OneOf(['er_erk', 'er_balanced', 'synflow', 'snip', 'just dont'])), required=True),
    prune_method=Param(And(OneOf(['random_erk', 'random_balanced', 'synflow', 'snip', 'mag']))))

Section('experiment_params', 'parameters to train model').params(
    epochs_per_level=Param(int, 'number of epochs per level', required=True),
    num_levels=Param(int, 'number of pruning levels', required=True),
    training_type=Param(And(str, OneOf(['imp', 'wr', 'lrr'])), required=True),
    expt_setup=Param(And(str, OneOf(['cispa', 'others'])), required=True),
    resume_from_level=Param(int, 'level to resume from', default=0)) 


Section('optimizer', 'data related stuff').params(
    lr=Param(float, 'Name of dataset', required=True),
    num_workers=Param(int, 'num_workers', default=8),
    momentum=Param(float, 'momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=1e-4),
    warmup_epochs=Param(int, 'warmup length', default=10),
    scheduler_type=Param(And(str, OneOf(['MultiStepLRWarmup', 'ImageNetLRDropsWarmup', 'CosineLRWarmup', 'CustomLRScheduler'])), required=True),
    lr_min=Param(float, 'minimum learning rate for cosine', default=0.01))

Section('dist_params', 'distributed parameters').params(
    distributed=Param(bool, 'use distributed training', default=True),
    address=Param(str, 'default address', default='localhost'),
    port=Param(int, 'default port', default=12350),)

