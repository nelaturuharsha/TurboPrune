import fastargs
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

def get_current_params():
    Section('model_params', 'model details').params(
    model_name=Param(str, 'model_choice', default='ResNet50', required=True),
    first_layer_type=Param(str, 'layer type'),
    conv_type=Param(And(str, OneOf(['ConvMask'])), required=True),
    bn_type=Param(And(str, OneOf(['LearnedBatchNorm'])), required=True),
    init=Param(And(str, OneOf(['kaiming_normal'])), required=True),
    nonlinearity=Param(And(str, OneOf(['relu', 'leaky_relu'])), required=True),
    mode=Param(And(str, OneOf(['fan_in'])), required=True),
    scale_fan=Param(bool, 'use scale fan', required=True))

Section('dataset', 'dataset configuration').params(
    dataset_name=Param(And(str, OneOf(['CIFAR10', 'CIFAR100', 'ImageNet'])),'Name of dataset', required=True),
    num_classes=Param(And(int, OneOf([10, 100, 1000])), 'number of classes',required=True),
    batch_size=Param(int, 'batch size', default=512),
    num_workers=Param(int, 'num_workers', default=8),)

Section('prune_params', 'pruning configuration').params(
    prune_rate=Param(float, 'percentage of parameters to remove',required=True),
    er_init=Param(float, 'sparse init percentage/target', required=True),
    er_method=Param(And(OneOf(['er_erk', 'er_balanced', 'synflow', 'snip', 'just dont'])), required=True),
    prune_method=Param(And(OneOf(['random_erk', 'random_balanced', 'synflow', 'snip', 'mag']))))

Section('experiment_params', 'parameters to train model').params(
    epochs_per_level=Param(int, 'number of epochs per level', required=True),
    num_levels=Param(int, 'number of pruning levels', required=True),
    training_type=Param(And(str, OneOf(['imp', 'wr', 'lrr'])), required=True))

Section('optimizer', 'data related stuff').params(
    lr=Param(float, 'Name of dataset', required=True),
    num_workers=Param(int, 'num_workers', default=8),
    momentum=Param(float, 'momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=1e-4),
    warmup_epochs=Param(int, 'warmup length', default=10),
    scheduler_type=Param(And(str, OneOf(['MultiStepLRWarmup', 'ImageNetLRDropsWarmup', 'CosineLRWarmup'])), required=True),
    lr_min=Param(float, 'minimum learning rate for cosine', default=0.01))

Section('dist', 'distributed parameters').params(
    distributed=Param(bool, 'use distributed training', default=True),
    address=Param(str, 'default address', default='localhost'),
    port=Param(str, 'default port', default='12345'),
)