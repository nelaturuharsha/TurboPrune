import os
import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from fastargs import get_current_config
from fastargs.decorators import param

import numpy as np

from typing import Optional

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224 / 256

config = get_current_config()

get_current_config()
try:
    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import (
        ToTensor,
        ToDevice,
        Squeeze,
        NormalizeImage,
        RandomHorizontalFlip,
        ToTorchImage,
    )
    from ffcv.fields.decoders import (
        RandomResizedCropRGBImageDecoder,
        CenterCropRGBImageDecoder,
    )
    from ffcv.fields.basics import IntDecoder
except ImportError:
    raise ImportError(
        'FFCV is not installed. Please install FFCV to train on ImageNet. '
        'You can install it by following the instructions provided in the repository'
    )
##else:
#    print('FFCV imports are ignored since you arent training on ImageNet')

class CIFARLoader:
    """Data loader for CIFAR-10 and CIFAR-100 datasets.

    Args:
        dataset_name (str): Name of the dataset ('CIFAR10' or 'CIFAR100').
        data_root (str): Directory to save the dataset which will be downloaded.
        batch_size (int): Batch size.
        distributed (bool, optional): Whether to use DistributedSampler for distributed training. Default is False.
    """

    @param("dataset.dataset_name")
    @param("dataset.data_root")
    @param("dataset.batch_size")
    def __init__(
        self,
        dataset_name: str,
        data_root: str,
        batch_size: int,
        distributed: bool = False
    ) -> None:
        super(CIFARLoader, self).__init__()

        self.dataset = dataset_name
        self.distributed = distributed

        if self.dataset == "CIFAR10":
            self.data_root = os.path.join(data_root, "cifar10")
            self.transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
            self.transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
            self.dataset_class = datasets.CIFAR10

        elif self.dataset == "CIFAR100":
            self.data_root = os.path.join(data_root, "cifar100")
            self.transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]
            )
            self.transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]
            )
            self.dataset_class = datasets.CIFAR100

        trainset = self.dataset_class(
            root=self.data_root,
            train=True,
            download=True,
            transform=self.transform_train,
        )

        testset = self.dataset_class(
            root=self.data_root,
            train=False,
            download=True,
            transform=self.transform_test,
        )

        if self.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                trainset, num_replicas=torch.cuda.device_count(), shuffle=True
            )
        else:
            self.train_sampler = None

        self.train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
        )

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False
        )


class imagenet:
    """
    Relevant DataLoaders and other functions for training on ImageNet.
    Uses FFCV. Please ensure that it is installed!

    Args:
        data_root (str): path to root folder which contains the train and validation betons.
        Ensure that the file name is the same as in the class.
        batch_size (int): batch size you'd like to train with
        distributed (bool, optional): whether you'd like to train on multiple GPUs or not, by default: False
    """

    @param('dataset.data_root')
    @param('dataset.batch_size')
    @param('dataset.num_workers')
    def __init__(self, data_root, batch_size, num_workers, this_device, distributed=False):
        self.this_device = this_device

        train_image_pipeline = [
                RandomResizedCropRGBImageDecoder((224, 224)),
                RandomHorizontalFlip(),
                ToTensor(),
                ToDevice(torch.device(self.this_device), non_blocking=True),
                ToTorchImage(),
                NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
            ]

        val_image_pipeline = [
            CenterCropRGBImageDecoder((256, 256), ratio=DEFAULT_CROP_RATIO),
            ToTensor(),
            ToDevice(torch.device(self.this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
        ]

        label_pipeline = [
                IntDecoder(),
                ToTensor(),
                Squeeze(),
                ToDevice(torch.device(self.this_device), non_blocking=True),
            ]


        self.train_loader = Loader(
            os.path.join(data_root, "train_500_0.50_90.beton"),
            batch_size=batch_size,
            num_workers=num_workers,
            order=OrderOption.RANDOM,
            os_cache=True,
            drop_last=True,
            pipelines={"image": train_image_pipeline, "label": label_pipeline},
            distributed=distributed,
        )

        self.test_loader = Loader(
            os.path.join(data_root, "val_500_0.50_90.beton"),
            batch_size=batch_size,
            num_workers=num_workers,
            order=OrderOption.SEQUENTIAL,
            drop_last=False,
            pipelines={"image": val_image_pipeline, "label": label_pipeline},
            distributed=distributed,
        )


