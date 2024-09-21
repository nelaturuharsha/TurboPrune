import os
import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split, Subset
from torch.utils.data.sampler import SubsetRandomSampler

from fastargs import get_current_config
from fastargs.decorators import param
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224 / 256

get_current_config()

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
        self.config = get_current_config()

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
            print('FFCV loaded, all good -- you can work on ImageNet!')
        except ImportError:
            raise ImportError(
                'FFCV is not installed. Please install FFCV to train on ImageNet. '
                'You can install it by following the instructions provided in the repository'
            )
        
        train_image_pipeline = [
                RandomResizedCropRGBImageDecoder((224, 224)),
                RandomHorizontalFlip(),
                ToTensor(),
                ToDevice(torch.device(self.this_device), non_blocking=True),
                ToTorchImage(),
                NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
            ]

        val_image_pipeline = [
            CenterCropRGBImageDecoder((224, 224), ratio=DEFAULT_CROP_RATIO),
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

########
class imagenet_subsampled:
    """
    Relevant DataLoaders and other functions for training on ImageNet.
    Uses FFCV. Please ensure that it is installed!
    Subsamples and uses a random subset of the dataset.
    Args:
        data_root (str): path to root folder which contains the train and validation betons.
        Ensure that the file name is the same as in the class.
        batch_size (int): batch size you'd like to train with
        distributed (bool, optional): whether you'd like to train on multiple GPUs or not, by default: False
    """

    @param('dataset.data_root')
    @param('dataset.batch_size')
    @param('dataset.num_workers')
    @param('dataset.subsample_frac')
    def __init__(self, data_root, batch_size, num_workers, subsample_frac, this_device, distributed=False):
        self.this_device = this_device
        self.config = get_current_config()
        # imagenet has these many train images
        
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
            print('FFCV loaded, all good -- you can work on ImageNet!')
        except ImportError:
            raise ImportError(
                'FFCV is not installed. Please install FFCV to train on ImageNet. '
                'You can install it by following the instructions provided in the repository'
            )
        
        train_image_pipeline = [
                RandomResizedCropRGBImageDecoder((224, 224)),
                RandomHorizontalFlip(),
                ToTensor(),
                ToDevice(torch.device(self.this_device), non_blocking=True),
                ToTorchImage(),
                NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
            ]

        val_image_pipeline = [
            CenterCropRGBImageDecoder((224, 224), ratio=DEFAULT_CROP_RATIO),
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

        num_images = 1281167
        indices = list(range(num_images))
        # # Shuffle indices and select a subset
        np.random.shuffle(indices)
        indices = indices[:int(subsample_frac * num_images)]

        print('Subsampled indiced are being chosen')
        self.train_loader = Loader(
            os.path.join(data_root, "train_500_0.50_90.beton"),
            batch_size=batch_size,
            num_workers=num_workers,
            order=OrderOption.RANDOM,
            os_cache=True,
            drop_last=True,
            pipelines={"image": train_image_pipeline, "label": label_pipeline},
            distributed=distributed,
            indices=indices
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

# CIFAR10 and CIFAR100 subsampled datasets
class CIFARLoader_subsampled:
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
    @param('dataset.subsample_frac')
    def __init__(
        self,
        dataset_name: str,
        data_root: str,
        batch_size: int,
        subsample_frac: float,
        distributed: bool = False
    ) -> None:
        super(CIFARLoader_subsampled, self).__init__()

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
        num_samples = int(len(trainset) * subsample_frac)
        indices = torch.randperm(len(trainset))[:num_samples].tolist()
        # subsampled dataset with only the indices that are listed above
        trainset = Subset(trainset, indices)
        print('Subsampled data before and after: {}, {}'.format(num_samples, len(trainset)))
        
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


import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

class imagenet_pytorch:
    """
    Relevant DataLoaders and other functions for training on ImageNet.
    Uses FFCV. Please ensure that it is installed!

    Args:
        data_root (str): path to root folder which contains the train and validation betons.
        Ensure that the file name is the same as in the class.
        batch_size (int): batch size you'd like to train with
        distributed (bool, optional): whether you'd like to train on multiple GPUs or not, by default: False
    """

    @param('dataset.batch_size')
    @param('dataset.num_workers')
    def __init__(self, batch_size, num_workers, world_size, rank, this_device, distributed=False):
        self.this_device = this_device
        self.config = get_current_config()

        # Define the transformations for the training and validation datasets
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Paths to the ImageNet data directories
        train_dir = '/home/c01adga/CISPA-projects/ImageNet-2022/tmp/train-data'
        val_dir = '/home/c01adga//CISPA-projects/ImageNet-2022/tmp/val-2011/val'
        # 
        # train_dir = '/tmp/Imagenet/train'
        # val_dir = '/tmp/Imagenet/val'
        # Create the ImageNet dataset objects
        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
        val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transforms)

        # Create DistributedSampler to handle distributed training
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

        # Create DataLoaders
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler, 
                                num_workers=8, pin_memory=True)
        self.test_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, sampler=val_sampler, 
                                num_workers=8, pin_memory=True)

