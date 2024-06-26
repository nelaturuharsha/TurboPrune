import os
import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from fastargs import get_current_config
from fastargs.decorators import param

from typing import Optional

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
        distributed: bool = False,
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
            download=False,
            transform=self.transform_train,
        )

        testset = self.dataset_class(
            root=self.data_root,
            train=False,
            download=False,
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
