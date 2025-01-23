# Standard library imports
import os
from math import ceil
from filelock import FileLock  ## need this for large-scale sweeps on the same machine

# Third party imports
import numpy as np
import os

## PyTorch imports
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms as T

# Third party imports
from omegaconf import DictConfig
from rich.console import Console
from rich.panel import Panel

import webdataset as wds

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224 / 256

CIFAR10_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR10_STD = torch.tensor((0.2470, 0.2435, 0.2616))
CIFAR100_MEAN = torch.tensor((0.5071, 0.4867, 0.4408))
CIFAR100_STD = torch.tensor((0.2675, 0.2565, 0.2761))


def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)


def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size) // 2
    shifts = torch.randint(-r, r + 1, size=(len(images), 2), device=images.device)
    images_out = torch.empty(
        (len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype
    )
    # The two cropping methods in this if-else produce equivalent results, but the second is faster for r > 2.
    if r <= 2:
        for sy in range(-r, r + 1):
            for sx in range(-r, r + 1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[
                    mask, :, r + sy : r + sy + crop_size, r + sx : r + sx + crop_size
                ]
    else:
        images_tmp = torch.empty(
            (len(images), 3, crop_size, crop_size + 2 * r),
            device=images.device,
            dtype=images.dtype,
        )
        for s in range(-r, r + 1):
            mask = shifts[:, 0] == s
            images_tmp[mask] = images[mask, :, r + s : r + s + crop_size, :]
        for s in range(-r, r + 1):
            mask = shifts[:, 1] == s
            images_out[mask] = images_tmp[mask, :, :, r + s : r + s + crop_size]
    return images_out


def make_random_square_masks(inputs, size):
    is_even = int(size % 2 == 0)
    n, c, h, w = inputs.shape

    # seed top-left corners of squares to cutout boxes from, in one dimension each
    corner_y = torch.randint(0, h - size + 1, size=(n,), device=inputs.device)
    corner_x = torch.randint(0, w - size + 1, size=(n,), device=inputs.device)

    # measure distance, using the center as a reference point
    corner_y_dists = torch.arange(h, device=inputs.device).view(
        1, 1, h, 1
    ) - corner_y.view(-1, 1, 1, 1)
    corner_x_dists = torch.arange(w, device=inputs.device).view(
        1, 1, 1, w
    ) - corner_x.view(-1, 1, 1, 1)

    mask_y = (corner_y_dists >= 0) * (corner_y_dists < size)
    mask_x = (corner_x_dists >= 0) * (corner_x_dists < size)

    final_mask = mask_y * mask_x

    return final_mask


def batch_cutout(inputs, size):
    cutout_masks = make_random_square_masks(inputs, size)
    return inputs.masked_fill(cutout_masks, 0)


class CifarLoader:
    def __init__(
        self,
        path,
        train=True,
        batch_size=500,
        aug=None,
        drop_last=None,
        shuffle=None,
        altflip=False,
        dataset="CIFAR10",
    ):
        if dataset == "CIFAR10":
            path = os.path.join(path, "cifar10")
            CIFAR_MEAN = CIFAR10_MEAN
            CIFAR_STD = CIFAR10_STD
        else:
            path = os.path.join(path, "cifar100")
            CIFAR_MEAN = CIFAR100_MEAN
            CIFAR_STD = CIFAR100_STD
        os.makedirs(path, exist_ok=True)
        data_path = os.path.join(
            path, f"{dataset}_train.pt" if train else f"{dataset}_test.pt"
        )
        lock_path = data_path + ".lock"  # Define the lock file path

        with FileLock(
            lock_path
        ):  # Use file lock to ensure only one process accesses the file at a time
            if not os.path.exists(data_path):
                if dataset == "CIFAR10":
                    dset = torchvision.datasets.CIFAR10(
                        path, download=True, train=train
                    )
                elif dataset == "CIFAR100":
                    dset = torchvision.datasets.CIFAR100(
                        path, download=True, train=train
                    )
                images = torch.tensor(dset.data)
                labels = torch.tensor(dset.targets)
                temp_data_path = data_path + ".tmp"
                torch.save(
                    {"images": images, "labels": labels, "classes": dset.classes},
                    temp_data_path,
                )
                os.rename(
                    temp_data_path, data_path
                )  # Rename temp file to final file atomically
            data = torch.load(data_path, map_location="cuda")

        self.epoch = 0
        self.images, self.labels, self.classes = (
            data["images"],
            data["labels"],
            data["classes"],
        )
        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        self.images = (
            (self.images / 255)
            .permute(0, 3, 1, 2)
            .to(memory_format=torch.channels_last)
        )

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = (
            {}
        )  # Saved results of image processing to be done on the first epoch

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ["flip", "translate", "cutout"], "Unrecognized key: %s" % k

        self.batch_size = batch_size
        self.drop_last = train if drop_last is None else drop_last
        self.shuffle = train if shuffle is None else shuffle
        self.altflip = altflip

    def __len__(self):
        return (
            len(self.images) // self.batch_size
            if self.drop_last
            else ceil(len(self.images) / self.batch_size)
        )

    def __setattr__(self, k, v):
        if k in ("images", "labels"):
            assert (
                self.epoch == 0
            ), "Changing images or labels is only unsupported before iteration."
        super().__setattr__(k, v)

    def __iter__(self):
        if self.epoch == 0:
            images = self.proc_images["norm"] = self.normalize(self.images)
            # Pre-flip images in order to do every-other epoch flipping scheme
            if self.aug.get("flip", False):
                images = self.proc_images["flip"] = batch_flip_lr(images)
            # Pre-pad images to save time when doing random translation
            pad = self.aug.get("translate", 0)
            if pad > 0:
                self.proc_images["pad"] = F.pad(images, (pad,) * 4, "reflect")

        if self.aug.get("translate", 0) > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]
        # Flip all images together every other epoch. This increases diversity relative to random flipping
        if self.aug.get("flip", False):
            if self.altflip:
                if self.epoch % 2 == 1:
                    images = images.flip(-1)
            else:
                images = batch_flip_lr(images)
        if self.aug.get("cutout", 0) > 0:
            images = batch_cutout(images, self.aug["cutout"])

        self.epoch += 1

        indices = (torch.randperm if self.shuffle else torch.arange)(
            len(images), device=images.device
        )
        for i in range(len(self)):
            idxs = indices[i * self.batch_size : (i + 1) * self.batch_size]
            yield (images[idxs], self.labels[idxs])


class AirbenchLoaders:
    """class for instantiating AirbenchLoaders for train and test"""

    def __init__(self, cfg: DictConfig):
        console = Console()
        console.print(
            Panel(
                "[bold green]Using Airbench CIFAR Loader with small modifications from https://github.com/KellerJordan/cifar10-airbench[/bold green]",
                border_style="green",
                expand=False,
            )
        )
        ## get total device count
        
        self.train_loader = CifarLoader(
            path=cfg.dataset_params.data_root_dir,
            batch_size=cfg.dataset_params.total_batch_size,
            train=True,
            aug={"flip": True, "translate": 2},
            altflip=True,
            dataset=cfg.dataset_params.dataset_name,
        )
        self.test_loader = CifarLoader(
            path=cfg.dataset_params.data_root_dir,
            batch_size=cfg.dataset_params.total_batch_size,
            train=False,
            dataset=cfg.dataset_params.dataset_name,
        )

'''
class StandardPyTorchCIFARLoader:
    """Data loader for CIFAR-10 and CIFAR-100 datasets."""

    def __init__(self, cfg: DictConfig) -> None:
        super(StandardPyTorchCIFARLoader, self).__init__()

        self.dataset = cfg.dataset_params.dataset_name
        self.distributed = cfg.experiment_params.distributed

        if self.dataset == "CIFAR10":
            self.data_root = os.path.join(cfg.dataset_params.data_root_dir, "cifar10")
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
            self.data_root = os.path.join(cfg.dataset_params.data_root_dir, "cifar100")
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
            batch_size=cfg.dataset_params.batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=cfg.dataset_params.num_workers,
        )

        self.test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=cfg.dataset_params.batch_size,
            shuffle=False,
            num_workers=cfg.dataset_params.num_workers,
        )
'''

class FFCVImagenet:
    """Uses FFCV. Please ensure that it is installed!"""

    def __init__(self, cfg: DictConfig, this_device):
        self.this_device = this_device
        self.rank = dist.get_rank()
        self.total_device_count = dist.get_world_size()
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

            console = Console()
            if self.rank == 0:
                panel = Panel(
                    "[bold green]FFCV loaded, all good -- you can work on ImageNet![/bold green]",
                    title="FFCV Status",
                    border_style="green",
                    expand=False,
                )
                console.print(panel)
        except ImportError:
            raise ImportError(
                "FFCV is not installed. Please install FFCV to train on ImageNet. "
                "You can install it by following the instructions provided in the repository"
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
            os.path.join(cfg.dataset_params.data_root_dir, "train_500_0.50_90.beton"),
            batch_size=cfg.dataset_params.total_batch_size // self.total_device_count,
            num_workers=cfg.dataset_params.num_workers,
            order=OrderOption.RANDOM,
            os_cache=True,
            drop_last=True,
            pipelines={"image": train_image_pipeline, "label": label_pipeline},
            distributed=cfg.experiment_params.distributed,
            seed=cfg.experiment_params.seed
        )

        self.test_loader = Loader(
            os.path.join(cfg.dataset_params.data_root_dir, "val_500_0.50_90.beton"),
            batch_size=cfg.dataset_params.total_batch_size // self.total_device_count,
            num_workers=cfg.dataset_params.num_workers,
            order=OrderOption.SEQUENTIAL,
            drop_last=False,
            pipelines={"image": val_image_pipeline, "label": label_pipeline},
            distributed=cfg.experiment_params.distributed,
            seed=cfg.experiment_params.seed
        )

'''
class WebDatasetImageNet:
    """Data loader for ImageNet using WebDataset format."""

    def __init__(self, cfg: DictConfig) -> None:
        super(WebDatasetImageNet, self).__init__()

        self.cfg = cfg
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        train_loader = self._make_loader(
            urls=f"{cfg.dataset_params.data_root_dir}/imagenet1k-train-{{0000..1023}}.tar",
            mode="train",
            batch_size=cfg.dataset_params.batch_size,
            num_workers=4,
            resampled=True,
        )

        test_loader = self._make_loader(
            urls=f"{cfg.dataset_params.data_root_dir}/imagenet1k-validation-{{00..63}}.tar",
            mode="val",
            batch_size=cfg.dataset_params.batch_size,
            num_workers=cfg.dataset_params.gpu_workers,
            resampled=False,
        )

        # Set number of batches for train/val
        nbatches = max(
            1,
            1281167 // (cfg.dataset_params.batch_size * cfg.dataset_params.gpu_workers),
        )
        self.train_loader = train_loader.with_epoch(nbatches)
        self.test_loader = test_loader.slice(
            50000 // cfg.dataset_params.batch_size * cfg.dataset_params.gpu_workers
        )

    def _make_transform(self, mode="train"):
        """Create transform pipeline."""
        if mode == "train":
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    self.normalize,
                ]
            )
        elif mode == "val":
            return transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    self.normalize,
                ]
            )

    def _nodesplitter(self, src, group=None):
        """Split data across nodes for distributed training."""
        if torch.distributed.is_initialized():
            if group is None:
                group = torch.distributed.group.WORLD
            rank = torch.distributed.get_rank(group=group)
            size = torch.distributed.get_world_size(group=group)
            print(f"nodesplitter: rank={rank} size={size}")
            count = 0
            for i, item in enumerate(src):
                if i % size == rank:
                    yield item
                    count += 1
            print(f"nodesplitter: rank={rank} size={size} count={count} DONE")
        else:
            yield from src

    def _make_loader(
        self,
        urls,
        mode="train",
        batch_size=64,
        num_workers=4,
        cache_dir=None,
        resampled=True,
    ):
        """Create WebDataset data loader."""
        training = mode == "train"
        transform = self._make_transform(mode=mode)

        dataset = (
            wds.WebDataset(
                urls,
                repeat=training,
                cache_dir=cache_dir,
                shardshuffle=1000 if training else False,
                resampled=resampled if training else False,
                handler=wds.ignore_and_continue,
                nodesplitter=None if (training and resampled) else self._nodesplitter,
            )
            .shuffle(5000 if training else 0)
            .decode("pil")
            .to_tuple("jpg;png;jpeg cls", handler=wds.ignore_and_continue)
            .map_tuple(transform)
            .batched(batch_size, partial=False)
        )

        loader = wds.WebLoader(
            dataset, batch_size=None, shuffle=False, num_workers=num_workers
        )
        loader.num_samples = 1281167 if training else 50000
        loader.num_batches = loader.num_samples // (batch_size * num_workers)

        return loader
'''