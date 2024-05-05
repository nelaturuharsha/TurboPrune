import numpy as np
import torch

from fastargs import get_current_config
from fastargs.decorators import param

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage, Convert
from ffcv.fields.decoders import RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder


class FFCVImageNet:
    @param('dataset.batch_size')
    @param('dataset.num_workers')
    def __init__(self, batch_size, num_workers, distributed=True):
        super(FFCVImageNet, self).__init__()
        data_root = '/home/harsha/v0.1/'

        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
        IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
        DEFAULT_CROP_RATIO = 224/256
        train_image_pipeline = [RandomResizedCropRGBImageDecoder((224, 224)),
                            RandomHorizontalFlip(),
                            ToTensor(),
                            ToDevice(torch.device('cuda'), non_blocking=True),
                            ToTorchImage(),
                            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)]

        val_image_pipeline = [CenterCropRGBImageDecoder((256, 256), ratio=DEFAULT_CROP_RATIO),
                              ToTensor(),
                              ToDevice(torch.device('cuda'), non_blocking=True),
                              ToTorchImage(),
                              NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)]

        label_pipeline = [IntDecoder(),
                            ToTensor(),
                            Squeeze(),
                            ToDevice(torch.device('cuda'), non_blocking=True)]



        self.train_loader = Loader(data_root + 'train_500_0.50_90.ffcv', 
                              batch_size  = batch_size,
                              num_workers = num_workers,
                              order       = OrderOption.RANDOM,
                              os_cache    = True,
                              drop_last   = True,
                              pipelines   = { 'image' : train_image_pipeline,
                                              'label' : label_pipeline},
                              distributed = distributed,
                              seed = 1
                              )

        self.val_loader = Loader(data_root + 'val_500_0.50_90.ffcv',
                            batch_size  = batch_size,
                            num_workers = num_workers,
                            order       = OrderOption.SEQUENTIAL,
                            drop_last   = False,
                            pipelines   = { 'image' : val_image_pipeline,
                                            'label' : label_pipeline},
                            distributed = distributed,
                            seed = 1
                            )