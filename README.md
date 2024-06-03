## Finding Lottery Tickets in Deep Neural Networks (+ Distributed Training)

In this repository, we implement a training harness which enables finding lottery tickets in deep CNNs on ImageNet and CIFAR datasets.

### Key Features
- PyTorch Distributed Data Parallel (DDP) based training harness for training the network (post-pruning) as fast as possible.
- [FFCV]() integration for super-fast training on ImageNet (1:09 mins/epoch on 4xA100 GPUs with ResNet18).
- Support for most (if not all) torchvision models. (Transformers will be added later).
- Multiple pruning techniques, listed below.
- Simple harness, with fastargs -- easily extensible.
- Logging to CSV (nothing fancy, but you can integrate easily).
- End to End pipeline easily configurarable using [fastargs]().

### Datasets Supported
1. CIFAR10
2. CIFAR100
3. ImageNet
4. SVHN (to be added)

### Networks supported
As it stands, ResNets, VGG variants should work out of the box. If you run into issues with any other variant happy to look into.
