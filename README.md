## TurboPrune: High-Speed Distributed Lottery Ticket Training

<p align="center">
  <img src="https://github.com/nelaturuharsha/TurboPrune/assets/22708963/51fb49f5-8ab5-4386-a4f2-57feec88c1dd" alt="circular_image_centered" width="400">
</p>

- PyTorch Distributed Data Parallel (DDP) based training harness for training the network (post-pruning) as fast as possible.
- [FFCV](https://github.com/libffcv/ffcv) integration for super-fast training on ImageNet (1:09 mins/epoch on 4xA100 GPUs with ResNet18).
- Support for most (if not all) torchvision models with limited testing of coverage with timm.
- Multiple pruning techniques, listed below.
- Simple harness, with hydra -- easily extensible.
- Logging to CSV and wandb (nothing fancy, but you can integrate wandb/comet/your own system easily).

An aim was also to make it easy to look through stuff, and I put in decent effort with logging via rich :D

### Timing Comparison

The numbers below were obtained on a cluster with similar computational configuration -- only variation was the dataloading method, AMP (enabled where specified) and the GPU model used was NVIDIA A100 (40GB).

The model used was ResNet50 and the effective batch size in each case was 512.

<p align="center">
  <img src="https://github.com/nelaturuharsha/TurboPrune/assets/22708963/f57bf823-e8b7-46dc-a1f6-a88d91b3e75e" alt="circular_image_centered" width="800">
</p>

### Datasets Supported
1. CIFAR10
2. CIFAR100
3. ImageNet

### Networks supported
As it stands, ResNets, VGG variants should work out of the box. If you run into issues with any other variant happy to look into. For CIFAR based datasets, there are modification to the basic architecture based on tuning and references such as this [repository](https://github.com/huyvnphan/PyTorch_CIFAR10/blob/master/cifar10_models/vgg.py).

There is additional support for Vision Transformers via timm, however as of this commit -- this is limited and has been tested only for DeIT.

### Pruning Algorithms included:
1. - **Name:** Iterative Magnitude Pruning (IMP)
   - **Type of Pruning:** Iterative
   - **Paper:** [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)
3. - **Name:** IMP with Weight Rewinding (IMP + WR)
   - **Type of Pruning:** Iterative
   - **Paper:** [Stabilizing the lottery ticket hypothesis](https://arxiv.org/abs/1903.01611)
4. - **Name:** IMP with Learning Rate Rewinding (IMP + LRR)
   - **Type of Pruning:** Iterative
   - **Paper:** [Comparing Rewinding and Fine-tuning in Neural Network Pruning](https://arxiv.org/abs/2003.02389)
5. - **Name:** SNIP
   - **Type of Pruning:** Pruning at Initialization (PaI), One-shot
   - **Paper:** [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://arxiv.org/abs/1810.02340)
6. - **Name:** SynFlow
   - **Type of Pruning:** Pruning at Initialization (PaI), One-shot
   - **Paper:** [Pruning neural networks without any data by iteratively conserving synaptic flow](https://arxiv.org/abs/2006.05467)
8. - **Name:** Random Balanced/ERK Pruning
   - **Type of Pruning:** Pruning at Initialization (PaI) One-shot + Iterative
   - **Paper:** [Why Random Pruning Is All We Need to Start Sparse](https://proceedings.mlr.press/v202/gadhikar23a/gadhikar23a.pdf)
9. - **Name:** Random Pruning
   - **Type of Pruning:** Iterative
   - **Paper:** [The Unreasonable Effectiveness of Random Pruning: Return of the Most Naive Baseline for Sparse Training](https://openreview.net/pdf?id=VBZJ_3tz-t)

### Repository structure:
1. **run_experiment.py** - This the main script for running pruning experiments, it uses the PruningHarness which is sub-classes BaseHarness and supports training all configurations currently possible in this repository. If you would like to modify the eventual running, I'd recommend using this.
2. **harness_definitions/base_harness.py**: Base training harness for running experiments, can be re-used for non-pruning experiments as well -- if you think its releveant and want the flexibility of modifying the forward pass and other componenets.
3. **utils/harness_params.py**: I realized hydra based config systems is more flexible, so now all experiment parameters are specified via hydra + easily extensible via dataclasses.
3. **utils/harness_utils.py**: This contains a lot of functions used for making the code run, logging metrics and other misc stuff. Let me know if you know how to cut it down :)
4. **utils/custom_models.py**: Model wrapper with additional functionalities that make your pruning experiments easier.
5. **utils/dataset.py**: definiton for CIFAR10/CIFAR100, ImageNet with FFCV but WebDatasets is a WIP.
6. **utils/schedulers.py**: learning rate schedulers, for when you need to use them.
7. **utils/pruning_utils.py**: Pruning functions + a simple function to apply the function.

Where necessary, pruning will use a single GPU/Dataset in the training precision chosen.

### Important Pre-requisites
- To run ImageNet experiments, you obviously need ImageNet downloaded -- in addition, since we use FFCV, you would need to generate .beton files as per the instructions [here](https://github.com/libffcv/ffcv-imagenet).
- CIFAR10, CIFAR100 and other stuff are handled using cifar10-airbench, but no change is required by the user. You do not need distributed training as its faster on a single GPU (lol) -- so there is no support for dist training with these datasets via airbench. But if you really want to you can modify the harness, train loop and use the Standard PT loaders.
- Have a look at the harness_params and the config structure to understand how to configure experiemnts. Its worth it.

## Usage

Now to the fun part:

### Running an Experiment

To start an experiment, ensure there is appropriate (sufficient) compute (or it might take a while -- its going to anyways) and in case of ImageNet the appropriate betons available.

```bash
pip install -r requirements.txt
python run_experiment.py --config-name=cifar10_er_erk dataset_params.data_root_dir=<PATH_TO_FOLDER>
```

For DDP (Only ImageNet)
```bash
torchrun --nproc_per_node=<num_gpus> run_experiment.py --config-name=imagenet_er_erk dataset_params.data_root_dir=<PATH_TO_FOLDER>
```

and it should start.

### Hydra Configuration

This is a bit detailed, coming soon - if you need any help -- open an issue or reach out.

## Baselines

The configs provided in conf/ are for some tuned baselines, but if you find a better configuration -- please feel free to make a pull request.
#### ImageNet Baseline
#### CIFAR10 Baseline
#### CIFAR100 Baseline

All baselines are coming soon!

If you use this code in your research, and find it useful in general -- please consider citing using:

```
@software{Nelaturu_TurboPrune_High-Speed_Distributed,
author = {Nelaturu, Sree Harsha and Gadhikar, Advait and Burkholz, Rebekka},
license = {Apache-2.0},
title = {{TurboPrune: High-Speed  Distributed Lottery Ticket Training}},
url = {https://github.com/nelaturuharsha/TurboPrune}}
```


----------------
#### Footnotes and Acknowledgments:
- This code is built using references to the substantial hard work put in by [Advait Gadhikar](https://advaitgadhikar.github.io/).
-  Thank you to [Dr. Rebekka Burkholz](https://cispa.de/de/people/c01rebu) for the opportunity to build this :)
-  I  was heavily influenced by the code style [here](https://github.com/libffcv/ffcv-imagenet). Just a general thanks and shout-out to the FFCV team for all they've done!
- All credit/references for the original methods and reference implementations are due to the original authors of the work :)
- Thank you Andrej, Bhavnick, Akanksha for feedback :)
