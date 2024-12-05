import os
from rich.console import Console
import torch
import torch.distributed as dist
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import sys

from utils.pruning_utils import *
from utils.harness_utils import *
from utils.distributed_utils import broadcast_object, setup_distributed
from harness import PruningHarness

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# torch._dynamo.config.guard_nn_modules = True


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main function for distributed training."""
    if cfg.experiment_params.dataset_name.lower().startswith("cifar"):
        if dist.get_world_size() > 1:
            if dist.get_rank() == 0:
                console = Console()
                console.print(
                    Panel(
                        f"[bold red]DISCLAIMER: Training with dataset '{cfg.experiment_params.dataset_name}' in a distributed setting is not supported (coz we dont need it).[/bold red] \n"
                        "[bold red]Please set CUDA_VISIBLE_DEVICES=0 or a different single device id to continue, this run will exit.[/bold red]",
                        border_style="red",
                        expand=False,
                    )
                )
            sys.exit(1)
    use_distributed = (
        cfg.experiment_params.distributed
        and torch.cuda.device_count() > 1
        and not cfg.experiment_params.dataset_name.lower().startswith("cifar")
    )

    set_seed(cfg)

    if use_distributed:
        setup_distributed()
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank, world_size = 0, 1

    console = Console()

    if rank == 0:
        console.print(OmegaConf.to_yaml(cfg))
        console.print(f"[bold green]Training on {world_size} GPUs[/bold green]")
        wandb.init(project=cfg.experiment_params.wandb_project_name)
        run_id = wandb.run.id
        wandb.run.tags += (cfg.experiment_params.dataset_name.lower(),)
        prefix, expt_dir = (
            resume_experiment(cfg)
            if cfg.experiment_params.resume_experiment
            else gen_expt_dir(cfg)
        )
        packaged = (prefix, expt_dir)
        save_config(expt_dir=expt_dir, cfg=OmegaConf.to_container(cfg, resolve=True))
    else:
        run_id, packaged = None, None

    if use_distributed:
        run_id = broadcast_object(run_id)
        packaged = broadcast_object(packaged)

    harness = PruningHarness(cfg=cfg, gpu_id=rank, expt_dir=packaged)
    model_in_question = harness.model.module if use_distributed else harness.model
    at_init = cfg.pruning_params.training_type == "at_init"
    densities = generate_densities(
        cfg=cfg, current_sparsity=model_in_question.get_overall_sparsity()
    )

    checkpoints_dir = os.path.join(packaged[1], "checkpoints")
    model_init_path = os.path.join(checkpoints_dir, "model_init.pt")

    for level, density in enumerate(densities):
        if rank == 0:
            if level == 0:
                if at_init:
                    console.print(
                        f"[bold cyan]Pruning Model at initialization[/bold cyan]"
                    )
                    prune_the_model(cfg=cfg, harness=harness, target_density=density)
                else:
                    save_model(model_in_question, model_init_path, distributed=False)
                    console.print(f"[bold cyan]Dense training homie![/bold cyan]")
            else:
                if not at_init:
                    console.print(
                        f"[bold cyan]Pruning Model at level: {level} to a target density of {density:.4f}[/bold cyan]"
                    )
                    model_level_path = os.path.join(
                        checkpoints_dir, f"model_level_{level-1}.pt"
                    )
                    model_in_question.load_model(model_level_path)
                    prune_the_model(cfg=cfg, harness=harness, target_density=density)
                    model_in_question.reset_weights(cfg=cfg, expt_dir=packaged[1])

        console.print(
            "[bold magenta]Model Sparsity check:[/bold magenta]",
            f"[cyan]{model_in_question.get_overall_sparsity():.2f}%[/cyan]",
        )

        harness = PruningHarness(
            cfg=cfg, model=model_in_question, expt_dir=packaged, gpu_id=rank
        )
        harness.train_one_level(
            num_cycles=cfg.cyclic_training.num_cycles,
            epochs_per_level=cfg.experiment_params.epochs_per_level,
            level=level,
        )

        if rank == 0:
            model_level_path = os.path.join(checkpoints_dir, f"model_level_{level}.pt")
            save_model(harness.model, model_level_path, distributed=use_distributed)
            console.print(
                f"[bold green]Training level {level} complete, moving on to {level+1}[/bold green]"
            )

    if rank == 0:
        wandb.finish()


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")

    main()
