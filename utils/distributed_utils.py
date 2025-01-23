import torch.distributed as dist
import torch
import hashlib
import os


def broadcast_object(obj):
    """Broadcast an object from rank 0 to all other processes."""
    object_list = [obj]
    dist.broadcast_object_list(object_list, src=0)
    return object_list[0]


def broadcast_model(model):
    """Broadcast model state dict from rank 0 to all other processes."""
    rank = dist.get_rank()

    if rank == 0:
        state_dict = model.state_dict()
    else:
        state_dict = None

    object_list = [state_dict]
    dist.broadcast_object_list(object_list, src=0)
    state_dict = object_list[0]

    model.load_state_dict(state_dict)
    dist.barrier()


def check_model_equality(model):
    """
    Check if the model is exactly the same across all GPUs.

    Args:
        model (nn.Module): The model to check.

    Returns:
        bool: True if all models are the same, False otherwise.
    """
    # Get world size and rank
    world_size = dist.get_world_size()

    # Flatten model parameters into a single tensor
    params = torch.cat([p.data.reshape(-1) for p in model.parameters()])

    # Calculate hash of the parameters
    param_hash = hashlib.sha256(params.cpu().numpy().tobytes()).hexdigest()

    # Gather hashes from all ranks
    all_hashes = [None for _ in range(world_size)]
    dist.all_gather_object(all_hashes, param_hash)

    # Check if all hashes are the same across all ranks
    are_equal = all(h == all_hashes[0] for h in all_hashes)

    if not are_equal and dist.get_rank() == 0:
        raise ValueError(f"Hashes do not match across ranks: {all_hashes}")

    return are_equal


def setup_distributed():
    """Setup distributed training environment."""
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
