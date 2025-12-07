# test.py — minimal test for PyTorch + NVSHMEM symmetric memory

import os
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

def main():
    # initialize process group (adjust backend / init_method as needed)
    dist.init_process_group(backend="nccl")  # or the backend your cluster expects
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[Rank {rank}/{world_size}] NVSHMEM available? {symm_mem.is_nvshmem_available()}")

    if not symm_mem.is_nvshmem_available():
        if rank == 0:
            print("NVSHMEM not available — aborting test.")
        return

    # set NVSHMEM as backend for symmetric memory
    symm_mem.set_backend("NVSHMEM")

    # create symmetric tensor
    # we'll make a small tensor, initialize it differently on each rank
    data = symm_mem.empty(4, dtype=torch.float32, device=f"cuda:{rank}")
    # fill data: e.g. rank + 1, so sum is non-trivial
    data.fill_(float(rank + 1))
    print(f"[Rank {rank}] before all-reduce: {data.tolist()}")

    # establish symmetric memory across all ranks
    symm_mem.rendezvous(data, dist.group.WORLD)

    # perform all-reduce (sum) on symmetric tensor
    torch.ops.symm_mem.one_shot_all_reduce(data, "sum", dist.group.WORLD)

    # synchronize / barrier
    dist.barrier()

    # now each rank should see the same summed result
    print(f"[Rank {rank}] after all-reduce: {data.tolist()}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
