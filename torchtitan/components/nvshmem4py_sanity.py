
def nvshmem4py_sanity_put(n=1024):
    import torch
    from torch.cuda import nvtx
    import torch.distributed as dist
    import nvshmem.core as nv
    from nvshmem.interop import torch as nv_torch

    rank = dist.get_rank()
    peer = (rank + 1) % dist.get_world_size()

    buf = nv_torch.empty((n,), dtype=torch.float32, device="cuda")
    src = torch.full((n,), float(rank), device="cuda")

    nvtx.range_push("nvshmem4py_put")
    nv.put(buf, src, pe=peer)
    nvtx.range_pop()

    nv.barrier(nv.Teams.TEAM_WORLD)
