import os
import torch
import torch.distributed as dist

# ---- Debug prints ----
print("COMM_BACKEND =", os.getenv("COMM_BACKEND", "nccl"))
print("GIN_ENABLE   =", os.getenv("GIN_ENABLE"))
print("GIN_TYPE     =", os.getenv("GIN_TYPE"))

# ---- Torchrun-provided env vars ----
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])

torch.cuda.set_device(local_rank)

# ---- Init process group (baseline NCCL path) ----
dist.init_process_group(backend="nccl")

# -------------------------------
# 1) Baseline NCCL collective
# -------------------------------
x = torch.ones(4, device="cuda") * rank

dist.barrier()
dist.all_reduce(x, op=dist.ReduceOp.SUM)
dist.barrier()

if rank == 0:
    print("[Baseline NCCL] all_reduce result:", x)

# -------------------------------
# 2) NCCL-GIN device-initiated path
# -------------------------------
# IMPORTANT: this must NOT call torch.distributed internally

from torchtitan.components.gin.ops import gin_all_gather_into_tensor

chunk = 4
gin_in  = torch.ones(chunk, device="cuda") * rank          # [chunk]
gin_out = torch.empty(world_size * chunk, device="cuda")   # [world_size * chunk]

dist.barrier()

gin_all_gather_into_tensor(
    gin_out,
    gin_in,
    world_size,
    rank,
)

torch.cuda.synchronize()
dist.barrier()

if rank == 0:
    print("[NCCL-GIN] all_gather result shape:", gin_out.shape)
    print("[NCCL-GIN] sample values:", gin_out[:8])

# ---- Cleanup ----
dist.destroy_process_group()
