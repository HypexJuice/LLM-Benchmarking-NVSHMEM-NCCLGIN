import os
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem



dist.init_process_group(backend="nccl")
rank = dist.get_rank()
device = torch.device("cuda", rank)

print(f"[rank {rank}] torch.__version__ = {torch.__version__}")
print(f"[rank {rank}] symm_mem.get_backend(): {symm_mem.get_backend(device)}")
print(f"[rank {rank}] symm_mem.is_nvshmem_available(): {symm_mem.is_nvshmem_available()}")

# try to import the ops
try:
    print(f"[rank {rank}] symm_mem ops: {hasattr(torch.ops, 'symm_mem')}")
except Exception as e:
    print("ops import error:", e)

dist.destroy_process_group()