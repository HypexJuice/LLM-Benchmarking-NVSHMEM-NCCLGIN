import os, torch

print("COMM_BACKEND =", os.getenv("COMM_BACKEND", "nccl"))
print("GIN_ENABLE   =", os.getenv("GIN_ENABLE"))
print("GIN_TYPE     =", os.getenv("GIN_TYPE"))

torch.distributed.init_process_group(
    backend="nccl",
    init_method="env://",
    world_size=1,
    rank=0,
)

x = torch.ones(1, device="cuda")
torch.distributed.all_reduce(x)
print("all_reduce ok, value:", x.item())
