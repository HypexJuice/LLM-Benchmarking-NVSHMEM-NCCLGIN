# from mpi4py import MPI

# comm = MPI.COMM_WORLD
# print("Rank", comm.Get_rank(), "of", comm.Get_size())

from mpi4py import MPI
import nvshmem.core as nv
import torch

from torch.cuda import nvtx

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world = comm.Get_size()

# Init NVSHMEM (MPI-based)
nv.init(mpi_comm=comm)

# Allocate symmetric memory (GPU)
N = 1024 * 1024
buf = nv.array((N,), dtype="float32")

# Local GPU buffer
src = torch.ones(N, device="cuda", dtype=torch.float32) * rank

nvtx.range_push("nvshmem_put")
peer = (rank + 1) % world
nv.put(buf, src, pe=peer)
nvtx.range_pop()

nv.barrier(nv.Teams.TEAM_WORLD)

if rank == 0:
    print("NVSHMEM put completed")

nv.free_array(buf)
nv.finalize()
