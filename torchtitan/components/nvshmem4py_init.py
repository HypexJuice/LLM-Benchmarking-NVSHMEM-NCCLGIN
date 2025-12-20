# # import os
# # import torch
# # import torch.distributed as dist

# # import nvshmem.core as nv
# # from cuda.core.experimental import Device

# # def nvshmem4py_init_from_torchrun() -> bool:
# #     """
# #     Initialize NVSHMEM4Py using torch.distributed to broadcast a UniqueID.
# #     This follows NVIDIA's torchrun UID init pattern. :contentReference[oaicite:3]{index=3}
# #     """
# #     if not dist.is_initialized():
# #         return False

# #     local_rank = int(os.environ["LOCAL_RANK"])
# #     torch.cuda.set_device(local_rank)
# #     device = torch.device("cuda", local_rank)

# #     # NVSHMEM4Py requires a cuda.core Device at init time. :contentReference[oaicite:4]{index=4}
# #     dev = Device(local_rank)
# #     dev.set_current()

# #     # Broadcast UniqueID via torch.distributed.broadcast_object_list. :contentReference[oaicite:5]{index=5}
# #     rank = dist.get_rank()
# #     world = dist.get_world_size()

# #     uniqueid = nv.get_unique_id(empty=True)
# #     if rank == 0:
# #         uniqueid = nv.get_unique_id()
# #         objs = [uniqueid]
# #     else:
# #         objs = [None]

# #     dist.broadcast_object_list(objs, src=0)
# #     dist.barrier()

# #     nv.init(device=dev, uid=objs[0], rank=rank, nranks=world, initializer_method="uid")
# #     return True

# # def nvshmem4py_finalize() -> None:
# #     try:
# #         nv.finalize()
# #     except Exception:
# #         pass

# """
# Safe NVSHMEM4Py initialization for multi-process PyTorch training.

# This module handles the tricky initialization order required when using
# NVSHMEM4Py with PyTorch Distributed (torchrun/elastic).
# """

# import os
# import torch
# import torch.distributed as dist


# # def nvshmem4py_init_from_env():
# #     """
# #     Initialize NVSHMEM4Py safely in a PyTorch distributed environment.
    
# #     CRITICAL: This must be called AFTER torch.distributed.init_process_group()
# #     but BEFORE any actual NVSHMEM operations.
# #     """
# #     if not dist.is_initialized():
# #         raise RuntimeError(
# #             "PyTorch distributed must be initialized before NVSHMEM4Py. "
# #             "Call this after dist.init_process_group()"
# #         )
    
# #     rank = dist.get_rank()
# #     world_size = dist.get_world_size()
# #     local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
# #     # Set CUDA device BEFORE importing nvshmem
# #     torch.cuda.set_device(local_rank)
    
# #     # Ensure all processes reach this point before any NVSHMEM imports
# #     dist.barrier()
    
# #     try:
# #         # Import MPI4Py first (if using MPI bootstrap)
# #         from mpi4py import MPI
        
# #         # Get MPI communicator
# #         mpi_comm = MPI.COMM_WORLD
# #         mpi_rank = mpi_comm.Get_rank()
# #         mpi_size = mpi_comm.Get_size()
        
# #         # Sanity check: MPI and PyTorch ranks should match
# #         if mpi_rank != rank or mpi_size != world_size:
# #             raise RuntimeError(
# #                 f"MPI rank/size ({mpi_rank}/{mpi_size}) doesn't match "
# #                 f"PyTorch rank/size ({rank}/{world_size})"
# #             )
        
# #         # NOW it's safe to import and initialize NVSHMEM
# #         import nvshmem.core as nvshmem
        
# #         # Initialize with MPI communicator
# #         nvshmem.init(mpi_comm=mpi_comm)
        
# #         # Verify initialization
# #         nv_rank = nvshmem.my_pe()
# #         nv_size = nvshmem.n_pes()
        
# #         if nv_rank != rank or nv_size != world_size:
# #             raise RuntimeError(
# #                 f"NVSHMEM rank/size ({nv_rank}/{nv_size}) doesn't match "
# #                 f"expected ({rank}/{world_size})"
# #             )
        
# #         if rank == 0:
# #             print(f"✓ NVSHMEM4Py initialized successfully with {world_size} PEs")
        
# #         return True
        
# #     except ImportError as e:
# #         if rank == 0:
# #             print(f"WARNING: Could not import MPI4Py or NVSHMEM4Py: {e}")
# #             print("Falling back to NCCL")
# #         return False
    
# #     except Exception as e:
# #         if rank == 0:
# #             print(f"ERROR: NVSHMEM4Py initialization failed: {e}")
# #             print("Falling back to NCCL")
# #         # Make sure all ranks fail together
# #         dist.barrier()
# #         return False

# def nvshmem4py_init_from_env():
#     """Initialize NVSHMEM4Py with automatic mode detection."""
    
#     if not dist.is_initialized():
#         raise RuntimeError("PyTorch distributed must be initialized first")
    
#     rank = dist.get_rank()
#     world_size = dist.get_world_size()
#     local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
#     torch.cuda.set_device(local_rank)
#     dist.barrier()
    
#     # Try MPI-based init first
#     try:
#         from mpi4py import MPI
#         mpi_comm = MPI.COMM_WORLD
        
#         # Verify MPI is actually initialized
#         if mpi_comm.Get_size() == world_size:
#             import nvshmem.core as nvshmem
#             nvshmem.init(mpi_comm=mpi_comm)
            
#             if rank == 0:
#                 print(f"✓ NVSHMEM initialized with MPI ({world_size} PEs)")
#             return True
#     except (ImportError, Exception) as e:
#         if rank == 0:
#             print(f"MPI-based init failed: {e}")
    
#     # Fallback: Try bootstrap-based init (for standalone torchrun)
#     try:
#         # Set bootstrap environment variables for standalone mode
#         os.environ['NVSHMEM_BOOTSTRAP'] = 'PMI'
#         os.environ['NVSHMEM_BOOTSTRAP_PMI'] = 'PMI-2'
        
#         # Map PyTorch environment to NVSHMEM
#         os.environ['NVSHMEM_BOOTSTRAP_TWO_STAGE'] = '1'
        
#         import nvshmem.core as nvshmem
#         nvshmem.init()  # No MPI comm needed with bootstrap
        
#         if rank == 0:
#             print(f"✓ NVSHMEM initialized with bootstrap ({world_size} PEs)")
#         return True
        
#     except Exception as e:
#         if rank == 0:
#             print(f"ERROR: All NVSHMEM init methods failed: {e}")
#         return False

# def nvshmem4py_finalize():
#     """
#     Safely finalize NVSHMEM4Py.
    
#     Call this before destroying the PyTorch process group.
#     """
#     try:
#         import nvshmem.core as nvshmem
#         nvshmem.finalize()
        
#         if dist.is_initialized() and dist.get_rank() == 0:
#             print("✓ NVSHMEM4Py finalized")
            
#     except Exception as e:
#         if dist.is_initialized() and dist.get_rank() == 0:
#             print(f"WARNING: NVSHMEM4Py finalize failed: {e}")


# # Alternative: Environment variable based initialization (no MPI)
# def nvshmem4py_init_bootstrap():
#     """
#     Initialize NVSHMEM using bootstrap instead of MPI.
    
#     This requires setting these environment variables BEFORE launching:
#     - NVSHMEM_BOOTSTRAP=PMI or NVSHMEM_BOOTSTRAP=PLUGIN
#     - Other bootstrap-specific variables
    
#     This is more complex and less reliable than MPI-based init.
#     """
#     if not dist.is_initialized():
#         raise RuntimeError("PyTorch distributed must be initialized first")
    
#     rank = dist.get_rank()
#     local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
#     # Set device
#     torch.cuda.set_device(local_rank)
#     dist.barrier()
    
#     try:
#         import nvshmem.core as nvshmem
        
#         # Initialize without MPI (requires bootstrap to be set up)
#         nvshmem.init()
        
#         if rank == 0:
#             print(f"✓ NVSHMEM4Py initialized via bootstrap")
        
#         return True
        
#     except Exception as e:
#         if rank == 0:
#             print(f"ERROR: NVSHMEM4Py bootstrap init failed: {e}")
#         dist.barrier()
#         return False

"""
NVSHMEM4Py initialization with multiple fallback modes.

Supports:
1. MPI-based initialization (best, for mpirun launches)
2. Bootstrap-based initialization (for standalone torchrun)
3. Graceful fallback to NCCL
"""

import os
import torch
import torch.distributed as dist


def nvshmem4py_init_from_env():
    """
    Initialize NVSHMEM4Py with automatic mode detection.
    
    Tries multiple initialization methods in order:
    1. MPI-based (requires mpirun)
    2. Bootstrap-based (for standalone torchrun)
    3. Falls back to NCCL if all fail
    
    Returns:
        bool: True if NVSHMEM initialized successfully, False otherwise
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "PyTorch distributed must be initialized before NVSHMEM4Py. "
            "Call this after dist.init_process_group()"
        )
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Set CUDA device BEFORE any NVSHMEM operations
    torch.cuda.set_device(local_rank)
    
    # Synchronize all processes
    dist.barrier()
    
    # =========================================================================
    # METHOD 1: MPI-based initialization (PREFERRED)
    # =========================================================================
    try:
        from mpi4py import MPI
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        mpi_size = mpi_comm.Get_size()
        
        # Verify MPI is properly initialized and matches PyTorch
        if mpi_rank == rank and mpi_size == world_size:
            # Import NVSHMEM core
            import nvshmem.core as nvshmem
            
            # Initialize with MPI communicator
            nvshmem.init(mpi_comm=mpi_comm)
            
            # Verify
            nv_rank = nvshmem.my_pe()
            nv_size = nvshmem.n_pes()
            
            if nv_rank != rank or nv_size != world_size:
                raise RuntimeError(
                    f"NVSHMEM PE mismatch: got {nv_rank}/{nv_size}, "
                    f"expected {rank}/{world_size}"
                )
            
            if rank == 0:
                print(f"✓ NVSHMEM initialized with MPI ({world_size} PEs)")
            
            dist.barrier()
            return True
        else:
            if rank == 0:
                print(f"MPI size mismatch: MPI({mpi_rank}/{mpi_size}) != "
                      f"PyTorch({rank}/{world_size})")
    
    except ImportError:
        if rank == 0:
            print("MPI4Py not available, trying bootstrap method...")
    except Exception as e:
        if rank == 0:
            print(f"MPI-based init failed: {e}")
    
    # =========================================================================
    # METHOD 2: Bootstrap-based initialization (for standalone torchrun)
    # =========================================================================
    try:
        if rank == 0:
            print("Attempting NVSHMEM bootstrap initialization...")
        
        # Set up bootstrap environment for NVSHMEM
        # This requires NVSHMEM to be built with PMI support
        os.environ['NVSHMEM_BOOTSTRAP'] = 'PMI'
        os.environ['NVSHMEM_BOOTSTRAP_PMI'] = 'PMI-2'
        
        # Map PyTorch distributed info to NVSHMEM bootstrap
        os.environ['NVSHMEM_BOOTSTRAP_TWO_STAGE'] = '1'
        
        # Try to set up PMI environment from PyTorch's info
        # This is a bit hacky but can work with some NVSHMEM builds
        if 'MASTER_ADDR' in os.environ and 'MASTER_PORT' in os.environ:
            os.environ['PMI_RANK'] = str(rank)
            os.environ['PMI_SIZE'] = str(world_size)
            os.environ['PMI_LOCAL_RANK'] = str(local_rank)
        
        dist.barrier()
        
        # Import and initialize NVSHMEM
        import nvshmem.core as nvshmem
        nvshmem.init()  # No MPI comm needed with bootstrap
        
        # Verify
        nv_rank = nvshmem.my_pe()
        nv_size = nvshmem.n_pes()
        
        if nv_rank != rank or nv_size != world_size:
            raise RuntimeError(
                f"NVSHMEM PE mismatch: got {nv_rank}/{nv_size}, "
                f"expected {rank}/{world_size}"
            )
        
        if rank == 0:
            print(f"✓ NVSHMEM initialized with bootstrap ({world_size} PEs)")
        
        dist.barrier()
        return True
    
    except ImportError as e:
        if rank == 0:
            print(f"Cannot import NVSHMEM: {e}")
    except Exception as e:
        if rank == 0:
            print(f"Bootstrap init failed: {e}")
            print(f"  This usually means NVSHMEM needs to be launched with MPI:")
            print(f"  mpirun -np {world_size} torchrun --nproc_per_node={world_size} ...")
    
    # =========================================================================
    # FALLBACK: All methods failed
    # =========================================================================
    if rank == 0:
        print("="*60)
        print("ERROR: NVSHMEM4Py initialization failed")
        print("="*60)
        print("All initialization methods failed. This can happen because:")
        print("1. NVSHMEM4Py is not properly installed")
        print("2. Using 'torchrun --standalone' without MPI")
        print("3. NVSHMEM was not built with PMI/bootstrap support")
        print()
        print("Solutions:")
        print("1. Launch with MPI: mpirun -np N torchrun --nproc_per_node=N ...")
        print("2. Install MPI4Py: pip install mpi4py")
        print("3. Rebuild NVSHMEM with bootstrap support")
        print()
        print("Falling back to NCCL for this run.")
        print("="*60)
    
    dist.barrier()
    return False


def nvshmem4py_finalize():
    """
    Safely finalize NVSHMEM4Py.
    
    Call this before destroying the PyTorch process group.
    """
    if not dist.is_initialized():
        return
    
    rank = dist.get_rank()
    
    try:
        import nvshmem.core as nvshmem
        nvshmem.finalize()
        
        if rank == 0:
            print("✓ NVSHMEM4Py finalized")
    
    except ImportError:
        pass  # NVSHMEM wasn't initialized
    except Exception as e:
        if rank == 0:
            print(f"WARNING: NVSHMEM4Py finalize failed: {e}")


def check_nvshmem_available():
    """
    Check if NVSHMEM4Py is available without initializing it.
    
    Returns:
        bool: True if NVSHMEM4Py can be imported
    """
    try:
        import nvshmem.core
        return True
    except ImportError:
        return False