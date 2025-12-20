# import subprocess
# import time
# import json
# import os
# from datetime import datetime

# # ---------------------------------------------------------------------
# # 1. ENVIRONMENT SETUP
# # ---------------------------------------------------------------------
# env = os.environ.copy()

# # TorchTitan repo
# env["PYTHONPATH"] = (
#     "/pscratch/sd/a/as4455/torchtitan:"
#     + env.get("PYTHONPATH", "")
# )
# env["NSYS_NVSHMEM_TRACE"] = "1"
# # env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# env["NVSHMEM_NVTX"] = "common"
# # env["COMM_BACKEND"] = "nvshmem"
# # env["TORCH_SYMMETRIC_MEMORY"] = "nvshmem"
# env["NVSHMEM_DEBUG"] = "INFO"
# env["NVSHMEM_SYMMETRIC_SIZE"] = "1G"
# # env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# # ✅ FORCE NCCL (TorchTitan-safe)
# env["COMM_BACKEND"] = "nvshmem"

# # ✅ Keep NVTX + CUDA tracing only
# env["NSYS_NVSHMEM_TRACE"] = "0"
# env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
# env["MPICH_GPU_SUPPORT_ENABLED"] = "1"


# # Shared config
# TOML_FILE = "./torchtitan/models/llama3/train_configs/llama3_8b_newwww.toml"

# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# # nsys_out = f"nsys_{backend}_{timestamp}"

# # Nsight Systems command template
# # NSYS_CMD = (
# #     "nsys profile "
# #     "--force-overwrite=true "
# #     "--sample=none "
# #     "--trace=cuda,nvtx,mpi,osrt "
# #     "--cuda-memory-usage=true "
# #     "--stats=true "
# #     "-o {nsys_out} "
# #     "torchrun --nproc_per_node=4 "
# #     "torchtitan/train.py "
# #     "--job.config_file {toml}"
# # )

# # NSYS_CMD = (
# #     "srun -N 1 -n 4 --gpus-per-task=1 --cpus-per-task=8 "
# #     "nsys profile "
# #     "--force-overwrite=true "
# #     "--sample=none "
# #     "--trace=cuda,nvtx,mpi,osrt "
# #     "--cuda-memory-usage=true "
# #     "--stats=true "
# #     "-o {nsys_out} "
# #     "torchrun --nproc_per_node=1 --standalone "
# #     "torchtitan/train.py "
# #     "--job.config_file {toml}"
# # )

# # NSYS_CMD = (
# #     "torchrun --nproc_per_node=1 --standalone "
# #     "torchtitan/train.py "
# #     "--job.config_file {toml}"
# # )

# NSYS_CMD = (
#     "mpirun -np 4 "
#     "torchrun --nproc_per_node=4 "  # Remove --standalone!
#     "torchtitan/train.py "
#     "--job.config_file {toml}"
# )
# # Backends
# # BACKENDS = ["nvshmem", "nccl"]
# BACKENDS = ["nvshmem"]

# # ---------------------------------------------------------------------
# # 2. Run training for a backend
# # ---------------------------------------------------------------------
# def run_training(backend):
#     print(f"\n================ BENCHMARK: {backend.upper()} ================\n")

#     env["COMM_BACKEND"] = backend

#     nsys_out = f"nsys_{backend}"
#     cmd = NSYS_CMD.format(
#         nsys_out=nsys_out,
#         toml=TOML_FILE,
#     )

#     print("Running:\n", cmd)

#     log_file = f"logs/benchmark_{backend}_{timestamp}.log"

#     start = time.time()
#     with open(log_file, "w") as lf:
#         subprocess.run(
#             cmd,
#             shell=True,   # 🔴 REQUIRED
#             stdout=lf,
#             stderr=lf,
#             env=env,
#         )
#     end = time.time()

#     return {
#         "backend": backend,
#         "time_sec": round(end - start, 3),
#         "log_file": log_file,
#         "nsys_report": f"{nsys_out}.qdrep",
#     }


# # ---------------------------------------------------------------------
# # 3. Extract throughput metrics
# # ---------------------------------------------------------------------
# def extract_metrics(log_file):
#     tokens = []
#     steps = []

#     with open(log_file, "r") as f:
#         for line in f:
#             if "tokens/sec:" in line:
#                 try:
#                     tokens.append(float(line.split("tokens/sec:")[-1].strip()))
#                 except ValueError:
#                     pass
#             if "step" in line and "loss" in line:
#                 steps.append(line)

#     return {
#         "avg_tokens_per_sec": round(sum(tokens) / len(tokens), 3) if tokens else 0.0,
#         "num_steps": len(steps),
#     }


# # ---------------------------------------------------------------------
# # 4. Main
# # ---------------------------------------------------------------------
# def main():
#     results = []

#     for backend in BACKENDS:
#         run_info = run_training(backend)
#         metrics = extract_metrics(run_info["log_file"])
#         results.append({**run_info, **metrics})

#     with open("benchmark_results.json", "w") as f:
#         json.dump(results, f, indent=4)

#     print("\nBenchmark completed.")
#     print(json.dumps(results, indent=4))


# if __name__ == "__main__":
#     main()


"""
SLURM-based benchmark script for TorchTitan NCCL vs NVSHMEM.

This version uses SLURM's srun instead of mpirun, which is the proper
way to launch on HPC clusters like NERSC Perlmutter.
"""

import subprocess
import time
import json
import os
from datetime import datetime

# ---------------------------------------------------------------------
# ENVIRONMENT SETUP
# ---------------------------------------------------------------------
env = os.environ.copy()

# Paths
env["PYTHONPATH"] = (
    "/pscratch/sd/a/as4455/torchtitan:"
    + env.get("PYTHONPATH", "")
)

# Memory optimization
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

# NVSHMEM settings
env["NVSHMEM_DEBUG"] = "INFO"
env["NVSHMEM_SYMMETRIC_SIZE"] = "2G"
env["NVSHMEM_NVTX"] = "1"

# NCCL settings
env["NCCL_DEBUG"] = "WARN"

# SLURM GPU settings
env["SLURM_GPU_BIND"] = "closest"

# Config
TOML_FILE = "./torchtitan/models/llama3/train_configs/llama3_8b_newwww.toml"
NUM_GPUS = 4
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================================
# OPTION 1: SLURM srun (RECOMMENDED for clusters)
# ============================================================================
SRUN_CMD_TEMPLATE = (
    "srun "
    "--nodes=1 "
    "--ntasks={nproc} "
    "--gpus-per-task=1 "
    "--cpus-per-task=8 "
    "nsys profile "
    "--force-overwrite=true "
    "--sample=none "
    "--trace=cuda,nvtx "
    "--cuda-memory-usage=true "
    "--stats=true "
    "-o {nsys_out}_rank%q{{SLURM_PROCID}} "
    "python -u torchtitan/train.py "
    "--job.config_file {toml}"
)

# ============================================================================
# OPTION 2: Pure torchrun (NO NVSHMEM SUPPORT)
# ============================================================================
TORCHRUN_CMD_TEMPLATE = (
    "torchrun "
    "--nproc_per_node={nproc} "
    "--rdzv_backend=c10d "
    "--rdzv_endpoint=localhost:29500 "
    "torchtitan/train.py "
    "--job.config_file {toml}"
)

# ============================================================================
# OPTION 3: SLURM without profiling (faster)
# ============================================================================
SRUN_SIMPLE_TEMPLATE = (
    "srun "
    "--nodes=1 "
    "--ntasks={nproc} "
    "--gpus-per-task=1 "
    "--cpus-per-task=8 "
    "python -u torchtitan/train.py "
    "--job.config_file {toml}"
)

# Choose which to use
USE_PROFILING = False  # Set to False initially to debug faster
USE_SLURM = True       # Set to False if not on SLURM cluster

if USE_SLURM:
    CMD_TEMPLATE = SRUN_CMD_TEMPLATE if USE_PROFILING else SRUN_SIMPLE_TEMPLATE
else:
    CMD_TEMPLATE = TORCHRUN_CMD_TEMPLATE
    print("⚠ WARNING: Using torchrun without SLURM")
    print("  NVSHMEM will NOT work without MPI or SLURM!")
    print("  Only NCCL backend will be tested.\n")

# BACKENDS = ["nccl"]  # Start with just NCCL
BACKENDS = [ "nvshmem"]  # Add nvshmem once NCCL works

# ---------------------------------------------------------------------
# Run training
# ---------------------------------------------------------------------
def run_training(backend):
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {backend.upper()}")
    print(f"{'='*70}\n")
    
    # Set backend
    env["COMM_BACKEND"] = backend
    
    # Output paths
    nsys_out = f"nsys_{backend}_{timestamp}"
    log_file = f"logs/benchmark_{backend}_{timestamp}.log"
    
    os.makedirs("logs", exist_ok=True)
    
    # Build command
    cmd = CMD_TEMPLATE.format(
        nproc=NUM_GPUS,
        nsys_out=nsys_out,
        toml=TOML_FILE,
    )
    
    print(f"Command:\n{cmd}\n")
    print(f"Log: {log_file}\n")
    
    # Run
    start = time.time()
    with open(log_file, "w") as lf:
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
        )
    end = time.time()
    
    # Check for errors
    success = result.returncode == 0
    error_msg = None
    
    if not success:
        with open(log_file, "r") as lf:
            lines = lf.readlines()
            for i, line in enumerate(lines):
                if any(x in line for x in ["ERROR", "Error", "FAILED", "OutOfMemoryError"]):
                    error_msg = "".join(lines[max(0, i-2):min(len(lines), i+5)])
                    break
    
    return {
        "backend": backend,
        "time_sec": round(end - start, 3),
        "log_file": log_file,
        "nsys_reports": [f"{nsys_out}_rank{i}.nsys-rep" for i in range(NUM_GPUS)] if USE_PROFILING and USE_SLURM else [],
        "success": success,
        "error": error_msg,
    }


def extract_metrics(log_file):
    """Parse metrics from log."""
    
    tokens_per_sec = []
    steps = []
    losses = []
    errors = {}
    
    try:
        with open(log_file, "r") as f:
            for line in f:
                # Throughput
                if "tokens/sec:" in line:
                    try:
                        val = float(line.split("tokens/sec:")[-1].split()[0])
                        tokens_per_sec.append(val)
                    except (ValueError, IndexError):
                        pass
                
                # Steps and loss
                if "step" in line.lower() and "loss" in line.lower():
                    steps.append(line.strip())
                    try:
                        if "loss:" in line:
                            loss = float(line.split("loss:")[-1].split()[0])
                            losses.append(loss)
                    except (ValueError, IndexError):
                        pass
                
                # Errors
                if "OutOfMemoryError" in line or "CUDA out of memory" in line:
                    errors['oom'] = True
                if "NVSHMEM" in line and ("ERROR" in line or "failed" in line):
                    errors['nvshmem'] = True
                if "command not found" in line:
                    errors['command_not_found'] = True
    
    except FileNotFoundError:
        pass
    
    return {
        "avg_tokens_per_sec": round(sum(tokens_per_sec) / len(tokens_per_sec), 3) if tokens_per_sec else 0.0,
        "peak_tokens_per_sec": max(tokens_per_sec) if tokens_per_sec else 0.0,
        "num_steps_completed": len(steps),
        "final_loss": losses[-1] if losses else None,
        "errors": errors,
        "sample_steps": steps[:3] if steps else [],
    }


def main():
    print("\n" + "="*70)
    print("TORCHTITAN BENCHMARK (SLURM)")
    print("="*70)
    print(f"System: {'SLURM' if USE_SLURM else 'standalone'}")
    print(f"GPUs: {NUM_GPUS}")
    print(f"Config: {TOML_FILE}")
    print(f"Profiling: {USE_PROFILING}")
    print(f"Backends: {BACKENDS}")
    print("="*70 + "\n")
    
    results = []
    
    for backend in BACKENDS:
        run_info = run_training(backend)
        metrics = extract_metrics(run_info["log_file"])
        result = {**run_info, **metrics}
        results.append(result)
        
        # Print summary
        print(f"\n{'-'*70}")
        print(f"{backend.upper()} RESULTS")
        print(f"{'-'*70}")
        print(f"  Runtime:         {run_info['time_sec']}s")
        print(f"  Success:         {run_info['success']}")
        print(f"  Steps:           {metrics['num_steps_completed']}")
        print(f"  Avg tokens/sec:  {metrics['avg_tokens_per_sec']}")
        print(f"  Peak tokens/sec: {metrics['peak_tokens_per_sec']}")
        print(f"  Final loss:      {metrics['final_loss']}")
        print(f"  Errors:          {metrics['errors']}")
        
        if run_info['error']:
            print(f"\n  Error preview:")
            for line in run_info['error'].split('\n')[:5]:
                print(f"    {line}")
        
        print(f"{'-'*70}\n")
    
    # Save results
    output_file = f"benchmark_results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✓ Results saved to: {output_file}\n")


if __name__ == "__main__":
    main()
