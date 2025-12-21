#!/usr/bin/env python3
"""
FAST Benchmark Script - Optimized for quick profiling
Profiles only 5 steps with minimal overhead
"""

import subprocess
import time
import json
import os
from datetime import datetime

env = os.environ.copy()

env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

env["NSYS_NVSHMEM_TRACE"] = "1"
# # env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# env["NVSHMEM_NVTX"] = "common"
# env["COMM_BACKEND"] = "nvshmem"
# env["TORCH_SYMMETRIC_MEMORY"] = "nvshmem"
# env["NVSHMEM_DEBUG"] = "INFO"nv["NVSHMEM_SYMMETRIC_SIZE"] = "1G"

# Config
TOML_FILE = "./torchtitan/models/llama3/train_configs/llama3_8bnew.toml"
NUM_GPUS = 4
PROFILE_DURATION = 30  # seconds - stops profiling automatically
NUM_STEPS = 5  # Only run 5 steps for quick comparison
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Environment
env["PYTHONPATH"] = "/pscratch/sd/a/as4455/torchtitan:" + env.get("PYTHONPATH", "")
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

# Fast profiling command - minimal overhead
# CMD_TEMPLATE = (
#     "torchrun "
#     "--nproc_per_node={nproc} "
#     "--rdzv_backend=c10d "
#     "--rdzv_endpoint=localhost:29500 "
#     "nsys profile "
#     "--force-overwrite=true "
#     "--trace=cuda,nvtx "  # Only CUDA and NVTX (no memory, no stats)
#     "--duration={duration} "  # Auto-stop after N seconds
#     "-o {nsys_out}_rank%q{{LOCAL_RANK}} "
#     "python -u torchtitan/train.py "
#     "--job.config_file {toml}"
# )

CMD_TEMPLATE = (
        "nsys profile "
        "--force-overwrite=true "
        "--sample=none "
        "--trace=cuda,nvtx "
        "--cuda-memory-usage=true "
        "--stats=true "
        "-o {nsys_out} "
        "torchrun --nproc_per_node=4 "
        "torchtitan/train.py "
        "--job.config_file {toml}"
    )


def run_benchmark(backend):
    """Run training for one backend."""
    print(f"\n{'='*60}")
    print(f"Testing {backend.upper()} (Fast - {NUM_STEPS} steps)")
    print(f"{'='*60}\n")
    
    # Set backend
    env["COMM_BACKEND"] = backend
    
    # Output paths
    nsys_out = f"nsys_{backend}_{timestamp}"
    log_file = f"logs/bench_{backend}_{timestamp}.log"
    
    os.makedirs("logs", exist_ok=True)
    
    # Build command
    cmd = CMD_TEMPLATE.format(
        nproc=NUM_GPUS,
        duration=PROFILE_DURATION,
        nsys_out=nsys_out,
        toml=TOML_FILE
    )
    
    print(f"Command: {cmd}")
    print(f"Log: {log_file}")
    print(f"Profile duration: {PROFILE_DURATION}s (auto-stops)\n")
    
    # Run
    start = time.time()
    with open(log_file, "w") as f:
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )
    elapsed = time.time() - start
    
    # Parse results
    tokens_per_sec = []
    steps = 0
    success = result.returncode == 0
    
    try:
        with open(log_file, "r") as f:
            for line in f:
                if "tps:" in line:
                    import re
                    match = re.search(r'tps:\s*([0-9.]+)', line)
                    if match:
                        tokens_per_sec.append(float(match.group(1)))
                if "step" in line.lower() and "loss" in line.lower():
                    steps += 1
    except FileNotFoundError:
        pass
    
    avg_throughput = sum(tokens_per_sec) / len(tokens_per_sec) if tokens_per_sec else 0
    
    # Print results
    print(f"Results:")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Success: {success}")
    print(f"  Steps completed: {steps}")
    print(f"  Avg throughput: {avg_throughput:.1f} tokens/sec")
    
    # List profile files
    profile_files = []
    for rank in range(NUM_GPUS):
        profile_file = f"{nsys_out}_rank{rank}.nsys-rep"
        if os.path.exists(profile_file):
            size_mb = os.path.getsize(profile_file) / 1024 / 1024
            print(f"  Profile: {profile_file} ({size_mb:.1f} MB)")
            profile_files.append(profile_file)
    
    # Cleanup sqlite files
    print(f"\nCleaning up temporary .sqlite files...")
    for rank in range(NUM_GPUS):
        sqlite_file = f"{nsys_out}_rank{rank}.sqlite"
        if os.path.exists(sqlite_file):
            os.remove(sqlite_file)
            print(f"  Removed: {sqlite_file}")
    
    return {
        "backend": backend,
        "success": success,
        "time_sec": round(elapsed, 2),
        "steps": steps,
        "avg_tokens_per_sec": round(avg_throughput, 1),
        "log_file": log_file,
        "profile_files": profile_files,
    }


def main():
    print("\n" + "="*60)
    print("FAST NCCL vs NVSHMEM BENCHMARK")
    print("="*60)
    print(f"GPUs: {NUM_GPUS}")
    print(f"Config: {TOML_FILE}")
    print(f"Profile duration: {PROFILE_DURATION}s per backend")
    print(f"Expected total time: ~{PROFILE_DURATION * 2 / 60:.1f} minutes")
    print("="*60)
    
    results = []
    
    # Test NCCL first (baseline)
    print("\n>>> Running NCCL baseline...")
    results.append(run_benchmark("nccl"))
    
    # Test NVSHMEM
    # print("\n>>> Running NVSHMEM...")
    # results.append(run_benchmark("nvshmem"))
    
    # Save results
    output = f"results_{timestamp}.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    
    # Final comparison
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    nccl = next(r for r in results if r["backend"] == "nccl")
    nvshmem = next(r for r in results if r["backend"] == "nvshmem")
    
    print(f"NCCL:    {nccl['avg_tokens_per_sec']:.1f} tokens/sec ({nccl['time_sec']}s)")
    print(f"NVSHMEM: {nvshmem['avg_tokens_per_sec']:.1f} tokens/sec ({nvshmem['time_sec']}s)")
    
    if nccl['success'] and nvshmem['success'] and nccl['avg_tokens_per_sec'] > 0:
        speedup = nvshmem['avg_tokens_per_sec'] / nccl['avg_tokens_per_sec']
        print(f"\nSpeedup: {speedup:.2f}x")
    elif not nccl['success']:
        print("\n⚠ NCCL failed - check logs")
    elif not nvshmem['success']:
        print("\n⚠ NVSHMEM failed - check logs")
    
    print(f"\nResults saved to: {output}")
    
    print("\n" + "="*60)
    print("PROFILE FILES")
    print("="*60)
    print("Open rank 0 profiles in Nsight Systems:")
    
    nccl_profile = next((f for f in nccl['profile_files'] if 'rank0' in f), None)
    nvshmem_profile = next((f for f in nvshmem['profile_files'] if 'rank0' in f), None)
    
    if nccl_profile:
        print(f"  NCCL:    {nccl_profile}")
    if nvshmem_profile:
        print(f"  NVSHMEM: {nvshmem_profile}")
    
    if nccl_profile and nvshmem_profile:
        print(f"\nCompare command:")
        print(f"  nsys-ui {nccl_profile} {nvshmem_profile}")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Open profiles in Nsight Systems GUI")
    print("2. Look for NVTX markers: train_step_*, forward_pass, backward_pass, etc.")
    print("3. Compare timeline gaps (less = better)")
    print("4. Compare optimizer_step duration")
    
    print(f"\nOr run log comparison:")
    print(f"  python compare_logs.py --nccl {nccl['log_file']} --nvshmem {nvshmem['log_file']}")
    print()


if __name__ == "__main__":
    main()