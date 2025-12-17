import subprocess
import time
import json
import os
from datetime import datetime

# ---------------------------------------------------------------------
# 1. ENVIRONMENT SETUP
# ---------------------------------------------------------------------
env = os.environ.copy()

# Ensure TorchTitan repo is discoverable
env["PYTHONPATH"] = (
    "/pscratch/sd/i/ishita/LLM-Benchmarking-NVSHMEM-NCCLGIN:"
    + env.get("PYTHONPATH", "")
)

# Config file used for BOTH NCCL and NVSHMEM
TOML_FILE = "./torchtitan/models/llama3/train_configs/llama3_8bnew.toml"

# Distributed training command
TRAIN_CMD = f"torchrun --nproc_per_node=4 torchtitan/train.py --job.config_file {TOML_FILE}"

# Backends to benchmark
BACKENDS = ["nvshmem", "nccl"]


# ---------------------------------------------------------------------
# 2. Run training for a backend
# ---------------------------------------------------------------------
def run_training(backend):
    print(f"\n================ BENCHMARK: {backend.upper()} ================\n")

    # Set backend for train.py to read
    env["COMM_BACKEND"] = backend

    start = time.time()

    log_file = f"benchmark_{backend}_2.log"

    print("Running:", TRAIN_CMD)

    with open(log_file, "w") as lf:
        subprocess.run(TRAIN_CMD.split(), stdout=lf, stderr=lf, env=env)

    end = time.time()

    return {
        "backend": backend,
        "time_sec": round(end - start, 3),
        "log_file": log_file,
    }


# ---------------------------------------------------------------------
# 3. Extract metrics from the training log
# ---------------------------------------------------------------------
def extract_metrics(log_file):
    tokens = []
    steps = []

    with open(log_file, "r") as f:
        for line in f:
            if "tokens/sec:" in line:
                try:
                    val = float(line.split("tokens/sec:")[-1].strip())
                    tokens.append(val)
                except:
                    pass
            if "step" in line and "loss" in line:
                steps.append(line)

    avg_tokens = sum(tokens) / len(tokens) if tokens else 0.0

    return {
        "avg_tokens_per_sec": round(avg_tokens, 3),
        "num_steps": len(steps),
    }


# ---------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------
def main():
    results = []

    for backend in BACKENDS:
        run_info = run_training(backend)
        metrics = extract_metrics(run_info["log_file"])
        results.append({**run_info, **metrics})

    # Save structured results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nBenchmark completed. Results saved to benchmark_results.json\n")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
