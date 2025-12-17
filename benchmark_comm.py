import subprocess
import time
import json
import os
from datetime import datetime

env = os.environ.copy()
env["PYTHONPATH"] = "/pscratch/sd/i/ishita/LLM-Benchmarking-NVSHMEM-NCCLGIN:" + env.get("PYTHONPATH", "")

# TOML_FILE = "./torchtitan/models/llama3/train_configs/llama3_8b_moe.toml"
TOML_FILE = "./torchtitan/models/llama3/train_configs/llama3_8bnew.toml"
TRAIN_CMD = f"torchrun --nproc_per_node=4 torchtitan/train.py --job.config_file {TOML_FILE}"
# BACKENDS = ["nccl", "nvshmem"]

BACKENDS = ["nvshmem"]

# def modify_toml(backend):
#     lines = []
#     with open(TOML_FILE, "r") as f:
#         for line in f:
#             if line.strip().startswith("comm_backend"):
#                 lines.append(f'comm_backend = "{backend}"\n')
#             else:
#                 lines.append(line)
#     with open(TOML_FILE, "w") as f:
#         f.writelines(lines)

def run_training(backend):
    print(f"\n================ BENCHMARK: {backend.upper()} ================\n")

    # modify_toml(backend)

    start = time.time()

    env["COMM_BACKEND"] = backend

    log_file = f"benchmark_{backend}.log"

    cmd = TRAIN_CMD
    print("Running:", cmd)

    with open(log_file, "w") as lf:
        subprocess.run(cmd.split(), stdout=lf, stderr=lf, env=env)
        # subprocess.run(cmd.split(), stdout=lf, stderr=lf)

    end = time.time()
    duration = end - start

    return {
        "backend": backend,
        "time_sec": duration,
        "log_file": log_file
    }

def extract_metrics(log_file):
    steps = []
    tokens = []
    with open(log_file, "r") as f:
        for line in f:
            if "tokens/sec" in line:
                tok = float(line.split("tokens/sec:")[-1].strip())
                tokens.append(tok)
            if "step" in line and "loss" in line:
                steps.append(line)

    avg_tokens = sum(tokens) / len(tokens) if tokens else 0

    return {
        "avg_tokens_per_sec": avg_tokens,
        "num_steps": len(steps)
    }

def main():
    results = []
    for backend in BACKENDS:
        run = run_training(backend)
        metrics = extract_metrics(run["log_file"])
        combined = {**run, **metrics}
        results.append(combined)

    out = "benchmark_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=4)

    print("\nBenchmark completed. Results saved to benchmark_results.json\n")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
