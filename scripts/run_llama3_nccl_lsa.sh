#!/usr/bin/bash
set -ex

NGPU=${NGPU:-"4"}
export LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"/pscratch/sd/a/aj689/project9/torchtitan/torchtitan/models/llama3/train_configs/llama3_8b_nccl_lsa.toml"}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.train"}

# CRITICAL: Force system to load NCCL 2.28.9 BEFORE anything else
NCCL_ROOT="/pscratch/sd/a/aj689/nccl"
export LD_PRELOAD="${NCCL_ROOT}/build/lib/libnccl.so.2.28.9"
export LD_LIBRARY_PATH="${NCCL_ROOT}/build/lib:${LD_LIBRARY_PATH}"

# Verify the library
echo "🔍 Verifying NCCL library:"
ldd ${NCCL_ROOT}/build/lib/libnccl.so.2.28.9 | head -5

export NCCL_DEBUG="INFO"
export NCCL_ALGO="AllReduce:Tree"
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL="PHB"
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_IFNAME="hsn"
export NCCL_MIN_NCHANNELS=4
export NCCL_MAX_NCHANNELS=16
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

echo "================================================"
echo "Llama 3 8B Training with NCCL-LSA 2.28.9 (FORCED)"
echo "================================================"
echo "NCCL Library: ${LD_PRELOAD}"
echo "NCCL_ALGO: ${NCCL_ALGO}"
echo "================================================"
echo ""

torchrun \
    --nproc_per_node=${NGPU} \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} \
    --role rank \
    --tee 3 \
    -m ${TRAIN_FILE} \
    --job.config_file ${CONFIG_FILE}

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Check logs above for 'NCCL version 2.28.9'"
else
    echo "❌ Failed with exit code: $EXIT_CODE"
fi
exit $EXIT_CODE
```

Save this script and run it. The key addition is **`LD_PRELOAD`** which forces the system to load your NCCL library before PyTorch's bundled version.

### What to Look For:

If it works, you should see:
```
NCCL version 2.28.9+cuda12.X  ← Must be 2.28.9
NCCL_ALGO set by environment to AllReduce:Tree
Connected all trees  ← Should say "trees" not "rings"
