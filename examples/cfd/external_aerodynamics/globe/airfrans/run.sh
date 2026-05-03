#!/bin/bash
#SBATCH -A accountname
#SBATCH -J accountname-%u.train_globe_airfrans_scarce
#SBATCH --time=4:00:00
#SBATCH -p batch
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --dependency=singleton
#SBATCH -o ./sbatch_logs/%x.log
#SBATCH -e ./sbatch_logs/%x.log
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120

### [Shell Setup]
set -euo pipefail
# Prevent torchrun worker processes from writing multi-GB core dumps on crash.
ulimit -c 0

### [User Configuration]
OUTPUT_NAME="${SLURM_JOB_NAME:-globe_airfrans_local}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
OUTPUT_DIR="${SCRIPT_DIR}/output/${OUTPUT_NAME}"

TRAIN_ARGS=(
    --output-name "${OUTPUT_NAME}"
    --airfrans-task "scarce"
    --amp
)

export AIRFRANS_DATA_DIR="${HOME}/datasets/airfrans/Dataset"  # Set this to your AirFRANS dataset
# This dataset can be downloaded from https://airfrans.readthedocs.io/en/latest/notes/dataset.html

export MLFLOW_TRACKING_URI="sqlite:///${SLURM_SUBMIT_DIR:-$(pwd)}/output/mlflow.db"

### [Run Information]
echo "SLURM Job ID: ${SLURM_JOB_ID:-n/a}"
echo "SLURM Job name: ${SLURM_JOB_NAME:-n/a}"
echo "Number of nodes: ${SLURM_NNODES:-1}"
echo "Node list: ${SLURM_NODELIST:-$(hostname)}"

### [Detect GPUs and CUDA Version]
# Parse nvidia-smi once to extract both GPU count and CUDA driver version.
NVIDIA_SMI_OUTPUT=$(nvidia-smi)
NUM_GPUS_PER_NODE=$(grep -cE '^\|[[:space:]]+[0-9]+[[:space:]]' <<< "$NVIDIA_SMI_OUTPUT")
CUDA_MAJOR=$(sed -n 's/.*CUDA Version: \([0-9]*\).*/\1/p' <<< "$NVIDIA_SMI_OUTPUT")
echo "Number of GPUs per node detected: $NUM_GPUS_PER_NODE"

### [Thread Configuration]
# OMP_NUM_THREADS=1: DataLoader workers use process-level parallelism
# (num_workers auto-computed from CPU affinity and LOCAL_WORLD_SIZE), so
# per-process threading is unnecessary and causes thread oversubscription.
CPUS_PER_NODE=${SLURM_CPUS_ON_NODE:-$(nproc)}
export OMP_NUM_THREADS=1
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS (process-level parallelism via DataLoader workers; ${CPUS_PER_NODE} CPUs / ${NUM_GPUS_PER_NODE} GPUs)"

### [Sync Dependencies]
# Select the right CUDA extra based on the detected driver version,
# then install both the project deps and example-specific requirements.
if [ -z "$CUDA_MAJOR" ]; then
    echo "ERROR: Could not detect CUDA version from nvidia-smi." >&2
    exit 1
elif [ "$CUDA_MAJOR" -ge 13 ]; then
    CUDA_EXTRA="cu13"
elif [ "$CUDA_MAJOR" -ge 12 ]; then
    CUDA_EXTRA="cu12"
else
    echo "ERROR: Unsupported CUDA major version ${CUDA_MAJOR} (need >= 12)." >&2
    exit 1
fi
echo "Detected CUDA major version ${CUDA_MAJOR} -> syncing with extra '${CUDA_EXTRA}'"
uv sync --inexact --extra "${CUDA_EXTRA}" --extra mesh-extras
uv pip install -r requirements.txt

### [Launch Training]
# Graceful shutdown mechanism:
#   1. SBATCH --signal=B:USR1@120 sends SIGUSR1 to this script 120s before
#      the wall-time limit.
#   2. The trap below catches USR1 and writes a SHUTDOWN sentinel file.
#   3. train.py polls for this file each epoch and checkpoints + exits
#      cleanly when it appears.
#
# The training process is backgrounded (&) so that this script remains the
# signal recipient. The double-wait pattern handles an edge case: the first
# `wait` can be interrupted by USR1, causing it to return immediately. After
# the trap fires, the second `wait` resumes waiting for the actual process exit.
rm -f "$OUTPUT_DIR/SHUTDOWN"

if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
    echo "Running multi-node training..."
    head_node=$(hostname -s)
    head_node_ip=$(hostname --ip-address)
    echo "Head node: $head_node"
    echo "Head node IP: $head_node_ip"
    # srun launches one torchrun per node; each torchrun spawns per-GPU workers.
    srun uv run --no-sync torchrun \
      --nnodes $SLURM_NNODES \
      --nproc-per-node $NUM_GPUS_PER_NODE \
      --rdzv_id $RANDOM \
      --rdzv_backend c10d \
      --rdzv_endpoint $head_node_ip:29500 \
      train.py \
      "${TRAIN_ARGS[@]}" &
else
    echo "Running single-node training..."
    uv run --no-sync torchrun \
      --nproc-per-node $NUM_GPUS_PER_NODE \
      train.py \
      "${TRAIN_ARGS[@]}" &
fi
TRAIN_PID=$!
trap 'touch "$OUTPUT_DIR/SHUTDOWN"' USR1
wait $TRAIN_PID || true
wait $TRAIN_PID 2>/dev/null
