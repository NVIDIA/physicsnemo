#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J coreai_modulus_cae-psharpe.train_globe_run_1_scarce
#SBATCH --time=4:00:00
#SBATCH -p batch
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --dependency=singleton
#SBATCH -o ./sbatch_logs/%x.log
#SBATCH -e ./sbatch_logs/%x.log
#SBATCH --open-mode=append

set -euo pipefail

TRAIN_ARGS=(
    --output-name ${SLURM_JOB_NAME#coreai_modulus_cae-psharpe.train_globe_}
    --airfrans-task "scarce"

)

### [Run Information]
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Job name: $SLURM_JOB_NAME"
echo "Number of nodes: $SLURM_NNODES"
echo "Node list: $SLURM_NODELIST"

### [Detect GPUs and CUDA version]
NVIDIA_SMI_OUTPUT=$(nvidia-smi)
NUM_GPUS_PER_NODE=$(grep -cE '^\|[[:space:]]+[0-9]+[[:space:]]' <<< "$NVIDIA_SMI_OUTPUT")
CUDA_MAJOR=$(sed -n 's/.*CUDA Version: \([0-9]*\).*/\1/p' <<< "$NVIDIA_SMI_OUTPUT")
echo "Number of GPUs per node detected: $NUM_GPUS_PER_NODE"


### [Sync dependencies]
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
uv sync --extra "${CUDA_EXTRA}" --extra mesh-extras
uv pip install -r requirements.txt

### [MLflow Configuration]
export MLFLOW_TRACKING_URI="sqlite:///${SLURM_SUBMIT_DIR:-$(pwd)}/output/mlflow.db"

### [Launch Training]
if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
    echo "Running multi-node training..."
    head_node=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
    echo "Head node: $head_node"
    echo "Head node IP: $head_node_ip"
    srun uv run --no-sync torchrun \
      --nnodes $SLURM_NNODES \
      --nproc-per-node $NUM_GPUS_PER_NODE \
      --rdzv_id $RANDOM \
      --rdzv_backend c10d \
      --rdzv_endpoint $head_node_ip:29500 \
      train.py \
      "${TRAIN_ARGS[@]}"
else
    echo "Running single-node training..."
    uv run --no-sync torchrun \
      --nproc-per-node $NUM_GPUS_PER_NODE \
      train.py \
      "${TRAIN_ARGS[@]}"
fi