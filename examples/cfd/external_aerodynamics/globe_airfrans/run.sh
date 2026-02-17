#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J coreai_modulus_cae-psharpe.train_globe_run_1
#SBATCH --time=4:00:00
#SBATCH -p batch
#SBATCH -N 10
#SBATCH --ntasks-per-node=1
#SBATCH --dependency=singleton
#SBATCH -o ./sbatch_logs/%x.log
#SBATCH -e ./sbatch_logs/%x.log
#SBATCH --open-mode=append

TRAIN_ARGS=(
    --output-name ${SLURM_JOB_NAME#coreai_modulus_cae-psharpe.train_globe_}
    --airfrans-task "full"
    --train-face-downsampling-ratio 0.5
)

### [Run Information]
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Job name: $SLURM_JOB_NAME"
echo "Number of nodes: $SLURM_NNODES"
echo "Node list: $SLURM_NODELIST"

### [Detect GPUs]
NUM_GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs detected: $NUM_GPUS_PER_NODE"

set -euxo pipefail

# uv sync --extra [cu12 or cu13] --extra mesh-extras
uv pip install -r requirements.txt

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