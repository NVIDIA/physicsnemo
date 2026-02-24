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

set -euo pipefail

TRAIN_ARGS=(
    --output-name ${SLURM_JOB_NAME:-globe_airfrans_local}
    --airfrans-task "scarce"

)

### [Run Information]
echo "SLURM Job ID: ${SLURM_JOB_ID:-n/a}"
echo "SLURM Job name: ${SLURM_JOB_NAME:-n/a}"
echo "Number of nodes: ${SLURM_NNODES:-1}"
echo "Node list: ${SLURM_NODELIST:-$(hostname)}"

### [Detect GPUs and CUDA version]
NVIDIA_SMI_OUTPUT=$(nvidia-smi)
NUM_GPUS_PER_NODE=$(grep -cE '^\|[[:space:]]+[0-9]+[[:space:]]' <<< "$NVIDIA_SMI_OUTPUT")
CUDA_MAJOR=$(sed -n 's/.*CUDA Version: \([0-9]*\).*/\1/p' <<< "$NVIDIA_SMI_OUTPUT")
echo "Number of GPUs per node detected: $NUM_GPUS_PER_NODE"

### [Thread Configuration]
CPUS_PER_NODE=${SLURM_CPUS_ON_NODE:-$(nproc)}
export OMP_NUM_THREADS=$((CPUS_PER_NODE / NUM_GPUS_PER_NODE))
OMP_NUM_THREADS=$((OMP_NUM_THREADS > 0 ? OMP_NUM_THREADS : 1))
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS (${CPUS_PER_NODE} CPUs / ${NUM_GPUS_PER_NODE} GPUs)"

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
uv sync --inexact --extra "${CUDA_EXTRA}" --extra mesh-extras
uv pip install -r requirements.txt

### [Dataset Path]
# Auto-detect AirFRANS dataset location by hostname if not already set.
if [ -z "${AIRFRANS_DATA_DIR:-}" ]; then
    HOSTNAME=$(hostname)
    if [[ "$HOSTNAME" == "NV-pds" ]]; then  # Local workstation
        export AIRFRANS_DATA_DIR="${HOME}/gh/aerodynamics_datasets/airfrans/Dataset"
    elif [[ "$HOSTNAME" == *"eos.clusters.nvidia.com" ]]; then  # EOS cluster
        export AIRFRANS_DATA_DIR="${HOME}/coreai_modulus_cae/datasets/airfrans/Dataset"
    elif [[ "$HOSTNAME" == "nvl72"* ]]; then  # HSG cluster
        export AIRFRANS_DATA_DIR="${HOME}/coreai_modulus_cae/datasets/airfrans/Dataset"
    else
        echo "WARNING: AIRFRANS_DATA_DIR is not set and hostname '$HOSTNAME' is not recognized." >&2
        echo "Continuing anyway -- train.py will fail unless --data-dir is in TRAIN_ARGS." >&2
    fi
    if [ -n "${AIRFRANS_DATA_DIR:-}" ]; then
        echo "Auto-detected AIRFRANS_DATA_DIR=$AIRFRANS_DATA_DIR"
    fi
fi

### [MLflow Configuration]
export MLFLOW_TRACKING_URI="sqlite:///${SLURM_SUBMIT_DIR:-$(pwd)}/output/mlflow.db"

### [Launch Training]
# torchrun's --signals-to-handle makes it catch SIGUSR1 and forward it to
# each worker's process group.  uv forwards signals transparently.
# The SBATCH --signal=B:USR1@120 directive sends SIGUSR1 to this script
# 120 seconds before the time limit; the trap below relays it to the child.
TORCHRUN_SIGNALS="SIGTERM,SIGINT,SIGHUP,SIGQUIT,SIGUSR1"

if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
    echo "Running multi-node training..."
    head_node=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
    echo "Head node: $head_node"
    echo "Head node IP: $head_node_ip"
    srun uv run --no-sync torchrun \
      --signals-to-handle="$TORCHRUN_SIGNALS" \
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
      --signals-to-handle="$TORCHRUN_SIGNALS" \
      --nproc-per-node $NUM_GPUS_PER_NODE \
      train.py \
      "${TRAIN_ARGS[@]}" &
fi
TRAIN_PID=$!
trap 'kill -USR1 $TRAIN_PID 2>/dev/null' USR1
wait $TRAIN_PID || true
wait $TRAIN_PID 2>/dev/null
