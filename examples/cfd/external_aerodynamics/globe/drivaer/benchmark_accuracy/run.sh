#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J globe_drivaer_benchmark_accuracy
#SBATCH --time=2:00:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --dependency=singleton
#SBATCH -o ./sbatch_logs/%x.log
#SBATCH -e ./sbatch_logs/%x.log
#SBATCH --open-mode=append

set -euo pipefail
export PATH="/cm/local/apps/slurm/current/bin:${PATH}"

### [User Configuration]
# Point GLOBE_OUTPUT_DIR at the trained model directory (containing
# best_model.mdlus and hyperparameters.yaml). If unset, auto-detects
# the most recent output/ subdirectory.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DRIVAER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export DRIVAER_DATA_DIR="${HOME}/coreai_modulus_cae/datasets/drivaer_aws/drivaer_data_full"
# export GLOBE_OUTPUT_DIR="${DRIVAER_DIR}/output/globe_drivaer_7_4n_1sh_256x3"

BENCH_ARGS=(
    --results-dir "${SCRIPT_DIR}"
)

### [Run Information]
echo "SLURM Job ID: ${SLURM_JOB_ID:-n/a}"
echo "SLURM Job name: ${SLURM_JOB_NAME:-n/a}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"

### [Detect GPUs and CUDA version]
NVIDIA_SMI_OUTPUT=$(nvidia-smi)
NUM_GPUS_PER_NODE=$(grep -cE '^\|[[:space:]]+[0-9]+[[:space:]]' <<< "$NVIDIA_SMI_OUTPUT")
CUDA_MAJOR=$(sed -n 's/.*CUDA Version: \([0-9]*\).*/\1/p' <<< "$NVIDIA_SMI_OUTPUT")
echo "GPUs per node: $NUM_GPUS_PER_NODE"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

### [Thread Configuration]
export OMP_NUM_THREADS=1

### [Sync Dependencies]
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
uv pip install -r "${DRIVAER_DIR}/requirements.txt"

### [Create log directory]
mkdir -p "${SCRIPT_DIR}/sbatch_logs"

### [Launch Distributed Benchmark]
# benchmark_accuracy.py imports from dataset.py in the parent drivaer/
# directory.  Python's sys.path[0] is the script's own directory, so we
# need PYTHONPATH to make drivaer/ importable.
export PYTHONPATH="${DRIVAER_DIR}:${PYTHONPATH:-}"
uv run --no-sync torchrun \
    --nproc-per-node "$NUM_GPUS_PER_NODE" \
    "${SCRIPT_DIR}/benchmark_accuracy.py" \
    "${BENCH_ARGS[@]}"
