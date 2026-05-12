#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J coreai_modulus_cae-psharpe.benchmark_globe
#SBATCH --time=1:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --dependency=singleton
#SBATCH -o ./sbatch_logs/%x.log
#SBATCH -e ./sbatch_logs/%x.log
#SBATCH --open-mode=append

set -euo pipefail

### [User Configuration]
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

BENCH_ARGS=(
    --subdivisions 5
    --n-prediction-points 2048
    --theta 1.0
    --leaf-size 32
    --save-json "${SCRIPT_DIR}/benchmark_results.json"
)

### [Run Information]
echo "SLURM Job ID: ${SLURM_JOB_ID:-n/a}"
echo "SLURM Job name: ${SLURM_JOB_NAME:-n/a}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"

### [Detect GPUs and CUDA version]
NVIDIA_SMI_OUTPUT=$(nvidia-smi)
NUM_GPUS_PER_NODE=$(grep -cE '^\|[[:space:]]+[0-9]+[[:space:]]' <<< "$NVIDIA_SMI_OUTPUT")
CUDA_MAJOR=$(sed -n 's/.*CUDA Version: \([0-9]*\).*/\1/p' <<< "$NVIDIA_SMI_OUTPUT")
echo "Number of GPUs per node detected: $NUM_GPUS_PER_NODE"
echo "Using 1 GPU for benchmark (single-process, no torchrun)"

### [Thread Configuration]
CPUS_PER_NODE=${SLURM_CPUS_ON_NODE:-$(nproc)}
export OMP_NUM_THREADS=$((CPUS_PER_NODE / NUM_GPUS_PER_NODE))
OMP_NUM_THREADS=$((OMP_NUM_THREADS > 0 ? OMP_NUM_THREADS : 1))
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS (${CPUS_PER_NODE} CPUs / ${NUM_GPUS_PER_NODE} GPUs)"

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
uv sync --inexact --compile-bytecode --extra "${CUDA_EXTRA}" --extra mesh-extras

### [Launch Benchmark]
echo ""
echo "Starting benchmark..."
uv run --no-sync benchmark.py "${BENCH_ARGS[@]}"
