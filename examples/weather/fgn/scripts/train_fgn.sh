#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# FGN training launcher — single-node multi-GPU (2x H100 by default).
#
# Usage:
#   sbatch scripts/train_fgn.sh                      # new run with defaults
#   sbatch scripts/train_fgn.sh --export=STEPS=5000  # override step count
#
# Environment variables (all optional, safe defaults below):
#   EXP_NAME   experiment name under rundir/  (default: fgn_2024_val)
#   RUN_ID     run sub-directory id           (default: 1)
#   STEPS      total training steps           (default: 5000)
#   CFG        Hydra config name              (default: fgn_arco)
#   STATS_PATH path to normalization stats    (default: rundir/fgn_2024_val/stats_2024.npz)
#   NGPU       number of GPUs                 (default: 2)

#SBATCH --job-name=fgn-long
#SBATCH --partition=hpc-low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --open-mode=append
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --chdir=/mnt/home/kashif/physicsnemo/examples/weather/fgn
#SBATCH --output=/mnt/home/kashif/physicsnemo/examples/weather/fgn/logs/%x_%j.log
#SBATCH --error=/mnt/home/kashif/physicsnemo/examples/weather/fgn/logs/%x_%j.log

set -euo pipefail

FGN_DIR="/mnt/home/kashif/physicsnemo/examples/weather/fgn"

# Defaults (override via environment or --export= on sbatch command line)
EXP_NAME="${EXP_NAME:-fgn_2024_val}"
RUN_ID="${RUN_ID:-1}"
STEPS="${STEPS:-5000}"
CFG="${CFG:-fgn_arco}"
STATS_PATH="${STATS_PATH:-rundir/fgn_2024_val/stats_2024.npz}"
NGPU="${NGPU:-2}"

mkdir -p "${FGN_DIR}/logs"

echo "[train_fgn.sh] exp=${EXP_NAME} run_id=${RUN_ID} steps=${STEPS} cfg=${CFG}"
echo "[train_fgn.sh] stats=${STATS_PATH}"
echo "Host  : $(hostname)"
echo "Date  : $(date)"
echo "GPUs  : ${NGPU} x H100"

cd "${FGN_DIR}"
export PYTHONPATH="${FGN_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Pick a free port — avoid 29500 which the first job may already hold.
MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "[train_fgn.sh] master_port=${MASTER_PORT}"

# Verify local datasets/ is importable before launching workers.
python3 -c "from datasets import dataset_classes; print('[train_fgn.sh] datasets OK')"

# torchrun manages multi-GPU process spawning directly (no srun wrapper).
# Using the full path ensures workers inherit the same Python as torchrun.
torchrun \
    --nproc_per_node="${NGPU}" \
    --nnodes=1 \
    --master_addr=localhost \
    --master_port="${MASTER_PORT}" \
    train.py \
    --config-name "${CFG}" \
    dataset.stats_path="${STATS_PATH}" \
    dataset.train_start="2024-01-01" \
    dataset.train_end="2024-10-01" \
    dataset.val_start="2024-10-01" \
    dataset.val_end="2025-01-01" \
    training.experiment_name="${EXP_NAME}" \
    "training.run_id='${RUN_ID}'" \
    training.total_train_steps="${STEPS}" \
    training.batch_size=1 \
    training.num_data_workers=0 \
    training.validation_steps=8 \
    training.validation_metrics=true \
    training.validation_ensemble_size=4 \
    training.checkpoint_freq=500 \
    training.validation_freq=500 \
    training.print_progress_freq=10 \
    training.domain_parallel_size=1 \
    training.resume_checkpoint=latest
