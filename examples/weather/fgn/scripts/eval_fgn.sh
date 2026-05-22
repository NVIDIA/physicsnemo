#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Evaluate a trained FGN checkpoint over the full validation split.
#
# Usage:
#   sbatch scripts/eval_fgn.sh
#
# Override defaults with --export:
#   sbatch --export=ALL,EXP_NAME=fgn_2024_long,RUN_ID=0,CKPT=latest scripts/eval_fgn.sh

#SBATCH --job-name=fgn_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fgn_eval_%j.log

# --- Configurable knobs (override with sbatch --export=ALL,VAR=val) ---
EXP_NAME=${EXP_NAME:-fgn_2024_long}
RUN_ID=${RUN_ID:-0}
FUTURE_STEPS=${FUTURE_STEPS:-20}
ENSEMBLE_SIZE=${ENSEMBLE_SIZE:-8}
STATS_PATH=${STATS_PATH:-rundir/fgn_2024_val/stats_2024.npz}
CKPT=${CKPT:-latest}
OUTDIR=${OUTDIR:-rundir/${EXP_NAME}/${RUN_ID}/eval}
CFG=${CFG:-eval_fgn}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

mkdir -p logs

echo "=== FGN eval ==="
echo "  EXP_NAME:      ${EXP_NAME}"
echo "  RUN_ID:        ${RUN_ID}"
echo "  CKPT:          ${CKPT}"
echo "  FUTURE_STEPS:  ${FUTURE_STEPS}"
echo "  ENSEMBLE_SIZE: ${ENSEMBLE_SIZE}"
echo "  STATS_PATH:    ${STATS_PATH}"
echo "  OUTDIR:        ${OUTDIR}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python eval.py \
    --config-name "${CFG}" \
    dataset.stats_path="${STATS_PATH}" \
    training.experiment_name="${EXP_NAME}" \
    training.run_id="${RUN_ID}" \
    training.rundir="rundir/${EXP_NAME}/${RUN_ID}" \
    eval.checkpoint="${CKPT}" \
    eval.future_steps="${FUTURE_STEPS}" \
    eval.ensemble_size="${ENSEMBLE_SIZE}" \
    eval.outdir="${OUTDIR}"
