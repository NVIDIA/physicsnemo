#!/usr/bin/env bash
set -euo pipefail

# Reproducible setup + train helper for flow reconstruction diffusion.
# Run this inside a PhysicsNeMo environment.
#
# Optional environment variables:
#   CONFIG_NAME        (default: config_dfsr_train)
#   TRAIN_EXTRA_ARGS   (default: empty; appended to train.py)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_NAME="${CONFIG_NAME:-config_dfsr_train}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

echo ">>> [0/3] Entering ${SCRIPT_DIR}"
cd "${SCRIPT_DIR}"

echo ">>> [1/3] Installing dependencies"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo ">>> [2/3] Starting training with --config-name ${CONFIG_NAME}"
if [[ -n "${TRAIN_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2086
  python train.py --config-name "${CONFIG_NAME}" ${TRAIN_EXTRA_ARGS}
else
  python train.py --config-name "${CONFIG_NAME}"
fi

echo ">>> [3/3] Done. Check configured output directory for logs and snapshots."
