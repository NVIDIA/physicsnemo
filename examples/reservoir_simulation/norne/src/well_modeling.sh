#!/usr/bin/env bash
# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# =============================================================================
# well_modeling.sh — NVIDIA PhysicsNeMo Reservoir Simulation Workflow
# =============================================================================
#
# End-to-end orchestration script for the black-oil history matching pipeline.
# Executes the following stages in sequence:
#
#   [1] (Optional) Data extraction          — extract_data.py
#   [2] Forward model training              — train.py
#   [3] Training metrics                    — plot_metrics_run.py
#   [4] Mixture-of-Experts CCR surrogate    — moe_ccr.py
#   [5] Numerical solver comparison         — inference.py
#   [6] Inverse history matching (α-REKI)   — run_inverse.py
#
# Usage:
#   bash well_modeling.sh [--ranks N] [--config CONFIG_YAML]
#
# Options:
#   --ranks  N       Number of GPU ranks for distributed training (default: prompt)
#   --config PATH    Path to YAML config (default: conf/DECK_CONFIG.yaml)
#   --dry-run        Print commands without executing them
#   -h, --help       Show this help message
#
# Requirements:
#   - CUDA-capable GPU(s) with nvidia-smi available
#   - PyTorch with torchrun
#   - Python 3 with pyyaml installed
#
# Author: Clement Etienam <cetienam@nvidia.com>
# =============================================================================

set -euo pipefail

# ── ANSI colour helpers ───────────────────────────────────────────────────────
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly RESET='\033[0m'

log_info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
log_ok()      { echo -e "${GREEN}[OK]${RESET}    $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
log_error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
log_section() {
    local title="$1"
    local width=70
    local line
    line=$(printf '%*s' "$width" '' | tr ' ' '─')
    echo -e "\n${BOLD}${CYAN}${line}${RESET}"
    printf "${BOLD}${CYAN}  %-$((width - 2))s${RESET}\n" "$title"
    echo -e "${BOLD}${CYAN}${line}${RESET}"
}

# ── Error / exit trap ─────────────────────────────────────────────────────────
on_error() {
    local exit_code=$?
    local line_no=$1
    log_error "Pipeline failed at line ${line_no} (exit code ${exit_code})."
    log_error "Check the log files in ${LOG_DIR} for details."
    exit "${exit_code}"
}
trap 'on_error ${LINENO}' ERR

# ── Elapsed-time helper ───────────────────────────────────────────────────────
PIPELINE_START=$(date +%s)

elapsed() {
    local secs=$(( $(date +%s) - PIPELINE_START ))
    printf '%02d:%02d:%02d' $(( secs/3600 )) $(( (secs%3600)/60 )) $(( secs%60 ))
}

step_timer() {
    local label="$1"
    local step_start="$2"
    local secs=$(( $(date +%s) - step_start ))
    printf '%s completed in %02d:%02d:%02d\n' "$label" \
        $(( secs/3600 )) $(( (secs%3600)/60 )) $(( secs%60 ))
}

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_FILE="${SCRIPT_DIR}/conf/DECK_CONFIG.yaml"
NUM_MPI_RANKS=""
DRY_RUN=false

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
    sed -n '/^# Usage:/,/^# Author:/{ /^# Author:/d; s/^# \{0,3\}//; p }' "$0"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ranks)  NUM_MPI_RANKS="$2"; shift 2 ;;
        --config) YAML_FILE="$2";     shift 2 ;;
        --dry-run) DRY_RUN=true;      shift   ;;
        -h|--help) usage ;;
        *) log_error "Unknown option: $1"; usage ;;
    esac
done

# ── Validate environment ──────────────────────────────────────────────────────
log_section "Environment Validation"

if [[ ! -f "$YAML_FILE" ]]; then
    log_error "Config not found: ${YAML_FILE}"
    exit 1
fi
log_ok "Config file     : ${YAML_FILE}"

if ! command -v python3 >/dev/null 2>&1; then
    log_error "python3 not found in PATH."
    exit 1
fi
log_ok "Python          : $(python3 --version)"

if ! command -v torchrun >/dev/null 2>&1; then
    log_error "torchrun not found. Install PyTorch or activate the correct conda environment."
    exit 1
fi
log_ok "torchrun        : $(torchrun --version 2>&1 | head -1)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    log_error "nvidia-smi not found. A GPU node is required."
    exit 1
fi

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "${NUM_GPUS:-0}" -lt 1 ]]; then
    log_error "No GPUs detected by nvidia-smi."
    exit 1
fi
log_ok "GPUs available  : ${NUM_GPUS}"

# ── Parse YAML configuration ──────────────────────────────────────────────────
log_section "Reading Configuration"

read -r INTEREST MODEL_DISTRIBUTED FNO_TYPE < <(python3 - <<PYEOF
import yaml, sys
with open('${YAML_FILE}') as fh:
    cfg = yaml.safe_load(fh).get('custom', {})
vals = [
    str(cfg.get('interest',          'No')),
    str(cfg.get('model_Distributed', 2)),
    str(cfg.get('fno_type',          'FNO')),
]
print(' '.join(vals))
PYEOF
)

log_info "interest          = ${INTEREST}"
log_info "model_Distributed = ${MODEL_DISTRIBUTED}"
log_info "fno_type          = ${FNO_TYPE}"

# ── Prompt for GPU ranks (if not provided via --ranks) ───────────────────────
if [[ -z "$NUM_MPI_RANKS" ]]; then
    echo -e "\n${BOLD}Enter number of GPU ranks to use [1–${NUM_GPUS}]:${RESET} \c"
    read -r NUM_MPI_RANKS
fi

if ! [[ "$NUM_MPI_RANKS" =~ ^[1-9][0-9]*$ ]] || \
   (( NUM_MPI_RANKS < 1 || NUM_MPI_RANKS > NUM_GPUS )); then
    log_error "GPU ranks must be an integer between 1 and ${NUM_GPUS}. Got: '${NUM_MPI_RANKS}'."
    exit 1
fi
log_ok "GPU ranks         = ${NUM_MPI_RANKS}"

# ── Environment variables ─────────────────────────────────────────────────────
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="12355"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TRITON_JIT=0

log_ok "MASTER_ADDR:PORT  = ${MASTER_ADDR}:${MASTER_PORT}"
log_ok "CUDA_HOME         = ${CUDA_HOME}"

# ── Log directory and file paths ──────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SCRIPT_DIR}/simulation_logs/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

LOG_EXTRACT="${LOG_DIR}/01_extract_data.log"
LOG_TRAIN="${LOG_DIR}/02_train_${FNO_TYPE}.log"
LOG_METRICS="${LOG_DIR}/03_plot_metrics.log"
LOG_MOE="${LOG_DIR}/04_moe_ccr.log"
LOG_INFER="${LOG_DIR}/05_inference.log"
LOG_INVERSE="${LOG_DIR}/06_inverse_${FNO_TYPE}.log"

log_ok "Logs directory    = ${LOG_DIR}"

# ── Script paths ──────────────────────────────────────────────────────────────
SCRIPT_EXTRACT="${SCRIPT_DIR}/extract_data.py"
SCRIPT_TRAIN="${SCRIPT_DIR}/train.py"
SCRIPT_METRICS="${SCRIPT_DIR}/plot_metrics_run.py"
SCRIPT_MOE="${SCRIPT_DIR}/moe_ccr.py"
SCRIPT_INFER="${SCRIPT_DIR}/inference.py"
SCRIPT_INVERSE="${SCRIPT_DIR}/run_inverse.py"

for script in "$SCRIPT_TRAIN" "$SCRIPT_METRICS" "$SCRIPT_MOE" \
              "$SCRIPT_INFER" "$SCRIPT_INVERSE"; do
    if [[ ! -f "$script" ]]; then
        log_error "Required script not found: ${script}"
        exit 1
    fi
done

# ── Core runner ───────────────────────────────────────────────────────────────
# Usage: run_stage LABEL LOGFILE NRANKS SCRIPT [extra args...]
run_stage() {
    local label="$1"
    local logfile="$2"
    local nranks="$3"
    local script="$4"
    shift 4

    local step_start
    step_start=$(date +%s)

    log_section "Stage: ${label}"
    log_info "Script : ${script}"
    log_info "Ranks  : ${nranks}"
    log_info "Log    : ${logfile}"

    local cmd=(
        torchrun
        --nproc_per_node="${nranks}"
        --nnodes=1
        --standalone
        "${script}"
        "$@"
    )

    if [[ "$DRY_RUN" == true ]]; then
        log_warn "[DRY-RUN] Would execute: ${cmd[*]}"
        return 0
    fi

    log_info "Starting at $(date '+%Y-%m-%d %H:%M:%S') ..."
    if "${cmd[@]}" 2>&1 | tee -a "${logfile}"; then
        log_ok "$(step_timer "${label}" "${step_start}")"
    else
        log_error "Stage '${label}' failed. See: ${logfile}"
        exit 1
    fi
}

# ── Pipeline ──────────────────────────────────────────────────────────────────
log_section "Pipeline Start  [$(date '+%Y-%m-%d %H:%M:%S')]"
log_info "FNO type       : ${FNO_TYPE}"
log_info "GPU ranks      : ${NUM_MPI_RANKS} / ${NUM_GPUS} available"
log_info "Distributed    : $([ "$MODEL_DISTRIBUTED" == "1" ] && echo Yes || echo No)"

# Stage 1 — Data extraction (optional)
if [[ "$INTEREST" == "Yes" ]]; then
    if [[ ! -f "$SCRIPT_EXTRACT" ]]; then
        log_error "extract_data.py not found but interest=Yes in config."
        exit 1
    fi
    run_stage "Data Extraction" "${LOG_EXTRACT}" 1 "${SCRIPT_EXTRACT}"
else
    log_info "Skipping data extraction (interest = No)."
fi

# Forward model training
TRAIN_RANKS=$([ "$MODEL_DISTRIBUTED" == "1" ] && echo "$NUM_MPI_RANKS" || echo 1)
run_stage "Forward Model Training" "${LOG_TRAIN}" "${TRAIN_RANKS}" "${SCRIPT_TRAIN}"

#Training metrics
run_stage "Training Metrics" "${LOG_METRICS}" 1 "${SCRIPT_METRICS}"

#Mixture-of-Experts CCR surrogate
run_stage "Mixture-of-Experts CCR" "${LOG_MOE}" 1 "${SCRIPT_MOE}"

#Numerical solver comparison
run_stage "Numerical Solver Comparison" "${LOG_INFER}" 1 "${SCRIPT_INFER}"


#Inverse history matching (multi-GPU)
run_stage "Inverse History Matching (α-REKI)" "${LOG_INVERSE}" "${NUM_MPI_RANKS}" "${SCRIPT_INVERSE}"

# ── Summary ───────────────────────────────────────────────────────────────────
log_section "Pipeline Complete"
log_ok "All stages finished successfully."
log_ok "Total elapsed time : $(elapsed)"
log_ok "Log directory      : ${LOG_DIR}"
