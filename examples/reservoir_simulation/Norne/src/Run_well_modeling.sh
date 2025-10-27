#!/bin/bash
cat <<'EOF'
#############################################################
# SPDX-FileCopyrightText: Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Clement Etienam (cetienam@nvidia.com)
#############################################################
EOF

set -euo pipefail

# Optional: better error message on failure
trap 'echo "❌ Script failed at line $LINENO"; exit 1' ERR

# === Script Directory & YAML Path ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_FILE="$SCRIPT_DIR/conf/DECK_CONFIG.yaml"

# === Check YAML file exists ===
if [[ ! -f "$YAML_FILE" ]]; then
  echo "❌ YAML config not found at $YAML_FILE"
  exit 1
fi

# === Ensure logs dir exists early ===
mkdir -p "$SCRIPT_DIR/simulation_logs"

# === Configuration ===
MASTER_ADDR="127.0.0.1"
MASTER_PORT="12355"
PYTHON_SCRIPT_1="Forward_problem_seq.py"
PYTHON_SCRIPT_2="Compare_fvm_surrogate_seq.py"
PYTHON_SCRIPT_3="Forward_problem_batch.py"
PYTHON_SCRIPT_4="Compare_fvm_surrogate_batch.py"
PYTHON_SCRIPT_5="Inverse_problem.py"
PYTHON_SCRIPT_6="Moe_ccr.py"
PYTHON_SCRIPT_7="Extract_data.py"

# === Parse YAML values using Python ===
read -r INTEREST MODEL_DISTRIBUTED FNO_TYPE <<< "$(python3 -c "
import yaml
with open('$YAML_FILE', 'r') as f:
    config = yaml.safe_load(f)
custom = config.get('custom', {})
print(custom.get('interest', 'No'), custom.get('model_Distributed', 2), custom.get('fno_type', 'PINO'))
")"

echo "📖 YAML Config Extracted:"
echo "• interest           = $INTEREST"
echo "• model_Distributed  = $MODEL_DISTRIBUTED"
echo "• fno_type           = $FNO_TYPE"

# === Dynamic log file names ===
ts="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SCRIPT_DIR/simulation_logs"
LOG_FILE1="$LOG_DIR/Forward_problem_seq_${FNO_TYPE}_${ts}.log"
LOG_FILE2="$LOG_DIR/Compare_FVM_surrogate_seq_${ts}.log"
LOG_FILE3="$LOG_DIR/Forward_problem_batch_${FNO_TYPE}_${ts}.log"
LOG_FILE4="$LOG_DIR/Compare_FVM_surrogate_batch_${ts}.log"
LOG_FILE5="$LOG_DIR/Inverse_problem_${FNO_TYPE}_${ts}.log" 
LOG_FILE6="$LOG_DIR/Moe_ccr_${ts}.log"  
LOG_FILE7="$LOG_DIR/Extract_data_${ts}.log"  

# === Detect number of available GPUs ===
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "❌ nvidia-smi not found. Is this a GPU node?"
  exit 1
fi
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | awk '{print $1}')
if [[ "${NUM_GPUS:-0}" -lt 1 ]]; then
  echo "❌ No GPUs detected on this node."
  exit 1
fi
echo "🖥️  Detected $NUM_GPUS GPU(s) on this node."

# === Prompt user to enter number of MPI ranks ===
echo "🔢 Please enter the number of MPI ranks to use (1 to $NUM_GPUS):"
read -r NUM_MPI_RANKS

# Validate input is a positive integer and within [1, NUM_GPUS]
if ! [[ "$NUM_MPI_RANKS" =~ ^[1-9][0-9]*$ ]]; then
  echo "❌ Error: Invalid number of MPI ranks. Please enter a positive integer."
  exit 1
fi
if (( NUM_MPI_RANKS < 1 || NUM_MPI_RANKS > NUM_GPUS )); then
  echo "❌ Error: MPI ranks must be between 1 and $NUM_GPUS."
  exit 1
fi

# === Ask user for operation type ===
echo "⚙️ What operation do you want to run?"
echo "1 - History Matching"
echo "2 - Well Optimisation and Placement"
read -r OPERATION_CHOICE

# Validate choice is 1, 2, or 3
if ! [[ "$OPERATION_CHOICE" =~ ^[1-3]$ ]]; then
  echo "❌ Error: Invalid choice. Please enter 1, 2, or 3."
  exit 1
fi
echo "✅ You selected option $OPERATION_CHOICE."


# === Environment Setup ===
export MASTER_ADDR
export MASTER_PORT
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO


# === Conditional run: Extraction ===
if [[ "$INTEREST" == "Yes" ]]; then
  echo "🧪 Running data extraction (interest = Yes)..."
  (
    torchrun \
      --nproc_per_node=1 \
      --nnodes=1 \
      --standalone \
      "$PYTHON_SCRIPT_7"
  ) 2>&1 | tee -a "$LOG_FILE7"
else
  echo "📦 Skipping data extraction (interest = No)"
fi


# Helper to run torch scripts
run_torch() {
  local nproc="$1"
  local script="$2"
  torchrun --nproc_per_node="$nproc" --nnodes=1 --standalone "$script"
}

# === Branching ===
if [[ "$OPERATION_CHOICE" == "1" ]]; then
  echo "🚀 Running the history matching problem......."
  echo "🚀*******************************************"
  echo "🚀 Training the Forward problem..............."
  if [[ "$MODEL_DISTRIBUTED" == "1" ]]; then
    echo "🚀 Running the workflow in Multi-GPU mode with $NUM_MPI_RANKS ranks..."
    ( run_torch "$NUM_MPI_RANKS" "$PYTHON_SCRIPT_3" ) 2>&1 | tee -a "$LOG_FILE3"
  else
    echo "🚀 Running forward problem in Single-GPU mode..."
    ( run_torch 1 "$PYTHON_SCRIPT_3" ) 2>&1 | tee -a "$LOG_FILE3"
  fi

  echo "🚀*******************************************"
  echo "🚀 Running the Mixture of Experts.................."
  ( run_torch 1 "$PYTHON_SCRIPT_6" ) 2>&1 | tee -a "$LOG_FILE6"

  echo "🚀*******************************************"
  echo "🚀 Running the Comparison with the numerical solver..."
  ( run_torch 1 "$PYTHON_SCRIPT_4" ) 2>&1 | tee -a "$LOG_FILE4"

  echo "🚀*******************************************"
  echo "🚀 Running the history matching loop.........."
  ( run_torch 1 "$PYTHON_SCRIPT_5" ) 2>&1 | tee -a "$LOG_FILE5"

else
  echo "🚀 Running the Well placement and optimisation step."
  echo "🚀*******************************************"
  echo "🚀 Training the Forward problem..."
  if [[ "$MODEL_DISTRIBUTED" == "1" ]]; then
    echo "🚀 Running the workflow in Multi-GPU mode with $NUM_MPI_RANKS ranks..."
    ( run_torch "$NUM_MPI_RANKS" "$PYTHON_SCRIPT_1" ) 2>&1 | tee -a "$LOG_FILE1"
  else
    echo "🚀 Running forward problem in Single-GPU mode..."
    ( run_torch 1 "$PYTHON_SCRIPT_1" ) 2>&1 | tee -a "$LOG_FILE1"
  fi
  echo "🚀 Running the Mixture of Experts.................."
  ( run_torch 1 "$PYTHON_SCRIPT_6" ) 2>&1 | tee -a "$LOG_FILE6"
  
  echo "🚀*******************************************"
  echo "🚀 Running the Comparison with the numerical solver..."
  ( run_torch 1 "$PYTHON_SCRIPT_2" ) 2>&1 | tee -a "$LOG_FILE2"

fi

echo "✅ All operations completed successfully!"
