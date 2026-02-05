#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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

# Run ASV benchmarks from the repository root directory.
# Usage: ./benchmarks/run_benchmarks.sh [additional asv arguments]

set -e

# Navigate to the repository root directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT_DIR"

echo -e "\033[0;32mRunning ASV benchmarks from: $REPO_ROOT_DIR\033[0m"

# Run ASV with spawn method for CUDA compatibility.
asv run --launch-method spawn "$@"
