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

""" 
Bash script to download the meshgraphnet dataset from deepmind's repo.
      -  Repo: https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
      -  Run: sh download_dataset.sh deforming_plate
"""

git clone https://github.com/deepmind/deepmind-research.git
set -e
DATASET_NAME="${1}"
OUTPUT_DIR="${DATASET_NAME}"
sh deepmind-research/meshgraphnets/download_dataset.sh ${DATASET_NAME} ${OUTPUT_DIR}
