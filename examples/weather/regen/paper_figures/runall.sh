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

set -e
if ! [[ -d figure_data ]]
then
    echo "figure_data/ not found. Did you extract the tar file here?"
    exit 1
fi

if ! [[ -f figure_data/evenmore_random_val_stations.npy ]]; then
    python3 val_stations.py
fi

set -e
project=/path/to/project
conditionalSamples=$project/inferences/peters_step_64_all.nc
unconditionalSamples=$project/inferences/uncondSample/samples.nc


# run these two in parallel
python3 score_inference.py figure_data/scores/peters_step_64_all &
# truth is misabled here. It is actually "hrrr"
python3 score_inference.py -g truth figure_data/scores/hrrr &
wait

export PYTHONPATH=$(pwd)

python3 figure-07/fig_metrics.py
python3 climate.py
