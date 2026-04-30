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
Bumper-beam simulation runner. Thin wrapper around `common.radioss_runner`.
Run AFTER `generate_dataset.py` has populated `./dataset/run*/` folders.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.radioss_runner import RunnerConfig, run_batch  # noqa: E402


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    openradioss_root = os.environ.get("OPENRADIOSS_ROOT")
    if not openradioss_root:
        raise SystemExit(
            "OPENRADIOSS_ROOT env var is required (path to your OpenRadioss build)."
        )
    cfg = RunnerConfig(
        openradioss_root=openradioss_root,
        dataset_dir=os.path.join(SCRIPT_DIR, "dataset"),
        input_base_name="Bumper_Beam_AP_meshed",
        max_parallel_jobs=int(os.environ.get("MAX_PARALLEL_JOBS", "2")),
        omp_num_threads=os.environ.get("OMP_NUM_THREADS", "8"),
        debug_mode=os.environ.get("DEBUG_MODE", "0") == "1",
    )
    run_batch(cfg)
