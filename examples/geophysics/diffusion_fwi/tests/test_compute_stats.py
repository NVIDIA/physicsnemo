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

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# As in data/generate_data.py: (1, nx, nz) models, (shots, time, receivers) records
_MODEL, _RECORD = (1, 7, 7), (5, 20, 6)
_SHAPES = {"vp": _MODEL, "vs": _MODEL, "rho": _MODEL, "vx": _RECORD, "vz": _RECORD}
_OFFSETS = {"vp": 3.0, "vs": 2.0, "rho": 2.5, "vx": 0.0, "vz": 0.3}


def test_compute_stats_matches_numpy(tmp_path):
    """Statistics in stats.json match NumPy over every element of each variable."""
    rng = np.random.default_rng(0)
    (tmp_path / "samples").mkdir()
    samples = {"train": [], "test": []}
    for i, phase in enumerate(["train"] * 5 + ["test"] * 2):
        sample = {
            k: (rng.standard_normal(shape) + _OFFSETS[k]).astype(np.float32)
            for k, shape in _SHAPES.items()
        }
        np.savez(tmp_path / "samples" / f"{phase}_sample_{i}.npz", **sample)
        samples[phase].append(sample)

    script = Path(__file__).parents[1] / "data" / "compute_stats.py"
    subprocess.run(  # noqa: S603 - interpreter and script are test constants
        [sys.executable, str(script), "--dir", str(tmp_path)]
        + ["--batch_size", "2", "--num_workers", "1"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=True,
    )

    stats = json.loads((tmp_path / "stats.json").read_text())
    assert set(stats) == set(_SHAPES)
    for name in _SHAPES:
        for phase in ["train", "test"]:
            x = np.stack([s[name] for s in samples[phase]]).astype(np.float64)
            ref = {"mean": x.mean(), "std": x.std(), "min": x.min(), "max": x.max()}
            assert stats[name][phase] == pytest.approx(ref, rel=1e-4, abs=1e-4), name
