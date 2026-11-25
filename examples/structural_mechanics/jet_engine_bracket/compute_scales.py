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

r"""Per-load-case normalization scales of the training designs.

The scale of a field under a load case is its zero-offset RMS over the training samples of that
load case, every sample weighted equally (per sample, the mean square over its nodes; for the
displacement, over its three components). ``stats/deepjeb_scales.json`` holds the scales of the
DeepJEB training split; the transfer-learning recipe keeps them, so that the adapted model
predicts in the units of the pre-trained one.

    python compute_scales.py --data <data>/deepjeb --split splits/deepjeb.json --output stats/deepjeb_scales.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from bracket_io import load_split, sample_path
from nastran import LOADS

from physicsnemo.mesh import DomainMesh


def main() -> None:
    r"""Compute the per-load-case scales of the DeepJEB training designs (command line)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shapes = load_split(args.split)["train"]
    squares = {f: {load: [] for load in LOADS} for f in ("displacement", "von_mises")}
    for i, shape in enumerate(shapes, 1):
        for load in LOADS:
            data = DomainMesh.load(
                sample_path(args.data, shape, load)
            ).interior.point_data
            squares["displacement"][load].append(
                float((data["displacement"].double() ** 2).mean())
            )
            squares["von_mises"][load].append(
                float((data["von_mises"].double() ** 2).mean())
            )
        if i % 200 == 0:
            print(f"{i}/{len(shapes)} designs", flush=True)
    scales = {
        f: {load: float(np.sqrt(np.mean(v))) for load, v in by_load.items()}
        for f, by_load in squares.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                **scales,
                "units": {"displacement": "mm", "von_mises": "MPa"},
                "designs": len(shapes),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(scales))


if __name__ == "__main__":
    main()
