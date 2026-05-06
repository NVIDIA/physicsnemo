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
Convert the drop-test `summary.json` (list of run metadata) into the per-run
dict format consumed by the crash recipe's datapipe
(`global_features.json` keyed by run ID).

Output keys (one row per run):
    e_scale_mat1, e_scale_mat4, e_scale_mat5, e_scale_mat8, e_scale_mat9,
    rwall_orientation_rx, rwall_orientation_ry, rwall_orientation_rz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MATERIAL_IDS = (1, 4, 5, 8, 9)


def convert_global_features(input_json: Path, output_json: Path) -> None:
    """Convert drop-test ``summary.json`` into a per-run-id ``global_features.json``."""
    with input_json.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("Input JSON must be a list of entries")

    out: dict[str, dict[str, float]] = {}
    for entry in data:
        run_id = entry.get("run_id")
        if run_id is None:
            raise KeyError("Missing run_id in entry")
        if run_id in out:
            raise ValueError(f"Duplicate run_id: {run_id}")

        params = entry.get("parameters")
        if params is None:
            raise KeyError(f"Missing parameters for run '{run_id}'")

        e_scales = params.get("e_scales") or {}
        orient = params.get("rwall_orientation_deg") or [0, 0, 0]

        features: dict[str, float] = {}
        for mat_id in MATERIAL_IDS:
            # JSON keys may come back as strings after a round-trip.
            val = e_scales.get(str(mat_id), e_scales.get(mat_id, 1.0))
            features[f"e_scale_mat{mat_id}"] = float(val)

        features["rwall_orientation_rx"] = float(orient[0]) if len(orient) > 0 else 0.0
        features["rwall_orientation_ry"] = float(orient[1]) if len(orient) > 1 else 0.0
        features["rwall_orientation_rz"] = float(orient[2]) if len(orient) > 2 else 0.0
        out[run_id] = features

    with output_json.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(out)} runs to {output_json}")


def main() -> None:
    """CLI entry point: parse ``--input`` / ``--output`` and write ``global_features.json``."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=here / "dataset" / "summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "global_features.json",
    )
    args = parser.parse_args()
    convert_global_features(args.input, args.output)


if __name__ == "__main__":
    main()
