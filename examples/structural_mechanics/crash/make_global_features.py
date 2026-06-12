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

"""Build the global-features JSON the crash datapipe expects from the bumper master CSV.

Mirrors the shipped bumper contract: per run, the 3 scalars
``velocity_x``, ``thickness_scale``, ``rwall_origin_y`` (``global_dim: 3``), keyed by run id
(``sim_name``). Mapping from the master CSV columns:

    velocity_x      <- velocity_mm_ms
    thickness_scale <- thickness_bb_mm   (bumper-beam thickness; --thickness-col to override)
    rwall_origin_y  <- pole_offset_y_mm

Usage:
    python make_global_features.py \
        --master-csv ./bumper_beam_master_with_split.csv \
        --out ./global_features.json
"""

import argparse
import csv
import json

# global-feature key -> master CSV column
COLUMN_MAP = {
    "velocity_x": "velocity_mm_ms",
    "thickness_scale": "thickness_bb_mm",
    "rwall_origin_y": "pole_offset_y_mm",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-csv", default="./bumper_beam_master_with_split.csv")
    ap.add_argument("--out", default="./global_features.json")
    ap.add_argument(
        "--thickness-col",
        default="thickness_bb_mm",
        help="CSV column mapped to thickness_scale (e.g. thickness_cb_mm).",
    )
    args = ap.parse_args()

    column_map = dict(COLUMN_MAP, thickness_scale=args.thickness_col)

    out = {}
    with open(args.master_csv, newline="") as f:
        for row in csv.DictReader(f):
            run_id = row.get("sim_name")
            if not run_id:
                continue
            out[run_id] = {
                key: float(row[col]) for key, col in column_map.items()
            }

    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"Wrote {len(out)} runs -> {args.out}  (keys: {list(column_map)})")


if __name__ == "__main__":
    main()
