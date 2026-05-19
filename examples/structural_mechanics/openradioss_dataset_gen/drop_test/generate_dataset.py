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
Cell-phone drop-test Design-of-Experiments generator.

Parameterises a base OpenRadioss drop-test starter deck
(`Cell_Phone_Drop_0000.rad`) over:
    - Young's modulus (E) scale per material (polymer, battery, glass, PCB,
      composites) — ±20% by default;
    - rigid-wall plane orientation — rotation around X, Y, Z (degrees).

Writes one case folder per DoE combination, each containing the mutated
starter, a copy of the engine deck, and a per-run metadata JSON.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import re
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.summary_utils import save_summaries  # noqa: E402


# Materials in the reference Cell_Phone_Drop_0000.rad deck.
MATERIAL_IDS = {
    1: "polymer_unfilled_plastic",  # MAT/PLAS_TAB
    4: "battery_elastic",  # MAT/ELAST
    5: "glass_elastic",  # MAT/ELAST
    8: "pcb_elastic",  # MAT/ELAST
    9: "comps_elastic",  # MAT/ELAST
}


def _rotate_vector_xyz(v, angles_deg):
    """Rotate vector by intrinsic (rx, ry, rz) degrees around X, Y, Z."""
    rx, ry, rz = angles_deg
    x, y, z = v
    if abs(rx) > 1e-9:
        r = math.radians(rx)
        c, s = math.cos(r), math.sin(r)
        y, z = y * c - z * s, y * s + z * c
    if abs(ry) > 1e-9:
        r = math.radians(ry)
        c, s = math.cos(r), math.sin(r)
        x, z = x * c + z * s, -x * s + z * c
    if abs(rz) > 1e-9:
        r = math.radians(rz)
        c, s = math.cos(r), math.sin(r)
        x, y = x * c - y * s, x * s + y * c
    return (x, y, z)


def modify_radioss_file(
    input_path,
    output_path,
    e_scales=None,
    rwall_orientation_deg=None,
):
    """Mutate material Young's moduli and rigid-wall orientation in a
    Radioss `.rad` starter file.

    - `/MAT/{ELAST,PLAS_TAB}/<id>` : if `<id>` is in `e_scales`, scale the
       next data line's first token (E) by `e_scales[<id>]`. Detection
       relies on the preceding comment line naming `E` and `nu`/`Nu`.
    - `/RWALL` (type PLANE) : rotate the M->M1 normal by `rwall_orientation_deg`
       about its base point M (line 4 xyz), updating line 5 xyz.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file '{input_path}' not found.")
    e_scales = e_scales or {}
    rwall_orientation_deg = rwall_orientation_deg or (0.0, 0.0, 0.0)

    processing_mat = False
    current_mat_id = None
    prev_line_was_e_comment = False

    processing_rwall = False
    rwall_data_line_count = 0
    rwall_M = None

    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            sline = line.strip()

            if sline.startswith("/"):
                prev_line_was_e_comment = False

                m = re.match(r"/MAT/(ELAST|PLAS_TAB)/(\d+)", sline, re.I)
                if m:
                    processing_mat = True
                    current_mat_id = int(m.group(2))
                else:
                    processing_mat = False
                    current_mat_id = None

                if sline.upper().startswith("/RWALL"):
                    processing_rwall = True
                    rwall_data_line_count = 0
                    rwall_M = None
                else:
                    processing_rwall = False

                f_out.write(line)
                continue

            if sline.startswith("#"):
                prev_line_was_e_comment = bool(
                    "E" in sline and ("nu" in sline or "Nu" in sline)
                )
                f_out.write(line)
                continue

            if not sline:
                prev_line_was_e_comment = False
                f_out.write(line)
                continue

            # A. Material E modification.
            if (
                processing_mat
                and prev_line_was_e_comment
                and current_mat_id in e_scales
            ):
                scale = e_scales[current_mat_id]
                if scale != 1.0:
                    clean = sline.split("#")[0].strip()
                    tokens = [t for t in re.split(r"\s+", clean) if t]
                    if tokens:
                        try:
                            new_e = float(tokens[0]) * scale
                            tokens[0] = f"{new_e:.6g}"
                            # OpenRadioss uses 20-char fixed-width columns here.
                            f_out.write("".join(f"{t:>20}" for t in tokens) + "\n")
                        except ValueError:
                            f_out.write(line)
                    else:
                        f_out.write(line)
                else:
                    f_out.write(line)
                prev_line_was_e_comment = False
                continue

            # B. Rigid-wall orientation.
            if processing_rwall and rwall_orientation_deg != (0, 0, 0):
                rwall_data_line_count += 1
                clean = sline.split("#")[0].strip()
                tokens = [t for t in re.split(r"\s+", clean) if t]

                if rwall_data_line_count == 4 and len(tokens) >= 3:
                    try:
                        rwall_M = (
                            float(tokens[0]),
                            float(tokens[1]),
                            float(tokens[2]),
                        )
                        f_out.write(line)
                    except ValueError:
                        f_out.write(line)
                elif rwall_data_line_count == 5 and len(tokens) >= 3 and rwall_M:
                    try:
                        M1 = (float(tokens[0]), float(tokens[1]), float(tokens[2]))
                        n = (M1[0] - rwall_M[0], M1[1] - rwall_M[1], M1[2] - rwall_M[2])
                        n_rot = _rotate_vector_xyz(n, rwall_orientation_deg)
                        M1_new = (
                            rwall_M[0] + n_rot[0],
                            rwall_M[1] + n_rot[1],
                            rwall_M[2] + n_rot[2],
                        )
                        f_out.write(
                            f"{M1_new[0]:>20.6f}{M1_new[1]:>20.6f}{M1_new[2]:>20.6f}\n"
                        )
                    except ValueError:
                        f_out.write(line)
                else:
                    f_out.write(line)
                continue

            f_out.write(line)


def _flatten_row(run: dict) -> dict:
    p = run["parameters"]
    e_scales = p.get("e_scales") or {}
    orient = p.get("rwall_orientation_deg") or (0, 0, 0)
    return {
        "Run_ID": run["run_id"],
        "Timestamp": run["timestamp"],
        "E_scale_mat1": e_scales.get(1, 1.0),
        "E_scale_mat4": e_scales.get(4, 1.0),
        "E_scale_mat5": e_scales.get(5, 1.0),
        "E_scale_mat8": e_scales.get(8, 1.0),
        "E_scale_mat9": e_scales.get(9, 1.0),
        "Orient_rx_deg": orient[0],
        "Orient_ry_deg": orient[1],
        "Orient_rz_deg": orient[2],
    }


def generate_dataset(base_file, engine_file, output_root, variations):
    """Write one case folder per drop-test DoE combination plus summary files."""
    os.makedirs(output_root, exist_ok=True)

    e_ranges = variations.get("e_scales", {})
    orient_range = variations.get("rwall_orientations", [(0, 0, 0)])

    mat_ids = sorted(e_ranges.keys())
    e_combos = [
        dict(zip(mat_ids, combo))
        for combo in itertools.product(*(e_ranges[mid] for mid in mat_ids))
    ] or [{}]

    combinations = list(itertools.product(e_combos, orient_range))
    total = len(combinations)
    print(f"\n TOTAL CASES TO GENERATE: {total}\n")

    if not os.path.exists(engine_file):
        print(f"WARNING: Engine file '{engine_file}' not found! It will not be copied.")

    all_runs = []
    for i, (e_scales, orient) in enumerate(combinations, 1):
        run_id = f"run{i:04d}"
        run_dir = os.path.join(output_root, run_id)
        os.makedirs(run_dir, exist_ok=True)

        out_file = os.path.join(run_dir, os.path.basename(base_file))
        modify_radioss_file(
            input_path=base_file,
            output_path=out_file,
            e_scales=e_scales if e_scales else None,
            rwall_orientation_deg=orient,
        )
        if os.path.exists(engine_file):
            shutil.copy(engine_file, run_dir)

        metadata = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "e_scales": e_scales,
                "rwall_orientation_deg": list(orient),
            },
        }
        with open(os.path.join(run_dir, f"{run_id}.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        all_runs.append(metadata)
        print(f"[{i}/{total}] {run_id} | E:{e_scales} | Orient:{orient}")

    save_summaries(output_root, all_runs, _flatten_row)


# Default DoE: 2^5 E-scale combinations x 6 orientations = 192 runs.
DEFAULT_EXPERIMENT_SETUP = {
    "e_scales": {
        1: [0.8, 1.2],  # polymer
        4: [0.8, 1.2],  # battery
        5: [0.8, 1.2],  # glass
        8: [0.8, 1.2],  # pcb
        9: [0.8, 1.2],  # comps
    },
    "rwall_orientations": [
        (0, 0, 0),
        (10, 0, 0),
        (-10, 0, 0),
        (0, 10, 0),
        (0, -10, 0),
        (0, 0, 10),
    ],
}


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(script_dir, "templates")

    STARTER_FILE = os.path.join(templates_dir, "Cell_Phone_Drop_0000.rad")
    ENGINE_FILE = os.path.join(templates_dir, "Cell_Phone_Drop_0001.rad")
    DATASET_DIR = os.path.join(script_dir, "dataset")

    try:
        generate_dataset(
            STARTER_FILE,
            ENGINE_FILE,
            DATASET_DIR,
            DEFAULT_EXPERIMENT_SETUP,
        )
        print("\nDataset generation complete.")
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        raise
