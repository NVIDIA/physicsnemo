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
Bumper-beam impact Design-of-Experiments generator.

Parameterises a base OpenRadioss bumper-beam starter deck
(`Bumper_Beam_AP_meshed_0000.rad`) over geometry scale, shell thickness,
impact velocity, and rigid-wall geometry/location, producing one case
folder per DoE combination.

Ported from the `fea-dataset-gen-gtc-lab` bumper generator; summary writing
is factored into `common.summary_utils`.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import shutil
import sys
from datetime import datetime

# Allow running as a script:  python bumper_beam/generate_dataset.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.summary_utils import save_summaries  # noqa: E402


def modify_radioss_file(
    input_path,
    output_path,
    geo_scales=(1.0, 1.0, 1.0),
    thick_scale=1.0,
    velocity_vector=None,
    rwall_updates=None,
):
    """Read a Radioss `.rad` file and apply bumper-beam parameter mutations.

    - `/NODE`         : scale nodal (x, y, z) by `geo_scales`.
    - `/PROP/SHELL`   : scale shell thickness (line 4, col 3) by `thick_scale`.
    - `/INIVEL`       : replace first 3 velocity components (line 2).
    - `/RWALL`        : overwrite diameter (line 3 col 3) and/or origin
                        (line 4 xyz); point on wall (line 5) is shifted by
                        the same delta applied to origin.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file '{input_path}' not found.")

    sx, sy, sz = geo_scales

    processing_nodes = False
    processing_prop = False
    processing_inivel = False
    processing_rwall = False

    prop_line_count = 0
    inivel_line_count = 0
    rwall_data_line_count = 0
    rwall_shift = [0.0, 0.0, 0.0]

    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            sline = line.strip()

            if sline.startswith("/"):
                processing_nodes = False
                processing_prop = False
                processing_inivel = False
                processing_rwall = False
                prop_line_count = 0
                inivel_line_count = 0
                rwall_data_line_count = 0

                if sline.upper().startswith("/NODE"):
                    processing_nodes = True
                elif sline.upper().startswith("/PROP/SHELL"):
                    processing_prop = True
                elif sline.upper().startswith("/INIVEL"):
                    processing_inivel = True
                elif sline.upper().startswith("/RWALL"):
                    processing_rwall = True

                f_out.write(line)
                continue

            if not sline or sline.startswith("#"):
                f_out.write(line)
                continue

            # A. Rigid wall
            if processing_rwall and rwall_updates:
                rwall_data_line_count += 1
                clean = sline.split("#")[0].strip()
                tokens = [t for t in re.split(r"[,\s]+", clean) if t]

                if rwall_data_line_count == 3 and "diameter" in rwall_updates:
                    if len(tokens) >= 3:
                        new_dia = rwall_updates["diameter"]
                        tokens[2] = f"{new_dia:>20.12g}"
                        f_out.write("".join(f"{t:>20}" for t in tokens) + "\n")
                    else:
                        f_out.write(line)

                elif rwall_data_line_count == 4 and "origin" in rwall_updates:
                    if len(tokens) >= 3:
                        old_x, old_y, old_z = (
                            float(tokens[0]),
                            float(tokens[1]),
                            float(tokens[2]),
                        )
                        new_x, new_y, new_z = rwall_updates["origin"]
                        rwall_shift = [new_x - old_x, new_y - old_y, new_z - old_z]
                        f_out.write(f"{new_x:>20.12g}{new_y:>20.12g}{new_z:>20.12g}\n")
                    else:
                        f_out.write(line)

                elif rwall_data_line_count == 5 and "origin" in rwall_updates:
                    if len(tokens) >= 3:
                        ox, oy, oz = (
                            float(tokens[0]),
                            float(tokens[1]),
                            float(tokens[2]),
                        )
                        nx = ox + rwall_shift[0]
                        ny = oy + rwall_shift[1]
                        nz = oz + rwall_shift[2]
                        f_out.write(f"{nx:>20.12g}{ny:>20.12g}{nz:>20.12g}\n")
                    else:
                        f_out.write(line)
                else:
                    f_out.write(line)

            # B. Nodal scaling
            elif processing_nodes:
                clean = sline.split("#")[0].strip()
                tokens = [t for t in re.split(r"[,\s]+", clean) if t]
                try:
                    if len(tokens) == 4:
                        nid = tokens[0]
                        nx = float(tokens[1]) * sx
                        ny = float(tokens[2]) * sy
                        nz = float(tokens[3]) * sz
                        f_out.write(f"{nid:>10}{nx:>20.12g}{ny:>20.12g}{nz:>20.12g}\n")
                    elif len(tokens) == 3:
                        nx = float(tokens[0]) * sx
                        ny = float(tokens[1]) * sy
                        nz = float(tokens[2]) * sz
                        f_out.write(f"{nx:>20.12g}{ny:>20.12g}{nz:>20.12g}\n")
                    else:
                        f_out.write(line)
                except ValueError:
                    f_out.write(line)

            # C. Thickness scaling
            elif processing_prop:
                prop_line_count += 1
                if prop_line_count == 4:
                    clean = sline.split("#")[0].strip()
                    tokens = [t for t in re.split(r"[,\s]+", clean) if t]
                    if len(tokens) >= 3:
                        try:
                            old_thick = float(tokens[2])
                            tokens[2] = f"{old_thick * thick_scale:.4f}"
                            f_out.write(
                                "".join(
                                    f"{t:>20}" if i == 2 else f"{t:>10}"
                                    for i, t in enumerate(tokens)
                                )
                                + "\n"
                            )
                        except ValueError:
                            f_out.write(line)
                    else:
                        f_out.write(line)
                else:
                    f_out.write(line)

            # D. Initial velocity
            elif processing_inivel and velocity_vector:
                inivel_line_count += 1
                if inivel_line_count == 2:
                    clean = sline.split("#")[0].strip()
                    tokens = [t for t in re.split(r"[,\s]+", clean) if t]
                    if len(tokens) >= 3:
                        vx, vy, vz = velocity_vector
                        out_tokens = [f"{vx:>20.12g}", f"{vy:>20.12g}", f"{vz:>20.12g}"]
                        for t in tokens[3:]:
                            out_tokens.append(f"{t:>10}")
                        f_out.write("".join(out_tokens) + "\n")
                    else:
                        f_out.write(line)
                else:
                    f_out.write(line)

            else:
                f_out.write(line)


def _flatten_row(run: dict) -> dict:
    p = run["parameters"]
    gx, gy, gz = p.get("geometry_scale", (1.0, 1.0, 1.0))
    vel = p.get("velocity_vector")
    vx, vy, vz = vel if vel else ("Default",) * 3
    w_orig = p.get("rwall_origin")
    wx, wy, wz = w_orig if w_orig else ("Default",) * 3
    return {
        "Run_ID": run["run_id"],
        "Timestamp": run["timestamp"],
        "Geo_Scale_X": gx,
        "Geo_Scale_Y": gy,
        "Geo_Scale_Z": gz,
        "Vel_X": vx,
        "Vel_Y": vy,
        "Vel_Z": vz,
        "Thickness_Factor": p.get("thickness_scale", 1.0),
        "Wall_Diameter": p.get("rwall_diameter", "Default"),
        "Wall_X": wx,
        "Wall_Y": wy,
        "Wall_Z": wz,
    }


def generate_dataset(base_file, engine_file, output_root, variations):
    """Write one case folder per bumper-beam DoE combination plus summary files."""
    os.makedirs(output_root, exist_ok=True)

    geo_range = variations.get("geometry_scales", [(1.0, 1.0, 1.0)])
    vel_range = variations.get("velocities", [None])
    thick_range = variations.get("thickness_scales", [1.0])
    dia_range = variations.get("rwall_diameters", [None])
    origin_range = variations.get("rwall_origins", [None])

    combinations = list(
        itertools.product(geo_range, vel_range, thick_range, dia_range, origin_range)
    )
    total = len(combinations)
    print(f"\n TOTAL CASES TO GENERATE: {total}\n")

    if not os.path.exists(engine_file):
        print(f"WARNING: Engine file '{engine_file}' not found! It will not be copied.")

    all_runs = []
    for i, (geo, vel, thick, dia, origin) in enumerate(combinations, 1):
        run_id = f"run{i}"
        run_dir = os.path.join(output_root, run_id)
        os.makedirs(run_dir, exist_ok=True)

        out_file = os.path.join(run_dir, os.path.basename(base_file))
        rwall_cfg = {}
        if dia is not None:
            rwall_cfg["diameter"] = dia
        if origin is not None:
            rwall_cfg["origin"] = origin
        if not rwall_cfg:
            rwall_cfg = None

        modify_radioss_file(
            input_path=base_file,
            output_path=out_file,
            geo_scales=geo,
            thick_scale=thick,
            velocity_vector=vel,
            rwall_updates=rwall_cfg,
        )

        if os.path.exists(engine_file):
            shutil.copy(engine_file, run_dir)

        metadata = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "geometry_scale": geo,
                "velocity_vector": vel,
                "thickness_scale": thick,
                "rwall_diameter": dia,
                "rwall_origin": origin,
            },
        }
        with open(os.path.join(run_dir, f"{run_id}.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        all_runs.append(metadata)
        print(
            f"[{i}/{total}] {run_id} | Geo:{geo} | Vel:{vel} "
            f"| Thk:{thick} | Wall:{origin}"
        )

    print("\n--- GENERATING SUMMARIES ---")
    save_summaries(output_root, all_runs, _flatten_row)


# Default bumper-beam DoE: 5 geo x 3 vel x 3 thk x 1 diameter x 3 origins = 135.
DEFAULT_EXPERIMENT_SETUP = {
    "geometry_scales": [
        (1.0, 1.0, 1.0),
        (1.0, 0.5, 1.0),
        (1.0, 1.0, 0.5),
        (1.0, 2.0, 1.0),
        (1.0, 1.0, 2.0),
    ],
    "velocities": [
        (-5.0, 0.0, 0.0),
        (-3.0, 0.0, 0.0),
        (-7.0, 0.0, 0.0),
    ],
    "thickness_scales": [1.0, 0.7, 1.3],
    "rwall_diameters": [254.0],
    "rwall_origins": [
        (-170.0, 0.0, 0.0),
        (-170.0, 120.0, 0.0),
        (-170.0, 240.0, 0.0),
    ],
}


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(script_dir, "templates")

    STARTER_FILE = os.path.join(templates_dir, "Bumper_Beam_AP_meshed_0000.rad")
    ENGINE_FILE = os.path.join(templates_dir, "Bumper_Beam_AP_meshed_0001.rad")
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
