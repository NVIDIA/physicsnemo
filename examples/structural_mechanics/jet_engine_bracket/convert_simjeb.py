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

r"""Convert SimJEB (4-node tetrahedral OptiStruct results) to the layout of ``bracket_io.py``.

Reads, per design, the volume mesh ``SimJEB_volmesh_(vtk)/<id>.vtk``, the field table
``SimJEB_simresults_(csv)_*/<id>field.csv`` (rows in the order of the mesh nodes, with the
region of every node) and the deck ``SimJEB_fea_(fem)_*/<id>.fem`` (loads and material) of the
extracted dataset. Designs listed as excluded in ``splits/simjeb.json`` are skipped.

    python convert_simjeb.py --input <SimJEB>/extracted --output <data>/simjeb --workers 16
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from bracket_io import save_design
from nastran import LOADS, parse_deck, require


def _one(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    require(len(matches) == 1, f"Expected one file {pattern}, found {len(matches)}")
    return matches[0]


def convert(root: Path, output: Path, shape: int) -> int:
    r"""Convert one design; returns its id."""
    table = pd.read_csv(_one(root, f"SimJEB_simresults_(csv)_*/{shape}field.csv"))
    volume = pv.read(_one(root, f"SimJEB_volmesh_(vtk)/{shape}.vtk"))
    deck = parse_deck(_one(root, f"SimJEB_fea_(fem)_*/{shape}.fem"))
    require(
        bool(np.all(volume.celltypes == pv.CellType.TETRA)),
        "Expected 4-node tetrahedra",
    )
    tets = np.asarray(volume.cells).reshape(-1, 5)[:, 1:].astype(np.int64)
    points = np.asarray(volume.points, dtype=np.float64)
    require(
        len(points) == len(table)
        and np.abs(points - table[["x", "y", "z"]].to_numpy()).max() <= 1e-3,
        "Mesh and field table rows differ",
    )
    region = table["surf"].to_numpy(dtype=np.int64)
    ids = table["id"].to_numpy(dtype=np.int64)
    require(
        set(ids[region == 2]) == deck["supports"]
        and set(ids[region == 3]) == deck["loaded"],
        "Interface regions differ from the deck",
    )
    fields = {}
    for load in LOADS:
        displacement = table[[f"{load}_{a}disp" for a in "xyz"]].to_numpy(
            dtype=np.float64
        )
        stress = table[f"{load}_stress"].to_numpy(dtype=np.float64)
        require(
            np.isfinite(displacement).all()
            and np.isfinite(stress).all()
            and (stress >= 0).all(),
            f"Invalid field ({load})",
        )
        fields[load] = {"displacement": displacement, "von_mises": stress}
    save_design(output, shape, points, tets, region, fields, deck)
    return shape


def _job(args):
    root, output, shape = args
    try:
        return shape, convert(Path(root), Path(output), shape), None
    except Exception:  # noqa: BLE001 - reported per design
        return shape, None, traceback.format_exc().strip().splitlines()[-1]


def main() -> None:
    r"""Convert every SimJEB design to one DomainMesh per load case (command line)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="extracted SimJEB folder (the Dataverse archives unzipped)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", type=Path, default=Path(__file__).parent / "splits" / "simjeb.json"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--designs",
        type=int,
        nargs="*",
        help="design ids (default: every design of the split file)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    split = json.loads(args.split.read_text())
    excluded = {int(k) for k in split.get("excluded", {})}
    shapes = args.designs or sorted(
        int(v) for k in ("train", "val", "test") for v in split[k]
    )
    shapes = [
        s
        for s in shapes
        if s not in excluded
        and (
            args.overwrite
            or not (args.output / "samples" / f"shape_{s}" / "tor.pdmsh").exists()
        )
    ]
    failures = {}
    with ProcessPoolExecutor(
        args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(_job, (str(args.input), str(args.output), s)) for s in shapes
        ]
        for i, future in enumerate(as_completed(futures), 1):
            shape, _, error = future.result()
            if error:
                failures[shape] = error
            if i % 50 == 0 or i == len(shapes):
                print(f"{i}/{len(shapes)} designs, {len(failures)} failed", flush=True)
    if failures:
        (args.output / "conversion_failures.json").write_text(
            json.dumps(failures, indent=1) + "\n"
        )
        raise SystemExit(
            f"{len(failures)} designs failed, see {args.output / 'conversion_failures.json'}"
        )


if __name__ == "__main__":
    main()
