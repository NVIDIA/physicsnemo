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

r"""Convert DeepJEB (10-node tetrahedral OptiStruct results) to the layout of ``bracket_io.py``.

Reads, per design ``A_B``, the deck ``SolverDeck/A_B.fem`` (mesh, interfaces, loads, material)
and the field table ``Field/A_B.csv`` (node coordinates, displacement and signed von Mises
stress of the four load cases) of the extracted dataset. The design id is ``1000 * A + B``.

    python convert_deepjeb.py --input <DeepJEB>/extracted --output <data>/deepjeb --workers 16
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
from bracket_io import save_design
from nastran import LOADS, parse_deck, require


def design_id(name: str) -> int:
    r"""Integer id ``1000 * A + B`` of DeepJEB design ``A_B``."""
    a, b = (int(v) for v in name.split("_"))
    require(0 <= b < 1000, f"Unexpected design name {name}")
    return 1000 * a + b


def convert(root: Path, output: Path, name: str) -> int:
    r"""Convert one design; returns its id."""
    deck = parse_deck(root / "SolverDeck" / f"{name}.fem")
    table = pd.read_csv(root / "Field" / f"{name}.csv")
    ids = table["nodeID"].to_numpy(dtype=np.int64)
    require(
        set(ids) == set(deck["grids"]) - deck["references"],
        "Field nodes differ from the mesh nodes of the deck",
    )
    points = table[["coord_x(mm)", "coord_y(mm)", "coord_z(mm)"]].to_numpy(
        dtype=np.float64
    )
    require(
        np.abs(points - np.array([deck["grids"][int(i)] for i in ids])).max() <= 1e-3,
        "Field and deck coordinates differ",
    )
    index = np.full(max(deck["grids"]) + 1, -1, dtype=np.int64)
    index[ids] = np.arange(len(ids))
    tets = index[deck["tets"]]
    require(
        tets.shape[1] == 10 and (tets >= 0).all(),
        "Expected 10-node tetrahedra on field nodes",
    )
    region = np.zeros(len(ids), dtype=np.int64)
    corner_faces = np.concatenate(
        [
            tets[:, [0, 1, 2, 4, 5, 6]],
            tets[:, [0, 1, 3, 4, 8, 7]],
            tets[:, [0, 2, 3, 6, 9, 7]],
            tets[:, [1, 2, 3, 5, 9, 8]],
        ]
    )
    _, first, counts = np.unique(
        np.sort(corner_faces[:, :3], axis=1),
        axis=0,
        return_index=True,
        return_counts=True,
    )
    region[np.unique(corner_faces[first[counts == 1]])] = 1
    region[np.isin(ids, list(deck["supports"]))] = 2
    region[np.isin(ids, list(deck["loaded"]))] = 3
    fields = {}
    for load in LOADS:
        displacement = table[[f"{load}_{a}_disp(mm)" for a in "xyz"]].to_numpy(
            dtype=np.float64
        )
        stress = np.abs(
            table[f"{load}_stress(MPa)"].to_numpy(dtype=np.float64)
        )  # the table holds a signed von Mises stress
        require(
            np.isfinite(displacement).all() and np.isfinite(stress).all(),
            f"Non-finite field ({load})",
        )
        fields[load] = {"displacement": displacement, "von_mises": stress}
    save_design(output, design_id(name), points, tets, region, fields, deck)
    return design_id(name)


def _job(args):
    root, output, name = args
    try:
        return name, convert(Path(root), Path(output), name), None
    except Exception:  # noqa: BLE001 - reported per design
        return name, None, traceback.format_exc().strip().splitlines()[-1]


def main() -> None:
    r"""Convert every DeepJEB design to one DomainMesh per load case (command line)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="extracted DeepJEB folder with SolverDeck/ and Field/",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--designs", nargs="*", help="design names A_B (default: every deck)"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    names = args.designs or sorted(
        p.stem for p in (args.input / "SolverDeck").glob("*.fem")
    )
    if not args.overwrite:
        names = [
            n
            for n in names
            if not (
                args.output / "samples" / f"shape_{design_id(n)}" / "tor.pdmsh"
            ).exists()
        ]
    failures = {}
    with ProcessPoolExecutor(
        args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(_job, (str(args.input), str(args.output), n)) for n in names
        ]
        for i, future in enumerate(as_completed(futures), 1):
            name, _, error = future.result()
            if error:
                failures[name] = error
            if i % 100 == 0 or i == len(names):
                print(f"{i}/{len(names)} designs, {len(failures)} failed", flush=True)
    if failures:
        (args.output / "conversion_failures.json").write_text(
            json.dumps(failures, indent=1) + "\n"
        )
        raise SystemExit(
            f"{len(failures)} designs failed, see {args.output / 'conversion_failures.json'}"
        )


if __name__ == "__main__":
    main()
