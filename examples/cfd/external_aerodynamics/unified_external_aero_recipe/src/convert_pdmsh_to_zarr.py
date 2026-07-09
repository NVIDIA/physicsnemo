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
Convert curated ``.pdmsh`` DomainMesh cases to the mesh-zarr schema.

Thin CLI over :func:`physicsnemo.datapipes.save_domain_mesh_to_zarr`: each
``run_*/domain_*.pdmsh`` case is written as one full-DomainMesh zarr group
(case-level ``global_data/`` + ``interior/`` + ``boundaries/<name>/``),
readable by ``ZarrMeshReader`` via the ``drivaer_ml_surface_zarr`` dataset
config. Nothing is baked in: case metadata stays single-sourced at the
group root and is merged at read time.

This is a transitional tool for datasets that were already curated to
``.pdmsh``. For new curation, prefer emitting this schema directly from
PhysicsNeMo-Curator (``DomainMeshZarrSink``) so raw CFD data converts to
zarr in one hop.

With ``--soup-boundaries`` (recommended for DrivAerML surface training)
boundary meshes are denormalized to a per-cell vertex soup so training-time
block subsamples read contiguously; the verified ``layout`` attr lets the
reader skip the ``cells`` arrays entirely.

Usage:
    python src/convert_pdmsh_to_zarr.py \
        --input <dataset dir with run_*/> --output <zarr dataset dir> \
        [--runs 8] [--soup-boundaries] [--chunk-cells 200000] [--no-compress]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from physicsnemo.datapipes import save_domain_mesh_to_zarr, to_cell_soup
from physicsnemo.mesh import DomainMesh, Mesh


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--input", type=Path, required=True,
                    help="dataset root containing run_*/domain_*.pdmsh")
    ap.add_argument("--output", type=Path, required=True,
                    help="output directory for <run>.zarr groups")
    ap.add_argument("--runs", type=int, default=None,
                    help="convert only the first N runs (numeric order)")
    ap.add_argument("--soup-boundaries", action="store_true",
                    help="denormalize boundary meshes to per-cell vertex "
                         "soup for contiguous block reads")
    ap.add_argument("--extra-boundary", action="append", default=[],
                    metavar="NAME=GLOB",
                    help="add a sibling .pmsh mesh (glob relative to the run "
                         "dir) as an extra boundary, stored indexed at full "
                         "resolution and never souped -- e.g. "
                         "stl_geometry='*_single_solid.stl.pmsh' for exact "
                         "SDF queries (mirrors DomainMeshReader's "
                         "extra_boundaries)")
    ap.add_argument("--chunk-cells", type=int, default=200_000,
                    help="zarr chunk length on the cell axis; align with "
                         "the reader's subsample_n_cells")
    ap.add_argument("--no-compress", action="store_true")
    args = ap.parse_args()

    extra_specs: list[tuple[str, str]] = []
    for spec in args.extra_boundary:
        name, _, pattern = spec.partition("=")
        if not name or not pattern:
            raise SystemExit(f"--extra-boundary expects NAME=GLOB, got {spec!r}")
        extra_specs.append((name, pattern))

    def _run_key(d: Path) -> tuple:
        # Numeric runs (run_42) sort numerically; anything else (run_ref,
        # run_0001_baseline) sorts lexicographically after them instead of
        # crashing the conversion.
        tail = d.name.split("_", 1)[1] if "_" in d.name else d.name
        return (0, int(tail), "") if tail.isdigit() else (1, 0, d.name)

    run_dirs = sorted(
        (d for d in args.input.glob("run_*") if d.is_dir()),
        key=_run_key,
    )
    if args.runs is not None:
        run_dirs = run_dirs[: args.runs]
    if not run_dirs:
        raise SystemExit(f"no run_* directories under {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    for i, run in enumerate(run_dirs):
        pdmsh_candidates = sorted(run.glob("domain_*.pdmsh"))
        if not pdmsh_candidates:
            print(f"  {run.name}: no domain_*.pdmsh, skipping")
            continue
        out_group = args.output / f"{run.name}.zarr"
        if out_group.exists():
            print(f"  {run.name}: {out_group.name} exists, skipping")
            continue
        domain = DomainMesh.load(str(pdmsh_candidates[0]))

        boundaries = {
            name: to_cell_soup(domain.boundaries[name]) if args.soup_boundaries
            else domain.boundaries[name]
            for name in domain.boundary_names
        }
        # Extra boundaries: full resolution, indexed (exact geometry for
        # SDF-style queries), never souped.
        for name, pattern in extra_specs:
            matches = sorted(run.glob(pattern))
            if not matches:
                raise FileNotFoundError(
                    f"no mesh matching {pattern!r} in {run} for extra "
                    f"boundary {name!r}"
                )
            boundaries[name] = Mesh.load(str(matches[0]))

        domain = DomainMesh(
            interior=domain.interior,
            boundaries=boundaries,
            global_data=domain.global_data,
        )
        save_domain_mesh_to_zarr(
            domain,
            out_group,
            chunk_cells=args.chunk_cells,
            chunk_points=3 * args.chunk_cells,
            compress=not args.no_compress,
        )
        print(f"  {run.name} -> {out_group.name} "
              f"(interior {domain.interior.n_points:,} pts, boundaries: "
              f"{', '.join(domain.boundary_names)}) [{i + 1}/{len(run_dirs)}]",
              flush=True)


if __name__ == "__main__":
    main()
