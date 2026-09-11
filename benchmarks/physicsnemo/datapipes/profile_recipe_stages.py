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
Stage-level profile of the unified_external_aero_recipe surface datapipe.

Mirrors MeshDataset._load exactly -- reader -> .to(cuda) -> transforms (GPU)
-- then collate + GeoTransolver fwd/bwd, timing every stage with CUDA sync
for attribution. Profiles both the .pdmsh and zarr variants, plus a
reader-internal breakdown for the zarr path (cells / points / fields reads).

Usage:
    PYTHONPATH=. python benchmarks/physicsnemo/datapipes/profile_recipe_stages.py \
        --pdmsh-dir .bench_data/pdmsh_8runs \
        --zarr-dir .bench_data/drivaer_ml_surface_zarr
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[3]
RECIPE = REPO / "examples/cfd/external_aerodynamics/unified_external_aero_recipe"
sys.path.insert(0, str(REPO / "benchmarks/physicsnemo/datapipes"))
sys.path.insert(0, str(RECIPE / "src"))

from benchmark_mesh_vs_zarr import evict_pages  # noqa: E402


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class StageClock:
    def __init__(self) -> None:
        self.times: dict[str, list[float]] = defaultdict(list)

    def stage(self, name: str, fn):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        self.times[name].append(1e3 * (time.perf_counter() - t0))
        return out

    def report(self, title: str, skip_first: bool = True) -> None:
        print(f"\n{title}")
        total = 0.0
        for name, vals in self.times.items():
            warm = vals[1:] if skip_first and len(vals) > 1 else vals
            med = statistics.median(warm)
            total += med
            cold = vals[0]
            print(f"  {name:<28s} median {med:8.2f} ms   (first/cold {cold:8.2f} ms)")
        print(f"  {'TOTAL (median stages)':<28s} {total:15.2f} ms")


def build_parts(yaml_name: str, path_overrides: dict, sampling_resolution: int):
    """Reader + transform list, via the recipe's own helpers."""
    import datasets as recipe_datasets
    import hydra.utils

    paths = OmegaConf.load(RECIPE / "datasets/dataset_paths.yaml")
    for k, v in path_overrides.items():
        paths[k] = str(v)
    cfg = OmegaConf.merge(
        OmegaConf.create(
            {"dataset_paths": paths, "sampling_resolution": sampling_resolution}
        ),
        OmegaConf.load(RECIPE / f"datasets/{yaml_name}"),
    )
    reader = hydra.utils.instantiate(cfg.pipeline.reader)
    target_names = list(OmegaConf.to_container(cfg.targets, resolve=True))
    transforms = []
    for t in cfg.pipeline.transforms:
        t = recipe_datasets._resolve_transform_paths(t, RECIPE)
        t = recipe_datasets._maybe_inject_targets(t, target_names)
        transforms.append(hydra.utils.instantiate(t))
    return reader, transforms


def build_model_and_collate():
    import hydra.utils
    from collate import build_collate_fn

    model_cfg = OmegaConf.merge(
        OmegaConf.create({"out_dim": 4}),
        OmegaConf.load(RECIPE / "conf/model/geotransolver_surface.yaml"),
    )
    collate = build_collate_fn(
        model_cfg.input_type,
        OmegaConf.to_container(model_cfg.forward_kwargs, resolve=True),
        {"pressure": "scalar", "wss": "vector"},
    )
    model = hydra.utils.instantiate(model_cfg.model, _convert_="partial").cuda()
    return model, collate


def profile_variant(label, yaml_name, overrides, store, model, collate, epochs, res):
    from physicsnemo.mesh import DomainMesh

    reader, transforms = build_parts(yaml_name, overrides, res)
    clock = StageClock()
    evict_pages(store)
    n = len(reader)
    for _ in range(epochs):
        for i in range(n):
            mesh = clock.stage("read (reader[i])", lambda: reader[i][0])
            mesh = clock.stage("H2D transfer (.to cuda)", lambda: mesh.to("cuda"))
            for t in transforms:
                name = type(t).__name__
                if isinstance(mesh, DomainMesh):
                    mesh = clock.stage(name, lambda: t.apply_to_domain(mesh))
                else:
                    mesh = clock.stage(name, lambda: t(mesh))
            batch = clock.stage("collate + fk resolve", lambda: collate([(mesh, {})]))
            fk = batch["forward_kwargs"]
            out = clock.stage("model forward", lambda: model(**fk))
            clock.stage("model backward", lambda: out.sum().backward())
            model.zero_grad(set_to_none=True)
    clock.report(f"=== {label} (median over {epochs * n - 1} warm samples) ===")
    return clock


def profile_zarr_reader_internals(zarr_dir: Path, res: int) -> None:
    """Break the ZarrMeshReader read into its component reads."""
    import tensorstore as ts

    group = sorted(zarr_dir.glob("*.zarr"))[0]
    handles = {
        name: ts.open(
            {
                "driver": "zarr3",
                "kvstore": {"driver": "file", "path": str(group / name)},
            },
            open=True,
        ).result()
        for name in (
            "points",
            "cells",
            "cell_data/pMeanTrim",
            "cell_data/wallShearStressMeanTrim",
            "cell_data/CpMeanTrim",
            "cell_data/pPrime2MeanTrim",
        )
    }
    n_cells = handles["cells"].shape[0]
    clock = StageClock()
    gen = torch.Generator().manual_seed(0)
    for _ in range(6):
        s = torch.randint(0, n_cells - res + 1, (1,), generator=gen).item()
        sl = slice(s, s + res)
        clock.stage("cells block (await)", lambda: handles["cells"][sl].read().result())
        clock.stage(
            "points range",
            lambda: handles["points"][3 * s : 3 * (s + res)].read().result(),
        )
        clock.stage(
            "4 fields (concurrent)",
            lambda: [
                f.result()
                for f in [
                    handles["cell_data/pMeanTrim"][sl].read(),
                    handles["cell_data/wallShearStressMeanTrim"][sl].read(),
                    handles["cell_data/CpMeanTrim"][sl].read(),
                    handles["cell_data/pPrime2MeanTrim"][sl].read(),
                ]
            ],
        )
    clock.report("=== zarr reader internals (warm, sequential for attribution) ===")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdmsh-dir", type=Path, required=True)
    ap.add_argument("--zarr-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--sampling-resolution", type=int, default=200_000)
    args = ap.parse_args()

    model, collate = build_model_and_collate()

    profile_variant(
        "pdmsh (MeshReaderWithGlobalData)",
        "drivaer_ml_surface.yaml",
        {"drivaer_ml": args.pdmsh_dir.resolve()},
        args.pdmsh_dir,
        model,
        collate,
        args.epochs,
        args.sampling_resolution,
    )
    profile_variant(
        "zarr (ZarrMeshReader)",
        "drivaer_ml_surface_zarr.yaml",
        {"drivaer_ml_zarr": args.zarr_dir.resolve()},
        args.zarr_dir,
        model,
        collate,
        args.epochs,
        args.sampling_resolution,
    )
    profile_zarr_reader_internals(args.zarr_dir, args.sampling_resolution)


if __name__ == "__main__":
    main()
