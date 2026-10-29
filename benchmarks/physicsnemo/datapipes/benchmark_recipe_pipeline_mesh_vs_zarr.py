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
Full-pipeline benchmark: unified_external_aero_recipe surface datapipe,
.pdmsh/MeshReader (shipping) vs zarr/ZarrMeshReader (new), on real DrivAerML.

Unlike benchmark_mesh_vs_zarr.py (reader-only), this times what the training
loop actually consumes: the recipe's own ``build_dataset`` output -- reader,
subsample, and the complete transform stack (DropMeshFields, CenterMesh,
NonDimensionalizeByMetadata, Rename/NormalizeMeshFields,
ComputeSurfaceNormals, SubsampleMesh, MeshToDomainMesh) producing a
DomainMesh per sample. It also times one GeoTransolver forward(+backward) on
GPU for context: whether datapipe latency can hide behind the training step.

Prerequisites:
  - a run-subset directory of the .pdmsh dataset (symlinked run_* dirs)
  - the matching zarr conversion (src/convert_pdmsh_to_zarr.py)

Usage:
    PYTHONPATH=. python benchmarks/physicsnemo/datapipes/\
benchmark_recipe_pipeline_mesh_vs_zarr.py \
        --pdmsh-dir .bench_data/pdmsh_8runs \
        --zarr-dir .bench_data/drivaer_ml_surface_zarr \
        [--workers 1 8] [--sampling-resolution 200000]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[3]
RECIPE = REPO / "examples/cfd/external_aerodynamics/unified_external_aero_recipe"

sys.path.insert(0, str(REPO / "benchmarks/physicsnemo/datapipes"))
sys.path.insert(0, str(RECIPE / "src"))

from benchmark_mesh_vs_zarr import evict_pages, store_size_mb  # noqa: E402


def build(dataset_yaml: str, path_overrides: dict, sampling_resolution: int):
    import datasets as recipe_datasets  # registers ${dp:} + recipe components

    paths = OmegaConf.load(RECIPE / "datasets/dataset_paths.yaml")
    for key, value in path_overrides.items():
        paths[key] = str(value)
    cfg = OmegaConf.merge(
        OmegaConf.create(
            {"dataset_paths": paths, "sampling_resolution": sampling_resolution}
        ),
        OmegaConf.load(RECIPE / f"datasets/{dataset_yaml}"),
    )
    return recipe_datasets.build_dataset(cfg, augment=False, device="cpu")


def touch_domain(domain) -> float:
    total = float(domain.interior.points.sum())
    for k in domain.interior.point_data.keys():
        total += float(domain.interior.point_data[k].sum())
    vehicle = domain.boundaries["vehicle"]
    total += float(vehicle.points.sum())
    for k in vehicle.cell_data.keys():
        total += float(vehicle.cell_data[k].sum())
    return total


def run_pass(dataset, order: list[int], workers: int) -> dict:
    def task(i: int) -> float:
        t0 = time.perf_counter()
        domain, _meta = dataset[i]
        touch_domain(domain)
        return time.perf_counter() - t0

    t0 = time.perf_counter()
    if workers == 1:
        latencies = [task(i) for i in order]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            latencies = list(pool.map(task, order))
    wall = time.perf_counter() - t0
    return {
        "wall_s": wall,
        "samples_per_s": len(order) / wall,
        "latency_p50_ms": 1e3 * statistics.median(latencies),
        "latency_p95_ms": 1e3
        * sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)],
    }


def gpu_step_time_ms(domain, out_dim: int = 4) -> float:
    """One GeoTransolver forward+backward, for datapipe-vs-step context."""
    import hydra.utils
    from collate import build_collate_fn

    model_cfg = OmegaConf.merge(
        OmegaConf.create({"out_dim": out_dim}),
        OmegaConf.load(RECIPE / "conf/model/geotransolver_surface.yaml"),
    )
    collate = build_collate_fn(
        model_cfg.input_type,
        OmegaConf.to_container(model_cfg.forward_kwargs, resolve=True),
        {"pressure": "scalar", "wss": "vector"},
    )
    batch = collate([(domain, {})])
    fk = {
        k: (v.cuda() if torch.is_tensor(v) else [x.cuda() for x in v])
        for k, v in batch["forward_kwargs"].items()
    }
    model = hydra.utils.instantiate(model_cfg.model, _convert_="partial").cuda()
    for _ in range(3):  # warmup
        model(**fk).sum().backward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 10
    for _ in range(n):
        model(**fk).sum().backward()
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t0) / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdmsh-dir", type=Path, required=True)
    ap.add_argument("--zarr-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--sampling-resolution", type=int, default=200_000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    variants = {
        "pdmsh (MeshReader, shipping)": (
            "drivaer_ml_surface.yaml",
            {"drivaer_ml": args.pdmsh_dir.resolve()},
            args.pdmsh_dir,
        ),
        "zarr (ZarrMeshReader, new)": (
            "drivaer_ml_surface_zarr.yaml",
            {"drivaer_ml_zarr": args.zarr_dir.resolve()},
            args.zarr_dir,
        ),
    }

    gen = torch.Generator().manual_seed(args.seed)
    step_ms = None
    for label, (yaml_name, overrides, store) in variants.items():
        dataset = build(yaml_name, overrides, args.sampling_resolution)
        n = len(dataset)
        print(f"\n=== {label}  ({n} samples, store {store_size_mb(store):.0f} MB) ===")
        for workers in args.workers:
            order = [int(x) for x in torch.randperm(n, generator=gen)]
            evict_pages(store)
            cold = run_pass(dataset, order, workers)
            warm = min(
                (
                    run_pass(
                        dataset,
                        [int(x) for x in torch.randperm(n, generator=gen)],
                        workers,
                    )
                    for _ in range(args.epochs)
                ),
                key=lambda r: r["wall_s"],
            )
            print(
                f"  workers={workers:<2d} "
                f"cold: {cold['samples_per_s']:6.2f} samp/s "
                f"(p50 {cold['latency_p50_ms']:7.1f} ms, "
                f"p95 {cold['latency_p95_ms']:7.1f} ms) | "
                f"warm: {warm['samples_per_s']:6.2f} samp/s "
                f"(p50 {warm['latency_p50_ms']:7.1f} ms)"
            )
        if step_ms is None and torch.cuda.is_available():
            domain, _ = dataset[0]
            step_ms = gpu_step_time_ms(domain)
            print(
                f"\n[context] GeoTransolver fwd+bwd GPU step: {step_ms:.1f} ms "
                f"({1e3 / step_ms:.1f} steps/s) -- datapipe must beat this "
                f"per-GPU to stay hidden"
            )


if __name__ == "__main__":
    main()
