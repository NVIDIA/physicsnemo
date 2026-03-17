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
Benchmark the PNM mesh datapipe for Ahmed body data loading.

Builds two separate DataLoader pipelines (surface + volume) entirely from
Hydra configuration using ``MeshReader`` -> ``MeshDataset`` -> ``DataLoader``.

Usage::

    python benchmark_datapipe.py --config-name benchmark data_dir=/path/to/data
    python benchmark_datapipe.py --config-name benchmark data_dir=/path/to/data max_samples=100
"""

from __future__ import annotations

import statistics
import time

import hydra
from omegaconf import DictConfig, OmegaConf

from physicsnemo.datapipes import DataLoader, FunctionCollator, MeshDataset


def _report_times(times: list[float], label: str) -> None:
    """Print summary statistics for a list of timings."""
    if not times:
        print(f"  {label}: no samples")
        return

    total = sum(times)
    mean = statistics.mean(times)
    med = statistics.median(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    p95 = sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else times[0]

    print(
        f"  {label} ({len(times)} samples): total={total:.2f}s  mean={mean:.4f}s  "
        f"median={med:.4f}s  std={std:.4f}s  p95={p95:.4f}s  "
        f"min={min(times):.4f}s  max={max(times):.4f}s"
    )


def benchmark_pipeline(name: str, dataloader: DataLoader, max_samples: int) -> None:
    """Benchmark a single DataLoader pipeline."""
    n_total = len(dataloader.dataset)
    n_batches = min(max_samples, len(dataloader))

    print("=" * 60)
    print(f"{name}  ({n_total} samples, benchmarking {n_batches} batches)")
    print("=" * 60)

    # Warmup
    warmup_iter = iter(dataloader)
    for _ in range(min(3, n_batches)):
        try:
            _ = next(warmup_iter)
        except StopIteration:
            break
    del warmup_iter

    times = []
    t0 = time.perf_counter()
    for i, batch in enumerate(dataloader):
        dt = time.perf_counter() - t0
        times.append(dt)

        if i == 0:
            data = batch[0] if isinstance(batch, tuple) else batch
            if hasattr(data, "points"):
                print(
                    f"  points={tuple(data.points.shape)}, "
                    f"cells={tuple(data.cells.shape)}, "
                    f"point_data={list(data.point_data.keys())}, "
                    f"cell_data={list(data.cell_data.keys())}"
                )
            elif hasattr(data, "keys"):
                for key in data.keys():
                    val = data[key]
                    if hasattr(val, "shape"):
                        print(f"  {key}: shape={tuple(val.shape)}")

        if i + 1 >= max_samples:
            break

        t0 = time.perf_counter()

    total_time = sum(times)
    if total_time > 0:
        print(f"  throughput: {len(times) / total_time:.2f} samples/sec")

    _report_times(times, name)
    print()


@hydra.main(version_base=None, config_path="conf", config_name="benchmark")
def main(cfg: DictConfig) -> None:
    print("Datapipe config:")
    print(OmegaConf.to_yaml(cfg))

    max_samples = cfg.get("max_samples", 50)

    for pipeline_name in ("surface", "volume"):
        pipeline_cfg = cfg[pipeline_name]

        reader = hydra.utils.instantiate(pipeline_cfg.reader)

        transforms = None
        if "transforms" in pipeline_cfg and pipeline_cfg.transforms:
            transforms = [hydra.utils.instantiate(t) for t in pipeline_cfg.transforms]

        dataset = MeshDataset(reader, transforms=transforms)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=FunctionCollator(lambda samples: samples[0]),
            use_streams=False,
            prefetch_factor=0,
        )

        benchmark_pipeline(pipeline_name, dataloader, max_samples)


if __name__ == "__main__":
    main()
