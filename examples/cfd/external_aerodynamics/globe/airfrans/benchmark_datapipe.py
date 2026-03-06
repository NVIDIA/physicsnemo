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
Benchmark the AirFRANS datapipe throughput via the physicsnemo DataLoader.

Instantiates the full pipeline from the Hydra config, wraps it in
physicsnemo.datapipes.DataLoader with the same collate as training, and
measures wall-clock time per batch over N iterations.

Usage
-----
    # Arrow reader (default; dataset_path from conf/config.yaml)
    python benchmark_datapipe.py

    # Override config from CLI
    python benchmark_datapipe.py dataset_path=/path/to/arrow +n_samples=50

    # VTK reader
    python benchmark_datapipe.py reader=vtk data_dir=/path/to/vtk
"""

from __future__ import annotations

import logging
import statistics
import time
from typing import Any, Sequence

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import SequentialSampler

from physicsnemo.datapipes import DataLoader as PhysicsnemoDataLoader
from physicsnemo.datapipes import Dataset as PhysicsnemoDataset
from tensordict import TensorDict

from physicsnemo_dataset import _structured_tensordict_to_airfrans_sample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_CUDA_AVAILABLE = torch.cuda.is_available()
_BYTES_PER_MB = 2**20


def collate_single(
    samples: Sequence[tuple[TensorDict, dict[str, Any]]],
):
    """Collate for batch_size=1: convert structured TensorDict to AirFRANSSample."""
    data, _ = samples[0]
    return _structured_tensordict_to_airfrans_sample(data)


def _gpu_memory_mb() -> dict[str, float] | None:
    """Return current GPU memory stats in MB, or None if CUDA not available."""
    if not _CUDA_AVAILABLE:
        return None
    torch.cuda.synchronize()
    return {
        "allocated_mb": torch.cuda.memory_allocated() / _BYTES_PER_MB,
        "reserved_mb": torch.cuda.memory_reserved() / _BYTES_PER_MB,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / _BYTES_PER_MB,
        "max_reserved_mb": torch.cuda.max_memory_reserved() / _BYTES_PER_MB,
    }


def benchmark(
    dataloader: PhysicsnemoDataLoader,
    n_samples: int,
) -> tuple[list[float], dict[str, float] | None]:
    dataset = dataloader.dataset
    actual_n = min(n_samples, len(dataset))
    if actual_n == 0:
        logger.warning("Dataset is empty — nothing to benchmark.")
        return [], _gpu_memory_mb() if _CUDA_AVAILABLE else None

    logger.info("Warming up (1 batch)...")
    _ = next(iter(dataloader))

    if _CUDA_AVAILABLE:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    logger.info("Timing %d batches...", actual_n)
    times: list[float] = []
    it = iter(dataloader)
    for _ in range(actual_n):
        t0 = time.perf_counter()
        _ = next(it)
        times.append(time.perf_counter() - t0)

    gpu_stats = _gpu_memory_mb()
    return times, gpu_stats


def print_results(
    times: list[float],
    gpu_stats: dict[str, float] | None = None,
) -> None:
    n = len(times)
    if n == 0:
        return
    total = sum(times)
    mean = statistics.mean(times)
    std = statistics.stdev(times) if n > 1 else 0.0
    median = statistics.median(times)
    throughput = n / total if total > 0 else 0.0

    header = (
        f"{'Samples':>8s} {'Total (s)':>10s} {'Mean (s)':>10s} "
        f"{'Median (s)':>11s} {'Std (s)':>10s} {'Throughput':>12s}"
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    print(
        f"{n:>8d} {total:>10.3f} {mean:>10.4f} "
        f"{median:>11.4f} {std:>10.4f} {throughput:>10.2f}/s"
    )
    print(sep)

    if gpu_stats is not None:
        print()
        print("GPU memory (peak during benchmark):")
        print(
            f"  max allocated: {gpu_stats['max_allocated_mb']:.2f} MB  "
            f"max reserved: {gpu_stats['max_reserved_mb']:.2f} MB"
        )
        print(
            f"  current allocated: {gpu_stats['allocated_mb']:.2f} MB  "
            f"current reserved: {gpu_stats['reserved_mb']:.2f} MB"
        )
    print()


@hydra.main(
    version_base=None,
    config_path="./conf",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    n_samples: int = cfg.get("n_samples", 100)

    print("=== AirFRANS Datapipe Benchmark ===")
    print()
    print(OmegaConf.to_yaml(cfg, resolve=True))

    logger.info("Building physicsnemo dataloader...")
    dataset: PhysicsnemoDataset = hydra.utils.instantiate(cfg.dataset)
    sampler = SequentialSampler(dataset)
    dataloader = PhysicsnemoDataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=collate_single,
    )
    logger.info("Dataset size: %d samples", len(dataset))

    times, gpu_stats = benchmark(dataloader, n_samples)
    print_results(times, gpu_stats)

    dataset.close()


if __name__ == "__main__":
    main()
