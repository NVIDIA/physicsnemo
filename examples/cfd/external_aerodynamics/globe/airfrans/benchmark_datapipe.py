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
Benchmark the AirFRANS datapipe throughput.

Instantiates the full pipeline from the Hydra config and measures
wall-clock time per sample over N iterations.

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

import hydra
from omegaconf import DictConfig, OmegaConf

from physicsnemo.datapipes import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark(dataset: Dataset, n_samples: int) -> list[float]:
    actual_n = min(n_samples, len(dataset))
    if actual_n == 0:
        logger.warning("Dataset is empty — nothing to benchmark.")
        return []

    logger.info("Warming up (1 sample)...")
    _ = dataset[0]

    logger.info("Timing %d samples...", actual_n)
    times: list[float] = []
    for i in range(actual_n):
        t0 = time.perf_counter()
        _ = dataset[i]
        times.append(time.perf_counter() - t0)

    return times


def print_results(times: list[float]) -> None:
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
    print()


@hydra.main(
    version_base=None,
    config_path="./conf",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    n_samples: int = cfg.get("n_samples", 10)

    print("=== AirFRANS Datapipe Benchmark ===")
    print()
    print(OmegaConf.to_yaml(cfg, resolve=True))

    logger.info("Building dataset...")
    dataset: Dataset = hydra.utils.instantiate(cfg.dataset)
    logger.info("Dataset size: %d samples", len(dataset))

    times = benchmark(dataset, n_samples)
    print_results(times)

    dataset.close()


if __name__ == "__main__":
    main()
