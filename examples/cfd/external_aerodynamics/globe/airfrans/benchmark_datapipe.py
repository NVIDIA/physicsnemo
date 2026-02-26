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
Benchmark the AirFRANS datapipe throughput with per-stage timing.

Measures wall-clock time for each pipeline stage independently:
  1. Reader-only (raw I/O)
  2. Each transform individually
  3. Full pipeline (reader + all transforms)

Usage
-----
    # Arrow reader
    python benchmark_datapipe.py --reader arrow \\
        --dataset-path data/airfrans_dataset --n-samples 10

    # VTK reader
    python benchmark_datapipe.py --reader vtk \\
        --data-dir /path/to/vtk --n-samples 10
"""

from __future__ import annotations

import argparse
import statistics
import time

from physicsnemo.datapipes import Dataset

from pipeline import (
    AirFRANSArrowReader,
    AirFRANSVTKReader,
    ComputeAirfoilNormals,
    ComputeForceCoefficients,
    ComputeFreestreamQuantities,
    ComputeGradients,
    NondimensionalizeFields,
    PatchNonPhysicalValues,
)


def build_reader(args: argparse.Namespace):
    if args.reader == "arrow":
        if args.dataset_path is None:
            raise ValueError("--dataset-path is required for the arrow reader")
        return AirFRANSArrowReader(
            dataset_path=args.dataset_path,
            task=args.task,
            split=args.split,
        )
    elif args.reader == "vtk":
        if args.data_dir is None:
            raise ValueError("--data-dir is required for the vtk reader")
        return AirFRANSVTKReader(
            data_dir=args.data_dir,
            task=args.task,
            split=args.split,
        )
    else:
        raise ValueError(f"Unknown reader: {args.reader}")


TRANSFORM_STAGES = [
    ("ComputeGradients", ComputeGradients),
    ("ComputeAirfoilNormals", ComputeAirfoilNormals),
    ("ComputeFreestreamQty", ComputeFreestreamQuantities),
    ("NondimensionalizeFields", NondimensionalizeFields),
    ("ComputeForceCoeffs", ComputeForceCoefficients),
    ("PatchNonPhysical", PatchNonPhysicalValues),
]


def build_transforms():
    return [cls() if cls != PatchNonPhysicalValues else cls(threshold=1.02)
            for _, cls in TRANSFORM_STAGES]


def time_reader(reader, n_samples: int) -> list[float]:
    actual_n = min(n_samples, len(reader))
    _ = reader[0]

    times: list[float] = []
    for i in range(actual_n):
        t0 = time.perf_counter()
        _ = reader[i]
        times.append(time.perf_counter() - t0)
    return times


def time_per_stage(reader, n_samples: int) -> dict[str, list[float]]:
    """Time each transform stage independently across n_samples."""
    actual_n = min(n_samples, len(reader))
    transforms = build_transforms()

    # Warm up
    data, _ = reader[0]
    for t in transforms:
        data = t(data)

    stage_times: dict[str, list[float]] = {name: [] for name, _ in TRANSFORM_STAGES}

    for i in range(actual_n):
        data, _ = reader[i]
        for (name, _), transform in zip(TRANSFORM_STAGES, transforms):
            t0 = time.perf_counter()
            data = transform(data)
            stage_times[name].append(time.perf_counter() - t0)

    return stage_times


def time_full_pipeline(dataset: Dataset, n_samples: int) -> list[float]:
    actual_n = min(n_samples, len(dataset))
    _ = dataset[0]

    times: list[float] = []
    for i in range(actual_n):
        t0 = time.perf_counter()
        _ = dataset[i]
        times.append(time.perf_counter() - t0)
    return times


def print_table(rows: list[tuple[str, list[float]]]) -> None:
    header = (
        f"{'Stage':<30s} {'Samples':>7s} {'Total (s)':>10s} "
        f"{'Mean (s)':>10s} {'Std (s)':>10s} {'Throughput':>12s}"
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for label, times in rows:
        n = len(times)
        if n == 0:
            continue
        total = sum(times)
        mean = statistics.mean(times)
        std = statistics.stdev(times) if n > 1 else 0.0
        throughput = n / total if total > 0 else 0.0
        print(
            f"{label:<30s} {n:>7d} {total:>10.3f} {mean:>10.3f} "
            f"{std:>10.3f} {throughput:>10.2f}/s"
        )
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark AirFRANS datapipe throughput"
    )
    parser.add_argument(
        "--reader",
        choices=["arrow", "vtk"],
        required=True,
        help="Which reader to benchmark",
    )
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--task", type=str, default="full")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of samples to time (clamped to dataset size)",
    )
    args = parser.parse_args()

    print(f"=== AirFRANS Datapipe Benchmark ===")
    print(f"  Reader:    {args.reader}")
    print(f"  Task:      {args.task}")
    print(f"  Split:     {args.split}")
    print(f"  N samples: {args.n_samples}")
    print()

    reader = build_reader(args)
    print(f"Reader: {reader}")
    print(f"Dataset size: {len(reader)} samples")
    print()

    # --- Reader-only ---
    print("Timing reader-only...")
    reader_times = time_reader(reader, args.n_samples)

    # --- Per-stage ---
    print("Timing per-stage transforms...")
    stage_times = time_per_stage(reader, args.n_samples)

    # --- Full pipeline ---
    print("Timing full pipeline...")
    transforms = build_transforms()
    dataset = Dataset(reader=reader, transforms=transforms, device="cpu")
    pipeline_times = time_full_pipeline(dataset, args.n_samples)

    # --- Summary ---
    rows: list[tuple[str, list[float]]] = [
        ("Reader-only (I/O)", reader_times),
    ]
    for name, _ in TRANSFORM_STAGES:
        rows.append((f"  + {name}", stage_times[name]))
    rows.append(("Full pipeline", pipeline_times))

    print_table(rows)

    dataset.close()


if __name__ == "__main__":
    main()
