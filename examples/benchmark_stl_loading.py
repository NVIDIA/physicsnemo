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
Benchmark script for STL loading performance comparison.

This script compares the performance of:
- trimesh (baseline)
- Fast Rust reader (if available)
- GPU acceleration (if available)

Usage:
    python benchmark_stl_loading.py /path/to/stl/directory
"""

import time
from pathlib import Path

import numpy as np


def benchmark_stl_loading(stl_dir: Path, n_files: int = 100):
    """
    Benchmark STL loading performance.

    Parameters
    ----------
    stl_dir : Path
        Directory containing STL files.
    n_files : int
        Number of files to process (limits test to first N files).
    """
    print(f"=== STL Loading Benchmark ===")
    print(f"Directory: {stl_dir}")
    print(f"Files to process: {n_files}\n")

    # Get list of STL files
    stl_files = sorted(stl_dir.glob("*.stl"))[:n_files]
    if not stl_files:
        print("No STL files found!")
        return

    actual_n = len(stl_files)
    print(f"Found {actual_n} STL files\n")

    # Benchmark 1: trimesh (baseline)
    print("[1/3] Benchmarking trimesh...")
    from physicsnemo.experimental.guardrails.geometry import load_features_from_dir

    start = time.time()
    features_trimesh, _ = load_features_from_dir(
        stl_dir, n_workers=8, use_fast_reader=False
    )
    time_trimesh = time.time() - start
    print(f"  Time: {time_trimesh:.2f}s ({actual_n / time_trimesh:.1f} files/sec)")
    print(f"  Features shape: {features_trimesh[0].shape}\n")

    # Benchmark 2: Fast Rust reader (if available)
    try:
        from physicsnemo.experimental.guardrails.geometry import (
            is_fast_reader_available,
        )

        if is_fast_reader_available():
            print("[2/3] Benchmarking fast Rust reader...")
            start = time.time()
            features_fast, _ = load_features_from_dir(
                stl_dir, n_workers=8, use_fast_reader=True
            )
            time_fast = time.time() - start
            speedup = time_trimesh / time_fast
            print(
                f"  Time: {time_fast:.2f}s ({actual_n / time_fast:.1f} files/sec)"
            )
            print(f"  Speedup: {speedup:.2f}x faster than trimesh")

            # Verify consistency
            diff = np.abs(features_trimesh[0] - features_fast[0])
            print(f"  Max feature difference: {np.max(diff):.6f}\n")
        else:
            print(
                "[2/3] Fast Rust reader not available (install 'stlreader' package)\n"
            )
    except ImportError:
        print("[2/3] Fast Rust reader module not found\n")

    # Benchmark 3: GPU (if available)
    try:
        import torch

        if torch.cuda.is_available():
            print("[3/3] Benchmarking GPU density model...")
            from physicsnemo.experimental.guardrails import GeometryGuardrail

            # Fit on CPU
            X_train = np.vstack(features_trimesh)

            # CPU baseline
            start = time.time()
            model_cpu = GeometryGuardrail(n_components=2, device="cpu")
            model_cpu.density.fit(X_train)
            scores_cpu = model_cpu.density.score(X_train[:100])
            time_cpu = time.time() - start

            # GPU
            start = time.time()
            model_gpu = GeometryGuardrail(n_components=2, device="cuda")
            model_gpu.density.fit(X_train)
            scores_gpu = model_gpu.density.score(X_train[:100])
            time_gpu = time.time() - start

            speedup = time_cpu / time_gpu
            print(f"  CPU time: {time_cpu:.2f}s")
            print(f"  GPU time: {time_gpu:.2f}s")
            print(f"  Speedup: {speedup:.2f}x faster on GPU\n")
        else:
            print("[3/3] CUDA not available\n")
    except ImportError:
        print("[3/3] PyTorch not installed\n")

    # Summary
    print("=== Summary ===")
    print(f"Baseline (trimesh): {time_trimesh:.2f}s")
    print(
        "For best performance:\n"
        "  - Use fast Rust reader for I/O (5-10x speedup)\n"
        "  - Use GPU for large datasets (2-10x speedup)\n"
        "  - Combine both for maximum speed!"
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python benchmark_stl_loading.py /path/to/stl/directory")
        sys.exit(1)

    stl_dir = Path(sys.argv[1])
    if not stl_dir.exists():
        print(f"Directory not found: {stl_dir}")
        sys.exit(1)

    benchmark_stl_loading(stl_dir)
