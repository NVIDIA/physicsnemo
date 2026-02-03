# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
ASV benchmarks for the knn function.
"""

import torch

from physicsnemo.nn.functional import knn


class KNNBenchmark:
    """Benchmark suite for the knn function."""

    bench_params = {
        "n_points": [10000, 100000],
        "n_queries": [1000, 5000],
        "k": [5, 16],
        "implementation": ["cuml", "scipy", "torch"],
    }

    # ASV benchmark attributes.
    # https://asv.readthedocs.io/en/latest/benchmarks.html#benchmark-attributes
    params = list(bench_params.values())
    param_names = list(bench_params.keys())

    # Timeout for each benchmark (seconds).
    timeout = 60

    def setup(self, n_points: int, n_queries: int, k: int, implementation: str) -> None:
        """Set up test data for the benchmark."""
        # CUDA is required for the cuML implementation.
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        # Determine device based on implementation.
        if implementation in ["cuml", "torch"]:
            self.device = "cuda"
        elif implementation == "scipy":
            self.device = "cpu"
        else:
            raise ValueError(f"Invalid implementation: {implementation}")

        # Generate random point clouds.
        self.points = torch.randn(n_points, 3, device=self.device, dtype=torch.float32)
        self.queries = torch.randn(
            n_queries, 3, device=self.device, dtype=torch.float32
        )
        self.k = k
        self.implementation = implementation

        if self.device == "cuda":
            torch.cuda.synchronize()

    def time_knn(
        self, n_points: int, n_queries: int, k: int, implementation: str
    ) -> None:
        """Benchmark the knn function execution time."""
        knn(self.points, self.queries, self.k, implementation=self.implementation)
        if self.device == "cuda":
            torch.cuda.synchronize()
