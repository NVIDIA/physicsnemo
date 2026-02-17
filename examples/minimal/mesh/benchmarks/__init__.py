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

"""Benchmark utilities for PhysicsNeMo-Mesh tutorial 6.

This package provides:
- ``benchmark()`` - a timing harness with CUDA synchronization and warmup
- ``compiled_ops`` - ``@torch.compile`` wrapped mesh operations
- ``raw_ops`` - uncompiled mesh operations (same logic, no ``torch.compile``)
- ``save_benchmark_results`` / ``load_benchmark_results`` - JSON serialization
- ``plot_speedup_chart`` - grouped bar chart of speedup vs. CPU-pyvista
"""

from . import compiled_ops, raw_ops
from .infrastructure import (
    BENCHMARK_DISPLAY_CONFIGS,
    VARIANT_CONFIGS,
    benchmark,
    collect_system_metadata,
    load_benchmark_results,
    plot_speedup_chart,
    save_benchmark_results,
)

__all__ = [
    "BENCHMARK_DISPLAY_CONFIGS",
    "VARIANT_CONFIGS",
    "benchmark",
    "collect_system_metadata",
    "compiled_ops",
    "load_benchmark_results",
    "plot_speedup_chart",
    "raw_ops",
    "save_benchmark_results",
]
