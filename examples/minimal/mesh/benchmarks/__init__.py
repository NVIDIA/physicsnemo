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
