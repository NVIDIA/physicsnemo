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

"""Benchmark timing, result serialization, and plotting utilities."""

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch

### Schema version for forward-compatible deserialization ###
_BENCHMARK_SCHEMA_VERSION = 1


def benchmark(
    name: str,
    func,
    n_runs: int = 50,
    warmup: int = 2,
    target_runtime: float = 3.0,
    verbose: bool = True,
    setup: callable = None,
):
    """Benchmark a function and return the minimum time.

    Using minimum (not mean) is standard practice for benchmarks because it
    represents the best achievable performance without system noise.

    The timed loop is **time-budgeted**: it runs up to ``n_runs`` iterations
    but stops early once cumulative timed runtime exceeds ``target_runtime``.
    This keeps fast operations well-sampled (up to 50 runs) while preventing
    slow operations from dominating total benchmark wall time.

    Warmup always runs in full regardless of the time budget.  The default
    ``warmup=2`` is sufficient for ``torch.compile`` (one tracing call plus
    one compiled execution).

    Parameters
    ----------
    name : str
        Display name for the benchmark.
    func : callable
        Function to benchmark.
    n_runs : int
        Maximum number of timed runs (cap).
    warmup : int
        Number of warmup runs (not timed).  The first warmup run triggers
        ``torch.compile`` tracing; the second executes compiled code.
    target_runtime : float
        Target cumulative runtime (seconds) for the timed loop.  The loop
        exits early once this budget is exceeded, even if fewer than
        ``n_runs`` iterations have completed.
    verbose : bool
        Whether to print results.
    setup : callable, optional
        Function to call before each run (not included in timing).
        Use this to strip caches from meshes to prevent false performance benefits.
    """
    ### Warmup runs ###
    for _ in range(warmup):
        if setup is not None:
            setup()
        func()

    ### Timed runs with proper CUDA synchronization ###
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    cumulative = 0.0
    for _ in range(n_runs):
        if setup is not None:
            setup()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = func()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - start
        times.append(dt)
        cumulative += dt

        if cumulative >= target_runtime:
            break

    min_time = min(times)
    if verbose:
        print(f"{name}: {min_time * 1000:.3f} ms ({len(times)} runs)")
    return min_time, result


# ---------------------------------------------------------------------------
# System metadata
# ---------------------------------------------------------------------------


def collect_system_metadata() -> dict:
    """Gather system information for benchmark reproducibility."""
    meta = {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "pyvista_version": pv.__version__,
        "cpu": platform.processor() or platform.machine(),
        "gpu": None,
        "cuda_version": None,
        "gpu_memory_gb": None,
    }
    if torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["cuda_version"] = ".".join(
            str(x) for x in torch.cuda.get_device_capability(0)
        )
        props = torch.cuda.get_device_properties(0)
        meta["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
    return meta


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------


def save_benchmark_results(
    results: dict,
    metadata: dict,
    config: dict,
    path: str | Path,
) -> Path:
    """Serialize benchmark results + metadata to a JSON file.

    Parameters
    ----------
    results : dict
        The ``benchmark_results`` dict produced by the benchmark cells.
        Each key maps to a dict of timing lists keyed by variant
        (``"pyvista"``, ``"pnm_cpu_raw"``, ``"pnm_cpu_compiled"``,
        ``"pnm_gpu_raw"``, ``"pnm_gpu_compiled"``) plus ``"sizes"``.
    metadata : dict
        System metadata from ``collect_system_metadata()``.
    config : dict
        Benchmark configuration, e.g.
        ``{"mesh_bucket": "small", "mesh_source": "Stanford Bunny"}``.
    path : str | Path
        Destination file path (should end in ``.json``).

    Returns
    -------
    Path
        The resolved path that was written.
    """
    path = Path(path)
    payload = {
        "schema_version": _BENCHMARK_SCHEMA_VERSION,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metadata,
        },
        "config": config,
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Benchmark results saved to {path}")
    return path


def load_benchmark_results(path: str | Path) -> dict:
    """Load benchmark results from a JSON file.

    Parameters
    ----------
    path : str | Path
        Path to a ``.json`` benchmark file produced by ``save_benchmark_results``.

    Returns
    -------
    dict
        The full payload including ``schema_version``, ``metadata``,
        ``config``, and ``results``.
    """
    path = Path(path)
    payload = json.loads(path.read_text())
    version = payload.get("schema_version", 0)
    if version != _BENCHMARK_SCHEMA_VERSION:
        print(
            f"Warning: benchmark file has schema version {version}, "
            f"expected {_BENCHMARK_SCHEMA_VERSION}. Results may need migration."
        )
    return payload


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

### Display configs: (result-dict key, axis label) ###
BENCHMARK_DISPLAY_CONFIGS = [
    ("normals", "Normal\nComputation"),
    ("curvature", "Gaussian\nCurvature"),
    ("gradient", "Gradient\nComputation"),
    ("subdivision", "Subdivision\n(Loop)"),
    ("neighbors_p2p", "Point-to-Point\nNeighbors"),
    ("neighbors_c2c", "Cell-to-Cell\nNeighbors"),
    ("sampling", "Random\nSampling"),
    ("smoothing", "Laplacian\nSmoothing"),
    ("transforms", "Geometric\nTransforms"),
]

### Variant configs: (result-dict key, chart label, bar color) ###
### Ordered left-to-right within each group on the chart. ###
VARIANT_CONFIGS = [
    ("pnm_gpu_compiled", "PNM GPU (compiled)", "#76B900"),  # NVIDIA green
    ("pnm_gpu_raw", "PNM GPU", "#A8D86E"),  # light green
    ("pnm_cpu_compiled", "PNM CPU (compiled)", "#2C5F9B"),  # dark blue
    ("pnm_cpu_raw", "PNM CPU", "#7CB3E8"),  # light blue
]


def plot_speedup_chart(
    results: dict,
    title_suffix: str = "",
    save_path: str | Path | None = "benchmark_results.png",
):
    """Plot a grouped bar chart of speedup over CPU-pyvista.

    Automatically includes only those variants that have data in ``results``.
    Each bar shows speedup relative to the PyVista (CPU) baseline.

    Parameters
    ----------
    results : dict
        Either the live ``benchmark_results`` dict, or the ``"results"``
        field from a loaded JSON payload.
    title_suffix : str
        Optional text appended to the chart title (e.g. hardware name).
    save_path : str | Path | None
        If not None, save the figure to this path.
    """
    ### Determine which variants have data across any operation ###
    active_variants = [
        (key, label, color)
        for key, label, color in VARIANT_CONFIGS
        if any(
            results.get(op_key, {}).get(key) for op_key, _ in BENCHMARK_DISPLAY_CONFIGS
        )
    ]

    if not active_variants:
        print("No benchmark results to plot.")
        return

    ### Collect per-operation speedups for each active variant ###
    operations: list[str] = []
    # speedups[variant_key] -> list of floats, one per operation
    speedups: dict[str, list[float]] = {key: [] for key, _, _ in active_variants}

    for op_key, display_name in BENCHMARK_DISPLAY_CONFIGS:
        op_data = results.get(op_key, {})
        pv_times = op_data.get("pyvista", [])
        if not pv_times:
            continue

        pv_time = pv_times[-1]  # largest mesh size
        operations.append(display_name)

        for var_key, _, _ in active_variants:
            var_times = op_data.get(var_key, [])
            sp = pv_time / var_times[-1] if var_times else 0.0
            speedups[var_key].append(sp)

    if not operations:
        print("No benchmark results to plot.")
        return

    ### Build figure ###
    n_variants = len(active_variants)
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(operations))
    total_group_width = 0.8
    bar_width = total_group_width / n_variants

    for i, (var_key, var_label, var_color) in enumerate(active_variants):
        offset = (i - (n_variants - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset,
            speedups[var_key],
            bar_width,
            label=var_label,
            color=var_color,
            edgecolor="black",
            linewidth=0.5,
        )

        ### Annotate speedup values on top of bars ###
        for bar, sp in zip(bars, speedups[var_key]):
            if sp <= 0:
                continue
            height = bar.get_height()
            text = f"{sp:.0f}x" if sp >= 10 else f"{sp:.1f}x"
            ax.annotate(
                text,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8 if n_variants > 3 else 9,
                fontweight="bold",
            )

    ax.set_ylabel("Speedup over PyVista (CPU)", fontsize=12)
    ax.set_yscale("log")
    chart_title = "Speedup over PyVista (CPU) \u2014 Largest Mesh Size"
    if title_suffix:
        chart_title += f"  [{title_suffix}]"
    ax.set_title(chart_title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(operations, fontsize=9)
    ax.axhline(
        y=1, color="gray", linestyle="--", alpha=0.5, label="PyVista baseline (1x)"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"\nBenchmark visualization saved to {save_path}")
    plt.show()
