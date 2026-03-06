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

"""Benchmark the CPU-bound data pipeline for DrivAerML GLOBE training.

Profiles every sub-operation of the data loading pipeline - disk I/O,
deserialization, boundary subsampling, mesh cleaning, padding, lazy
geometry computation, and host-to-device transfer - to identify exactly
where CPU time is spent and what can be optimized.

The pipeline under test (per-sample) is:

    1. torch.load(.pt cache file from Lustre)
    2. _subsample_boundary: randperm + slice_cells + clean + area recompute
    3. train.py main-thread preprocessing:
       a. slice_points (subsample surface prediction points)
       b. mesh.pad (pad boundary to fixed size)
       c. cell_centroids / cell_areas / cell_normals (lazy geometry)
       d. sample_random_points_on_cells (randomize face centers)
    4. sample.to(device) (pin_memory H2D transfer)

Usage::

    uv run benchmark_dataset.py
    uv run benchmark_dataset.py --n-samples 50 --boundary-n-faces 40000
    uv run benchmark_dataset.py --save-json dataset_bench.json
"""

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from tensordict import TensorDict

from dataset import DrivAerMLDataSet, DrivAerMLSample
from physicsnemo.mesh import Mesh

H_LINE = "\u2500"


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PhaseTime:
    """Wall-clock timing for one sub-operation of the pipeline."""

    name: str
    ms: float
    notes: str = ""


@dataclass
class SampleProfile:
    """Full timing breakdown for a single sample."""

    sample_name: str
    cache_file_mb: float
    surface_n_cells: int
    surface_n_points: int
    phases: list[PhaseTime]

    @property
    def total_ms(self) -> float:
        return sum(p.ms for p in self.phases)


@dataclass
class PipelineSummary:
    """Aggregate statistics across all profiled samples."""

    n_samples: int = 0
    per_phase_median_ms: dict[str, float] = field(default_factory=dict)
    per_phase_min_ms: dict[str, float] = field(default_factory=dict)
    per_phase_max_ms: dict[str, float] = field(default_factory=dict)
    total_median_ms: float = 0.0
    total_min_ms: float = 0.0
    total_max_ms: float = 0.0
    cache_file_median_mb: float = 0.0
    cache_file_min_mb: float = 0.0
    cache_file_max_mb: float = 0.0
    surface_n_cells_median: int = 0
    surface_n_cells_min: int = 0
    surface_n_cells_max: int = 0


@dataclass
class DataLoaderProfile:
    """Timing for DataLoader throughput measurement."""

    num_workers: int
    prefetch_factor: int
    n_samples: int
    total_s: float
    per_sample_s: float
    throughput_samples_per_s: float


@dataclass
class AllResults:
    """All benchmark results, serializable to JSON."""

    system_info: dict = field(default_factory=dict)
    sample_profiles: list[dict] = field(default_factory=list)
    pipeline_summary: dict = field(default_factory=dict)
    dataloader_profiles: list[dict] = field(default_factory=list)
    worker_scaling: list[dict] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Sub-operation timing
# ═══════════════════════════════════════════════════════════════════════════


def profile_single_sample(
    dataset: DrivAerMLDataSet,
    index: int,
    *,
    boundary_n_faces: int,
    points_per_iter: int,
    pad_n_points: int,
    pad_n_cells: int,
    device: torch.device | None,
) -> SampleProfile:
    """Profile every sub-operation for one sample.

    Replicates the exact pipeline from CachedPreprocessingDataset.__getitem__
    through the train.py main-thread preprocessing, timing each step.
    """
    sample_path = dataset.sample_paths[index]
    cache_dir = dataset.cache_dir
    phases: list[PhaseTime] = []

    ### Measure cache file size
    cache_path = (
        (cache_dir / sample_path.name).with_suffix(".pt") if cache_dir else None
    )
    if cache_path and cache_path.exists():
        cache_file_mb = cache_path.stat().st_size / 1024**2
    else:
        cache_file_mb = 0.0

    ### Phase 1: torch.load (disk I/O + deserialization)
    if cache_path and cache_path.exists():
        t0 = perf_counter()
        raw_sample: DrivAerMLSample = torch.load(cache_path, weights_only=False)
        t1 = perf_counter()
        phases.append(
            PhaseTime(
                "torch.load (cache)",
                (t1 - t0) * 1000,
                f"{cache_file_mb:.1f} MB",
            )
        )
    else:
        t0 = perf_counter()
        raw_sample = DrivAerMLDataSet.preprocess(sample_path=sample_path)
        t1 = perf_counter()
        phases.append(
            PhaseTime(
                "preprocess (no cache)",
                (t1 - t0) * 1000,
            )
        )

    surface_n_cells = raw_sample.surface_mesh.n_cells
    surface_n_points = raw_sample.surface_mesh.n_points

    ### Phase 2a: surface_mesh.cell_areas.sum() (in _subsample_boundary)
    t0 = perf_counter()
    total_area = raw_sample.surface_mesh.cell_areas.sum()
    t1 = perf_counter()
    phases.append(
        PhaseTime(
            "surface cell_areas",
            (t1 - t0) * 1000,
            f"{surface_n_cells:,} cells",
        )
    )

    ### Phase 2b: torch.randperm (select random cell indices)
    t0 = perf_counter()
    indices = torch.randperm(surface_n_cells)[:boundary_n_faces]
    t1 = perf_counter()
    phases.append(
        PhaseTime(
            "randperm",
            (t1 - t0) * 1000,
            f"{surface_n_cells:,} -> {boundary_n_faces:,}",
        )
    )

    ### Phase 2c: slice_cells (select cells + associated points)
    t0 = perf_counter()
    sliced = raw_sample.surface_mesh.slice_cells(indices)
    t1 = perf_counter()
    phases.append(
        PhaseTime(
            "slice_cells",
            (t1 - t0) * 1000,
            f"{sliced.n_cells:,} cells, {sliced.n_points:,} pts",
        )
    )

    ### Phase 2d: clean (remove unreferenced points only)
    t0 = perf_counter()
    boundary = sliced.clean(
        merge_points=False,
        remove_duplicate_cells=False,
        remove_unused_points=True,
    )
    t1 = perf_counter()
    phases.append(
        PhaseTime(
            "clean",
            (t1 - t0) * 1000,
            f"{sliced.n_points:,} -> {boundary.n_points:,} pts",
        )
    )

    ### Phase 2e: Mesh() constructor + cell_areas (strip data, recompute areas)
    t0 = perf_counter()
    boundary = Mesh(points=boundary.points, cells=boundary.cells)
    raw_areas = boundary.cell_areas
    boundary._cache["cell", "areas"] = raw_areas * (total_area / raw_areas.sum())
    t1 = perf_counter()
    phases.append(
        PhaseTime(
            "boundary cell_areas + rescale",
            (t1 - t0) * 1000,
            f"{boundary.n_cells:,} cells",
        )
    )

    ### Build the sample as __getitem__ would
    raw_sample.boundary_meshes["no_slip"] = boundary

    ### Phase 3a: slice_points (subsample surface prediction points)
    n_points = min(points_per_iter, raw_sample.surface_mesh.n_points)
    t0 = perf_counter()
    mask = torch.randperm(raw_sample.surface_mesh.n_points)[:n_points]
    subsampled_surface = raw_sample.surface_mesh.slice_points(mask)
    t1 = perf_counter()
    phases.append(
        PhaseTime(
            "slice_points (prediction)",
            (t1 - t0) * 1000,
            f"{raw_sample.surface_mesh.n_points:,} -> {n_points:,} pts",
        )
    )

    ### Phase 3b: mesh.pad (pad boundary to fixed size)
    t0 = perf_counter()
    padded = boundary.pad(
        target_n_points=pad_n_points,
        target_n_cells=pad_n_cells,
        data_padding_value=0.0,
    )
    t1 = perf_counter()
    phases.append(
        PhaseTime(
            "pad",
            (t1 - t0) * 1000,
            f"-> {pad_n_cells:,} cells, {pad_n_points:,} pts",
        )
    )

    ### Phase 3c: sample_random_points_on_cells (randomize face centers)
    t0 = perf_counter()
    _ = padded.sample_random_points_on_cells()
    t1 = perf_counter()
    phases.append(
        PhaseTime(
            "sample_random_points_on_cells",
            (t1 - t0) * 1000,
        )
    )

    ### Phase 3d: lazy geometry computation (cell_areas, cell_normals)
    padded_stripped = padded.strip_caches()

    t0 = perf_counter()
    _ = padded_stripped.cell_centroids
    t1 = perf_counter()
    phases.append(
        PhaseTime(
            "pad cell_centroids",
            (t1 - t0) * 1000,
            f"{padded_stripped.n_cells:,} cells",
        )
    )

    padded_stripped2 = padded.strip_caches()
    t0 = perf_counter()
    _ = padded_stripped2.cell_areas
    t1 = perf_counter()
    phases.append(PhaseTime("pad cell_areas", (t1 - t0) * 1000))

    padded_stripped3 = padded.strip_caches()
    t0 = perf_counter()
    _ = padded_stripped3.cell_normals
    t1 = perf_counter()
    phases.append(PhaseTime("pad cell_normals", (t1 - t0) * 1000))

    ### Phase 4: .to(device) (H2D transfer, if CUDA available)
    if device is not None and device.type == "cuda":
        assembled = DrivAerMLSample(
            surface_mesh=subsampled_surface,
            boundary_meshes=TensorDict({"no_slip": padded}),
            reference_lengths=raw_sample.reference_lengths,
            dimensional_constants=raw_sample.dimensional_constants,
            aero_coefficients=raw_sample.aero_coefficients,
        )
        torch.cuda.synchronize(device)
        t0 = perf_counter()
        _ = assembled.to(device)
        torch.cuda.synchronize(device)
        t1 = perf_counter()
        phases.append(PhaseTime("to(cuda)", (t1 - t0) * 1000))

    return SampleProfile(
        sample_name=sample_path.name,
        cache_file_mb=cache_file_mb,
        surface_n_cells=surface_n_cells,
        surface_n_points=surface_n_points,
        phases=phases,
    )


def summarize_profiles(profiles: list[SampleProfile]) -> PipelineSummary:
    """Compute aggregate statistics across all sample profiles."""
    if not profiles:
        return PipelineSummary()

    phase_names: list[str] = [p.name for p in profiles[0].phases]
    phase_times: dict[str, list[float]] = {name: [] for name in phase_names}
    totals: list[float] = []
    cache_sizes: list[float] = []
    cell_counts: list[int] = []

    for prof in profiles:
        totals.append(prof.total_ms)
        cache_sizes.append(prof.cache_file_mb)
        cell_counts.append(prof.surface_n_cells)
        for phase in prof.phases:
            if phase.name in phase_times:
                phase_times[phase.name].append(phase.ms)

    return PipelineSummary(
        n_samples=len(profiles),
        per_phase_median_ms={k: median(v) for k, v in phase_times.items() if v},
        per_phase_min_ms={k: min(v) for k, v in phase_times.items() if v},
        per_phase_max_ms={k: max(v) for k, v in phase_times.items() if v},
        total_median_ms=median(totals),
        total_min_ms=min(totals),
        total_max_ms=max(totals),
        cache_file_median_mb=median(cache_sizes),
        cache_file_min_mb=min(cache_sizes),
        cache_file_max_mb=max(cache_sizes),
        surface_n_cells_median=int(median(cell_counts)),
        surface_n_cells_min=min(cell_counts),
        surface_n_cells_max=max(cell_counts),
    )


# ═══════════════════════════════════════════════════════════════════════════
# DataLoader throughput measurement
# ═══════════════════════════════════════════════════════════════════════════


def profile_dataloader(
    sample_paths: list[Path],
    cache_dir: Path,
    *,
    num_workers: int,
    prefetch_factor: int,
    boundary_n_faces: int,
    n_epochs: int = 1,
) -> DataLoaderProfile:
    """Measure DataLoader throughput with given worker configuration."""
    dataset = DrivAerMLDataSet(
        sample_paths=sample_paths,
        cache_dir=cache_dir,
        boundary_n_faces=boundary_n_faces,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        collate_fn=lambda x: x,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )

    ### Warmup: iterate once to populate Lustre page cache
    for _ in loader:
        pass

    ### Timed iteration
    n_samples = 0
    t0 = perf_counter()
    for _ in range(n_epochs):
        for _ in loader:
            n_samples += 1
    t1 = perf_counter()

    elapsed = t1 - t0
    return DataLoaderProfile(
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        n_samples=n_samples,
        total_s=elapsed,
        per_sample_s=elapsed / max(1, n_samples),
        throughput_samples_per_s=n_samples / max(1e-9, elapsed),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Printing
# ═══════════════════════════════════════════════════════════════════════════


def print_sample_profile(prof: SampleProfile) -> None:
    """Print a single sample's phase breakdown."""
    total = prof.total_ms
    name_w = max(len(p.name) for p in prof.phases) + 2
    for p in prof.phases:
        pct = f"{p.ms / total * 100:5.1f}%" if total > 0 else "    \u2014"
        print(f"    {p.name:<{name_w}} {p.ms:>9.1f} ms  {pct:>6}  {p.notes}")
    print(f"    {'TOTAL':<{name_w}} {total:>9.1f} ms")


def print_summary(summary: PipelineSummary) -> None:
    """Print aggregate pipeline statistics."""
    print(f"\n  Aggregate across {summary.n_samples} samples:")
    print(
        f"  Cache file size:  median={summary.cache_file_median_mb:.1f} MB  "
        f"min={summary.cache_file_min_mb:.1f} MB  max={summary.cache_file_max_mb:.1f} MB"
    )
    print(
        f"  Surface cells:    median={summary.surface_n_cells_median:,}  "
        f"min={summary.surface_n_cells_min:,}  max={summary.surface_n_cells_max:,}"
    )
    print(
        f"  Total load time:  median={summary.total_median_ms:.1f} ms  "
        f"min={summary.total_min_ms:.1f} ms  max={summary.total_max_ms:.1f} ms"
    )

    total_median = summary.total_median_ms
    name_w = (
        max(len(k) for k in summary.per_phase_median_ms) + 2
        if summary.per_phase_median_ms
        else 20
    )

    print(
        f"\n  {'Phase':<{name_w}}  {'Median':>9}  {'Min':>9}  {'Max':>9}  {'% Total':>7}"
    )
    print(
        f"  {H_LINE * name_w}  {H_LINE * 9}  {H_LINE * 9}  {H_LINE * 9}  {H_LINE * 7}"
    )

    for name in summary.per_phase_median_ms:
        med = summary.per_phase_median_ms[name]
        mn = summary.per_phase_min_ms.get(name, 0)
        mx = summary.per_phase_max_ms.get(name, 0)
        pct = f"{med / total_median * 100:5.1f}%" if total_median > 0 else "    \u2014"
        print(f"  {name:<{name_w}}  {med:>8.1f}ms {mn:>8.1f}ms {mx:>8.1f}ms  {pct:>7}")

    print(
        f"\n  Pipeline throughput: 1 sample / {summary.total_median_ms / 1000:.2f}s "
        f"= {1000 / summary.total_median_ms:.2f} samples/s (single-threaded)"
    )

    gpu_step_s = 14.0
    for nw in [1, 4, 8, 16, 28]:
        t_load = summary.total_median_ms / 1000
        stall = max(0, t_load - nw * gpu_step_s)
        effective = gpu_step_s + stall / max(1, 14 // nw + 1)
        print(
            f"  With {nw:>2} workers: stall/cycle = {stall:.1f}s, "
            f"effective ~{effective:.1f}s/sample "
            f"{'(no stall)' if stall < 0.1 else ''}"
        )


def print_dataloader_table(profiles: list[DataLoaderProfile]) -> None:
    """Print DataLoader throughput comparison."""
    print(
        f"  {'Workers':>7} {'Prefetch':>8} {'Samples':>7} {'Total':>8} "
        f"{'Per Sample':>10} {'Throughput':>12}"
    )
    print(
        f"  {H_LINE * 7} {H_LINE * 8} {H_LINE * 7} {H_LINE * 8} "
        f"{H_LINE * 10} {H_LINE * 12}"
    )
    for p in profiles:
        print(
            f"  {p.num_workers:>7} {p.prefetch_factor:>8} {p.n_samples:>7} "
            f"{p.total_s:>7.1f}s {p.per_sample_s:>9.2f}s "
            f"{p.throughput_samples_per_s:>9.2f} samp/s"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Recommendations
# ═══════════════════════════════════════════════════════════════════════════


def generate_recommendations(
    summary: PipelineSummary,
    dl_profiles: list[DataLoaderProfile],
) -> list[str]:
    """Generate actionable recommendations from benchmark data."""
    recs: list[str] = []
    t_load = summary.total_median_ms / 1000
    gpu_step = 14.0

    ### Identify dominant phase
    if summary.per_phase_median_ms:
        dominant_name = max(
            summary.per_phase_median_ms,
            key=summary.per_phase_median_ms.get,
        )
        dominant_frac = (
            summary.per_phase_median_ms[dominant_name] / summary.total_median_ms
        )
        if dominant_frac > 0.3:
            recs.append(
                f"DOMINANT PHASE: '{dominant_name}' accounts for "
                f"{dominant_frac * 100:.0f}% of per-sample CPU time. "
                f"Focus optimization here."
            )

    ### torch.load dominance
    load_ms = summary.per_phase_median_ms.get("torch.load (cache)", 0)
    if load_ms > 0.5 * summary.total_median_ms:
        recs.append(
            f"DISK I/O: torch.load is {load_ms / summary.total_median_ms * 100:.0f}% "
            f"of load time ({load_ms / 1000:.1f}s). The bottleneck is Lustre read "
            f"bandwidth. Consider: moving cache to local NVMe, using memory-mapped "
            f"tensors (torch.load with mmap=True), or RAM caching "
            f"(use_ram_caching=True)."
        )

    ### clean() cost
    clean_ms = summary.per_phase_median_ms.get("clean", 0)
    if clean_ms > 0.1 * summary.total_median_ms:
        recs.append(
            f"MESH CLEAN: clean() takes {clean_ms:.0f}ms "
            f"({clean_ms / summary.total_median_ms * 100:.0f}% of load time). "
            f"Consider skipping merge_points when not needed: "
            f"`.clean(merge_points=False)` or using "
            f"`remove_unused_points=True` only."
        )

    ### Worker count recommendation
    min_workers = int(t_load / gpu_step) + 1
    recs.append(
        f"WORKERS: T_load={t_load:.1f}s, T_train~{gpu_step:.0f}s. "
        f"Need >= {min_workers} workers to avoid pipeline stalls "
        f"(currently {min_workers} workers eliminates period-{min_workers} stall)."
    )

    ### Cache file size
    if summary.cache_file_median_mb > 100:
        recs.append(
            f"CACHE SIZE: Median .pt file is {summary.cache_file_median_mb:.0f} MB. "
            f"Large files amplify Lustre read latency. Consider storing only "
            f"the boundary-relevant subset, or using compressed formats."
        )

    ### Scaling efficiency from DataLoader profiles
    if len(dl_profiles) >= 2:
        baseline = dl_profiles[0]
        best = min(dl_profiles, key=lambda p: p.per_sample_s)
        speedup = baseline.per_sample_s / max(1e-9, best.per_sample_s)
        ideal = best.num_workers / max(1, baseline.num_workers)
        efficiency = speedup / max(1e-9, ideal)
        recs.append(
            f"SCALING: {best.num_workers} workers gives {speedup:.1f}x speedup "
            f"over {baseline.num_workers} workers "
            f"(efficiency={efficiency * 100:.0f}% of ideal {ideal:.0f}x). "
            + (
                "Diminishing returns - likely I/O bandwidth limited."
                if efficiency < 0.5
                else "Good parallel efficiency."
            )
        )

    return recs


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main(
    data_dir: Path | None = None,
    n_samples: int | None = None,
    boundary_n_faces: int = 20_000,
    points_per_iter: int = 2048,
    pad_n_points: int = 60_000,
    pad_n_cells: int = 20_000,
    skip_per_sample: bool = False,
    skip_dataloader: bool = False,
    skip_worker_scaling: bool = False,
    worker_counts: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 28),
    prefetch_factor: int = 4,
    save_json: str | None = None,
):
    """Benchmark the DrivAerML data loading pipeline.

    Profiles every sub-operation of the data pipeline to identify CPU-bound
    bottlenecks: disk I/O (torch.load from Lustre cache), boundary mesh
    subsampling (slice_cells + clean), padding, lazy geometry computation,
    and H2D transfer.

    Args:
        data_dir: Path to the DrivAerML dataset root.  Falls back to
            ``DRIVAER_DATA_DIR`` env var.
        n_samples: Number of samples to profile.  None = all training samples.
        boundary_n_faces: Number of cells for boundary subsampling.
        points_per_iter: Surface prediction points per training iteration.
        pad_n_points: Padding target for boundary mesh points.
        pad_n_cells: Padding target for boundary mesh cells.
        skip_per_sample: Skip per-sample phase breakdown.
        skip_dataloader: Skip DataLoader throughput measurement.
        skip_worker_scaling: Skip worker count scaling test.
        worker_counts: Worker counts to test in the scaling sweep.
        prefetch_factor: Prefetch factor for DataLoader tests.
        save_json: Path to save machine-readable results as JSON.
    """
    if data_dir is None:
        if _data_env := os.environ.get("DRIVAER_DATA_DIR"):
            data_dir = Path(_data_env)
        else:
            print(
                "ERROR: DrivAerML data directory not specified.  Pass --data-dir "
                "or set DRIVAER_DATA_DIR.",
                file=sys.stderr,
            )
            sys.exit(1)

    data_dir = Path(data_dir)
    cache_dir = Path(__file__).parent / "cache"
    device = torch.device("cuda") if torch.cuda.is_available() else None

    sample_paths = DrivAerMLDataSet.get_split_paths(data_dir, "train")
    if n_samples is not None:
        sample_paths = sample_paths[:n_samples]

    n_cpus = os.cpu_count() or 1
    has_gpu = torch.cuda.is_available()

    all_results = AllResults()

    ### System info
    hdr_w = 90
    print(f"\n{'=' * hdr_w}")
    print(f"  DrivAerML Data Pipeline Benchmark")
    print(f"{'=' * hdr_w}")
    print(f"  CPUs:               {n_cpus}")
    if has_gpu:
        print(f"  GPU:                {torch.cuda.get_device_name()}")
        print(f"  GPUs:               {torch.cuda.device_count()}")
    print(f"  Dataset:            {len(sample_paths)} training samples")
    print(f"  Cache dir:          {cache_dir}")
    print(f"  boundary_n_faces:   {boundary_n_faces:,}")
    print(f"  points_per_iter:    {points_per_iter:,}")
    print(f"  Pad targets:        {pad_n_cells:,} cells, {pad_n_points:,} points")
    print(f"  OMP_NUM_THREADS:    {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    print(f"{'=' * hdr_w}")

    all_results.system_info = {
        "n_cpus": n_cpus,
        "n_gpus": torch.cuda.device_count() if has_gpu else 0,
        "gpu_name": torch.cuda.get_device_name() if has_gpu else "N/A",
        "n_samples": len(sample_paths),
        "boundary_n_faces": boundary_n_faces,
        "points_per_iter": points_per_iter,
        "pad_n_cells": pad_n_cells,
        "pad_n_points": pad_n_points,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "not set"),
    }

    # ── Section 1: Per-sample phase breakdown ──────────────────────────
    if not skip_per_sample:
        n_sections = 3 - skip_dataloader - skip_worker_scaling
        section_num = 1
        print(f"\n[{section_num}/{n_sections}] Per-Sample Phase Breakdown")
        print(f"  {H_LINE * 78}")

        dataset = DrivAerMLDataSet(
            sample_paths=sample_paths,
            cache_dir=cache_dir,
            boundary_n_faces=boundary_n_faces,
        )

        ### Warmup: load one sample to populate Lustre page cache metadata
        print(f"  Warming up (loading sample 0)...")
        _ = dataset[0]

        profiles: list[SampleProfile] = []
        for i in range(len(sample_paths)):
            print(f"\n  Sample {i}: {sample_paths[i].name}")
            prof = profile_single_sample(
                dataset,
                i,
                boundary_n_faces=boundary_n_faces,
                points_per_iter=points_per_iter,
                pad_n_points=pad_n_points,
                pad_n_cells=pad_n_cells,
                device=device,
            )
            print_sample_profile(prof)
            profiles.append(prof)

        summary = summarize_profiles(profiles)
        print_summary(summary)

        all_results.sample_profiles = [
            {
                "sample_name": p.sample_name,
                "cache_file_mb": p.cache_file_mb,
                "surface_n_cells": p.surface_n_cells,
                "surface_n_points": p.surface_n_points,
                "total_ms": p.total_ms,
                "phases": [asdict(ph) for ph in p.phases],
            }
            for p in profiles
        ]
        all_results.pipeline_summary = asdict(summary)

    # ── Section 2: DataLoader throughput ───────────────────────────────
    if not skip_dataloader:
        n_sections_so_far = 1 + (not skip_per_sample)
        n_sections = 3 - skip_per_sample - skip_worker_scaling
        print(
            f"\n[{n_sections_so_far}/{n_sections}] DataLoader Throughput "
            f"(single-process, varying workers)"
        )
        print(f"  {H_LINE * 78}")

        n_gpus = max(1, torch.cuda.device_count()) if has_gpu else 1
        auto_workers = n_cpus // n_gpus
        test_configs = sorted(set([0, 4, auto_workers]))

        dl_profiles: list[DataLoaderProfile] = []
        for nw in test_configs:
            pf = prefetch_factor if nw > 0 else 2
            print(
                f"  Testing num_workers={nw}, prefetch_factor={pf}...",
                end="",
                flush=True,
            )
            prof = profile_dataloader(
                sample_paths,
                cache_dir,
                num_workers=nw,
                prefetch_factor=pf,
                boundary_n_faces=boundary_n_faces,
                n_epochs=2,
            )
            print(
                f" {prof.per_sample_s:.2f}s/sample, "
                f"{prof.throughput_samples_per_s:.2f} samp/s"
            )
            dl_profiles.append(prof)

        print()
        print_dataloader_table(dl_profiles)
        all_results.dataloader_profiles = [asdict(p) for p in dl_profiles]

    # ── Section 3: Worker count scaling ────────────────────────────────
    if not skip_worker_scaling:
        n_sections = 3 - skip_per_sample - skip_dataloader
        print(f"\n[{n_sections}/{n_sections}] Worker Count Scaling")
        print(f"  {H_LINE * 78}")

        scaling_profiles: list[DataLoaderProfile] = []
        for nw in worker_counts:
            pf = prefetch_factor if nw > 0 else 2
            print(f"  num_workers={nw:>3}, prefetch={pf}...", end="", flush=True)
            prof = profile_dataloader(
                sample_paths,
                cache_dir,
                num_workers=nw,
                prefetch_factor=pf,
                boundary_n_faces=boundary_n_faces,
                n_epochs=2,
            )
            print(
                f"  {prof.per_sample_s:.2f}s/sample  "
                f"{prof.throughput_samples_per_s:.2f} samp/s"
            )
            scaling_profiles.append(prof)

        print()
        print_dataloader_table(scaling_profiles)
        all_results.worker_scaling = [asdict(p) for p in scaling_profiles]

    # ── Recommendations ────────────────────────────────────────────────
    print(f"\n  Recommendations")
    print(f"  {H_LINE * 78}")

    summary_for_recs = (
        summarize_profiles(profiles) if not skip_per_sample else PipelineSummary()
    )
    dl_for_recs = (dl_profiles if not skip_dataloader else []) + (
        scaling_profiles if not skip_worker_scaling else []
    )
    recs = generate_recommendations(summary_for_recs, dl_for_recs)
    all_results.recommendations = recs
    for i, rec in enumerate(recs, 1):
        print(f"  {i}. {rec}")
    print()

    ### Save JSON
    if save_json:
        Path(save_json).write_text(json.dumps(asdict(all_results), indent=2))
        print(f"  Results saved to {save_json}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
