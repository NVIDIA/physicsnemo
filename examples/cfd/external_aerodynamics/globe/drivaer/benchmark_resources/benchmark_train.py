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

"""Comprehensive GPU-side training benchmark for DrivAerML GLOBE.

Measures how forward+backward training step performance scales with
n_faces_per_boundary, theta, leaf_size, n_prediction_points, and other
GLOBE hyperparameters using real DrivAerML car body geometry.

All sweep cases are distributed round-robin across available GPUs, so
the benchmark parallelizes naturally on an 8-GPU node.

Usage:

    uv run torchrun --nproc-per-node 8 benchmark_train.py
    uv run torchrun --nproc-per-node 8 benchmark_train.py --quick
    uv run benchmark_train.py                             # single GPU
    uv run benchmark_train.py --save-json results.json
"""

import gc
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field
from math import log
from pathlib import Path
from time import perf_counter

import torch
from dataset import DrivAerMLDataSet, DrivAerMLSample
from tensordict import TensorDict

from physicsnemo.distributed import DistributedManager
from physicsnemo.experimental.models.globe.cluster_tree import ClusterTree
from physicsnemo.experimental.models.globe.field_kernel import BarnesHutKernel
from physicsnemo.experimental.models.globe.model import GLOBE
from physicsnemo.mesh import Mesh

# ═══════════════════════════════════════════════════════════════════════════
# Data classes for structured results
# ═══════════════════════════════════════════════════════════════════════════

### DrivAerML reference area (aRefRef = 2.170 m² from the DrivAerML spec)
DRIVAER_REFERENCE_AREA = 2.170


@dataclass
class PhaseResult:
    """Timing result for a single phase of the forward pass."""

    name: str
    wall_ms: float
    gpu_ms: float
    mem_delta_mb: float = 0.0
    notes: str = ""


@dataclass
class TrainingStepResult:
    """Timing result for a full training step."""

    forward_ms: float
    backward_ms: float
    zero_grad_ms: float
    peak_alloc_gb: float
    peak_reserved_gb: float

    @property
    def total_ms(self) -> float:
        return self.forward_ms + self.backward_ms + self.zero_grad_ms


@dataclass
class SweepPoint:
    """One data point in a parameter sweep."""

    sweep_name: str
    label: str
    config: dict
    sort_key: float = 0.0
    oom: bool = False
    forward_ms: float = 0.0
    backward_ms: float = 0.0
    total_ms: float = 0.0
    peak_alloc_gb: float = 0.0
    peak_reserved_gb: float = 0.0
    n_near: int = 0
    n_far_nodes: int = 0
    n_nf: int = 0
    n_fn: int = 0
    compression_ratio: float = 0.0
    tree_depth: int = 0
    tree_nodes: int = 0
    tree_leaves: int = 0
    n_faces: int = 0
    n_prediction_points: int = 0


@dataclass
class ProfileRegion:
    """One region from torch.profiler output."""

    name: str
    cpu_ms: float
    cuda_ms: float
    count: int


@dataclass
class BenchmarkCase:
    """Specification for one benchmark case."""

    sweep_name: str
    label: str
    model_overrides: dict
    n_faces_per_boundary: int
    n_prediction_points: int
    amp: bool
    compile_mode: str | None
    use_grad_checkpointing: bool
    sort_key: float = 0.0


@dataclass
class AllResults:
    """Container for all benchmark results (serializable to JSON)."""

    system_info: dict = field(default_factory=dict)
    drivaer_info: dict = field(default_factory=dict)
    phase_results: list[dict] = field(default_factory=list)
    profiler_fwd: list[dict] = field(default_factory=list)
    profiler_fwd_bwd: list[dict] = field(default_factory=list)
    profiler_top_bwd_ops: list[dict] = field(default_factory=list)
    training_step: dict = field(default_factory=dict)
    sweeps: dict[str, list[dict]] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Timing primitives
# ═══════════════════════════════════════════════════════════════════════════


def time_fn(
    fn,
    device: torch.device,
    n_warmup: int = 3,
    n_trials: int = 10,
) -> tuple[float, float]:
    """Run *fn* with warmup, return median ``(wall_ms, gpu_ms)``.

    Uses paired ``torch.cuda.Event`` for GPU timing and ``perf_counter``
    for wall-clock.  Synchronizes before and after every trial so the two
    clocks bracket exactly the same work.
    """
    for _ in range(n_warmup):
        fn()
        torch.cuda.synchronize(device)

    wall_samples: list[float] = []
    gpu_samples: list[float] = []
    for _ in range(n_trials):
        torch.cuda.synchronize(device)
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        t0 = perf_counter()
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize(device)
        t1 = perf_counter()
        wall_samples.append((t1 - t0) * 1000)
        gpu_samples.append(start_ev.elapsed_time(end_ev))

    wall_samples.sort()
    gpu_samples.sort()
    mid = n_trials // 2
    return wall_samples[mid], gpu_samples[mid]


def time_training_step(
    model: GLOBE,
    boundary_meshes: dict[str, Mesh],
    prediction_points: torch.Tensor,
    reference_lengths: dict[str, torch.Tensor],
    *,
    device: torch.device,
    amp: bool = False,
    n_warmup: int = 3,
    n_trials: int = 5,
    max_step_ms: float = 300_000,
) -> TrainingStepResult | None:
    """Time a full training step (forward + backward + zero_grad).

    Returns median timings across *n_trials*, or ``None`` on OOM or if
    a single step exceeds *max_step_ms* (prevents NCCL timeouts when one
    rank gets a pathologically slow case).
    """
    model.train()

    def _step() -> tuple[float, float, float]:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            torch.cuda.synchronize(device)
            t0 = perf_counter()
            pred_mesh = model(
                prediction_points=prediction_points,
                boundary_meshes=boundary_meshes,
                reference_lengths=reference_lengths,
            )
            loss = sum(
                v.float().sum()
                for v in pred_mesh.point_data.values(
                    include_nested=True, leaves_only=True
                )
            )
            torch.cuda.synchronize(device)
            t1 = perf_counter()
            loss.backward()
            torch.cuda.synchronize(device)
            t2 = perf_counter()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        t3 = perf_counter()
        return (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000

    for i in range(n_warmup):
        try:
            f, b, z = _step()
            if f + b + z > max_step_ms:
                return None
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return None

    torch.cuda.reset_peak_memory_stats(device)
    fwd_times: list[float] = []
    bwd_times: list[float] = []
    zg_times: list[float] = []

    for _ in range(n_trials):
        try:
            f, b, z = _step()
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return None
        fwd_times.append(f)
        bwd_times.append(b)
        zg_times.append(z)

    fwd_times.sort()
    bwd_times.sort()
    zg_times.sort()
    mid = n_trials // 2

    return TrainingStepResult(
        forward_ms=fwd_times[mid],
        backward_ms=bwd_times[mid],
        zero_grad_ms=zg_times[mid],
        peak_alloc_gb=torch.cuda.max_memory_allocated(device) / 1024**3,
        peak_reserved_gb=torch.cuda.max_memory_reserved(device) / 1024**3,
    )


def _time_compiled_step(
    model: GLOBE,
    boundary_meshes: dict[str, Mesh],
    prediction_points: torch.Tensor,
    reference_lengths: dict[str, torch.Tensor],
    *,
    compile_mode: str,
    device: torch.device,
    amp: bool,
    n_warmup: int,
    n_trials: int,
    max_step_ms: float = 300_000,
) -> TrainingStepResult | None:
    """Time a training step with ``torch.compile`` wrapping the forward pass.

    Compilation happens during warmup; timed iterations use the compiled
    graph.  Returns ``None`` on OOM or per-step timeout.
    """
    model.train()

    @torch.compile(dynamic=True, mode=compile_mode)
    def compiled_fwd(pp, bm, rl):
        pred_mesh = model(
            prediction_points=pp,
            boundary_meshes=bm,
            reference_lengths=rl,
        )
        return sum(
            v.float().sum()
            for v in pred_mesh.point_data.values(include_nested=True, leaves_only=True)
        )

    for _ in range(n_warmup):
        try:
            torch.cuda.synchronize(device)
            t0 = perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                loss = compiled_fwd(
                    prediction_points, boundary_meshes, reference_lengths
                )
                loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            if (perf_counter() - t0) * 1000 > max_step_ms:
                return None
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return None

    torch.cuda.reset_peak_memory_stats(device)
    fwd_times: list[float] = []
    bwd_times: list[float] = []
    zg_times: list[float] = []

    for _ in range(n_trials):
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                torch.cuda.synchronize(device)
                t0 = perf_counter()
                loss = compiled_fwd(
                    prediction_points, boundary_meshes, reference_lengths
                )
                torch.cuda.synchronize(device)
                t1 = perf_counter()
                loss.backward()
                torch.cuda.synchronize(device)
                t2 = perf_counter()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            t3 = perf_counter()
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return None
        fwd_times.append((t1 - t0) * 1000)
        bwd_times.append((t2 - t1) * 1000)
        zg_times.append((t3 - t2) * 1000)

    fwd_times.sort()
    bwd_times.sort()
    zg_times.sort()
    mid = n_trials // 2

    return TrainingStepResult(
        forward_ms=fwd_times[mid],
        backward_ms=bwd_times[mid],
        zero_grad_ms=zg_times[mid],
        peak_alloc_gb=torch.cuda.max_memory_allocated(device) / 1024**3,
        peak_reserved_gb=torch.cuda.max_memory_reserved(device) / 1024**3,
    )


def profiler_run(
    fn,
    device: torch.device,
    n_warmup: int = 2,
) -> list[ProfileRegion]:
    """Run *fn* under torch.profiler, return record_function regions."""
    for _ in range(n_warmup):
        fn()
        torch.cuda.synchronize(device)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
    ) as prof:
        fn()
        torch.cuda.synchronize(device)

    PREFIXES = ("globe::", "multiscale_kernel::", "bh_kernel::", "kernel::")
    regions: list[ProfileRegion] = [
        ProfileRegion(
            name=evt.key,
            cpu_ms=evt.cpu_time_total / 1000,
            cuda_ms=evt.device_time_total / 1000,
            count=evt.count,
        )
        for evt in prof.key_averages()
        if any(evt.key.startswith(p) for p in PREFIXES)
    ]
    regions.sort(key=lambda r: r.cpu_ms, reverse=True)
    return regions


def profiler_top_backward_ops(
    fn,
    device: torch.device,
    n_warmup: int = 2,
    top_n: int = 15,
) -> list[ProfileRegion]:
    """Run *fn* under torch.profiler, return top-N backward CUDA ops by time."""
    for _ in range(n_warmup):
        fn()
        torch.cuda.synchronize(device)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
    ) as prof:
        fn()
        torch.cuda.synchronize(device)

    ops: list[ProfileRegion] = [
        ProfileRegion(
            name=evt.key,
            cpu_ms=evt.cpu_time_total / 1000,
            cuda_ms=evt.device_time_total / 1000,
            count=evt.count,
        )
        for evt in prof.key_averages()
        if evt.device_time_total > 0
    ]
    ops.sort(key=lambda r: r.cuda_ms, reverse=True)
    return ops[:top_n]


# ═══════════════════════════════════════════════════════════════════════════
# GPU state management
# ═══════════════════════════════════════════════════════════════════════════


def clean_gpu(device: torch.device) -> None:
    """Reset GPU state between experiments."""
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def mem_mb(device: torch.device) -> float:
    """Current GPU memory allocated in MB."""
    return torch.cuda.memory_allocated(device) / 1024**2


# ═══════════════════════════════════════════════════════════════════════════
# Formatting utilities
# ═══════════════════════════════════════════════════════════════════════════

H_LINE = "\u2500"


def section_header(num: int, total: int, title: str) -> None:
    print(f"\n[{num}/{total}] {title}")
    print(f"  {H_LINE * 78}")


def fmt_overhead(wall: float, gpu: float) -> str:
    if wall <= 0:
        return "  \u2014"
    return f"{(wall - gpu) / wall * 100:3.0f}%"


def fmt_pct(part: float, total: float) -> str:
    if total <= 0:
        return "  \u2014"
    return f"{part / total * 100:5.1f}%"


def fmt_mem(mb: float) -> str:
    if abs(mb) < 0.05:
        return "    \u2014"
    return f"{mb:+7.1f}"


# ═══════════════════════════════════════════════════════════════════════════
# DrivAer data preparation
# ═══════════════════════════════════════════════════════════════════════════


def load_drivaer_sample(data_dir: Path, cache_dir: Path) -> DrivAerMLSample:
    """Load the first DrivAerML training sample from cache or raw data.

    Attempts cache first for speed, falls back to full preprocessing.
    Returns the full-resolution sample (no boundary subsampling applied).
    """
    sample_paths = DrivAerMLDataSet.get_split_paths(data_dir, "train")
    if not sample_paths:
        raise FileNotFoundError(f"No training samples found in {data_dir}")
    sample_path = sample_paths[0]

    cache_pt = (cache_dir / sample_path.name).with_suffix(".pt")
    if cache_pt.exists():
        return torch.load(cache_pt, weights_only=False)
    return DrivAerMLDataSet.preprocess(sample_path)


def prepare_case_data(
    raw_sample: DrivAerMLSample,
    n_faces_per_boundary: int,
    n_prediction_points: int,
    device: torch.device,
    seed: int = 0,
) -> tuple[dict[str, Mesh], torch.Tensor, dict[str, torch.Tensor]]:
    """Create boundary mesh and prediction points for one benchmark case.

    Subsamples the full-resolution prediction mesh to create boundary
    geometry at the specified face count, precomputes lazy geometry
    (matching the main-thread preprocessing in train.py), then transfers
    everything to *device*.
    """
    torch.manual_seed(seed)

    boundary = DrivAerMLDataSet._subsample_mesh(
        raw_sample.prediction_mesh, n_faces_per_boundary
    )
    ### Precompute lazy geometry on CPU before transfer (same as train.py)
    _ = boundary.cell_centroids
    _ = boundary.cell_areas
    _ = boundary.cell_normals
    boundary_meshes: dict[str, Mesh] = {"no_slip": boundary.to(device)}

    n_pts = min(n_prediction_points, raw_sample.prediction_mesh.n_points)
    mask = torch.randperm(raw_sample.prediction_mesh.n_points)[:n_pts]
    prediction_points = raw_sample.prediction_mesh.points[mask].to(device)

    ref_lengths = {
        k: raw_sample.reference_lengths[k].to(device)
        for k in raw_sample.reference_lengths.keys()
    }
    return boundary_meshes, prediction_points, ref_lengths


def collect_tree_stats(
    boundary_meshes: dict[str, Mesh],
    theta: float,
    leaf_size: int,
) -> dict:
    """Build a ClusterTree and collect dual-tree interaction statistics.

    Mimics GLOBE's internal tree construction (area normalization by
    reference_area) to get accurate near/far counts and compression
    ratios for the DrivAer car body geometry.
    """
    mesh = boundary_meshes["no_slip"]
    areas = mesh.cell_areas / DRIVAER_REFERENCE_AREA
    source_points = mesh.cell_centroids

    tree = ClusterTree.from_points(source_points, leaf_size=leaf_size, areas=areas)
    plan = tree.find_dual_interaction_pairs(target_tree=tree, theta=theta)

    n_faces = mesh.n_cells
    n_a2a = n_faces * n_faces
    n_kernel_evals = plan.n_near + plan.n_nf + plan.n_fn + plan.n_far_nodes

    return {
        "tree_depth": int(tree.max_depth.item()),
        "tree_nodes": tree.n_nodes,
        "tree_leaves": int((tree.leaf_count > 0).sum()),
        "n_near": plan.n_near,
        "n_far_nodes": plan.n_far_nodes,
        "n_nf": plan.n_nf,
        "n_fn": plan.n_fn,
        "compression_ratio": n_a2a / max(1, n_kernel_evals),
        "n_faces": n_faces,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase-level breakdown
# ═══════════════════════════════════════════════════════════════════════════


def run_phase_breakdown(
    raw_sample: DrivAerMLSample,
    *,
    n_faces_per_boundary: int,
    n_prediction_points: int,
    theta: float,
    leaf_size: int,
    model_kwargs: dict,
    device: torch.device,
    amp: bool,
    n_warmup: int,
    n_trials: int,
) -> tuple[list[PhaseResult], dict]:
    """Per-phase timing of a training step using DrivAer data.

    Times tree construction, dual interaction planning, full GLOBE
    forward pass, and a complete training step (fwd+bwd) independently
    to identify where GPU time is spent with real car body geometry.
    """
    boundary_meshes, prediction_points, ref_lengths = prepare_case_data(
        raw_sample, n_faces_per_boundary, n_prediction_points, device
    )
    mesh = boundary_meshes["no_slip"]
    n_faces = mesh.n_cells
    source_points = mesh.cell_centroids
    areas = mesh.cell_areas / DRIVAER_REFERENCE_AREA

    rows: list[PhaseResult] = []
    stats: dict = {}

    ### Phase 1: Tree construction
    tree = ClusterTree.from_points(source_points, leaf_size=leaf_size, areas=areas)
    m0 = mem_mb(device)
    w, g = time_fn(
        lambda: ClusterTree.from_points(
            source_points, leaf_size=leaf_size, areas=areas
        ),
        device,
        n_warmup,
        n_trials,
    )
    m1 = mem_mb(device)
    depth = int(tree.max_depth.item())
    n_nodes = tree.n_nodes
    n_leaves = int((tree.leaf_count > 0).sum())
    rows.append(
        PhaseResult(
            "Tree construction",
            w,
            g,
            m1 - m0,
            f"depth={depth} nodes={n_nodes} leaves={n_leaves}",
        )
    )
    stats.update(tree_depth=depth, tree_nodes=n_nodes, tree_leaves=n_leaves)

    ### Phase 2: Dual interaction planning (comm: boundary -> boundary)
    comm_plan = tree.find_dual_interaction_pairs(target_tree=tree, theta=theta)
    m0 = mem_mb(device)
    w, g = time_fn(
        lambda: tree.find_dual_interaction_pairs(target_tree=tree, theta=theta),
        device,
        n_warmup,
        n_trials,
    )
    m1 = mem_mb(device)
    n_a2a = n_faces * n_faces
    n_evals = comm_plan.n_near + comm_plan.n_nf + comm_plan.n_fn + comm_plan.n_far_nodes
    comp = n_a2a / max(1, n_evals)
    rows.append(
        PhaseResult(
            "Dual plan (comm)",
            w,
            g,
            m1 - m0,
            f"near={comm_plan.n_near:,} nf={comm_plan.n_nf:,} "
            f"fn={comm_plan.n_fn:,} far={comm_plan.n_far_nodes:,} "
            f"ratio={comp:.1f}x",
        )
    )
    stats.update(
        comm_n_near=comm_plan.n_near,
        comm_n_far_nodes=comm_plan.n_far_nodes,
        comm_n_nf=comm_plan.n_nf,
        comm_n_fn=comm_plan.n_fn,
        comm_compression=comp,
    )

    ### Phase 3: Dual interaction planning (pred: boundary -> prediction)
    pred_tree = ClusterTree.from_points(prediction_points, leaf_size=leaf_size)
    pred_plan = tree.find_dual_interaction_pairs(target_tree=pred_tree, theta=theta)
    m0 = mem_mb(device)
    w, g = time_fn(
        lambda: tree.find_dual_interaction_pairs(target_tree=pred_tree, theta=theta),
        device,
        n_warmup,
        n_trials,
    )
    m1 = mem_mb(device)
    n_a2a_pred = prediction_points.shape[0] * n_faces
    n_evals_pred = (
        pred_plan.n_near + pred_plan.n_nf + pred_plan.n_fn + pred_plan.n_far_nodes
    )
    comp_pred = n_a2a_pred / max(1, n_evals_pred)
    rows.append(
        PhaseResult(
            "Dual plan (pred)",
            w,
            g,
            m1 - m0,
            f"near={pred_plan.n_near:,} nf={pred_plan.n_nf:,} "
            f"fn={pred_plan.n_fn:,} far={pred_plan.n_far_nodes:,} "
            f"ratio={comp_pred:.1f}x",
        )
    )

    ### Phase 4: Source aggregation
    source_data = TensorDict(
        {"normals": mesh.cell_normals},
        batch_size=[n_faces],
        device=device,
    )
    m0 = mem_mb(device)
    w, g = time_fn(
        lambda: tree.compute_source_aggregates(
            source_points=source_points,
            areas=areas,
            source_data=source_data,
        ),
        device,
        n_warmup,
        n_trials,
    )
    m1 = mem_mb(device)
    rows.append(PhaseResult("Source aggregation", w, g, m1 - m0))

    ### Phase 5: Full GLOBE.forward (inference)
    model = GLOBE(**model_kwargs).to(device).eval()

    def globe_fwd():
        return model(
            prediction_points=prediction_points,
            boundary_meshes=boundary_meshes,
            reference_lengths=ref_lengths,
        )

    with torch.no_grad():
        globe_fwd()
        m0 = mem_mb(device)
        w, g = time_fn(globe_fwd, device, n_warmup, n_trials)
        m1 = mem_mb(device)
    n_comm = model_kwargs.get("n_communication_hyperlayers", 2)
    rows.append(
        PhaseResult(
            "GLOBE.forward (inference)",
            w,
            g,
            m1 - m0,
            f"{n_comm} comm + 1 final hyperlayer",
        )
    )

    ### Phase 6: Full training step (forward + backward)
    del model
    clean_gpu(device)
    model_train = GLOBE(**model_kwargs).to(device)
    ts_result = time_training_step(
        model_train,
        boundary_meshes,
        prediction_points,
        ref_lengths,
        device=device,
        amp=amp,
        n_warmup=n_warmup,
        n_trials=n_trials,
    )
    if ts_result is not None:
        bwd_fwd = ts_result.backward_ms / max(ts_result.forward_ms, 1e-6)
        rows.append(
            PhaseResult(
                "Training step (fwd)",
                ts_result.forward_ms,
                ts_result.forward_ms,
            )
        )
        rows.append(
            PhaseResult(
                "Training step (bwd)",
                ts_result.backward_ms,
                ts_result.backward_ms,
                notes=f"bwd/fwd = {bwd_fwd:.1f}x",
            )
        )
        stats["training_step"] = asdict(ts_result)
    else:
        rows.append(PhaseResult("Training step", 0, 0, notes="OOM"))

    stats["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
    stats["n_faces"] = n_faces
    stats["n_prediction_points"] = prediction_points.shape[0]

    del model_train
    clean_gpu(device)
    return rows, stats


# ═══════════════════════════════════════════════════════════════════════════
# Profiler analysis
# ═══════════════════════════════════════════════════════════════════════════


def run_profiler_analysis(
    raw_sample: DrivAerMLSample,
    *,
    n_faces_per_boundary: int,
    n_prediction_points: int,
    model_kwargs: dict,
    device: torch.device,
    amp: bool,
    n_warmup: int,
) -> tuple[list[ProfileRegion], list[ProfileRegion], list[ProfileRegion]]:
    """Deep profiler analysis: record_function regions and top backward ops."""
    boundary_meshes, prediction_points, ref_lengths = prepare_case_data(
        raw_sample, n_faces_per_boundary, n_prediction_points, device
    )
    model = GLOBE(**model_kwargs).to(device)

    def fwd_only():
        with torch.no_grad():
            model.eval()
            model(
                prediction_points=prediction_points,
                boundary_meshes=boundary_meshes,
                reference_lengths=ref_lengths,
            )

    def fwd_bwd():
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            pred_mesh = model(
                prediction_points=prediction_points,
                boundary_meshes=boundary_meshes,
                reference_lengths=ref_lengths,
            )
            loss = sum(
                v.float().sum()
                for v in pred_mesh.point_data.values(
                    include_nested=True, leaves_only=True
                )
            )
            loss.backward()
        model.zero_grad(set_to_none=True)

    regions_fwd: list[ProfileRegion] = []
    regions_fwd_bwd: list[ProfileRegion] = []
    top_bwd_ops: list[ProfileRegion] = []

    try:
        regions_fwd = profiler_run(fwd_only, device, n_warmup=n_warmup)
    except torch.cuda.OutOfMemoryError:
        print("    OOM during inference profiling, skipping.", flush=True)
        clean_gpu(device)

    try:
        regions_fwd_bwd = profiler_run(fwd_bwd, device, n_warmup=n_warmup)
    except torch.cuda.OutOfMemoryError:
        print("    OOM during fwd+bwd profiling, skipping.", flush=True)
        clean_gpu(device)

    try:
        top_bwd_ops = profiler_top_backward_ops(fwd_bwd, device, n_warmup=n_warmup)
    except torch.cuda.OutOfMemoryError:
        print("    OOM during backward op profiling, skipping.", flush=True)
        clean_gpu(device)

    del model
    clean_gpu(device)
    return regions_fwd, regions_fwd_bwd, top_bwd_ops


# ═══════════════════════════════════════════════════════════════════════════
# Sweep case definition
# ═══════════════════════════════════════════════════════════════════════════


def build_cases(
    *,
    baseline_n_faces: int,
    baseline_n_pred: int,
    baseline_theta: float,
    baseline_leaf_size: int,
    baseline_n_comm: int,
    baseline_hidden: tuple[int, ...],
    baseline_n_sph: int,
    baseline_n_latent_scalars: int,
    baseline_n_latent_vectors: int,
    baseline_amp: bool,
    quick: bool,
    n_faces_values: tuple[int, ...] | None = None,
    n_pred_values: tuple[int, ...] | None = None,
    theta_values: tuple[float, ...] | None = None,
    leaf_size_values: tuple[int, ...] | None = None,
    n_comm_values: tuple[int, ...] | None = None,
    hidden_values: tuple[tuple[int, ...], ...] | None = None,
    n_sph_values: tuple[int, ...] | None = None,
    n_latent_scalar_values: tuple[int, ...] | None = None,
    n_latent_vector_values: tuple[int, ...] | None = None,
) -> list[BenchmarkCase]:
    """Build the full set of benchmark cases.

    Each sweep varies one parameter from baseline and holds the rest
    constant.  Returns a flat list suitable for round-robin distribution
    across GPUs.
    """
    ### Default sweep ranges
    if n_faces_values is None:
        n_faces_values = (
            (10_000, 40_000, 80_000)
            if quick
            else (5_000, 10_000, 20_000, 40_000, 80_000, 120_000)
        )
    if n_pred_values is None:
        n_pred_values = (
            (4_096, 40_000, 80_000) if quick else (1_024, 4_096, 16_384, 40_000, 80_000)
        )
    if theta_values is None:
        theta_values = (0.7, 1.0, 2.0) if quick else (0.7, 1.0, 1.5, 2.0, 3.0)
    if leaf_size_values is None:
        leaf_size_values = (1, 2, 4, 8) if quick else (1, 2, 4, 8)
    if n_comm_values is None:
        n_comm_values = (1, 2, 4) if quick else (1, 2, 3, 4)
    if hidden_values is None:
        hidden_values = (
            ((64, 64, 64), (256, 256, 256))
            if quick
            else ((32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256))
        )
    if n_sph_values is None:
        n_sph_values = (1, 4, 8) if quick else (1, 2, 4, 8)
    if n_latent_scalar_values is None:
        n_latent_scalar_values = (4, 16) if quick else (2, 4, 8, 16)
    if n_latent_vector_values is None:
        n_latent_vector_values = (2, 8) if quick else (1, 2, 4, 8)

    cases: list[BenchmarkCase] = []

    def _base(**kw) -> dict:
        """Defaults shared by every case."""
        return dict(
            model_overrides=kw.pop("model_overrides", {}),
            n_faces_per_boundary=kw.pop("n_faces_per_boundary", baseline_n_faces),
            n_prediction_points=kw.pop("n_prediction_points", baseline_n_pred),
            amp=kw.pop("amp", baseline_amp),
            compile_mode=kw.pop("compile_mode", None),
            use_grad_checkpointing=kw.pop("use_grad_checkpointing", True),
            **kw,
        )

    ### n_faces_per_boundary sweep (n_pred matched to n_faces)
    for val in n_faces_values:
        cases.append(
            BenchmarkCase(
                sweep_name="n_faces_per_boundary",
                label=f"n_faces={val:,}",
                sort_key=float(val),
                **_base(n_faces_per_boundary=val, n_prediction_points=val),
            )
        )

    ### n_prediction_points sweep
    for val in n_pred_values:
        cases.append(
            BenchmarkCase(
                sweep_name="n_prediction_points",
                label=f"n_pred={val:,}",
                sort_key=float(val),
                **_base(n_prediction_points=val),
            )
        )

    ### theta sweep
    for val in theta_values:
        cases.append(
            BenchmarkCase(
                sweep_name="theta",
                label=f"theta={val}",
                sort_key=val,
                **_base(model_overrides={"theta": val}),
            )
        )

    ### leaf_size sweep
    for val in leaf_size_values:
        cases.append(
            BenchmarkCase(
                sweep_name="leaf_size",
                label=f"leaf_size={val}",
                sort_key=float(val),
                **_base(model_overrides={"leaf_size": val}),
            )
        )

    ### n_communication_hyperlayers sweep
    for val in n_comm_values:
        cases.append(
            BenchmarkCase(
                sweep_name="n_communication_hyperlayers",
                label=f"n_comm={val}",
                sort_key=float(val),
                **_base(model_overrides={"n_communication_hyperlayers": val}),
            )
        )

    ### hidden_layer_sizes sweep
    for val in hidden_values:
        cases.append(
            BenchmarkCase(
                sweep_name="hidden_layer_sizes",
                label=f"hidden={list(val)}",
                sort_key=float(val[0]),
                **_base(model_overrides={"hidden_layer_sizes": list(val)}),
            )
        )

    ### n_spherical_harmonics sweep
    for val in n_sph_values:
        cases.append(
            BenchmarkCase(
                sweep_name="n_spherical_harmonics",
                label=f"n_sph={val}",
                sort_key=float(val),
                **_base(model_overrides={"n_spherical_harmonics": val}),
            )
        )

    ### n_latent_scalars sweep
    for val in n_latent_scalar_values:
        cases.append(
            BenchmarkCase(
                sweep_name="n_latent_scalars",
                label=f"n_ls={val}",
                sort_key=float(val),
                **_base(model_overrides={"n_latent_scalars": val}),
            )
        )

    ### n_latent_vectors sweep
    for val in n_latent_vector_values:
        cases.append(
            BenchmarkCase(
                sweep_name="n_latent_vectors",
                label=f"n_lv={val}",
                sort_key=float(val),
                **_base(model_overrides={"n_latent_vectors": val}),
            )
        )

    ### AMP comparison
    for amp_val in [False, True]:
        cases.append(
            BenchmarkCase(
                sweep_name="amp",
                label=f"amp={'on' if amp_val else 'off'}",
                sort_key=float(amp_val),
                **_base(amp=amp_val),
            )
        )

    ### Gradient checkpointing comparison
    for gc_val in [True, False]:
        cases.append(
            BenchmarkCase(
                sweep_name="grad_checkpointing",
                label=f"grad_ckpt={'on' if gc_val else 'off'}",
                sort_key=float(gc_val),
                **_base(use_grad_checkpointing=gc_val),
            )
        )

    ### torch.compile comparison (placed last - compile has persistent state)
    compile_modes: list[str | None] = (
        [None, "max-autotune-no-cudagraphs"]
        if quick
        else [None, "default", "max-autotune-no-cudagraphs"]
    )
    for i, mode in enumerate(compile_modes):
        label = "uncompiled" if mode is None else f"compile({mode})"
        cases.append(
            BenchmarkCase(
                sweep_name="torch_compile",
                label=label,
                sort_key=float(i),
                **_base(compile_mode=mode),
            )
        )

    ### n_faces + AMP cross-sweep (does bfloat16 enable larger meshes?)
    for val in n_faces_values:
        cases.append(
            BenchmarkCase(
                sweep_name="n_faces_amp",
                label=f"n_faces={val:,} amp",
                sort_key=float(val),
                **_base(n_faces_per_boundary=val, n_prediction_points=val, amp=True),
            )
        )

    ### leaf_size x theta cross-sweep (are these independent speed knobs?)
    ls_theta_grid: list[tuple[int, float]] = (
        [(4, 1.0), (4, 2.0), (8, 1.0), (8, 2.0)]
        if quick
        else [(ls, th) for ls in (4, 8) for th in (0.7, 1.0, 1.5, 2.0)]
    )
    for ls, th in ls_theta_grid:
        cases.append(
            BenchmarkCase(
                sweep_name="leaf_size_theta",
                label=f"ls={ls} th={th}",
                sort_key=ls + th / 10,
                **_base(model_overrides={"leaf_size": ls, "theta": th}),
            )
        )

    ### Frontier sweep (maximum feasible resolution with all knobs turned)
    frontier_n_faces: tuple[int, ...] = (
        (40_000, 80_000) if quick else (40_000, 60_000, 80_000, 120_000)
    )
    for val in frontier_n_faces:
        cases.append(
            BenchmarkCase(
                sweep_name="frontier",
                label=f"n_faces={val:,} frontier",
                sort_key=float(val),
                **_base(
                    n_faces_per_boundary=val,
                    n_prediction_points=val,
                    amp=True,
                    model_overrides={"theta": 1.5, "leaf_size": 1},
                ),
            )
        )

    return cases


# ═══════════════════════════════════════════════════════════════════════════
# Case execution
# ═══════════════════════════════════════════════════════════════════════════


def run_case(
    case: BenchmarkCase,
    raw_sample: DrivAerMLSample,
    baseline_model_kwargs: dict,
    device: torch.device,
    n_warmup: int,
    n_trials: int,
    rank: int = 0,
) -> SweepPoint:
    """Execute one benchmark case and return timing results."""
    sp = SweepPoint(
        sweep_name=case.sweep_name,
        label=case.label,
        sort_key=case.sort_key,
        config={
            "n_faces_per_boundary": case.n_faces_per_boundary,
            "n_prediction_points": case.n_prediction_points,
            "amp": case.amp,
            "compile_mode": case.compile_mode,
            "use_grad_checkpointing": case.use_grad_checkpointing,
            **case.model_overrides,
        },
        n_faces=case.n_faces_per_boundary,
        n_prediction_points=case.n_prediction_points,
    )
    clean_gpu(device)

    try:
        ### Prepare data
        boundary_meshes, prediction_points, ref_lengths = prepare_case_data(
            raw_sample, case.n_faces_per_boundary, case.n_prediction_points, device
        )
        sp.n_prediction_points = prediction_points.shape[0]

        ### Collect tree stats
        theta = case.model_overrides.get("theta", baseline_model_kwargs["theta"])
        leaf_size = case.model_overrides.get(
            "leaf_size", baseline_model_kwargs["leaf_size"]
        )
        stats = collect_tree_stats(boundary_meshes, theta, leaf_size)
        sp.n_faces = stats["n_faces"]
        sp.tree_depth = stats["tree_depth"]
        sp.tree_nodes = stats["tree_nodes"]
        sp.tree_leaves = stats["tree_leaves"]
        sp.n_near = stats["n_near"]
        sp.n_far_nodes = stats["n_far_nodes"]
        sp.n_nf = stats["n_nf"]
        sp.n_fn = stats["n_fn"]
        sp.compression_ratio = stats["compression_ratio"]

        ### Build model
        kwargs = {**baseline_model_kwargs, **case.model_overrides}
        model = GLOBE(**kwargs).to(device)
        if not case.use_grad_checkpointing:
            for module in model.modules():
                if isinstance(module, BarnesHutKernel):
                    module.use_gradient_checkpointing = False

        ### Time training step
        if case.compile_mode is not None:
            torch._dynamo.reset()
            result = _time_compiled_step(
                model,
                boundary_meshes,
                prediction_points,
                ref_lengths,
                compile_mode=case.compile_mode,
                device=device,
                amp=case.amp,
                n_warmup=max(n_warmup, 5),
                n_trials=n_trials,
            )
            torch._dynamo.reset()
        else:
            result = time_training_step(
                model,
                boundary_meshes,
                prediction_points,
                ref_lengths,
                device=device,
                amp=case.amp,
                n_warmup=n_warmup,
                n_trials=n_trials,
            )

        if result is None:
            sp.oom = True
        else:
            sp.forward_ms = result.forward_ms
            sp.backward_ms = result.backward_ms
            sp.total_ms = result.total_ms
            sp.peak_alloc_gb = result.peak_alloc_gb
            sp.peak_reserved_gb = result.peak_reserved_gb

        del model
    except torch.cuda.OutOfMemoryError:
        sp.oom = True
        print(f"  [rank {rank}] OOM: {case.label}", flush=True)

    clean_gpu(device)
    return sp


# ═══════════════════════════════════════════════════════════════════════════
# Multi-GPU dispatch
# ═══════════════════════════════════════════════════════════════════════════


def dispatch_and_gather(
    all_cases: list[BenchmarkCase],
    raw_sample: DrivAerMLSample,
    baseline_model_kwargs: dict,
    device: torch.device,
    rank: int,
    world_size: int,
    n_warmup: int,
    n_trials: int,
) -> list[SweepPoint]:
    """Distribute cases across GPUs, run locally, gather results on rank 0."""
    my_cases = all_cases[rank::world_size]
    print(
        f"  [rank {rank}/{world_size}] Running {len(my_cases)}/{len(all_cases)} cases",
        flush=True,
    )

    my_results: list[SweepPoint] = []
    for i, case in enumerate(my_cases):
        print(
            f"  [rank {rank}] Case {i + 1}/{len(my_cases)}: "
            f"{case.sweep_name}/{case.label}",
            flush=True,
        )
        result = run_case(
            case, raw_sample, baseline_model_kwargs, device, n_warmup, n_trials, rank
        )
        my_results.append(result)
        status = "OOM" if result.oom else f"{result.total_ms:,.0f}ms"
        print(f"  [rank {rank}]   -> {status}", flush=True)

    ### Gather results on rank 0
    if world_size > 1:
        gathered: list[list[SweepPoint] | None] = [None] * world_size
        torch.distributed.gather_object(
            my_results, gathered if rank == 0 else None, dst=0
        )
        if rank == 0:
            merged: list[SweepPoint] = []
            for shard in gathered:
                if shard is not None:
                    merged.extend(shard)
            return merged
        return []
    return my_results


# ═══════════════════════════════════════════════════════════════════════════
# Printing utilities
# ═══════════════════════════════════════════════════════════════════════════

PROFILER_NESTING = {
    "globe::": 0,
    "multiscale_kernel::": 1,
    "bh_kernel::": 2,
    "kernel::": 3,
}


def _indent_for(name: str) -> int:
    for prefix, level in PROFILER_NESTING.items():
        if name.startswith(prefix):
            return level
    return 0


def print_phase_table(rows: list[PhaseResult], stats: dict) -> None:
    """Print the phase-level breakdown table."""
    globe_wall = 0.0
    for r in rows:
        if r.name == "GLOBE.forward (inference)":
            globe_wall = r.wall_ms

    name_w = max(len(r.name) for r in rows) + 2
    print(
        f"  {'Phase':<{name_w}}  {'Wall':>8}  {'GPU':>8}  "
        f"{'Overhead':>8}  {'% Fwd':>7}  {'Mem':>7}  Notes"
    )
    print(
        f"  {H_LINE * name_w}  {H_LINE * 8}  {H_LINE * 8}  "
        f"{H_LINE * 8}  {H_LINE * 7}  {H_LINE * 7}  {H_LINE * 5}"
    )
    for r in rows:
        print(
            f"  {r.name:<{name_w}}  {r.wall_ms:7.1f}ms {r.gpu_ms:7.1f}ms "
            f"  {fmt_overhead(r.wall_ms, r.gpu_ms):>8}  "
            f"{fmt_pct(r.wall_ms, globe_wall):>7}  "
            f"{fmt_mem(r.mem_delta_mb):>7}  {r.notes}"
        )

    print(f"\n  Peak GPU memory: {stats.get('peak_mem_gb', 0):.2f} GB")
    if "comm_compression" in stats:
        print(
            f"  Comm compression: {stats['comm_compression']:.1f}x "
            f"[near={stats['comm_n_near']:,}, nf={stats.get('comm_n_nf', 0):,}, "
            f"fn={stats.get('comm_n_fn', 0):,}, "
            f"far_nodes={stats.get('comm_n_far_nodes', 0):,}]"
        )
    if "tree_depth" in stats:
        print(
            f"  Tree: depth={stats['tree_depth']}  "
            f"nodes={stats['tree_nodes']}  leaves={stats['tree_leaves']}"
        )
    print(
        f"  DrivAer: {stats.get('n_faces', 0):,} boundary faces, "
        f"{stats.get('n_prediction_points', 0):,} prediction points"
    )


def print_profiler_table(regions: list[ProfileRegion]) -> None:
    """Print profiler regions as a table with conceptual nesting."""
    if not regions:
        print("  (no record_function regions captured)")
        return

    total_cpu = sum(r.cpu_ms for r in regions if r.name.startswith("globe::"))
    if total_cpu <= 0:
        total_cpu = max(r.cpu_ms for r in regions) if regions else 1.0

    sorted_regions = sorted(
        regions,
        key=lambda r: (
            0
            if r.name.startswith("globe::")
            else 1
            if r.name.startswith("multiscale_kernel::")
            else 2
            if r.name.startswith("bh_kernel::")
            else 3,
            r.name,
        ),
    )

    print(f"  {'Region':<55} {'CPU':>8}  {'CUDA':>8}  {'Calls':>5}  {'% Top':>6}")
    print(f"  {H_LINE * 55} {H_LINE * 8}  {H_LINE * 8}  {H_LINE * 5}  {H_LINE * 6}")

    for r in sorted_regions:
        indent = "  " * _indent_for(r.name)
        short_name = r.name.split("::")[-1] if "::" in r.name else r.name
        display = f"{indent}{short_name}"
        pct = f"{r.cpu_ms / total_cpu * 100:5.1f}%" if total_cpu > 0 else "    \u2014"
        print(
            f"  {display:<55} {r.cpu_ms:7.1f}ms {r.cuda_ms:7.1f}ms "
            f"{r.count:>5}  {pct:>6}"
        )


def print_top_ops_table(ops: list[ProfileRegion]) -> None:
    """Print top backward ops by CUDA time."""
    if not ops:
        print("  (no CUDA ops captured)")
        return
    total_cuda = sum(r.cuda_ms for r in ops)
    print(f"  {'Operation':<55} {'CUDA':>8}  {'Calls':>5}  {'% Total':>7}")
    print(f"  {H_LINE * 55} {H_LINE * 8}  {H_LINE * 5}  {H_LINE * 7}")
    for r in ops:
        name = r.name[:55]
        pct = (
            f"{r.cuda_ms / total_cuda * 100:5.1f}%" if total_cuda > 0 else "    \u2014"
        )
        print(f"  {name:<55} {r.cuda_ms:7.1f}ms {r.count:>5}  {pct:>7}")


def print_sweep_table(
    points: list[SweepPoint],
    title: str,
    show_tree: bool = False,
) -> None:
    """Print a parameter sweep table with timing, memory, and tree stats."""
    if not points:
        print(f"  (no results for {title})")
        return

    ### Sort by sweep parameter value
    points = sorted(points, key=lambda sp: sp.sort_key)

    baseline_ms: float | None = None
    for sp in points:
        if not sp.oom and sp.total_ms > 0:
            baseline_ms = sp.total_ms
            break

    header_parts = [f"  {'Config':<30} {'Forward':>10} {'Backward':>10} {'Total':>10}"]
    header_parts.append(f" {'Alloc':>6} {'Rsvd':>6}")
    header_parts.append(f" {'Near':>10} {'Far':>8} {'Compress':>8}")
    if show_tree:
        header_parts.append(f" {'Depth':>5} {'Nodes':>6} {'Leaves':>6}")
    header_parts.append(f" {'Speedup':>7}")
    print("".join(header_parts))

    sep_parts = [f"  {H_LINE * 30} {H_LINE * 10} {H_LINE * 10} {H_LINE * 10}"]
    sep_parts.append(f" {H_LINE * 6} {H_LINE * 6}")
    sep_parts.append(f" {H_LINE * 10} {H_LINE * 8} {H_LINE * 8}")
    if show_tree:
        sep_parts.append(f" {H_LINE * 5} {H_LINE * 6} {H_LINE * 6}")
    sep_parts.append(f" {H_LINE * 7}")
    print("".join(sep_parts))

    for sp in points:
        if sp.oom:
            oom_row = (
                f"  {sp.label:<30}        OOM        OOM        OOM"
                f"    OOM    OOM"
                f" {'':>10} {'':>8} {'':>8}"
            )
            if show_tree:
                oom_row += f" {'':>5} {'':>6} {'':>6}"
            oom_row += "     ---"
            print(oom_row)
            continue
        spd = (
            f"{baseline_ms / sp.total_ms:>5.1f}x"
            if baseline_ms and sp.total_ms > 0
            else "   ---"
        )
        n_far_total = sp.n_far_nodes + sp.n_nf + sp.n_fn
        parts = [
            f"  {sp.label:<30} {sp.forward_ms:>8,.0f}ms {sp.backward_ms:>8,.0f}ms "
            f"{sp.total_ms:>8,.0f}ms",
            f" {sp.peak_alloc_gb:>5.1f}G {sp.peak_reserved_gb:>5.1f}G",
            f" {sp.n_near:>10,} {n_far_total:>8,} {sp.compression_ratio:>7.1f}x",
        ]
        if show_tree:
            parts.append(f" {sp.tree_depth:>5} {sp.tree_nodes:>6} {sp.tree_leaves:>6}")
        parts.append(f" {spd:>7}")
        print("".join(parts))


def print_scale_analysis(points: list[SweepPoint], param_name: str) -> None:
    """Log-log regression to estimate scaling exponent."""
    non_oom = [sp for sp in points if not sp.oom and sp.total_ms > 0]
    if len(non_oom) < 2:
        return

    if param_name == "n_faces_per_boundary":
        vals = [float(sp.n_faces) for sp in non_oom]
    elif param_name == "n_prediction_points":
        vals = [float(sp.n_prediction_points) for sp in non_oom]
    else:
        return

    if any(v <= 0 for v in vals):
        return

    log_n = [log(v) for v in vals]
    log_t = [log(sp.total_ms) for sp in non_oom]
    n = len(log_n)
    sum_x = sum(log_n)
    sum_y = sum(log_t)
    sum_xy = sum(x * y for x, y in zip(log_n, log_t))
    sum_xx = sum(x * x for x in log_n)
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) > 1e-12:
        alpha = (n * sum_xy - sum_x * sum_y) / denom
        if alpha < 1.2:
            regime = "sub-linear or linear"
        elif alpha < 1.6:
            regime = "N-log-N"
        elif alpha < 2.2:
            regime = "quadratic"
        else:
            regime = "super-quadratic"
        print(
            f"\n  Estimated scaling exponent ({param_name}): "
            f"alpha = {alpha:.2f}  (t ~ N^alpha, {regime})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Recommendations
# ═══════════════════════════════════════════════════════════════════════════


def generate_recommendations(
    results: AllResults,
    sweep_groups: dict[str, list[SweepPoint]],
) -> list[str]:
    """Generate actionable recommendations from benchmark results."""
    recs: list[str] = []

    ### CPU overhead from phase breakdown
    for pr in results.phase_results:
        wall, gpu = pr.get("wall_ms", 0), pr.get("gpu_ms", 0)
        if wall > 0 and (wall - gpu) / wall > 0.5:
            name = pr.get("name", "?")
            overhead = (wall - gpu) / wall * 100
            recs.append(
                f"CPU OVERHEAD: '{name}' has {overhead:.0f}% overhead "
                f"(wall={wall:.1f}ms vs gpu={gpu:.1f}ms). "
                f"Bottleneck is Python/CPU, not GPU compute."
            )

    ### Backward/forward ratio
    ts = results.training_step
    if ts and ts.get("forward_ms", 0) > 0:
        ratio = ts.get("backward_ms", 0) / ts["forward_ms"]
        if ratio > 3.0:
            recs.append(
                f"BACKWARD RATIO: backward/forward = {ratio:.1f}x (unusually high). "
                f"Check gradient checkpointing settings and autocast."
            )

    ### n_faces_per_boundary scaling
    if "n_faces_per_boundary" in sweep_groups:
        pts = sweep_groups["n_faces_per_boundary"]
        non_oom = [
            sp for sp in pts if not sp.oom and sp.total_ms > 0 and sp.n_faces > 0
        ]
        if len(non_oom) >= 2:
            log_n = [log(sp.n_faces) for sp in non_oom]
            log_t = [log(sp.total_ms) for sp in non_oom]
            n = len(log_n)
            s_x, s_y = sum(log_n), sum(log_t)
            s_xy = sum(x * y for x, y in zip(log_n, log_t))
            s_xx = sum(x * x for x in log_n)
            denom = n * s_xx - s_x * s_x
            if abs(denom) > 1e-12:
                alpha = (n * s_xy - s_x * s_y) / denom
                regime = (
                    "sub-linear or linear"
                    if alpha < 1.2
                    else "N-log-N"
                    if alpha < 1.6
                    else "quadratic"
                    if alpha < 2.2
                    else "super-quadratic"
                )
                recs.append(
                    f"SCALING (n_faces): alpha={alpha:.2f} ({regime}). "
                    f"The non-convex DrivAer car body has less favorable "
                    f"Barnes-Hut compression than convex shapes."
                )

    ### Theta sensitivity
    if "theta" in sweep_groups:
        pts = sweep_groups["theta"]
        non_oom = [sp for sp in pts if not sp.oom and sp.total_ms > 0]
        if len(non_oom) >= 2:
            slowest = max(non_oom, key=lambda sp: sp.total_ms)
            fastest = min(non_oom, key=lambda sp: sp.total_ms)
            if slowest.total_ms > 0:
                speedup = slowest.total_ms / fastest.total_ms
                recs.append(
                    f"THETA: {fastest.label} is {speedup:.1f}x faster than "
                    f"{slowest.label} (compression "
                    f"{fastest.compression_ratio:.1f}x vs "
                    f"{slowest.compression_ratio:.1f}x). "
                    f"Accuracy degrades with larger theta - validate on held-out data."
                )

    ### Leaf size
    if "leaf_size" in sweep_groups:
        pts = sweep_groups["leaf_size"]
        non_oom = [sp for sp in pts if not sp.oom and sp.total_ms > 0]
        if len(non_oom) >= 2:
            best = min(non_oom, key=lambda sp: sp.total_ms)
            recs.append(
                f"LEAF SIZE: Fastest at {best.label} "
                f"({best.total_ms:,.0f}ms, {best.peak_alloc_gb:.1f}GB alloc). "
                f"Larger leaves reduce tree overhead but increase near-field cost."
            )

    ### AMP
    if "amp" in sweep_groups:
        pts = sweep_groups["amp"]
        non_oom = {sp.label: sp for sp in pts if not sp.oom and sp.total_ms > 0}
        amp_on = non_oom.get("amp=on")
        amp_off = non_oom.get("amp=off")
        if amp_on and amp_off and amp_off.total_ms > 0:
            speedup = amp_off.total_ms / amp_on.total_ms
            mem_saved = amp_off.peak_alloc_gb - amp_on.peak_alloc_gb
            recs.append(
                f"AMP: bfloat16 gives {speedup:.2f}x speedup and "
                f"saves {mem_saved:.1f} GB."
                + (" Worth enabling." if speedup > 1.1 else " Marginal benefit.")
            )

    ### Gradient checkpointing
    if "grad_checkpointing" in sweep_groups:
        pts = sweep_groups["grad_checkpointing"]
        non_oom = {sp.label: sp for sp in pts if not sp.oom and sp.total_ms > 0}
        gc_on = non_oom.get("grad_ckpt=on")
        gc_off = non_oom.get("grad_ckpt=off")
        if gc_on and gc_off:
            speed_ratio = gc_on.total_ms / max(gc_off.total_ms, 1e-6)
            mem_saved = gc_off.peak_alloc_gb - gc_on.peak_alloc_gb
            recs.append(
                f"GRADIENT CHECKPOINTING: Saves {mem_saved:.2f} GB "
                f"but is {speed_ratio:.2f}x the speed. "
                + (
                    "Worth keeping for memory savings."
                    if mem_saved > 0.5
                    else "Consider disabling for speed."
                )
            )
        oom_labels = {sp.label for sp in pts if sp.oom}
        if "grad_ckpt=off" in oom_labels and "grad_ckpt=on" not in oom_labels:
            recs.append("GRADIENT CHECKPOINTING: Required to avoid OOM. Keep enabled.")

    ### torch.compile
    if "torch_compile" in sweep_groups:
        pts = sweep_groups["torch_compile"]
        non_oom = [sp for sp in pts if not sp.oom and sp.total_ms > 0]
        if len(non_oom) >= 2:
            uncompiled = next((sp for sp in non_oom if sp.label == "uncompiled"), None)
            if uncompiled:
                for sp in non_oom:
                    if sp is not uncompiled and sp.total_ms > 0:
                        speedup = uncompiled.total_ms / sp.total_ms
                        if speedup > 1.3:
                            recs.append(
                                f"COMPILE: '{sp.label}' gives {speedup:.1f}x speedup. "
                                f"Worth enabling."
                            )

    ### n_communication_hyperlayers
    if "n_communication_hyperlayers" in sweep_groups:
        pts = sweep_groups["n_communication_hyperlayers"]
        non_oom = sorted(
            [sp for sp in pts if not sp.oom and sp.total_ms > 0],
            key=lambda sp: sp.sort_key,
        )
        if len(non_oom) >= 2:
            baseline = non_oom[0]
            for sp in non_oom[1:]:
                cost_per_layer = (sp.total_ms - baseline.total_ms) / max(
                    sp.sort_key - baseline.sort_key, 1
                )
                if cost_per_layer > baseline.total_ms * 0.5:
                    recs.append(
                        f"COMM LAYERS: Each extra layer costs ~{cost_per_layer:,.0f}ms. "
                        f"Verify the accuracy benefit justifies the cost."
                    )
                    break

    if not recs:
        recs.append("No obvious bottlenecks detected. Performance looks healthy.")

    return recs


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main(
    data_dir: Path | None = None,
    n_faces_per_boundary: int = 40_000,
    n_prediction_points: int = 40_000,
    theta: float = 1.0,
    leaf_size: int = 1,
    hidden_layer_sizes: tuple[int, ...] = (128, 128, 128),
    n_communication_hyperlayers: int = 2,
    n_latent_scalars: int = 8,
    n_latent_vectors: int = 4,
    n_spherical_harmonics: int = 4,
    n_warmup: int = 3,
    n_trials: int = 5,
    amp: bool = False,
    quick: bool = False,
    skip_phase_breakdown: bool = False,
    skip_profiler: bool = False,
    skip_sweeps: bool = False,
    n_faces_values: tuple[int, ...] | None = None,
    n_pred_values: tuple[int, ...] | None = None,
    theta_values: tuple[float, ...] | None = None,
    leaf_size_values: tuple[int, ...] | None = None,
    n_comm_values: tuple[int, ...] | None = None,
    hidden_values: tuple[tuple[int, ...], ...] | None = None,
    n_sph_values: tuple[int, ...] | None = None,
    n_latent_scalar_values: tuple[int, ...] | None = None,
    n_latent_vector_values: tuple[int, ...] | None = None,
    save_json: str | None = None,
):
    """Comprehensive GPU-side training benchmark for DrivAerML GLOBE.

    Measures how forward+backward training step performance scales with
    n_faces_per_boundary, theta, leaf_size, and other model parameters
    using real DrivAerML car body geometry.  Distributes sweep cases
    across all available GPUs for parallel execution.

    Args:
        data_dir: Path to the DrivAerML dataset root.  Falls back to
            ``DRIVAER_DATA_DIR`` env var.
        n_faces_per_boundary: Baseline boundary mesh face count.
        n_prediction_points: Baseline prediction point count.
        theta: Baseline Barnes-Hut opening angle.
        leaf_size: Baseline max sources per tree leaf.
        hidden_layer_sizes: Baseline kernel MLP hidden sizes.
        n_communication_hyperlayers: Baseline GLOBE comm layers.
        n_latent_scalars: Scalar latent channels.
        n_latent_vectors: Vector latent channels.
        n_spherical_harmonics: Legendre polynomial terms.
        n_warmup: Warmup iterations (not timed).
        n_trials: Timed iterations (reports median).
        amp: Baseline AMP setting.
        quick: Reduced sweep ranges and fewer trials.
        skip_phase_breakdown: Skip phase-level analysis (rank 0 only).
        skip_profiler: Skip torch.profiler analysis (rank 0 only).
        skip_sweeps: Skip all parameter sweeps.
        n_faces_values: Override n_faces_per_boundary sweep values.
        n_pred_values: Override n_prediction_points sweep values.
        theta_values: Override theta sweep values.
        leaf_size_values: Override leaf_size sweep values.
        n_comm_values: Override n_communication_hyperlayers sweep values.
        hidden_values: Override hidden_layer_sizes sweep values.
        n_sph_values: Override n_spherical_harmonics sweep values.
        n_latent_scalar_values: Override n_latent_scalars sweep values.
        n_latent_vector_values: Override n_latent_vectors sweep values.
        save_json: Path to save machine-readable results as JSON.
    """
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required.", file=sys.stderr)
        sys.exit(1)

    if quick:
        n_warmup = min(n_warmup, 1)
        n_trials = min(n_trials, 3)
        skip_profiler = True

    ### Resolve data directory
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

    ### Distributed setup (works with torchrun or standalone)
    DistributedManager.initialize()
    dist = DistributedManager()
    rank = dist.rank
    world_size = dist.world_size
    device = dist.device
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")

    is_rank0 = rank == 0
    if not is_rank0:
        warnings.filterwarnings("ignore")

    ### Load DrivAer sample (all ranks load independently from Lustre cache)
    if is_rank0:
        print(f"Loading DrivAer sample from {data_dir}...", flush=True)
    raw_sample = load_drivaer_sample(data_dir, cache_dir)
    original_n_cells = raw_sample.prediction_mesh.n_cells
    original_n_points = raw_sample.prediction_mesh.n_points

    ### Baseline GLOBE configuration
    ref_names = list(raw_sample.reference_lengths.keys())
    model_kwargs = dict(
        n_spatial_dims=3,
        output_field_ranks={"C_p": 0, "C_f": 1},
        boundary_source_data_ranks={"no_slip": {}},
        reference_length_names=ref_names,
        reference_area=DRIVAER_REFERENCE_AREA,
        n_communication_hyperlayers=n_communication_hyperlayers,
        hidden_layer_sizes=list(hidden_layer_sizes),
        n_latent_scalars=n_latent_scalars,
        n_latent_vectors=n_latent_vectors,
        n_spherical_harmonics=n_spherical_harmonics,
        theta=theta,
        leaf_size=leaf_size,
    )

    all_results = AllResults()
    n_sections = 4 - skip_phase_breakdown - skip_profiler - skip_sweeps
    section_num = 0

    # ── Section 1: System and DrivAer info ─────────────────────────────
    if is_rank0:
        gpu_name = torch.cuda.get_device_name(device)
        total_vram = torch.cuda.get_device_properties(device).total_memory / 1024**3
        cc = torch.cuda.get_device_properties(device).major

        hdr_w = 90
        print(f"\n{'=' * hdr_w}")
        print("  DrivAerML GLOBE GPU Training Benchmark")
        print(f"{'=' * hdr_w}")
        print(f"  GPU:                 {gpu_name}  ({total_vram:.1f} GB, SM {cc}x)")
        print(f"  GPUs (world_size):   {world_size}")
        print(
            f"  DrivAer mesh:        {original_n_cells:,} cells, "
            f"{original_n_points:,} points"
        )
        print(f"  Reference lengths:   {ref_names}")
        print(f"  Reference area:      {DRIVAER_REFERENCE_AREA}")
        print(
            f"  Baseline:            n_faces={n_faces_per_boundary:,}  "
            f"n_pred={n_prediction_points:,}  theta={theta}  leaf_size={leaf_size}"
        )
        print(
            f"  Model:               hidden={list(hidden_layer_sizes)}  "
            f"n_comm={n_communication_hyperlayers}  n_sph={n_spherical_harmonics}"
        )
        print(
            f"  Latent channels:     {n_latent_scalars} scalar, "
            f"{n_latent_vectors} vector"
        )
        print(f"  Benchmark:           {n_warmup} warmup, {n_trials} trials (median)")
        print(f"  AMP (bfloat16):      {'yes' if amp else 'no'}")
        print(f"{'=' * hdr_w}")

        all_results.system_info = {
            "gpu_name": gpu_name,
            "total_vram_gb": total_vram,
            "world_size": world_size,
            "n_warmup": n_warmup,
            "n_trials": n_trials,
            "amp": amp,
        }
        all_results.drivaer_info = {
            "original_n_cells": original_n_cells,
            "original_n_points": original_n_points,
            "reference_lengths": ref_names,
            "reference_area": DRIVAER_REFERENCE_AREA,
            "baseline_n_faces_per_boundary": n_faces_per_boundary,
            "baseline_n_prediction_points": n_prediction_points,
            "baseline_theta": theta,
            "baseline_leaf_size": leaf_size,
            "baseline_hidden_layer_sizes": list(hidden_layer_sizes),
            "baseline_n_communication_hyperlayers": n_communication_hyperlayers,
        }

    # ── Section 2: Phase-level breakdown (rank 0 only) ─────────────────
    if not skip_phase_breakdown:
        if world_size > 1:
            torch.distributed.barrier()
        section_num += 1
        if is_rank0:
            section_header(
                section_num,
                n_sections,
                "Phase-Level Breakdown (DrivAer, baseline config)",
            )
            phase_rows, phase_stats = run_phase_breakdown(
                raw_sample,
                n_faces_per_boundary=n_faces_per_boundary,
                n_prediction_points=n_prediction_points,
                theta=theta,
                leaf_size=leaf_size,
                model_kwargs=model_kwargs,
                device=device,
                amp=amp,
                n_warmup=n_warmup,
                n_trials=n_trials,
            )
            print_phase_table(phase_rows, phase_stats)
            all_results.phase_results = [asdict(r) for r in phase_rows]
            if "training_step" in phase_stats:
                all_results.training_step = phase_stats["training_step"]
        if world_size > 1:
            torch.distributed.barrier()

    # ── Section 3: Profiler analysis (rank 0 only) ─────────────────────
    clean_gpu(device)
    if not skip_profiler:
        if world_size > 1:
            torch.distributed.barrier()
        section_num += 1
        if is_rank0:
            section_header(section_num, n_sections, "Deep Profiler Analysis (DrivAer)")
            regions_fwd, regions_fwd_bwd, top_bwd_ops = run_profiler_analysis(
                raw_sample,
                n_faces_per_boundary=n_faces_per_boundary,
                n_prediction_points=n_prediction_points,
                model_kwargs=model_kwargs,
                device=device,
                amp=amp,
                n_warmup=n_warmup,
            )
            print("\n  Forward pass (inference) - record_function regions:")
            print_profiler_table(regions_fwd)
            all_results.profiler_fwd = [asdict(r) for r in regions_fwd]

            print("\n  Forward + backward (train) - record_function regions:")
            print_profiler_table(regions_fwd_bwd)
            all_results.profiler_fwd_bwd = [asdict(r) for r in regions_fwd_bwd]

            print("\n  Top CUDA ops during forward + backward (by CUDA time):")
            print_top_ops_table(top_bwd_ops)
            all_results.profiler_top_bwd_ops = [asdict(r) for r in top_bwd_ops]
        if world_size > 1:
            torch.distributed.barrier()

    # ── Section 4: Distributed parameter sweeps ────────────────────────
    clean_gpu(device)
    sweep_results: list[SweepPoint] = []

    if not skip_sweeps:
        section_num += 1
        if is_rank0:
            section_header(
                section_num,
                n_sections,
                f"Parameter Sweeps ({world_size} GPU{'s' if world_size > 1 else ''})",
            )

        all_cases = build_cases(
            baseline_n_faces=n_faces_per_boundary,
            baseline_n_pred=n_prediction_points,
            baseline_theta=theta,
            baseline_leaf_size=leaf_size,
            baseline_n_comm=n_communication_hyperlayers,
            baseline_hidden=hidden_layer_sizes,
            baseline_n_sph=n_spherical_harmonics,
            baseline_n_latent_scalars=n_latent_scalars,
            baseline_n_latent_vectors=n_latent_vectors,
            baseline_amp=amp,
            quick=quick,
            n_faces_values=n_faces_values,
            n_pred_values=n_pred_values,
            theta_values=theta_values,
            leaf_size_values=leaf_size_values,
            n_comm_values=n_comm_values,
            hidden_values=hidden_values,
            n_sph_values=n_sph_values,
            n_latent_scalar_values=n_latent_scalar_values,
            n_latent_vector_values=n_latent_vector_values,
        )

        if is_rank0:
            print(f"  Total cases: {len(all_cases)}", flush=True)

        sweep_results = dispatch_and_gather(
            all_cases,
            raw_sample,
            model_kwargs,
            device,
            rank,
            world_size,
            n_warmup,
            n_trials,
        )

        ### Print results grouped by sweep dimension (rank 0 only)
        if is_rank0:
            sweep_groups: dict[str, list[SweepPoint]] = {}
            for sp in sweep_results:
                sweep_groups.setdefault(sp.sweep_name, []).append(sp)

            SWEEP_ORDER = [
                "n_faces_per_boundary",
                "n_faces_amp",
                "frontier",
                "n_prediction_points",
                "theta",
                "leaf_size",
                "leaf_size_theta",
                "n_communication_hyperlayers",
                "hidden_layer_sizes",
                "n_spherical_harmonics",
                "n_latent_scalars",
                "n_latent_vectors",
                "amp",
                "grad_checkpointing",
                "torch_compile",
            ]
            SHOW_TREE = {
                "leaf_size",
                "n_faces_per_boundary",
                "n_faces_amp",
                "leaf_size_theta",
                "frontier",
            }
            SHOW_SCALE = {
                "n_faces_per_boundary",
                "n_faces_amp",
                "n_prediction_points",
                "frontier",
            }
            for sweep_name in SWEEP_ORDER:
                if sweep_name not in sweep_groups:
                    continue
                pts = sweep_groups[sweep_name]
                print(f"\n  --- {sweep_name} ---")
                print_sweep_table(pts, sweep_name, show_tree=sweep_name in SHOW_TREE)
                if sweep_name in SHOW_SCALE:
                    print_scale_analysis(pts, sweep_name)

            all_results.sweeps = {
                name: [asdict(sp) for sp in pts] for name, pts in sweep_groups.items()
            }

    # ── Summary and recommendations ────────────────────────────────────
    if is_rank0:
        section_num += 1
        section_header(section_num, n_sections + 1, "Summary and Recommendations")
        sweep_groups_for_recs: dict[str, list[SweepPoint]] = {}
        for sp in sweep_results:
            sweep_groups_for_recs.setdefault(sp.sweep_name, []).append(sp)

        recs = generate_recommendations(all_results, sweep_groups_for_recs)
        all_results.recommendations = recs
        for i, rec in enumerate(recs, 1):
            print(f"  {i}. {rec}")
        print()

        if save_json:
            Path(save_json).write_text(json.dumps(asdict(all_results), indent=2))
            print(f"  Results saved to {save_json}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
