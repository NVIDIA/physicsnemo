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

"""Comprehensive GLOBE Barnes-Hut benchmark.

Combines phase-level profiling (wall + GPU timing with overhead analysis),
deep torch.profiler-based sub-operation tracing, training step analysis,
gradient-checkpointing comparison, parameter sensitivity sweeps (theta,
leaf_size), torch.compile comparison, and mesh-scale analysis into a single
diagnostic tool.

Usage::

    uv run benchmark.py                          # full analysis
    uv run benchmark.py --quick                  # quick diagnostic
    uv run benchmark.py --subdivisions 6         # larger mesh
    uv run benchmark.py --save-json results.json # machine-readable output
"""

import gc
import json
import sys
from dataclasses import asdict, dataclass, field
from math import log
from time import perf_counter

import torch
from tensordict import TensorDict

from physicsnemo.experimental.models.globe.cluster_tree import ClusterTree
from physicsnemo.experimental.models.globe.field_kernel import (
    BarnesHutKernel,
    MultiscaleKernel,
)
from physicsnemo.experimental.models.globe.model import GLOBE
from physicsnemo.mesh.primitives.procedural import lumpy_sphere

# ═══════════════════════════════════════════════════════════════════════════
# Data classes for structured results
# ═══════════════════════════════════════════════════════════════════════════


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

    label: str
    config: dict
    oom: bool = False
    forward_ms: float = 0.0
    backward_ms: float = 0.0
    total_ms: float = 0.0
    peak_alloc_gb: float = 0.0
    peak_reserved_gb: float = 0.0
    n_near: int = 0
    n_far: int = 0
    compression_ratio: float = 0.0
    tree_depth: int = 0
    tree_nodes: int = 0
    tree_leaves: int = 0
    n_faces: int = 0


@dataclass
class ProfileRegion:
    """One region from torch.profiler output."""

    name: str
    cpu_ms: float
    cuda_ms: float
    count: int


@dataclass
class AllResults:
    """Container for all benchmark results (serializable to JSON)."""

    system_info: dict = field(default_factory=dict)
    phase_results: list[dict] = field(default_factory=list)
    profiler_fwd: list[dict] = field(default_factory=list)
    profiler_fwd_bwd: list[dict] = field(default_factory=list)
    profiler_top_bwd_ops: list[dict] = field(default_factory=list)
    training_step: dict = field(default_factory=dict)
    grad_ckpt_comparison: dict = field(default_factory=dict)
    theta_sweep: list[dict] = field(default_factory=list)
    leaf_size_sweep: list[dict] = field(default_factory=list)
    compile_comparison: list[dict] = field(default_factory=list)
    scale_sweep: list[dict] = field(default_factory=list)
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
    mesh,
    prediction_points: torch.Tensor,
    reference_lengths: dict[str, torch.Tensor],
    *,
    device: torch.device,
    amp: bool = False,
    n_warmup: int = 3,
    n_trials: int = 5,
) -> TrainingStepResult | None:
    """Time a full training step (forward + backward + zero_grad).

    Returns median timings across *n_trials*, or ``None`` on OOM.
    """
    model.train()
    boundary_meshes = {"no_slip": mesh}

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
            _step()
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


def profiler_run(
    fn,
    device: torch.device,
    n_warmup: int = 2,
) -> list[ProfileRegion]:
    """Run *fn* under torch.profiler, return record_function regions.

    Runs *n_warmup* untimed iterations, then one profiled iteration.
    Returns a flat list of ``ProfileRegion`` for every ``record_function``
    region observed (identified by the ``globe::``, ``multiscale_kernel::``,
    ``bh_kernel::``, or ``kernel::`` prefix).
    """
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
# Benchmark sections
# ═══════════════════════════════════════════════════════════════════════════


def run_phase_breakdown(
    mesh,
    prediction_points: torch.Tensor,
    *,
    theta: float,
    leaf_size: int,
    hidden_layer_sizes: list[int],
    n_spherical_harmonics: int,
    n_communication_hyperlayers: int,
    n_latent_scalars: int,
    n_latent_vectors: int,
    device: torch.device,
    n_warmup: int,
    n_trials: int,
) -> tuple[list[PhaseResult], dict]:
    """Section 2: phase-level forward pass breakdown with manual timing."""
    n_faces = mesh.n_cells
    source_points = mesh.cell_centroids
    source_areas = mesh.cell_areas
    _ = mesh.cell_normals

    n_prediction_points = prediction_points.shape[0]
    rows: list[PhaseResult] = []
    stats: dict = {}

    ### Phase 1: Tree construction
    tree = ClusterTree.from_points(
        source_points,
        leaf_size=leaf_size,
        areas=source_areas,
    )
    m0 = mem_mb(device)
    w, g = time_fn(
        lambda: ClusterTree.from_points(
            source_points,
            leaf_size=leaf_size,
            areas=source_areas,
        ),
        device,
        n_warmup,
        n_trials,
    )
    m1 = mem_mb(device)
    n_nodes = tree.n_nodes
    n_leaves = int((tree.leaf_count > 0).sum())
    depth = int(tree.max_depth.item())
    rows.append(
        PhaseResult(
            "Tree construction",
            w,
            g,
            m1 - m0,
            f"depth={depth} nodes={n_nodes} leaves={n_leaves}",
        )
    )
    stats["tree_depth"] = depth
    stats["tree_nodes"] = n_nodes
    stats["tree_leaves"] = n_leaves

    ### Phase 2: Interaction planning (communication)
    comm_plan = tree.find_interaction_pairs(source_points, theta=theta)
    m0 = mem_mb(device)
    w, g = time_fn(
        lambda: tree.find_interaction_pairs(source_points, theta=theta),
        device,
        n_warmup,
        n_trials,
    )
    m1 = mem_mb(device)
    n_a2a_comm = n_faces * n_faces
    comp_comm = n_a2a_comm / max(1, comm_plan.n_total)
    rows.append(
        PhaseResult(
            "Find pairs (comm)",
            w,
            g,
            m1 - m0,
            f"near={comm_plan.n_near:,} far={comm_plan.n_far:,} "
            f"total={comm_plan.n_total:,} ratio={comp_comm:.1f}x",
        )
    )
    stats["comm_n_near"] = comm_plan.n_near
    stats["comm_n_far"] = comm_plan.n_far
    stats["comm_n_total"] = comm_plan.n_total
    stats["comm_compression"] = comp_comm

    ### Phase 3: Interaction planning (prediction)
    pred_plan = tree.find_interaction_pairs(prediction_points, theta=theta)
    m0 = mem_mb(device)
    w, g = time_fn(
        lambda: tree.find_interaction_pairs(prediction_points, theta=theta),
        device,
        n_warmup,
        n_trials,
    )
    m1 = mem_mb(device)
    n_a2a_pred = n_prediction_points * n_faces
    comp_pred = n_a2a_pred / max(1, pred_plan.n_total)
    rows.append(
        PhaseResult(
            "Find pairs (pred)",
            w,
            g,
            m1 - m0,
            f"near={pred_plan.n_near:,} far={pred_plan.n_far:,} "
            f"total={pred_plan.n_total:,} ratio={comp_pred:.1f}x",
        )
    )

    ### Phase 4: Source aggregation
    source_data = TensorDict(
        {"normals": mesh.cell_normals},
        batch_size=[n_faces],
        device=device,
    )
    m0 = mem_mb(device)
    agg = tree.compute_source_aggregates(
        source_points=source_points,
        areas=source_areas,
        source_data=source_data,
    )
    w, g = time_fn(
        lambda: tree.compute_source_aggregates(
            source_points=source_points,
            areas=source_areas,
            source_data=source_data,
        ),
        device,
        n_warmup,
        n_trials,
    )
    m1 = mem_mb(device)
    rows.append(PhaseResult("Source aggregation", w, g, m1 - m0))

    ### Phase 5: Node strengths
    output_ranks = {"C_p": 0, "C_f": 1}
    bh_kernel = (
        BarnesHutKernel(
            n_spatial_dims=3,
            output_field_ranks=output_ranks,
            source_data_ranks={"normals": 1},
            hidden_layer_sizes=hidden_layer_sizes,
            n_spherical_harmonics=n_spherical_harmonics,
            leaf_size=leaf_size,
            use_gradient_checkpointing=False,
        )
        .to(device)
        .eval()
    )

    strengths = torch.ones(n_faces, device=device)
    m0 = mem_mb(device)
    w, g = time_fn(
        lambda: bh_kernel._compute_node_strengths(tree, strengths),
        device,
        n_warmup,
        n_trials,
    )
    m1 = mem_mb(device)
    rows.append(PhaseResult("Node strengths", w, g, m1 - m0))

    ### Phase 6: BarnesHutKernel.forward (comm config)
    ref_len = torch.tensor(1.0, device=device)
    bh_kwargs = dict(
        reference_length=ref_len,
        source_points=source_points,
        target_points=source_points,
        source_strengths=strengths,
        source_data=source_data,
        theta=theta,
        cluster_tree=tree,
        interaction_plan=comm_plan,
        source_areas=source_areas,
        source_aggregates=agg,
    )
    with torch.no_grad():
        bh_kernel(**bh_kwargs)
        m0 = mem_mb(device)
        w, g = time_fn(lambda: bh_kernel(**bh_kwargs), device, n_warmup, n_trials)
        m1 = mem_mb(device)
    chunk_sz = bh_kernel._auto_chunk_size(comm_plan.n_total, device)
    n_chunks = max(1, -(-comm_plan.n_total // chunk_sz))
    rows.append(
        PhaseResult(
            "BH kernel (comm)",
            w,
            g,
            m1 - m0,
            f"chunk_size={chunk_sz:,} n_chunks={n_chunks}",
        )
    )
    stats["chunk_size"] = chunk_sz
    stats["n_chunks"] = n_chunks

    ### Phase 7: MultiscaleKernel.forward (comm config)
    ref_names = ["L_ref", "sqrt_A_ref"]
    ms_kernel = (
        MultiscaleKernel(
            n_spatial_dims=3,
            output_field_ranks=output_ranks,
            reference_length_names=ref_names,
            source_data_ranks={"normals": 1},
            hidden_layer_sizes=hidden_layer_sizes,
            n_spherical_harmonics=n_spherical_harmonics,
            leaf_size=leaf_size,
            use_gradient_checkpointing=False,
        )
        .to(device)
        .eval()
    )

    ref_lengths = {n: torch.tensor(1.0, device=device) for n in ref_names}
    ms_strengths = TensorDict(
        {n: strengths.clone() for n in ref_names},
        batch_size=[n_faces],
        device=device,
    )
    ms_kwargs = dict(
        reference_lengths=ref_lengths,
        source_points=source_points,
        target_points=source_points,
        source_strengths=ms_strengths,
        source_data=source_data,
        theta=theta,
        cluster_tree=tree,
        interaction_plan=comm_plan,
        source_areas=source_areas,
    )
    with torch.no_grad():
        ms_kernel(**ms_kwargs)
        m0 = mem_mb(device)
        w, g = time_fn(lambda: ms_kernel(**ms_kwargs), device, n_warmup, n_trials)
        m1 = mem_mb(device)
    rows.append(
        PhaseResult(
            "MultiscaleKernel (comm)",
            w,
            g,
            m1 - m0,
            f"{len(ref_names)} branches",
        )
    )

    ### Phase 8: Full GLOBE.forward (inference)
    model = (
        GLOBE(
            n_spatial_dims=3,
            output_field_ranks={"C_p": 0, "C_f": 1},
            boundary_source_data_ranks={"no_slip": {}},
            reference_length_names=ref_names,
            reference_area=1.0,
            n_communication_hyperlayers=n_communication_hyperlayers,
            hidden_layer_sizes=hidden_layer_sizes,
            n_latent_scalars=n_latent_scalars,
            n_latent_vectors=n_latent_vectors,
            n_spherical_harmonics=n_spherical_harmonics,
            theta=theta,
            leaf_size=leaf_size,
        )
        .to(device)
        .eval()
    )

    def globe_call():
        return model(
            prediction_points=prediction_points,
            boundary_meshes={"no_slip": mesh},
            reference_lengths=ref_lengths,
        )
    with torch.no_grad():
        globe_call()
        m0 = mem_mb(device)
        w, g = time_fn(globe_call, device, n_warmup, n_trials)
        m1 = mem_mb(device)
    rows.append(
        PhaseResult(
            "GLOBE.forward (inference)",
            w,
            g,
            m1 - m0,
            f"{n_communication_hyperlayers} comm + 1 final",
        )
    )

    stats["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3

    del model, ms_kernel, bh_kernel
    clean_gpu(device)

    return rows, stats


def run_deep_profiler(
    mesh,
    prediction_points: torch.Tensor,
    ref_lengths: dict[str, torch.Tensor],
    *,
    model_kwargs: dict,
    device: torch.device,
    n_warmup: int,
) -> tuple[list[ProfileRegion], list[ProfileRegion], list[ProfileRegion]]:
    """Section 3: deep profiler analysis (inference + forward+backward)."""
    model = GLOBE(**model_kwargs).to(device)
    boundary_meshes = {"no_slip": mesh}

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
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
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


def run_training_step_analysis(
    mesh,
    prediction_points: torch.Tensor,
    ref_lengths: dict[str, torch.Tensor],
    *,
    model_kwargs: dict,
    device: torch.device,
    amp: bool,
    n_warmup: int,
    n_trials: int,
) -> TrainingStepResult | None:
    """Section 4: training step analysis."""
    model = GLOBE(**model_kwargs).to(device)
    result = time_training_step(
        model,
        mesh,
        prediction_points,
        ref_lengths,
        device=device,
        amp=amp,
        n_warmup=n_warmup,
        n_trials=n_trials,
    )
    del model
    clean_gpu(device)
    return result


def run_grad_ckpt_comparison(
    mesh,
    prediction_points: torch.Tensor,
    ref_lengths: dict[str, torch.Tensor],
    *,
    model_kwargs: dict,
    device: torch.device,
    amp: bool,
    n_warmup: int,
    n_trials: int,
) -> dict:
    """Section 5: gradient checkpointing comparison."""
    results = {}
    for ckpt_enabled in [True, False]:
        label = "with_grad_ckpt" if ckpt_enabled else "without_grad_ckpt"
        try:
            model = GLOBE(**model_kwargs).to(device)
            for module in model.modules():
                if isinstance(module, BarnesHutKernel):
                    module.use_gradient_checkpointing = ckpt_enabled
            result = time_training_step(
                model,
                mesh,
                prediction_points,
                ref_lengths,
                device=device,
                amp=amp,
                n_warmup=n_warmup,
                n_trials=n_trials,
            )
            if result is not None:
                results[label] = asdict(result)
            else:
                results[label] = {"oom": True}
            del model
        except torch.cuda.OutOfMemoryError:
            results[label] = {"oom": True}
            print(f"    OOM for {label}, skipping.", flush=True)
        clean_gpu(device)

    return results


def run_theta_sweep(
    mesh,
    prediction_points: torch.Tensor,
    ref_lengths: dict[str, torch.Tensor],
    *,
    model_kwargs: dict,
    theta_values: tuple[float, ...],
    device: torch.device,
    amp: bool,
    n_warmup: int,
    n_trials: int,
) -> list[SweepPoint]:
    """Section 6: theta sensitivity sweep."""
    n_faces = mesh.n_cells
    source_points = mesh.cell_centroids
    source_areas = mesh.cell_areas
    n_a2a = n_faces * n_faces
    points: list[SweepPoint] = []

    for theta in theta_values:
        clean_gpu(device)
        sp = SweepPoint(
            label=f"theta={theta}", config={"theta": theta}, n_faces=n_faces
        )
        try:
            tree = ClusterTree.from_points(
                source_points,
                leaf_size=model_kwargs["leaf_size"],
                areas=source_areas,
            )
            plan = tree.find_interaction_pairs(source_points, theta=theta)
            sp.compression_ratio = n_a2a / max(1, plan.n_total)
            sp.tree_depth = int(tree.max_depth.item())
            sp.n_near = plan.n_near
            sp.n_far = plan.n_far
            sp.tree_nodes = tree.n_nodes
            sp.tree_leaves = int((tree.leaf_count > 0).sum())

            kwargs = {**model_kwargs, "theta": theta}
            model = GLOBE(**kwargs).to(device)
            result = time_training_step(
                model,
                mesh,
                prediction_points,
                ref_lengths,
                device=device,
                amp=amp,
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
            print(f"    OOM at theta={theta}, skipping.", flush=True)
        points.append(sp)
        clean_gpu(device)

    return points


def run_leaf_size_sweep(
    mesh,
    prediction_points: torch.Tensor,
    ref_lengths: dict[str, torch.Tensor],
    *,
    model_kwargs: dict,
    leaf_size_values: tuple[int, ...],
    device: torch.device,
    amp: bool,
    n_warmup: int,
    n_trials: int,
) -> list[SweepPoint]:
    """Section 7: leaf size sensitivity sweep."""
    n_faces = mesh.n_cells
    source_points = mesh.cell_centroids
    source_areas = mesh.cell_areas
    n_a2a = n_faces * n_faces
    theta = model_kwargs["theta"]
    points: list[SweepPoint] = []

    for leaf_size in leaf_size_values:
        clean_gpu(device)
        sp = SweepPoint(
            label=f"leaf_size={leaf_size}",
            config={"leaf_size": leaf_size},
            n_faces=n_faces,
        )
        try:
            tree = ClusterTree.from_points(
                source_points,
                leaf_size=leaf_size,
                areas=source_areas,
            )
            plan = tree.find_interaction_pairs(source_points, theta=theta)
            sp.compression_ratio = n_a2a / max(1, plan.n_total)
            sp.tree_depth = int(tree.max_depth.item())
            sp.n_near = plan.n_near
            sp.n_far = plan.n_far
            sp.tree_nodes = tree.n_nodes
            sp.tree_leaves = int((tree.leaf_count > 0).sum())

            kwargs = {**model_kwargs, "leaf_size": leaf_size}
            model = GLOBE(**kwargs).to(device)
            result = time_training_step(
                model,
                mesh,
                prediction_points,
                ref_lengths,
                device=device,
                amp=amp,
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
            print(f"    OOM at leaf_size={leaf_size}, skipping.", flush=True)
        points.append(sp)
        clean_gpu(device)

    return points


def run_compile_test(
    mesh,
    prediction_points: torch.Tensor,
    ref_lengths: dict[str, torch.Tensor],
    *,
    model_kwargs: dict,
    device: torch.device,
    amp: bool,
    n_warmup: int,
    n_trials: int,
) -> list[SweepPoint]:
    """Section 8: torch.compile comparison."""
    configs = [
        ("Uncompiled", None),
        ("compile(default)", "default"),
        ("compile(max-autotune)", "max-autotune-no-cudagraphs"),
    ]
    points: list[SweepPoint] = []
    boundary_meshes = {"no_slip": mesh}

    for label, compile_mode in configs:
        clean_gpu(device)
        torch._dynamo.reset()
        sp = SweepPoint(label=label, config={"compile_mode": compile_mode})
        try:
            model = GLOBE(**model_kwargs).to(device)
            model.train()

            if compile_mode is not None:

                def make_step_fn(mdl, compile_mode_str):
                    @torch.compile(dynamic=True, mode=compile_mode_str)
                    def _step(pp, bm, rl):
                        pred_mesh = mdl(
                            prediction_points=pp,
                            boundary_meshes=bm,
                            reference_lengths=rl,
                        )
                        return sum(
                            v.float().sum()
                            for v in pred_mesh.point_data.values(
                                include_nested=True, leaves_only=True
                            )
                        )

                    return _step

                step_fn = make_step_fn(model, compile_mode)

                warmup_count = max(n_warmup, 5)
                print(
                    f"    {label}: compiling ({warmup_count} warmup iters)...",
                    end="",
                    flush=True,
                )
                for _ in range(warmup_count):
                    try:
                        with torch.autocast(
                            device_type="cuda",
                            dtype=torch.bfloat16,
                            enabled=amp,
                        ):
                            loss = step_fn(
                                prediction_points,
                                boundary_meshes,
                                ref_lengths,
                            )
                            loss.backward()
                        model.zero_grad(set_to_none=True)
                        torch.cuda.synchronize(device)
                    except torch.cuda.OutOfMemoryError:
                        model.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        break
                print(" done.", flush=True)

            result = time_training_step(
                model,
                mesh,
                prediction_points,
                ref_lengths,
                device=device,
                amp=amp,
                n_warmup=0,
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
            print(f"    OOM for {label}, skipping.", flush=True)
        points.append(sp)
        clean_gpu(device)
        torch._dynamo.reset()

    return points


def run_scale_sweep(
    prediction_points: torch.Tensor,
    ref_lengths: dict[str, torch.Tensor],
    *,
    model_kwargs: dict,
    subdivision_values: tuple[int, ...],
    device: torch.device,
    amp: bool,
    n_warmup: int,
    n_trials: int,
) -> list[SweepPoint]:
    """Section 9: mesh scale analysis."""
    theta = model_kwargs["theta"]
    leaf_size = model_kwargs["leaf_size"]
    points: list[SweepPoint] = []

    for subdiv in subdivision_values:
        clean_gpu(device)
        sp = SweepPoint(
            label=f"subdiv={subdiv}",
            config={"subdivisions": subdiv},
        )
        try:
            mesh = lumpy_sphere.load(subdivisions=subdiv, device="cuda")
            n_faces = mesh.n_cells
            sp.n_faces = n_faces
            source_points = mesh.cell_centroids
            source_areas = mesh.cell_areas
            _ = mesh.cell_normals

            n_a2a = n_faces * n_faces
            tree = ClusterTree.from_points(
                source_points,
                leaf_size=leaf_size,
                areas=source_areas,
            )
            plan = tree.find_interaction_pairs(source_points, theta=theta)
            sp.compression_ratio = n_a2a / max(1, plan.n_total)
            sp.tree_depth = int(tree.max_depth.item())
            sp.n_near = plan.n_near
            sp.n_far = plan.n_far
            sp.tree_nodes = tree.n_nodes
            sp.tree_leaves = int((tree.leaf_count > 0).sum())

            model = GLOBE(**model_kwargs).to(device)
            result = time_training_step(
                model,
                mesh,
                prediction_points,
                ref_lengths,
                device=device,
                amp=amp,
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
            del model, mesh
        except torch.cuda.OutOfMemoryError:
            sp.oom = True
            print(f"    OOM at subdiv={subdiv}, skipping.", flush=True)
        points.append(sp)
        clean_gpu(device)

    return points


# ═══════════════════════════════════════════════════════════════════════════
# Summary and recommendations
# ═══════════════════════════════════════════════════════════════════════════


def generate_recommendations(results: AllResults) -> list[str]:
    """Generate actionable recommendations from benchmark results."""
    recs: list[str] = []

    ### CPU overhead
    for pr in results.phase_results:
        wall, gpu = pr.get("wall_ms", 0), pr.get("gpu_ms", 0)
        if wall > 0 and (wall - gpu) / wall > 0.5:
            name = pr.get("name", "?")
            overhead = (wall - gpu) / wall * 100
            recs.append(
                f"CPU OVERHEAD: '{name}' has {overhead:.0f}% overhead "
                f"(wall={wall:.1f}ms vs gpu={gpu:.1f}ms). "
                f"The bottleneck is Python control flow, not GPU compute."
            )

    ### Tree/plan cost
    globe_row = None
    tree_plan_total = 0.0
    for pr in results.phase_results:
        name = pr.get("name", "")
        wall = pr.get("wall_ms", 0)
        if name == "GLOBE.forward (inference)":
            globe_row = pr
        if name in ("Tree construction", "Find pairs (comm)", "Find pairs (pred)"):
            tree_plan_total += wall
    if globe_row and globe_row.get("wall_ms", 0) > 0:
        frac = tree_plan_total / globe_row["wall_ms"]
        if frac > 0.20:
            recs.append(
                f"TREE/PLAN COST: Tree construction + interaction planning = "
                f"{frac * 100:.0f}% of forward pass. Trees are already cached "
                f"across communication layers; consider also caching across "
                f"training iterations when the mesh geometry is fixed."
            )

    ### Backward/forward ratio
    ts = results.training_step
    if ts and ts.get("forward_ms", 0) > 0:
        ratio = ts["backward_ms"] / ts["forward_ms"]
        if ratio > 3.0:
            recs.append(
                f"BACKWARD RATIO: backward/forward = {ratio:.1f}x (unusually high). "
                f"Check gradient checkpointing settings and autocast."
            )

    ### Gradient checkpointing
    gc_data = results.grad_ckpt_comparison
    with_gc = gc_data.get("with_grad_ckpt", {})
    without_gc = gc_data.get("without_grad_ckpt", {})
    if (
        not with_gc.get("oom")
        and not without_gc.get("oom")
        and with_gc.get("total_ms", 0) > 0
        and without_gc.get("total_ms", 0) > 0
    ):
        speed_diff = with_gc["total_ms"] / without_gc["total_ms"]
        mem_saved = without_gc["peak_alloc_gb"] - with_gc["peak_alloc_gb"]
        recs.append(
            f"GRADIENT CHECKPOINTING: Enabled saves {mem_saved:.2f} GB "
            f"but is {speed_diff:.2f}x the speed. "
            + (
                "Worth keeping for memory savings."
                if mem_saved > 0.5
                else "Consider disabling for speed."
            )
        )
    elif without_gc.get("oom") and not with_gc.get("oom"):
        recs.append("GRADIENT CHECKPOINTING: Required to avoid OOM. Keep enabled.")

    ### Theta sensitivity
    if results.theta_sweep:
        non_oom = [sp for sp in results.theta_sweep if not sp.get("oom")]
        if len(non_oom) >= 2:
            baseline = non_oom[0]
            for sp in non_oom[1:]:
                if (
                    baseline["total_ms"] > 0
                    and sp["total_ms"] > 0
                    and baseline["total_ms"] / sp["total_ms"] > 2.0
                ):
                    speedup = baseline["total_ms"] / sp["total_ms"]
                    recs.append(
                        f"THETA: {sp['label']} is {speedup:.1f}x faster than "
                        f"{baseline['label']} (compression "
                        f"{sp['compression_ratio']:.1f}x vs "
                        f"{baseline['compression_ratio']:.1f}x). "
                        f"If accuracy permits, increase theta."
                    )

    ### Compile
    if results.compile_comparison:
        non_oom = [sp for sp in results.compile_comparison if not sp.get("oom")]
        if len(non_oom) >= 2:
            uncompiled = non_oom[0]
            for sp in non_oom[1:]:
                if (
                    uncompiled["total_ms"] > 0
                    and sp["total_ms"] > 0
                    and uncompiled["total_ms"] / sp["total_ms"] > 1.3
                ):
                    speedup = uncompiled["total_ms"] / sp["total_ms"]
                    recs.append(
                        f"COMPILE: '{sp['label']}' gives {speedup:.1f}x speedup. "
                        f"Worth enabling."
                    )

    ### Scaling exponent
    if results.scale_sweep:
        non_oom = [sp for sp in results.scale_sweep if not sp.get("oom")]
        if len(non_oom) >= 2:
            log_n = [log(sp["n_faces"]) for sp in non_oom]
            log_t = [log(sp["total_ms"]) for sp in non_oom]
            n = len(log_n)
            sum_x = sum(log_n)
            sum_y = sum(log_t)
            sum_xy = sum(x * y for x, y in zip(log_n, log_t))
            sum_xx = sum(x * x for x in log_n)
            denom = n * sum_xx - sum_x * sum_x
            if abs(denom) > 1e-12:
                alpha = (n * sum_xy - sum_x * sum_y) / denom
                if alpha < 1.2:
                    label = "sub-linear or linear"
                elif alpha < 1.6:
                    label = "N-log-N"
                elif alpha < 2.2:
                    label = "quadratic"
                else:
                    label = "super-quadratic"
                recs.append(
                    f"SCALING: Estimated exponent alpha={alpha:.2f} "
                    f"(t ~ N^alpha), consistent with {label} scaling."
                )

    if not recs:
        recs.append("No obvious bottlenecks detected. Performance looks healthy.")

    return recs


# ═══════════════════════════════════════════════════════════════════════════
# Printing
# ═══════════════════════════════════════════════════════════════════════════


def print_phase_table(rows: list[PhaseResult], stats: dict) -> None:
    """Print the phase-level breakdown table."""
    globe_wall = 0.0
    for r in rows:
        if r.name == "GLOBE.forward (inference)":
            globe_wall = r.wall_ms

    name_w = max(len(r.name) for r in rows) + 2
    print(
        f"  {'Phase':<{name_w}}  {'Wall':>8}  {'GPU':>8}  "
        f"{'Overhead':>8}  {'% Total':>7}  {'Mem':>7}  Notes"
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
            f"[near={stats['comm_n_near']:,}, far={stats['comm_n_far']:,}]"
        )
    if "chunk_size" in stats:
        print(
            f"  Chunking: {stats['n_chunks']} chunk(s) of size {stats['chunk_size']:,} "
            f"({stats['comm_n_total']:,} total pairs)"
        )
    if "tree_depth" in stats:
        print(
            f"  Tree: depth={stats['tree_depth']}  "
            f"nodes={stats['tree_nodes']}  leaves={stats['tree_leaves']}"
        )


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


def print_profiler_table(regions: list[ProfileRegion], title: str) -> None:
    """Print profiler regions as a table with conceptual nesting indentation."""
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


def print_training_step(result: TrainingStepResult | None) -> None:
    """Print training step timing."""
    if result is None:
        print("  OOM - could not complete training step.")
        return
    ratio = result.backward_ms / result.forward_ms if result.forward_ms > 0 else 0
    print(f"  Forward:       {result.forward_ms:>10,.1f} ms")
    print(f"  Backward:      {result.backward_ms:>10,.1f} ms")
    print(f"  Zero grad:     {result.zero_grad_ms:>10,.1f} ms")
    print(f"  Total:         {result.total_ms:>10,.1f} ms")
    print(f"  Bwd/Fwd ratio: {ratio:>10.2f}x")
    print(f"  Peak alloc:    {result.peak_alloc_gb:>10.2f} GB")
    print(f"  Peak reserved: {result.peak_reserved_gb:>10.2f} GB")


def print_grad_ckpt(data: dict) -> None:
    """Print gradient checkpointing comparison."""
    with_gc = data.get("with_grad_ckpt", {})
    without_gc = data.get("without_grad_ckpt", {})

    print(
        f"  {'Config':<28} {'Forward':>10} {'Backward':>10} "
        f"{'Total':>10} {'Alloc':>7} {'Rsvd':>7}"
    )
    print(
        f"  {H_LINE * 28} {H_LINE * 10} {H_LINE * 10} "
        f"{H_LINE * 10} {H_LINE * 7} {H_LINE * 7}"
    )

    for label, d in [
        ("With grad checkpointing", with_gc),
        ("Without grad checkpointing", without_gc),
    ]:
        if d.get("oom"):
            print(f"  {label:<28}        OOM        OOM        OOM     OOM     OOM")
        else:
            t = (
                d.get("forward_ms", 0)
                + d.get("backward_ms", 0)
                + d.get("zero_grad_ms", 0)
            )
            print(
                f"  {label:<28} {d.get('forward_ms', 0):>8,.0f}ms "
                f"{d.get('backward_ms', 0):>8,.0f}ms "
                f"{t:>8,.0f}ms "
                f"{d.get('peak_alloc_gb', 0):>5.1f}G "
                f"{d.get('peak_reserved_gb', 0):>5.1f}G"
            )


def print_sweep_table(points: list[SweepPoint], show_tree: bool = False) -> None:
    """Print a parameter sweep table."""
    baseline_ms: float | None = None
    for sp in points:
        if not sp.oom and sp.total_ms > 0:
            baseline_ms = sp.total_ms
            break

    header_parts = [f"  {'Config':<28} {'Forward':>10} {'Backward':>10} {'Total':>10}"]
    header_parts.append(f" {'Alloc':>6} {'Rsvd':>6}")
    header_parts.append(f" {'Near':>10} {'Far':>10} {'Compress':>8}")
    if show_tree:
        header_parts.append(f" {'Depth':>5} {'Nodes':>6} {'Leaves':>6}")
    header_parts.append(f" {'Speedup':>7}")
    print("".join(header_parts))

    sep_parts = [f"  {H_LINE * 28} {H_LINE * 10} {H_LINE * 10} {H_LINE * 10}"]
    sep_parts.append(f" {H_LINE * 6} {H_LINE * 6}")
    sep_parts.append(f" {H_LINE * 10} {H_LINE * 10} {H_LINE * 8}")
    if show_tree:
        sep_parts.append(f" {H_LINE * 5} {H_LINE * 6} {H_LINE * 6}")
    sep_parts.append(f" {H_LINE * 7}")
    print("".join(sep_parts))

    for sp in points:
        if sp.oom:
            print(
                f"  {sp.label:<28}        OOM        OOM        OOM"
                f"    OOM    OOM"
                f" {'':>10} {'':>10} {'':>8}"
                + (f" {'':>5} {'':>6} {'':>6}" if show_tree else "")
                + "     ---"
            )
            continue
        spd = (
            f"{baseline_ms / sp.total_ms:>5.1f}x"
            if baseline_ms and sp.total_ms > 0
            else "   ---"
        )
        parts = [
            f"  {sp.label:<28} {sp.forward_ms:>8,.0f}ms {sp.backward_ms:>8,.0f}ms "
            f"{sp.total_ms:>8,.0f}ms",
            f" {sp.peak_alloc_gb:>5.1f}G {sp.peak_reserved_gb:>5.1f}G",
            f" {sp.n_near:>10,} {sp.n_far:>10,} {sp.compression_ratio:>7.1f}x",
        ]
        if show_tree:
            parts.append(f" {sp.tree_depth:>5} {sp.tree_nodes:>6} {sp.tree_leaves:>6}")
        parts.append(f" {spd:>7}")
        print("".join(parts))


def print_scale_table(points: list[SweepPoint]) -> None:
    """Print the mesh scale sweep with scaling exponent estimate."""
    print_sweep_table(points)

    non_oom = [sp for sp in points if not sp.oom and sp.total_ms > 0]
    if len(non_oom) >= 2:
        log_n = [log(sp.n_faces) for sp in non_oom]
        log_t = [log(sp.total_ms) for sp in non_oom]
        n = len(log_n)
        sum_x = sum(log_n)
        sum_y = sum(log_t)
        sum_xy = sum(x * y for x, y in zip(log_n, log_t))
        sum_xx = sum(x * x for x in log_n)
        denom = n * sum_xx - sum_x * sum_x
        if abs(denom) > 1e-12:
            alpha = (n * sum_xy - sum_x * sum_y) / denom
            print(f"\n  Estimated scaling exponent: alpha = {alpha:.2f}  (t ~ N^alpha)")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main(
    subdivisions: int = 5,
    n_prediction_points: int = 2048,
    theta: float = 1.0,
    leaf_size: int = 1,
    hidden_layer_sizes: tuple[int, ...] = (64, 64, 64),
    n_communication_hyperlayers: int = 2,
    n_latent_scalars: int = 8,
    n_latent_vectors: int = 4,
    n_spherical_harmonics: int = 4,
    n_warmup: int = 3,
    n_trials: int = 10,
    amp: bool = False,
    quick: bool = False,
    skip_phase_breakdown: bool = False,
    skip_profiler: bool = False,
    skip_training_step: bool = False,
    skip_grad_ckpt: bool = False,
    skip_theta_sweep: bool = False,
    skip_leaf_size_sweep: bool = False,
    skip_compile_test: bool = False,
    skip_scale_sweep: bool = False,
    theta_values: tuple[float, ...] = (0.3, 0.5, 0.7, 1.0, 1.5, 2.0),
    leaf_size_values: tuple[int, ...] = (8, 16, 32, 64, 128),
    subdivision_values: tuple[int, ...] = (3, 4, 5),
    save_json: str | None = None,
):
    """Comprehensive GLOBE Barnes-Hut benchmark.

    Runs phase-level profiling, deep torch.profiler analysis, training step
    timing, gradient-checkpointing comparison, parameter sensitivity sweeps,
    torch.compile comparison, and mesh-scale analysis.

    Args:
        subdivisions: Lumpy-sphere subdivision level (5 ~ 20K faces).
        n_prediction_points: Volume query points for final evaluation.
        theta: Barnes-Hut opening angle for the reference configuration.
        leaf_size: Max sources per tree leaf for the reference configuration.
        hidden_layer_sizes: Kernel MLP hidden layer sizes.
        n_communication_hyperlayers: GLOBE boundary-to-boundary comm layers.
        n_latent_scalars: Scalar latent channels between hyperlayers.
        n_latent_vectors: Vector latent channels between hyperlayers.
        n_spherical_harmonics: Legendre polynomial terms.
        n_warmup: Warmup iterations per experiment (not timed).
        n_trials: Timed iterations per experiment (reports median).
        amp: Test with bfloat16 autocast.
        quick: Reduced sweep ranges and fewer trials for faster results.
        skip_phase_breakdown: Skip phase-level forward pass breakdown.
        skip_profiler: Skip deep torch.profiler analysis.
        skip_training_step: Skip training step analysis.
        skip_grad_ckpt: Skip gradient checkpointing comparison.
        skip_theta_sweep: Skip theta sensitivity sweep.
        skip_leaf_size_sweep: Skip leaf size sensitivity sweep.
        skip_compile_test: Skip torch.compile comparison.
        skip_scale_sweep: Skip mesh scale analysis.
        theta_values: Theta values for the sensitivity sweep.
        leaf_size_values: Leaf sizes for the sensitivity sweep.
        subdivision_values: Subdivision levels for the scale sweep.
        save_json: Path to save machine-readable results as JSON.
    """
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required.", file=sys.stderr)
        sys.exit(1)

    ### Quick mode overrides
    if quick:
        n_warmup = min(n_warmup, 1)
        n_trials = min(n_trials, 3)
        theta_values = (0.5, 1.0, 2.0)
        leaf_size_values = (16, 32, 64)
        skip_compile_test = True
        skip_scale_sweep = True
        skip_profiler = True

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    ref_names = ["L_ref", "sqrt_A_ref"]
    model_kwargs = dict(
        n_spatial_dims=3,
        output_field_ranks={"C_p": 0, "C_f": 1},
        boundary_source_data_ranks={"no_slip": {}},
        reference_length_names=ref_names,
        reference_area=1.0,
        n_communication_hyperlayers=n_communication_hyperlayers,
        hidden_layer_sizes=list(hidden_layer_sizes),
        n_latent_scalars=n_latent_scalars,
        n_latent_vectors=n_latent_vectors,
        n_spherical_harmonics=n_spherical_harmonics,
        theta=theta,
        leaf_size=leaf_size,
    )
    ref_lengths = {n: torch.tensor(1.0, device=device) for n in ref_names}

    n_sections = 10 - sum(
        [
            skip_phase_breakdown,
            skip_profiler,
            skip_training_step,
            skip_grad_ckpt,
            skip_theta_sweep,
            skip_leaf_size_sweep,
            skip_compile_test,
            skip_scale_sweep,
        ]
    )
    section_num = 0

    all_results = AllResults()

    # ── Section 1: System and problem info ─────────────────────────────
    print(f"\nCreating lumpy-sphere mesh (subdivisions={subdivisions})...")
    mesh = lumpy_sphere.load(subdivisions=subdivisions, device="cuda")
    n_faces = mesh.n_cells
    _ = mesh.cell_centroids
    _ = mesh.cell_areas
    _ = mesh.cell_normals

    generator = torch.Generator(device=device).manual_seed(0)
    prediction_points = torch.randn(
        n_prediction_points,
        3,
        generator=generator,
        device=device,
    )

    gpu_name = torch.cuda.get_device_name(device)
    total_vram_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
    cc = torch.cuda.get_device_properties(device).major

    hdr_w = 90
    print(f"\n{'=' * hdr_w}")
    print("  GLOBE Comprehensive Benchmark")
    print(f"{'=' * hdr_w}")
    print(f"  GPU:                 {gpu_name}  ({total_vram_gb:.1f} GB, SM {cc}x)")
    print(f"  Mesh:                {n_faces:,} faces, {mesh.n_points:,} points")
    print(f"  Prediction points:   {n_prediction_points:,}")
    print(
        f"  Reference config:    theta={theta}  leaf_size={leaf_size}  "
        f"hidden={list(hidden_layer_sizes)}"
    )
    print(f"  Comm hyperlayers:    {n_communication_hyperlayers}")
    print(
        f"  Latent channels:     {n_latent_scalars} scalar, {n_latent_vectors} vector"
    )
    print(f"  Benchmark:           {n_warmup} warmup, {n_trials} trials (median)")
    print(f"  AMP (bfloat16):      {'yes' if amp else 'no'}")
    print(f"{'=' * hdr_w}")

    all_results.system_info = {
        "gpu_name": gpu_name,
        "total_vram_gb": total_vram_gb,
        "n_faces": n_faces,
        "n_points": mesh.n_points,
        "n_prediction_points": n_prediction_points,
        "theta": theta,
        "leaf_size": leaf_size,
        "hidden_layer_sizes": list(hidden_layer_sizes),
        "n_communication_hyperlayers": n_communication_hyperlayers,
        "n_warmup": n_warmup,
        "n_trials": n_trials,
        "amp": amp,
        "subdivisions": subdivisions,
    }

    # ── Section 2: Phase-level forward pass breakdown ──────────────────
    if not skip_phase_breakdown:
        section_num += 1
        section_header(
            section_num, n_sections, "Phase-Level Forward Pass Breakdown (inference)"
        )
        phase_rows, phase_stats = run_phase_breakdown(
            mesh,
            prediction_points,
            theta=theta,
            leaf_size=leaf_size,
            hidden_layer_sizes=list(hidden_layer_sizes),
            n_spherical_harmonics=n_spherical_harmonics,
            n_communication_hyperlayers=n_communication_hyperlayers,
            n_latent_scalars=n_latent_scalars,
            n_latent_vectors=n_latent_vectors,
            device=device,
            n_warmup=n_warmup,
            n_trials=n_trials,
        )
        print_phase_table(phase_rows, phase_stats)
        all_results.phase_results = [asdict(r) for r in phase_rows]

    # ── Section 3: Deep profiler analysis ──────────────────────────────
    clean_gpu(device)
    if not skip_profiler:
        section_num += 1
        section_header(section_num, n_sections, "Deep Profiler Analysis")

        regions_fwd, regions_fwd_bwd, top_bwd_ops = run_deep_profiler(
            mesh,
            prediction_points,
            ref_lengths,
            model_kwargs=model_kwargs,
            device=device,
            n_warmup=n_warmup,
        )

        print("\n  Forward pass (inference) - record_function regions:")
        print_profiler_table(regions_fwd, "Forward (inference)")
        all_results.profiler_fwd = [asdict(r) for r in regions_fwd]

        print("\n  Forward + backward (train) - record_function regions:")
        print_profiler_table(regions_fwd_bwd, "Forward + backward")
        all_results.profiler_fwd_bwd = [asdict(r) for r in regions_fwd_bwd]

        print("\n  Top CUDA ops during forward + backward (by CUDA time):")
        print_top_ops_table(top_bwd_ops)
        all_results.profiler_top_bwd_ops = [asdict(r) for r in top_bwd_ops]

    # ── Section 4: Training step analysis ──────────────────────────────
    clean_gpu(device)
    if not skip_training_step:
        section_num += 1
        section_header(
            section_num,
            n_sections,
            f"Training Step Analysis {'(AMP)' if amp else '(FP32)'}",
        )
        ts_result = run_training_step_analysis(
            mesh,
            prediction_points,
            ref_lengths,
            model_kwargs=model_kwargs,
            device=device,
            amp=amp,
            n_warmup=n_warmup,
            n_trials=n_trials,
        )
        print_training_step(ts_result)
        if ts_result is not None:
            all_results.training_step = asdict(ts_result)

    # ── Section 5: Gradient checkpointing comparison ───────────────────
    clean_gpu(device)
    if not skip_grad_ckpt:
        section_num += 1
        section_header(section_num, n_sections, "Gradient Checkpointing Comparison")
        gc_data = run_grad_ckpt_comparison(
            mesh,
            prediction_points,
            ref_lengths,
            model_kwargs=model_kwargs,
            device=device,
            amp=amp,
            n_warmup=n_warmup,
            n_trials=n_trials,
        )
        print_grad_ckpt(gc_data)
        all_results.grad_ckpt_comparison = gc_data

    # ── Section 6: Theta sensitivity sweep ─────────────────────────────
    clean_gpu(device)
    if not skip_theta_sweep:
        section_num += 1
        section_header(section_num, n_sections, "Theta Sensitivity Sweep")
        theta_pts = run_theta_sweep(
            mesh,
            prediction_points,
            ref_lengths,
            model_kwargs=model_kwargs,
            theta_values=theta_values,
            device=device,
            amp=amp,
            n_warmup=n_warmup,
            n_trials=n_trials,
        )
        print_sweep_table(theta_pts)
        all_results.theta_sweep = [asdict(sp) for sp in theta_pts]

    # ── Section 7: Leaf size sensitivity sweep ─────────────────────────
    clean_gpu(device)
    if not skip_leaf_size_sweep:
        section_num += 1
        section_header(section_num, n_sections, "Leaf Size Sensitivity Sweep")
        ls_pts = run_leaf_size_sweep(
            mesh,
            prediction_points,
            ref_lengths,
            model_kwargs=model_kwargs,
            leaf_size_values=leaf_size_values,
            device=device,
            amp=amp,
            n_warmup=n_warmup,
            n_trials=n_trials,
        )
        print_sweep_table(ls_pts, show_tree=True)
        all_results.leaf_size_sweep = [asdict(sp) for sp in ls_pts]

    # ── Section 8: torch.compile comparison ────────────────────────────
    clean_gpu(device)
    if not skip_compile_test:
        section_num += 1
        section_header(section_num, n_sections, "torch.compile Comparison")
        compile_pts = run_compile_test(
            mesh,
            prediction_points,
            ref_lengths,
            model_kwargs=model_kwargs,
            device=device,
            amp=amp,
            n_warmup=n_warmup,
            n_trials=n_trials,
        )
        print_sweep_table(compile_pts)
        all_results.compile_comparison = [asdict(sp) for sp in compile_pts]

    # ── Section 9: Mesh scale analysis ─────────────────────────────────
    clean_gpu(device)
    if not skip_scale_sweep:
        section_num += 1
        section_header(section_num, n_sections, "Mesh Scale Analysis")
        scale_pts = run_scale_sweep(
            prediction_points,
            ref_lengths,
            model_kwargs=model_kwargs,
            subdivision_values=subdivision_values,
            device=device,
            amp=amp,
            n_warmup=n_warmup,
            n_trials=n_trials,
        )
        print_scale_table(scale_pts)
        all_results.scale_sweep = [asdict(sp) for sp in scale_pts]

    # ── Section 10: Summary and recommendations ────────────────────────
    section_num += 1
    section_header(section_num, n_sections, "Summary and Recommendations")
    recs = generate_recommendations(all_results)
    all_results.recommendations = recs
    for i, rec in enumerate(recs, 1):
        print(f"  {i}. {rec}")
    print()

    # ── Save JSON ──────────────────────────────────────────────────────
    if save_json:
        from pathlib import Path

        Path(save_json).write_text(json.dumps(asdict(all_results), indent=2))
        print(f"  Results saved to {save_json}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
