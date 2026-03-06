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

"""Benchmark each phase of the Barnes-Hut GLOBE pipeline.

Creates synthetic mesh data (lumpy sphere) at configurable scale and profiles
tree construction, traversal, aggregation, kernel evaluation, and the full
GLOBE forward pass independently.  Reports wall-clock time, GPU time,
interaction statistics, and peak memory usage.

The "Overhead" column shows the fraction of wall-clock time *not* spent in
GPU kernels: ``(wall - gpu) / wall``.  High overhead (>50%) indicates the
bottleneck is Python control flow (BFS loops, TensorDict operations, kernel
launch latency) rather than GPU compute.

Usage::

    uv run benchmark_barnes_hut.py
    uv run benchmark_barnes_hut.py --subdivisions 6 --theta 0.5
    uv run benchmark_barnes_hut.py --leaf-size 64 --n-trials 20
"""

import sys
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


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    subdivisions: int = 5,
    theta: float = 1.0,
    leaf_size: int = 32,
    n_warmup: int = 3,
    n_trials: int = 10,
    hidden_layer_sizes: tuple[int, ...] = (64, 64, 64),
    n_communication_hyperlayers: int = 2,
    n_latent_scalars: int = 8,
    n_latent_vectors: int = 4,
    n_spherical_harmonics: int = 4,
    n_prediction_points: int = 2048,
    skip_globe: bool = False,
):
    """Benchmark Barnes-Hut GLOBE pipeline on a synthetic lumpy-sphere mesh.

    Each icosahedron subdivision quadruples the face count:
    0 -> 20, 1 -> 80, ..., 5 -> 20480, 6 -> 81920.

    Args:
        subdivisions: Icosahedron subdivision level (5 matches DrivAerML scale).
        theta: Barnes-Hut opening angle (larger = more aggressive).
        leaf_size: Max sources per tree leaf node.
        n_warmup: Warmup iterations (not timed).
        n_trials: Timed iterations; reports the median.
        hidden_layer_sizes: Kernel MLP layer sizes.
        n_communication_hyperlayers: GLOBE boundary-to-boundary comm layers.
        n_latent_scalars: Scalar latent channels.
        n_latent_vectors: Vector latent channels.
        n_spherical_harmonics: Legendre polynomial terms.
        n_prediction_points: Volume query points for the final evaluation.
        skip_globe: Skip the full GLOBE.forward benchmark (faster iteration).
    """
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")

    ### Create synthetic mesh
    print(f"Creating lumpy-sphere mesh (subdivisions={subdivisions})...")
    mesh = lumpy_sphere.load(subdivisions=subdivisions, device="cuda")
    n_faces = mesh.n_cells

    source_points = mesh.cell_centroids
    source_areas = mesh.cell_areas
    _ = mesh.cell_normals  # trigger lazy computation

    generator = torch.Generator(device=device).manual_seed(0)
    prediction_points = torch.randn(
        n_prediction_points, 3, generator=generator, device=device,
    )

    print(
        f"  {n_faces:,} faces, {mesh.n_points:,} points, "
        f"{n_prediction_points:,} prediction points\n"
    )

    ### Collect rows: (name, wall_ms, gpu_ms, notes)
    rows: list[tuple[str, float, float, str]] = []

    def emit(name: str, wall: float, gpu: float, notes: str = "") -> None:
        rows.append((name, wall, gpu, notes))
        overhead = f"{(wall - gpu) / wall * 100:.0f}%" if wall > 0 else "—"
        print(f"  {name}: wall={wall:.1f}ms  gpu={gpu:.1f}ms  overhead={overhead}  {notes}")

    torch.cuda.reset_peak_memory_stats(device)

    # ── Phase 1: Tree construction ────────────────────────────────────
    print("[1/7] Tree construction")
    tree = ClusterTree.from_points(
        source_points, leaf_size=leaf_size, areas=source_areas,
    )
    w, g = time_fn(
        lambda: ClusterTree.from_points(
            source_points, leaf_size=leaf_size, areas=source_areas,
        ),
        device, n_warmup, n_trials,
    )
    n_nodes = tree.n_nodes
    n_leaves = int((tree.leaf_count > 0).sum())
    depth = int(tree.max_depth.item())
    emit("Tree construction", w, g,
         f"depth={depth} nodes={n_nodes} leaves={n_leaves}")

    # ── Phase 2: Interaction plans ────────────────────────────────────
    print("[2/7] Interaction plans")

    ### 2a: Communication plan (targets = sources, the expensive case)
    comm_plan = tree.find_interaction_pairs(source_points, theta=theta)
    w, g = time_fn(
        lambda: tree.find_interaction_pairs(source_points, theta=theta),
        device, n_warmup, n_trials,
    )
    n_a2a_comm = n_faces * n_faces
    comp_comm = n_a2a_comm / max(1, comm_plan.n_total)
    emit(
        "Find pairs (comm)", w, g,
        f"near={comm_plan.n_near:,} far={comm_plan.n_far:,} "
        f"total={comm_plan.n_total:,} ratio={comp_comm:.1f}x",
    )

    ### 2b: Prediction plan (targets = volume points, well-separated)
    pred_plan = tree.find_interaction_pairs(prediction_points, theta=theta)
    w, g = time_fn(
        lambda: tree.find_interaction_pairs(prediction_points, theta=theta),
        device, n_warmup, n_trials,
    )
    n_a2a_pred = n_prediction_points * n_faces
    comp_pred = n_a2a_pred / max(1, pred_plan.n_total)
    emit(
        "Find pairs (pred)", w, g,
        f"near={pred_plan.n_near:,} far={pred_plan.n_far:,} "
        f"total={pred_plan.n_total:,} ratio={comp_pred:.1f}x",
    )

    # ── Phase 3: Source aggregation ───────────────────────────────────
    print("[3/7] Source aggregation")
    source_data = TensorDict(
        {"normals": mesh.cell_normals},
        batch_size=[n_faces], device=device,
    )
    agg = tree.compute_source_aggregates(
        source_points=source_points, areas=source_areas,
        source_data=source_data,
    )
    w, g = time_fn(
        lambda: tree.compute_source_aggregates(
            source_points=source_points, areas=source_areas,
            source_data=source_data,
        ),
        device, n_warmup, n_trials,
    )
    emit("Source aggregation", w, g)

    # ── Phase 4: Node strengths ───────────────────────────────────────
    print("[4/7] Node strengths")
    output_ranks = {"C_p": 0, "C_f": 1}
    bh_kernel = BarnesHutKernel(
        n_spatial_dims=3,
        output_field_ranks=output_ranks,
        source_data_ranks={"normals": 1},
        hidden_layer_sizes=list(hidden_layer_sizes),
        n_spherical_harmonics=n_spherical_harmonics,
        leaf_size=leaf_size,
        use_gradient_checkpointing=False,
    ).to(device).eval()

    strengths = torch.ones(n_faces, device=device)
    w, g = time_fn(
        lambda: bh_kernel._compute_node_strengths(tree, strengths),
        device, n_warmup, n_trials,
    )
    emit("Node strengths", w, g)

    # ── Phase 5: BarnesHutKernel.forward (single branch) ─────────────
    print("[5/7] BarnesHutKernel.forward (comm config)")
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
        bh_kernel(**bh_kwargs)  # trigger any lazy compilation
        w, g = time_fn(lambda: bh_kernel(**bh_kwargs), device, n_warmup, n_trials)
    chunk_sz = bh_kernel._auto_chunk_size(comm_plan.n_total, device)
    emit("BH kernel (comm)", w, g, f"chunk_size={chunk_sz:,}")

    # ── Phase 6: MultiscaleKernel.forward ─────────────────────────────
    print("[6/7] MultiscaleKernel.forward (comm config)")
    ref_names = ["L_ref", "sqrt_A_ref"]
    ms_kernel = MultiscaleKernel(
        n_spatial_dims=3,
        output_field_ranks=output_ranks,
        reference_length_names=ref_names,
        source_data_ranks={"normals": 1},
        hidden_layer_sizes=list(hidden_layer_sizes),
        n_spherical_harmonics=n_spherical_harmonics,
        leaf_size=leaf_size,
        use_gradient_checkpointing=False,
    ).to(device).eval()

    ref_lengths = {n: torch.tensor(1.0, device=device) for n in ref_names}
    ms_strengths = TensorDict(
        {n: strengths.clone() for n in ref_names},
        batch_size=[n_faces], device=device,
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
        w, g = time_fn(lambda: ms_kernel(**ms_kwargs), device, n_warmup, n_trials)
    emit("MultiscaleKernel (comm)", w, g, f"{len(ref_names)} branches")

    # ── Phase 7: Full GLOBE.forward ───────────────────────────────────
    if not skip_globe:
        print("[7/7] GLOBE.forward")
        model = GLOBE(
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
        ).to(device).eval()

        globe_call = lambda: model(
            prediction_points=prediction_points,
            boundary_meshes={"no_slip": mesh},
            reference_lengths=ref_lengths,
        )
        with torch.no_grad():
            globe_call()
            w, g = time_fn(globe_call, device, n_warmup, n_trials)
        emit(
            "GLOBE.forward", w, g,
            f"{n_communication_hyperlayers} comm + 1 final",
        )
    else:
        print("[7/7] GLOBE.forward — skipped")

    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3

    # ── Summary ───────────────────────────────────────────────────────
    name_w = max(len(r[0]) for r in rows) + 2
    hdr = (
        f"Barnes-Hut GLOBE Benchmark  |  {n_faces:,} faces  "
        f"theta={theta}  leaf_size={leaf_size}  "
        f"hidden={list(hidden_layer_sizes)}"
    )
    print(f"\n{'=' * max(80, len(hdr) + 4)}")
    print(f"  {hdr}")
    print(f"{'=' * max(80, len(hdr) + 4)}")
    print(
        f"  {'Phase':<{name_w}}  {'Wall (ms)':>10}  {'GPU (ms)':>10}  "
        f"{'Overhead':>9}  Notes"
    )
    print(f"  {chr(0x2500) * name_w}  {chr(0x2500) * 10}  {chr(0x2500) * 10}  {chr(0x2500) * 9}  {chr(0x2500) * 5}")
    for name, wall, gpu, notes in rows:
        overhead = f"{(wall - gpu) / wall * 100:.0f}%" if wall > 0 else "—"
        print(
            f"  {name:<{name_w}}  {wall:10.1f}  {gpu:10.1f}  "
            f"{overhead:>9}  {notes}"
        )

    print(f"\n  Peak GPU memory: {peak_mem_gb:.2f} GB")
    print(
        f"  Comm: {comm_plan.n_total:,} pairs vs {n_a2a_comm:,} all-to-all "
        f"({comp_comm:.1f}x compression)  "
        f"[near={comm_plan.n_near:,}, far={comm_plan.n_far:,}]"
    )
    print(
        f"  Pred: {pred_plan.n_total:,} pairs vs {n_a2a_pred:,} all-to-all "
        f"({comp_pred:.1f}x compression)  "
        f"[near={pred_plan.n_near:,}, far={pred_plan.n_far:,}]"
    )
    print(f"  Tree: depth={depth}  nodes={n_nodes}  leaves={n_leaves}")
    print()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
