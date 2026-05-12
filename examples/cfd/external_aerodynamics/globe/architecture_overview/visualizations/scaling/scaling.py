"""Empirical wall-clock scaling of GLOBE's BarnesHutKernel on synthetic 3D data.

Produces a single log-log figure (``scaling.{pdf,png}``) that compares the
per-forward-pass wall-clock time of:

- The dense baseline (``BarnesHutKernel`` with ``theta=0`` - all interactions
  exact, expected slope :math:`\\sim 2`)
- Barnes-Hut with ``theta in {0.5, 1.0, 2.0}`` (expected slope :math:`\\sim 1`)

The script is split out from ``theta_effect.py`` because it is by far the
slowest visualization in this directory and benefits from being run on its
own when iterating on benchmark methodology.

Methodology notes
-----------------
- A fixed ``near_chunk_size`` is queried once at the start of the script,
  while VRAM is still uncluttered, and pinned for every subsequent forward
  call. ``BarnesHutKernel`` defaults to a memory-aware ``_auto_chunk_size``
  that *shrinks* chunks under runtime memory pressure; that re-acts as
  pathological launch-overhead-driven thrashing rather than a clean OOM
  (single forward passes taking 30+ s instead of failing fast).
- ``BarnesHutKernel.forward`` only chunks the near-field phase. The other
  three phases (far-far, near-far, far-near) evaluate in single batched
  calls. Above some N these single calls trigger memory pressure that the
  kernel masks rather than OOMs; we detect that via a wall-clock sentinel
  on the first warmup pass and abort the curve.
- Per-(theta, N) timing reports the **min** across ``N_TRIALS`` trials. Min
  is the right estimator for the operation's intrinsic cost: jitter from
  background processes, GPU power state transitions, or PyTorch caching
  allocator hiccups can only ever inflate timings, never deflate them.
- Both wall-clock (``perf_counter``) and GPU-event (``torch.cuda.Event``)
  timings are recorded per trial. Wall is what users see; the wall - GPU
  gap is CPU/dispatch overhead, which dominates at small N. The full
  per-trial distribution is printed to stdout for diagnosis.
"""

import os

### [Allocator config: select expandable_segments BEFORE any torch/CUDA import.]
# `BarnesHutKernel.forward` empirically reserves ~30% more GPU memory than it
# truly needs when the chunked Phase A loop fragments the default caching
# allocator's free list.  Selecting PyTorch's expandable-segments allocator
# eliminates that overhead with negligible wall-time cost; see the Notes
# section of `BarnesHutKernel`'s docstring.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path
from time import perf_counter

import aerosandbox.tools.pretty_plots as p
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensordict import TensorDict

from physicsnemo.experimental.models.globe import BarnesHutKernel
from physicsnemo.experimental.models.globe.cluster_tree import ClusterTree

device = "cuda" if torch.cuda.is_available() else "cpu"
USE_CUDA = device == "cuda"
OUTPUT_DIR = Path(__file__).parent

### [Configuration]
SEED = 39
# BH curves run from N=500 to N=100k (Phase-B's unchunked far-field
# allocation, ~n_far_nodes * floats_per_interaction * 4 bytes, is what
# eventually OOMs - and that's the quantity that scales like O(N) for
# BH, so 100k is roughly where theta=2.0 hits a 16-32 GB GPU's limit).
# Dense (theta=0) is capped at 5k because Phase A's chunked loop holds
# ~5x per_chunk_peak in cached-but-not-reusable PyTorch blocks (this is
# documented in BarnesHutKernel's Notes); above 5k the cumulative
# reservation across an N sweep exhausts the GPU.  13 geometrically-
# spaced points gives a per-step ratio of ~1.55x for nicely sampled curves.
N_VALUES_BH = sorted(
    set(np.round(np.geomspace(500, 100_000, 13)).astype(int).tolist())
)
N_VALUES_DENSE = [n for n in N_VALUES_BH if n <= 5_000]
THETA_SCALING = [0.5, 1.0, 2.0]
N_WARMUP = 3
N_TRIALS = 4
# A single forward pass exceeding this duration almost certainly means the
# kernel is fighting fragmentation/eviction in the unchunked Phase B
# allocation.  The sentinel is applied to the *second* warmup run (not the
# first), so cold-start costs from the absolute first call at a new shape
# (cudaMalloc, cuBLAS plan caching) don't trigger a false positive.  We
# treat genuine sentinel trips as effectively-OOM and abort the curve to
# avoid burning N_TRIALS * TIMEOUT_S of slow runs after the first.
TIMEOUT_S = 15.0


### [Helpers]
def make_3d_problem(
    n: int, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, TensorDict]:
    """Random unit-cube source/target points + unit normals at scale ``n``.

    A *fixed* seed is used for every N so that successive N values share a
    common random prefix (the first 100 points at N=100 are the same
    physical points as the first 100 of the N=215 sweep, etc.).  This
    keeps the cluster-tree shape statistics consistent across N and removes
    one source of run-to-run jitter (different seeds would put points in
    radically different positions, producing wildly different dual_plan
    fan-outs even at the same N).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    src = torch.rand(n, 3, generator=g, device=device) * 2.0 - 1.0
    tgt = torch.rand(n, 3, generator=g, device=device) * 2.0 - 1.0
    strengths = torch.full((n,), 1.0 / n, device=device)
    normals = torch.randn(n, 3, generator=g, device=device)
    normals = normals / normals.norm(dim=-1, keepdim=True)
    data = TensorDict(
        {"normal": normals, "other": torch.zeros_like(normals)},
        batch_size=torch.Size([n]),
        device=device,
    )
    return src, tgt, strengths, data


def memory_mb() -> float:
    """Currently-allocated GPU memory in MB (0.0 on CPU)."""
    return torch.cuda.memory_allocated() / 1e6 if USE_CUDA else 0.0


def time_forward(
    *,
    kernel: BarnesHutKernel,
    src: torch.Tensor,
    tgt: torch.Tensor,
    strengths: torch.Tensor,
    data: TensorDict,
    src_tree: ClusterTree,
    tgt_tree: ClusterTree,
    theta_val: float,
    near_chunk_size: int,
) -> dict[str, float | list[float]] | None:
    """One (theta, N) measurement: cold absorber + warmup + N_TRIALS trials.

    Returns a dict with min wall-clock and GPU-event times (both ms) plus
    the full per-trial distributions, or ``None`` if the *second* warmup
    run exceeded ``TIMEOUT_S`` (a thrash sentinel).  The absolute first
    call at a new (theta, N) shape pays one-time cold-start costs
    (cudaMalloc for new pair-storage shapes, cuBLAS plan-cache misses)
    that can comfortably exceed the genuine compute time, so we run an
    untimed cold-absorber call before applying the sentinel.
    """

    def _run() -> None:
        with torch.no_grad():
            kernel(
                reference_length=torch.tensor(1.0, device=device),
                source_points=src,
                target_points=tgt,
                source_strengths=strengths,
                source_data=data,
                theta=theta_val,
                cluster_tree=src_tree,
                target_tree=tgt_tree,
                near_chunk_size=near_chunk_size,
            )

    ### [Cold-start absorber: untimed, just to fault in shape-specific caches]
    _run()
    if USE_CUDA:
        torch.cuda.synchronize()

    ### [Sentinel warmup: now that caches are warm, this run reflects steady
    ### state, so a wall-clock budget genuinely catches Phase-B thrashing.]
    if USE_CUDA:
        torch.cuda.synchronize()
    t0 = perf_counter()
    _run()
    if USE_CUDA:
        torch.cuda.synchronize()
    if perf_counter() - t0 > TIMEOUT_S:
        return None

    ### [Remaining warmups]
    for _ in range(max(0, N_WARMUP - 2)):
        _run()
    if USE_CUDA:
        torch.cuda.synchronize()

    ### [Timed runs - record both wall-clock and GPU-event time per trial]
    wall_ms: list[float] = []
    gpu_ms: list[float] = []
    for _ in range(N_TRIALS):
        if USE_CUDA:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            t0 = perf_counter()
            start_ev.record()
            _run()
            end_ev.record()
            torch.cuda.synchronize()
            wall_ms.append((perf_counter() - t0) * 1e3)
            gpu_ms.append(start_ev.elapsed_time(end_ev))
        else:
            t0 = perf_counter()
            _run()
            dt_ms = (perf_counter() - t0) * 1e3
            wall_ms.append(dt_ms)
            gpu_ms.append(dt_ms)
    return {
        "wall_min_ms": min(wall_ms),
        "wall_ms": wall_ms,
        "gpu_min_ms": min(gpu_ms),
        "gpu_ms": gpu_ms,
    }


def save_figure(fig: plt.Figure, *, stem: str) -> None:
    """Save figure as both PDF and PNG under ``OUTPUT_DIR``."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = OUTPUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05, dpi=200)
        print(f"Saved {path}")


# =====================================================================
# Build kernel and pin chunk size
# =====================================================================

print(f"\n=== Empirical scaling test (3D, device={device}) ===")
torch.manual_seed(SEED)
np.random.seed(SEED)

kernel_3d = BarnesHutKernel(
    n_spatial_dims=3,
    output_field_ranks={"phi": 0, "u": 1},
    source_data_ranks={"normal": 1, "other": 1},
    hidden_layer_sizes=[64, 64],
    n_spherical_harmonics=4,
    network_type="pade",
    spectral_norm=False,
    use_gradient_checkpointing=False,
    leaf_size=1,
).to(device)
kernel_3d.eval()

NEAR_CHUNK_SIZE = kernel_3d._auto_chunk_size(
    n_total_pairs=10_000_000,
    device=torch.device(device),
)
print(f"  Pinned near_chunk_size = {NEAR_CHUNK_SIZE:,}")

### [Phase-B memory budget]
# `BarnesHutKernel.forward` only chunks the near-field (Phase A); Phases B,
# C, and D evaluate a single unchunked (n_pairs, floats_per_interaction)
# tensor.  Under the default PyTorch caching allocator we observed a sharp
# performance cliff when this allocation exceeded ~1.5-1.8 GB: on a 17 GB-
# free GPU, ~180 ms forward passes at 1.5 GB Phase B *jumped* to ~1.1 s at
# 2.0 GB Phase B (5-10x slowdown).  The cliff was too sharp to be memory
# pressure (15+ GB free); it is almost certainly cuBLAS picking a different
# GEMM algorithm above some M-dimension threshold, or the caching
# allocator's `max_split_size` behavior at ~2 GB.  Setting
# `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (done at the top of
# this script) eliminates the cliff, so we cap Phase B at the GPU's
# actual total VRAM (queried dynamically via `torch.cuda.mem_get_info`)
# rather than at a fixed safety value.  The safe budget is then the
# minimum of (80% of free GPU memory) and that total-VRAM cap.
FLOATS_PER_INTERACTION = kernel_3d._floats_per_interaction
if USE_CUDA:
    free_bytes, total_bytes = torch.cuda.mem_get_info(torch.device(device))
else:
    free_bytes, total_bytes = 0, 0
PHASE_B_HARD_CAP_BYTES = total_bytes if USE_CUDA else 10**12
PHASE_B_SAFE_BYTES = (
    min(int(free_bytes * 0.80), PHASE_B_HARD_CAP_BYTES) if USE_CUDA else 10**12
)
PHASE_B_MAX_PAIRS = (
    PHASE_B_SAFE_BYTES // (FLOATS_PER_INTERACTION * 4) if USE_CUDA else 10**12
)
if USE_CUDA:
    print(
        f"  GPU free / total = {free_bytes / 1e9:.2f} / {total_bytes / 1e9:.2f} GB; "
        f"floats_per_interaction = {FLOATS_PER_INTERACTION}"
    )
    print(
        f"  Phase-B safe budget = {PHASE_B_SAFE_BYTES / 1e9:.2f} GB "
        f"(min of 80% of free and {PHASE_B_HARD_CAP_BYTES / 1e9:.1f} GB hard cap)"
        f"  ->  max n_far = {PHASE_B_MAX_PAIRS:,}"
    )


# =====================================================================
# Global warmup: amortize CUDA / cuBLAS / cuDNN init once before the sweep
# =====================================================================

print("  Performing global warmup...")
_w_src, _w_tgt, _w_str, _w_data = make_3d_problem(500, seed=SEED + 99)
_w_src_tree = ClusterTree.from_points(_w_src, leaf_size=1)
_w_tgt_tree = ClusterTree.from_points(_w_tgt, leaf_size=1)
for _theta_warm in (0.0, 1.0):
    for _ in range(3):
        with torch.no_grad():
            kernel_3d(
                reference_length=torch.tensor(1.0, device=device),
                source_points=_w_src,
                target_points=_w_tgt,
                source_strengths=_w_str,
                source_data=_w_data,
                theta=_theta_warm,
                cluster_tree=_w_src_tree,
                target_tree=_w_tgt_tree,
                near_chunk_size=NEAR_CHUNK_SIZE,
            )
if USE_CUDA:
    torch.cuda.synchronize()
del _w_src, _w_tgt, _w_str, _w_data, _w_src_tree, _w_tgt_tree
# Deliberately *not* calling empty_cache here either - we want the cache
# warm and populated before the first sweep iteration starts.
print("  Global warmup done.\n")


# =====================================================================
# Sweep over (theta, N)
# =====================================================================

all_thetas = [0.0, *THETA_SCALING]
scaling_results: dict[float, dict[str, list]] = {}
mem_baseline_mb = memory_mb()
print(f"  Memory baseline = {mem_baseline_mb:.1f} MB")
for theta_val in all_thetas:
    is_dense = theta_val == 0.0
    n_values = N_VALUES_DENSE if is_dense else N_VALUES_BH
    label = "dense (theta=0)" if is_dense else f"theta={theta_val}"
    Ns_used: list[int] = []
    wall_mins: list[float] = []
    gpu_mins: list[float] = []
    for n in n_values:
        src, tgt, strengths, data = make_3d_problem(n, seed=SEED)
        src_tree = ClusterTree.from_points(src, leaf_size=1)
        tgt_tree = ClusterTree.from_points(tgt, leaf_size=1)
        ### [Plan counts - lets us see the per-phase fan-out]
        # Wrap in try/except: the dual-plan computation itself can OOM at
        # very large N because its O(n_pairs) intermediates (e.g. the
        # `_ragged_arange` cumsum) are allocated *before* the explicit
        # Phase-B budget check below has any chance to short-circuit.
        try:
            dual_plan = src_tree.find_dual_interaction_pairs(
                target_tree=tgt_tree, theta=theta_val,
            )
            n_near = dual_plan.n_near
            n_far_nodes = dual_plan.n_far_nodes
            n_nf = dual_plan.n_nf
            n_fn = dual_plan.n_fn
            del dual_plan
        except torch.cuda.OutOfMemoryError:
            print(
                f"  {label}, N={n}: OOM during dual-plan computation, "
                f"stopping curve."
            )
            del src, tgt, strengths, data, src_tree, tgt_tree
            break
        ### [Phase-B memory check - abort before allocator pressure kicks in]
        phase_b_bytes = n_far_nodes * FLOATS_PER_INTERACTION * 4
        if phase_b_bytes > PHASE_B_SAFE_BYTES:
            print(
                f"  {label}, N={n}: Phase-B would need ~{phase_b_bytes / 1e9:.2f} GB "
                f"(> {PHASE_B_SAFE_BYTES / 1e9:.2f} GB budget), stopping curve."
            )
            del src, tgt, strengths, data, src_tree, tgt_tree
            break
        mem_before_mb = memory_mb()
        try:
            res = time_forward(
                kernel=kernel_3d,
                src=src,
                tgt=tgt,
                strengths=strengths,
                data=data,
                src_tree=src_tree,
                tgt_tree=tgt_tree,
                theta_val=theta_val,
                near_chunk_size=NEAR_CHUNK_SIZE,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  {label}, N={n}: OOM during forward pass, stopping curve.")
            del src, tgt, strengths, data, src_tree, tgt_tree
            break
        if res is None:
            print(
                f"  {label}, N={n}: warmup exceeded {TIMEOUT_S:.1f}s "
                f"(memory pressure / chunking thrash), stopping curve."
            )
            del src, tgt, strengths, data, src_tree, tgt_tree
            break
        mem_after_mb = memory_mb()
        Ns_used.append(n)
        wall_mins.append(res["wall_min_ms"])
        gpu_mins.append(res["gpu_min_ms"])
        ### [Per-trial logging - exposes jitter so spikes are diagnosable]
        wall_str = ", ".join(f"{x:.1f}" for x in sorted(res["wall_ms"]))
        gpu_str = ", ".join(f"{x:.1f}" for x in sorted(res["gpu_ms"]))
        print(
            f"  {label}, N={n}: "
            f"wall_min={res['wall_min_ms']:6.2f}ms  "
            f"gpu_min={res['gpu_min_ms']:6.2f}ms  "
            f"plan(n_near={n_near:>9,d} n_far={n_far_nodes:>7,d} "
            f"n_nf={n_nf:>7,d} n_fn={n_fn:>7,d})  "
            f"phase_B={phase_b_bytes / 1e9:5.2f}GB  "
            f"mem={mem_after_mb:.1f}MB ({mem_after_mb - mem_before_mb:+.1f})"
        )
        print(f"    wall trials [ms]: [{wall_str}]")
        print(f"    gpu  trials [ms]: [{gpu_str}]")
        del src, tgt, strengths, data, src_tree, tgt_tree
        # NOTE: deliberately *not* calling torch.cuda.empty_cache() between
        # iterations.  Empirically, doing so triggers a fresh-cudaMalloc
        # cycle that takes several warmup runs to amortize, producing
        # bimodal or monotonically-growing per-trial timings (especially
        # at small N where Phase B's unchunked far-field allocation has to
        # be re-issued from scratch).  Letting PyTorch's caching allocator
        # reuse blocks across (theta, N) keeps the per-trial distributions
        # tight; the trade-off is we OOM slightly sooner at large N, which
        # is fine because the sentinel already aborts those curves.
    scaling_results[theta_val] = {
        "N": Ns_used,
        "wall_ms": wall_mins,
        "gpu_ms": gpu_mins,
    }


# =====================================================================
# Plot: log-log min wall-clock vs N
# =====================================================================

fig, ax = plt.subplots(figsize=(7, 5))

THETA_PLOT_COLORS = {0.0: "k", 0.5: "C0", 1.0: "C1", 2.0: "C2"}
THETA_PLOT_LABELS = {
    0.0: r"Dense ($\theta=0$)",
    0.5: r"Barnes-Hut ($\theta=0.5$)",
    1.0: r"Barnes-Hut ($\theta=1.0$)",
    2.0: r"Barnes-Hut ($\theta=2.0$)",
}

for theta_val in all_thetas:
    res = scaling_results[theta_val]
    if not res["N"]:
        continue
    ax.loglog(
        res["N"],
        np.array(res["wall_ms"]) / 1e3,
        "o-",
        color=THETA_PLOT_COLORS[theta_val],
        label=THETA_PLOT_LABELS[theta_val],
    )

### [Reference slopes anchored to actual data so they sit alongside curves]
N_ref = np.array([N_VALUES_BH[0], N_VALUES_BH[-1]])
dense_res = scaling_results[0.0]
if dense_res["N"]:
    n_a = dense_res["N"][-1]
    t_a = dense_res["wall_ms"][-1] / 1e3
    ax.loglog(
        N_ref,
        t_a * (N_ref / n_a) ** 2,
        "--",
        color="gray",
        alpha=0.5,
        label=r"$\propto N^2$",
    )
bh_ref_res = scaling_results[1.0]
if bh_ref_res["N"]:
    n_a = bh_ref_res["N"][-1]
    t_a = bh_ref_res["wall_ms"][-1] / 1e3
    ax.loglog(
        N_ref,
        t_a * (N_ref * np.log(N_ref)) / (n_a * np.log(n_a)),
        ":",
        color="gray",
        alpha=0.5,
        label=r"$\propto N \log N$",
    )

ax.set_xlabel(r"$N$ (sources $=$ targets)")
ax.set_ylabel("Min wall-clock time per forward pass [s]")
ax.set_title("Empirical scaling on synthetic 3D point distributions")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
p.show_plot(show=False)
save_figure(fig, stem="scaling")
plt.show()
