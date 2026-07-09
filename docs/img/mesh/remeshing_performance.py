"""Benchmark CPU and GPU remeshing and render the documentation plot."""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.mesh.remeshing import remesh

OUTPUT = Path(__file__).parent / "remeshing_performance.png"
SUBDIVISIONS = (4, 5, 6, 7, 8, 9)
REPEATS = 3


def measure(mesh, n_clusters: int, implementation: str) -> float:
    """Return median warmed end-to-end latency in milliseconds."""
    remesh(mesh, n_clusters, implementation=implementation)
    if mesh.points.is_cuda:
        torch.cuda.synchronize(mesh.points.device)

    samples = []
    for _ in range(REPEATS):
        if mesh.points.is_cuda:
            torch.cuda.synchronize(mesh.points.device)
        start = time.perf_counter()
        result = remesh(mesh, n_clusters, implementation=implementation)
        if mesh.points.is_cuda:
            torch.cuda.synchronize(mesh.points.device)
        samples.append(1_000.0 * (time.perf_counter() - start))
        # Keep the result alive through synchronization and until the sample is
        # recorded, matching the ASV benchmark's end-to-end lifetime.
        if result.n_cells == 0:
            raise RuntimeError("remeshing returned an empty surface")
    return statistics.median(samples)


if not torch.cuda.is_available():
    raise RuntimeError("A CUDA device is required to benchmark Warp remeshing")

vertex_counts = []
cpu_ms = []
gpu_ms = []
for subdivisions in SUBDIVISIONS:
    cpu_mesh = sphere_icosahedral.load(subdivisions=subdivisions, device="cpu")
    gpu_mesh = cpu_mesh.to("cuda")
    target = max(3, cpu_mesh.n_points // 8)

    vertex_counts.append(cpu_mesh.n_points)
    gpu_ms.append(measure(gpu_mesh, target, "warp"))
    cpu_ms.append(measure(cpu_mesh, target, "pyacvd"))
    print(
        f"{cpu_mesh.n_points:>8,} vertices -> {target:>7,}: "
        f"Warp {gpu_ms[-1]:8.3f} ms, PyACVD {cpu_ms[-1]:8.3f} ms, "
        f"{cpu_ms[-1] / gpu_ms[-1]:6.1f}x"
    )

positions = range(len(vertex_counts))


def compact_count(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 10_000:
        return f"{count / 1_000:.0f}K"
    return f"{count / 1_000:.1f}K"


labels = [compact_count(count) for count in vertex_counts]

fig, axis = plt.subplots(figsize=(7.2, 4.2), dpi=150)
axis.plot(
    positions,
    cpu_ms,
    marker="o",
    markersize=7,
    linewidth=2.5,
    color="#4C78A8",
    label="PyACVD · CPU",
)
axis.plot(
    positions,
    gpu_ms,
    marker="o",
    markersize=7,
    linewidth=2.5,
    color="#76B900",
    label="Warp · GPU",
)

axis.set_yscale("log")
axis.set_xticks(list(positions), labels)
axis.set_xlabel("Input vertices", fontsize=11)
axis.set_ylabel("Median latency (ms)", fontsize=11)
axis.set_title("CPU vs GPU remeshing", fontsize=14, pad=12)
axis.grid(True, axis="y", which="major", color="#dddddd", linewidth=0.8)
axis.set_axisbelow(True)
axis.legend(frameon=False, fontsize=11, loc="upper left")
axis.spines[["top", "right"]].set_visible(False)
axis.tick_params(colors="#303030")
fig.patch.set_facecolor("white")
axis.set_facecolor("white")
fig.tight_layout()
fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Saved {OUTPUT}")
