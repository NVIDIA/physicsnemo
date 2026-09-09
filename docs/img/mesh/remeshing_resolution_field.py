# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render a linear-resolution field and its effect on remeshing density."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from physicsnemo.mesh import Mesh

OUTPUT = Path(__file__).parent / "remeshing_resolution_field.png"

X_LIMITS = (-1.65, 1.65)
Y_LIMITS = (-1.02, 1.02)
PLOT_X_LIMITS = (-1.35, 1.35)
PLOT_Y_LIMITS = (-0.75, 0.75)
SOURCE_COLUMNS = 145
SOURCE_ROWS = 91
TARGET_VERTICES = 320
RESOLUTION_KEY = "front_resolution"
FRONT_WIDTH = 0.100
FRONT_RESOLUTION_GAIN = 3.0
SUPPORT_RESOLUTION_GAIN = 0.25
FOCUS_RESOLUTION_THRESHOLD = 2.5

EDGE_COLOR = "#ffffff"
BACKGROUND_COLOR = "#f7f9fa"
TEXT_COLOR = "#23313a"
RESOLUTION_CMAP = "viridis"


def _device() -> torch.device:
    """Use CUDA when available and retain a portable CPU path."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_sheet(device: torch.device) -> Mesh:
    """Create a dense, consistently oriented triangle sheet."""

    x_axis = torch.linspace(
        X_LIMITS[0],
        X_LIMITS[1],
        SOURCE_COLUMNS,
        dtype=torch.float32,
        device=device,
    )
    y_axis = torch.linspace(
        Y_LIMITS[0],
        Y_LIMITS[1],
        SOURCE_ROWS,
        dtype=torch.float32,
        device=device,
    )
    y_grid, x_grid = torch.meshgrid(y_axis, x_axis, indexing="ij")
    points = torch.stack(
        (x_grid, y_grid, torch.zeros_like(x_grid)),
        dim=-1,
    ).reshape(-1, 3)

    rows = torch.arange(SOURCE_ROWS - 1, device=device)
    columns = torch.arange(SOURCE_COLUMNS - 1, device=device)
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    lower_left = row_grid * SOURCE_COLUMNS + column_grid
    lower_right = lower_left + 1
    upper_left = lower_left + SOURCE_COLUMNS
    upper_right = upper_left + 1
    cells = torch.stack(
        (
            torch.stack((lower_left, lower_right, upper_right), dim=-1),
            torch.stack((lower_left, upper_right, upper_left), dim=-1),
        ),
        dim=-2,
    ).reshape(-1, 3)
    return Mesh(points=points, cells=cells)


def _front_center(y: torch.Tensor) -> torch.Tensor:
    """Return the curved centerline of the requested high-resolution band."""

    return 0.08 + 0.24 * torch.sin(2.1 * y)


def _resolution_field(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return a linear-resolution request concentrated near the curved band."""

    signed_front_distance = x - _front_center(y)
    front_band = torch.exp(
        -0.5 * (signed_front_distance / FRONT_WIDTH).square() - 0.5 * (y / 0.82).pow(8)
    )
    support_region = torch.exp(
        -0.5 * ((x - 0.08) / 0.58).square() - 0.5 * ((y + 0.04) / 0.62).square()
    )
    return (
        1.0
        + FRONT_RESOLUTION_GAIN * front_band
        + SUPPORT_RESOLUTION_GAIN * support_region
    )


def _triangulation(mesh: Mesh) -> mtri.Triangulation:
    """Convert a planar mesh to a Matplotlib triangulation."""

    points = mesh.points.detach().cpu().to(torch.float64).numpy()
    cells = mesh.cells.detach().cpu().to(torch.int64).numpy()
    return mtri.Triangulation(points[:, 0], points[:, 1], cells)


def _focus_edge_length(mesh: Mesh) -> float:
    """Return the median edge length inside the high-resolution region."""

    cells = mesh.cells.to(torch.int64)
    edges = torch.cat(
        (
            cells[:, (0, 1)],
            cells[:, (1, 2)],
            cells[:, (2, 0)],
        ),
        dim=0,
    )
    edges = torch.unique(torch.sort(edges, dim=1).values, dim=0)
    starts = mesh.points[edges[:, 0]]
    ends = mesh.points[edges[:, 1]]
    midpoints = 0.5 * (starts + ends)
    midpoint_resolution = _resolution_field(midpoints[:, 0], midpoints[:, 1])
    focus = midpoint_resolution >= FOCUS_RESOLUTION_THRESHOLD
    if not bool(focus.any()):
        raise RuntimeError("The remeshed surface has no edges in the focus region")
    lengths = torch.linalg.vector_norm(ends - starts, dim=1)
    return float(lengths[focus].median().detach().cpu())


def _draw_resolution(
    axis: plt.Axes,
    mesh: Mesh,
    values: torch.Tensor,
    normalization: Normalize,
    *,
    show_edges: bool,
) -> None:
    """Draw one resolution panel with shared limits and styling."""

    triangulation = _triangulation(mesh)
    axis.tripcolor(
        triangulation,
        values.detach().cpu().to(torch.float64).numpy(),
        cmap=RESOLUTION_CMAP,
        norm=normalization,
        shading="gouraud",
        rasterized=True,
    )
    if show_edges:
        axis.triplot(
            triangulation,
            color=EDGE_COLOR,
            linewidth=0.44,
            alpha=0.84,
        )
    axis.set_xlim(PLOT_X_LIMITS)
    axis.set_ylim(PLOT_Y_LIMITS)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor(BACKGROUND_COLOR)
    for spine in axis.spines.values():
        spine.set_color("#cbd3d8")
        spine.set_linewidth(0.7)


def main() -> None:
    """Generate the resolution-field documentation image."""

    source = _make_sheet(_device())
    x = source.points[:, 0]
    y = source.points[:, 1]
    source.point_data[RESOLUTION_KEY] = _resolution_field(x, y)

    uniform = source.remesh(
        TARGET_VERTICES,
        transfer_point_data=[RESOLUTION_KEY],
    )
    adaptive = source.remesh(
        TARGET_VERTICES,
        transfer_point_data=[RESOLUTION_KEY],
        resolution_field=RESOLUTION_KEY,
    )

    uniform_focus_edge = _focus_edge_length(uniform)
    adaptive_focus_edge = _focus_edge_length(adaptive)
    if not torch.isfinite(
        torch.tensor([uniform_focus_edge, adaptive_focus_edge])
    ).all():
        raise RuntimeError("Remeshing produced nonfinite edge lengths")
    if not adaptive_focus_edge < 0.7 * uniform_focus_edge:
        raise RuntimeError(
            "The resolution field did not materially shorten focus-region edges"
        )

    source_resolution = source.point_data[RESOLUTION_KEY]
    lower = float(source_resolution.amin().detach().cpu())
    upper = float(source_resolution.amax().detach().cpu())
    normalization = Normalize(vmin=lower, vmax=upper)

    figure = plt.figure(figsize=(14.4, 4.25), dpi=180)
    grid = figure.add_gridspec(
        1,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.035),
        left=0.018,
        right=0.975,
        bottom=0.055,
        top=0.80,
        wspace=0.055,
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    colorbar_axis = figure.add_subplot(grid[0, 3])

    _draw_resolution(
        axes[0],
        source,
        source_resolution,
        normalization,
        show_edges=False,
    )
    _draw_resolution(
        axes[1],
        uniform,
        uniform.point_data[RESOLUTION_KEY],
        normalization,
        show_edges=True,
    )
    _draw_resolution(
        axes[2],
        adaptive,
        adaptive.point_data[RESOLUTION_KEY],
        normalization,
        show_edges=True,
    )

    axes[0].set_title(
        "Attached linear-resolution field\n"
        f"{lower:.1f}× to {upper:.1f}× relative resolution",
        color=TEXT_COLOR,
        fontsize=11.5,
        pad=8,
    )
    axes[1].set_title(
        "Uniform remesh\n"
        f"{uniform.n_points:,} vertices  |  median focus edge "
        f"{uniform_focus_edge:.3f}",
        color=TEXT_COLOR,
        fontsize=11.5,
        pad=8,
    )
    axes[2].set_title(
        "Resolution-controlled remesh\n"
        f"{adaptive.n_points:,} vertices  |  median focus edge "
        f"{adaptive_focus_edge:.3f}",
        color=TEXT_COLOR,
        fontsize=11.5,
        pad=8,
    )

    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=normalization, cmap=RESOLUTION_CMAP),
        cax=colorbar_axis,
    )
    colorbar.set_label("Relative linear resolution", color=TEXT_COLOR, fontsize=10)
    colorbar.ax.tick_params(labelsize=8.5, colors=TEXT_COLOR)
    colorbar.outline.set_linewidth(0.6)
    colorbar.outline.set_edgecolor("#9daab2")

    figure.suptitle(
        f"One resolution field redirects the same {TARGET_VERTICES}-vertex budget",
        x=0.49,
        y=0.965,
        color=TEXT_COLOR,
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.49,
        0.875,
        "A larger multiplier requests shorter edges near the curved front",
        ha="center",
        va="center",
        color="#52636d",
        fontsize=9.3,
    )

    figure.patch.set_facecolor("white")
    figure.savefig(
        OUTPUT,
        dpi=180,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(figure)

    print(f"Saved {OUTPUT}")
    print(
        "Focus-region median edge length: "
        f"uniform={uniform_focus_edge:.4f}, adaptive={adaptive_focus_edge:.4f}"
    )


if __name__ == "__main__":
    main()
