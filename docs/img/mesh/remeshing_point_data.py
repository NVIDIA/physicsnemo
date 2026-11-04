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

"""Render field preservation under uniform and field-aware mesh reduction."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
from _style import clean_axis, mesh_label, save  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from physicsnemo.mesh import Mesh

OUTPUT = Path(__file__).parent / "remeshing_point_data.png"

X_LIMITS = (-1.65, 1.65)
Y_LIMITS = (-1.02, 1.02)
PLOT_X_LIMITS = (-1.35, 1.35)
PLOT_Y_LIMITS = (-0.75, 0.75)
SOURCE_COLUMNS = 145
SOURCE_ROWS = 91
TARGET_VERTICES = 400
FIELD_KEY = "center_field"
RESOLUTION_KEY = "center_resolution"
FIELD_WIDTH = 0.46
MAX_RESOLUTION_MULTIPLIER = 4.0

EDGE_COLOR = "#ffffff"
BACKGROUND_COLOR = "#f7f9fa"
TEXT_COLOR = "#23313a"
FIELD_CMAP = "viridis"


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


def _input_field(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Evaluate one smooth sphere-like scalar field at the mesh center."""

    squared_radius = x.square() + y.square()
    return torch.exp(-0.5 * squared_radius / FIELD_WIDTH**2)


def _triangulation(mesh: Mesh) -> mtri.Triangulation:
    """Convert a planar mesh to a Matplotlib triangulation."""

    points = mesh.points.detach().cpu().to(torch.float64).numpy()
    cells = mesh.cells.detach().cpu().to(torch.int64).numpy()
    return mtri.Triangulation(points[:, 0], points[:, 1], cells)


def _interpolate_field(
    mesh: Mesh,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> np.ma.MaskedArray:
    """Interpolate one mesh field on the supplied evaluation grid."""
    triangulation = _triangulation(mesh)
    values = mesh.point_data[FIELD_KEY].detach().cpu().to(torch.float64).numpy()
    interpolator = mtri.LinearTriInterpolator(triangulation, values)
    return interpolator(x_grid, y_grid)


def _preservation_rmse(
    uniform: Mesh,
    adaptive: Mesh,
    reference: Mesh,
) -> tuple[float, float]:
    """Compare both reduced fields with the input on one shared grid subset."""
    x_axis = np.linspace(PLOT_X_LIMITS[0], PLOT_X_LIMITS[1], 260)
    y_axis = np.linspace(PLOT_Y_LIMITS[0], PLOT_Y_LIMITS[1], 160)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    uniform_field = _interpolate_field(uniform, x_grid, y_grid)
    adaptive_field = _interpolate_field(adaptive, x_grid, y_grid)
    reference_field = _interpolate_field(reference, x_grid, y_grid)
    valid = (
        ~np.ma.getmaskarray(uniform_field)
        & ~np.ma.getmaskarray(adaptive_field)
        & ~np.ma.getmaskarray(reference_field)
    )
    if not np.any(valid):
        raise RuntimeError("The reduced fields do not share an evaluation region")
    if float(valid.mean()) < 0.9:
        raise RuntimeError("The shared evaluation region covers too little of the grid")

    reference_values = np.asarray(reference_field)[valid]
    errors = (
        np.asarray(uniform_field)[valid] - reference_values,
        np.asarray(adaptive_field)[valid] - reference_values,
    )
    return tuple(float(np.sqrt(np.mean(error * error))) for error in errors)


def _draw_field(
    axis: plt.Axes,
    mesh: Mesh,
    values: torch.Tensor,
    normalization: Normalize,
    *,
    edge_linewidth: float = 0.42,
    edge_alpha: float = 0.82,
) -> None:
    """Draw one field panel with common limits and compact styling."""

    triangulation = _triangulation(mesh)
    axis.tripcolor(
        triangulation,
        values.detach().cpu().to(torch.float64).numpy(),
        cmap=FIELD_CMAP,
        norm=normalization,
        shading="gouraud",
        rasterized=True,
    )
    axis.triplot(
        triangulation,
        color=EDGE_COLOR,
        linewidth=edge_linewidth,
        alpha=edge_alpha,
    )
    clean_axis(axis, (PLOT_X_LIMITS, PLOT_Y_LIMITS))


def main() -> None:
    """Generate the field-preserving reduction documentation image."""

    source = _make_sheet(_device())
    x = source.points[:, 0]
    y = source.points[:, 1]
    source.point_data[FIELD_KEY] = _input_field(x, y)
    source.point_data[RESOLUTION_KEY] = (
        1.0 + (MAX_RESOLUTION_MULTIPLIER - 1.0) * source.point_data[FIELD_KEY]
    )

    uniform = source.remesh(
        TARGET_VERTICES,
        transfer_point_data=[FIELD_KEY],
    )
    adaptive = source.remesh(
        TARGET_VERTICES,
        transfer_point_data=[FIELD_KEY],
        resolution_field=RESOLUTION_KEY,
    )

    uniform_rmse, adaptive_rmse = _preservation_rmse(uniform, adaptive, source)
    if not np.isfinite([uniform_rmse, adaptive_rmse]).all():
        raise RuntimeError("Mesh reduction produced nonfinite field errors")
    if not adaptive_rmse < 0.85 * uniform_rmse:
        raise RuntimeError(
            "Field-aware reduction did not materially reduce preservation error"
        )

    source_values = source.point_data[FIELD_KEY]
    lower = float(source_values.amin().detach().cpu())
    upper = float(source_values.amax().detach().cpu())
    normalization = Normalize(vmin=lower, vmax=upper)

    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))

    _draw_field(
        axes[0],
        source,
        source.point_data[FIELD_KEY],
        normalization,
        edge_linewidth=0.18,
        edge_alpha=0.42,
    )
    _draw_field(
        axes[1],
        uniform,
        uniform.point_data[FIELD_KEY],
        normalization,
    )
    _draw_field(
        axes[2],
        adaptive,
        adaptive.point_data[FIELD_KEY],
        normalization,
    )

    axes[0].set_title(
        f"source sheet\npoint_data[{FIELD_KEY!r}]\n"
        f"{mesh_label(source)}, {source.n_points:,} points"
    )
    axes[1].set_title(
        f"remesh({TARGET_VERTICES},\ntransfer_point_data=[{FIELD_KEY!r}])\n"
        f"{uniform.n_points:,} points, field RMSE {uniform_rmse:.4f}"
    )
    axes[2].set_title(
        f"remesh({TARGET_VERTICES}, transfer_point_data=[...],\n"
        f"resolution_field={RESOLUTION_KEY!r})\n"
        f"{adaptive.n_points:,} points, field RMSE {adaptive_rmse:.4f}"
    )

    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=normalization, cmap=FIELD_CMAP),
        ax=axes,
        orientation="horizontal",
        fraction=0.07,
        pad=0.05,
        shrink=0.4,
    )
    colorbar.set_label("attached scalar point field")

    figure.subplots_adjust(left=0.005, right=0.995, bottom=0.2, top=0.86, wspace=0.04)
    save(figure, OUTPUT, tight=True)
    print(
        "Field preservation RMSE: "
        f"uniform={uniform_rmse:.3e}, adaptive={adaptive_rmse:.3e}"
    )


if __name__ == "__main__":
    main()
