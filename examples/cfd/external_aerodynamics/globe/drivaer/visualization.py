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

"""Visualization helpers for the DrivAerML GLOBE case study.

Provides PyVista and Matplotlib backends for rendering pred/true/error
surface field comparisons on 3D car body meshes.
"""

from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from tensordict import TensorDict

from physicsnemo.mesh import Mesh
from physicsnemo.utils.logging import PythonLogger

logger = PythonLogger("globe.drivaer.visualization")


def visualize_pyvista(
    combined: Mesh[2, 3],
    kind_data: dict[str, TensorDict],
    kinds: dict[str, str],
    fields: list[str],
    save_path: Path,
    show: bool,
) -> None:
    """PyVista backend for :meth:`DrivAerMLDataSet.visualize_comparison`."""
    from physicsnemo.mesh.io import to_pyvista

    n_rows, n_cols = len(kinds), len(fields)
    plotter = pv.Plotter(
        shape=(n_rows, n_cols),
        off_screen=not show,
        window_size=(600 * n_cols, 500 * n_rows),
    )

    combined_pv = to_pyvista(combined.to("cpu"))

    for col, field_name in enumerate(fields):
        true_vals = kind_data["true"][field_name].float().cpu().numpy()
        pred_vals = kind_data["pred"][field_name].float().cpu().numpy()

        is_vector = true_vals.ndim > 1 and true_vals.shape[-1] > 1
        if is_vector:
            true_scalars = np.linalg.norm(true_vals, axis=-1)
            pred_scalars = np.linalg.norm(pred_vals, axis=-1)
            label = f"|{field_name}|"
        else:
            true_scalars = true_vals.ravel()
            pred_scalars = pred_vals.ravel()
            label = field_name

        ### Shared color limits across truth and prediction
        finite_all = np.concatenate(
            [
                true_scalars[np.isfinite(true_scalars)],
                pred_scalars[np.isfinite(pred_scalars)],
            ]
        )
        shared_clim = [float(finite_all.min()), float(finite_all.max())]

        for row, (key, title) in enumerate(kinds.items()):
            plotter.subplot(row, col)
            vals: torch.Tensor = kind_data[key][field_name]  # ty: ignore[invalid-assignment]

            if is_vector:
                scalars = np.linalg.norm(vals.float().cpu().numpy(), axis=-1)
            else:
                scalars = vals.float().cpu().numpy().ravel()

            if key == "error":
                emax = float(np.abs(scalars[np.isfinite(scalars)]).max())
                if is_vector:
                    cmap, clim = "Reds", [0.0, emax]
                else:
                    cmap, clim = "RdBu_r", [-emax, emax]
            else:
                cmap, clim = "turbo", shared_clim

            plotter.add_mesh(
                combined_pv.copy(),
                scalars=scalars,
                cmap=cmap,
                clim=clim,
                show_edges=False,
                scalar_bar_args={"title": label if row == 0 else ""},
            )
            plotter.add_text(
                f"{title}\n{label}" if row == 0 else title,
                font_size=10,
            )
            plotter.camera_position = "xy"

    plotter.screenshot(str(save_path), scale=2)
    logger.info(f"Saved comparison to {save_path}")
    if show:
        plotter.show()
    plotter.close()


def draw_disk_cells(
    ax,
    mesh: Mesh[2, 3],
    cell_scalars: np.ndarray,
    *,
    cmap: str = "turbo",
    vmin: float | None = None,
    vmax: float | None = None,
    n_sides: int = 6,
) -> None:
    """Render mesh cells as oriented disks in a matplotlib 3D axes.

    Each cell is drawn as a regular ``n_sides``-gon (default: hexagon)
    centred at the cell centroid, lying in the tangent plane defined by the
    cell normal, with area scaled so that the total disk area equals the
    total area the subsampled cells would have if they covered the full
    original surface (i.e. ``raw_area * n_original / n_subsampled``).

    This is useful when the mesh is a sparse subsample of a much finer
    surface: the original triangles are sub-pixel at typical plot
    resolution, but the scaled disks are large enough to be visible and
    convey spatial field variation.

    Args:
        ax: A matplotlib ``Axes3D`` instance.
        mesh: Surface Mesh with triangular cells (``n_manifold_dims == 2``,
            ``n_spatial_dims == 3``).
        cell_scalars: 1-D array of shape ``(n_cells,)`` with a scalar
            value per cell used for colour-mapping.
        cmap: Matplotlib colourmap name.
        vmin: Colourmap minimum.  ``None`` uses data min.
        vmax: Colourmap maximum.  ``None`` uses data max.
        n_sides: Number of polygon sides per disk (6 = hexagon).
    """
    import importlib

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    Poly3DCollection = importlib.import_module(
        "mpl_toolkits.mplot3d.art3d"
    ).Poly3DCollection

    centroids = mesh.cell_centroids.float().cpu().numpy()  # (C, 3)
    normals = mesh.cell_normals.float().cpu().numpy()  # (C, 3)
    radii = np.sqrt(mesh.cell_areas.float().cpu().numpy() / np.pi)  # (C,)

    ### Build a tangent-plane basis (u, v) for each cell from its normal.
    # Choose an arbitrary vector *not* parallel to the normal, cross it
    # with the normal to get u, then v = normal x u.
    n_cells = len(centroids)
    abs_n = np.abs(normals)
    # Pick the coordinate axis least aligned with the normal to avoid
    # degeneracy.  For each cell, the axis with the smallest |n_i|.
    aux = np.zeros_like(normals)
    aux[np.arange(n_cells), abs_n.argmin(axis=1)] = 1.0

    u = np.cross(normals, aux)
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-30
    v = np.cross(normals, u)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-30

    ### Generate polygon vertices for every cell.
    # theta: (S,)  with S = n_sides angular samples around the disk.
    theta = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)  # (S,)
    cos_t = np.cos(theta)  # (S,)
    sin_t = np.sin(theta)  # (S,)

    # offset[c, s, :] = r[c] * (cos(t[s]) * u[c] + sin(t[s]) * v[c])
    offsets = radii[:, None, None] * (
        cos_t[None, :, None] * u[:, None, :] + sin_t[None, :, None] * v[:, None, :]
    )  # (C, S, 3)
    disk_verts = centroids[:, None, :] + offsets  # (C, S, 3)

    ### Colour-map the cell scalars.
    norm = Normalize(
        vmin=vmin if vmin is not None else float(cell_scalars.min()),
        vmax=vmax if vmax is not None else float(cell_scalars.max()),
    )
    sm = ScalarMappable(norm=norm, cmap=cmap)
    facecolors = sm.to_rgba(cell_scalars)  # (C, 4)

    ### Render as a single Poly3DCollection.
    pc = Poly3DCollection(
        disk_verts,
        facecolors=facecolors,
        edgecolors="none",
        linewidths=0,
        alpha=1.0,
        zorder=1,
    )
    ax.add_collection3d(pc)

    ### Set axis limits from centroids.
    margin = 0.01 * (centroids.max() - centroids.min())
    ax.set_xlim(centroids[:, 0].min() - margin, centroids[:, 0].max() + margin)
    ax.set_ylim(centroids[:, 1].min() - margin, centroids[:, 1].max() + margin)
    ax.set_zlim(centroids[:, 2].min() - margin, centroids[:, 2].max() + margin)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))

    ### Add colourbar.
    import matplotlib.pyplot as plt

    plt.colorbar(sm, ax=ax, label="Scalars")


def visualize_matplotlib(
    combined: Mesh[2, 3],
    kind_data: dict[str, TensorDict],
    kinds: dict[str, str],
    fields: list[str],
    save_path: Path,
    show: bool,
) -> None:
    """Matplotlib backend for :meth:`DrivAerMLDataSet.visualize_comparison`.

    Renders each subsampled cell as an oriented hexagonal disk whose area
    is scaled so that the disks collectively approximate the full car
    surface.  This avoids the sub-pixel triangle problem that occurs when
    ``slice_cells`` picks a small number of cells from a very fine mesh.
    """
    import matplotlib.pyplot as plt

    n_rows, n_cols = len(kinds), len(fields)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        subplot_kw={"projection": "3d"},
        squeeze=False,
    )

    ### Pre-compute per-cell scalars for every (kind, field) combination.
    # The combined mesh has per-*point* data; we average to per-cell here
    # so draw_disk_cells can colour each disk uniformly.
    cells = combined.cells  # (n_cells, 3)

    for col, field_name in enumerate(fields):
        true_vals: torch.Tensor = kind_data["true"][field_name]  # ty: ignore[invalid-assignment]
        pred_vals: torch.Tensor = kind_data["pred"][field_name]  # ty: ignore[invalid-assignment]

        is_vector = true_vals.ndim > 1 and true_vals.shape[-1] > 1

        ### Reduce to scalar magnitudes for color-mapping
        if is_vector:
            true_scalars = true_vals.float().norm(dim=-1)
            pred_scalars = pred_vals.float().norm(dim=-1)
        else:
            true_scalars = true_vals.float().reshape(-1)
            pred_scalars = pred_vals.float().reshape(-1)

        ### Shared color limits across truth and prediction
        finite_mask_t = torch.isfinite(true_scalars)
        finite_mask_p = torch.isfinite(pred_scalars)
        finite_all = torch.cat(
            [
                true_scalars[finite_mask_t],
                pred_scalars[finite_mask_p],
            ]
        )
        shared_vmin = float(finite_all.min())
        shared_vmax = float(finite_all.max())

        for row, (key, title) in enumerate(kinds.items()):
            ax = axes[row, col]
            vals: torch.Tensor = kind_data[key][field_name]  # ty: ignore[invalid-assignment]

            if is_vector:
                pt_scalars = vals.float().norm(dim=-1)
            else:
                pt_scalars = vals.float().reshape(-1)

            cell_scalars = pt_scalars[cells].mean(dim=1).cpu().numpy()

            if key == "error":
                finite_err = cell_scalars[np.isfinite(cell_scalars)]
                emax = float(np.abs(finite_err).max()) if len(finite_err) > 0 else 1.0
                if is_vector:
                    cmap, vmin, vmax = "Reds", 0.0, emax
                else:
                    cmap, vmin, vmax = "RdBu_r", -emax, emax
            else:
                cmap, vmin, vmax = "turbo", shared_vmin, shared_vmax

            draw_disk_cells(
                ax,
                combined,
                cell_scalars=cell_scalars,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_title(
                f"{title}\n{field_name}" if row == 0 else title,
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    logger.info(f"Saved comparison to {save_path}")
    if show:
        plt.show()
    plt.close(fig)
