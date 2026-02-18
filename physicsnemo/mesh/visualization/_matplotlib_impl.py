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

"""Matplotlib backend for mesh visualization."""

import importlib
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

if TYPE_CHECKING:
    from physicsnemo.mesh import Mesh

# Dynamic imports for optional matplotlib dependency (invisible to static analysis)
plt = importlib.import_module("matplotlib.pyplot")
_cm = importlib.import_module("matplotlib.cm")
ScalarMappable = _cm.ScalarMappable
_collections = importlib.import_module("matplotlib.collections")
LineCollection = _collections.LineCollection
PolyCollection = _collections.PolyCollection
_colors = importlib.import_module("matplotlib.colors")
Normalize = _colors.Normalize


def draw_mesh_matplotlib(
    mesh: "Mesh",
    point_scalar_values: torch.Tensor | None,
    cell_scalar_values: torch.Tensor | None,
    active_scalar_source: Literal["points", "cells", None],
    scalar_label: str | None,
    show: bool,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    alpha_points: float,
    alpha_cells: float,
    alpha_edges: float,
    show_edges: bool,
    ax=None,
):
    """Draw mesh using matplotlib backend.

    Supports 0D, 1D, 2D, and 3D spatial dimensions with appropriate matplotlib primitives.

    Parameters
    ----------
    mesh : Mesh
        Mesh object to visualize.
    point_scalar_values : torch.Tensor or None
        Processed point scalar values (1D tensor or None).
    cell_scalar_values : torch.Tensor or None
        Processed cell scalar values (1D tensor or None).
    active_scalar_source : {"points", "cells", None}
        Which scalar source is active ("points", "cells", or None).
    scalar_label : str or None
        Human-readable label for the colorbar.
    show : bool
        Whether to call plt.show().
    cmap : str
        Colormap name.
    vmin : float or None
        Minimum value for colormap normalization.
    vmax : float or None
        Maximum value for colormap normalization.
    alpha_points : float
        Opacity for points (0-1).
    alpha_cells : float
        Opacity for cells (0-1).
    alpha_edges : float
        Opacity for edges (0-1).
    show_edges : bool
        Whether to draw cell edges.
    ax : matplotlib.axes.Axes or None
        Existing matplotlib axes (if None, creates new figure).

    Returns
    -------
    matplotlib.axes.Axes
        Matplotlib axes object.
    """
    ### For volume meshes (3D+ manifold), reduce to a surface mesh.
    ### Matplotlib can only render 2D facets (polygons), not volumetric cells
    ### like tetrahedra. Extract boundary facets for clean surface visualization.
    if mesh.n_manifold_dims >= 3:
        _VIZ_KEY = "_viz_cell_scalars"

        ### If cell scalars are active, inject them into a cloned cell_data so
        ### get_facet_mesh can propagate them to boundary facets via averaging.
        ### We clone to avoid mutating the caller's mesh.
        if cell_scalar_values is not None:
            from physicsnemo.mesh import Mesh

            augmented_cell_data = mesh.cell_data.clone()
            augmented_cell_data[_VIZ_KEY] = cell_scalar_values
            mesh = Mesh(
                points=mesh.points,
                cells=mesh.cells,
                point_data=mesh.point_data,
                cell_data=augmented_cell_data,
                global_data=mesh.global_data,
            )

        mesh = mesh.get_facet_mesh(
            manifold_codimension=mesh.n_manifold_dims - 2,
            data_source="cells",
            data_aggregation="mean",
            target_counts="boundary",
        )

        ### Extract propagated cell scalars from the facet mesh
        if cell_scalar_values is not None:
            cell_scalar_values = mesh.cell_data[_VIZ_KEY]

    ### Convert mesh data to numpy
    points_np = mesh.points.cpu().detach().numpy()
    cells_np = mesh.cells.cpu().detach().numpy()

    ### Determine neutral colors based on active_scalar_source
    point_neutral_color = "black"
    if active_scalar_source is None:
        cell_neutral_color = "lightblue"
    elif active_scalar_source == "points":
        cell_neutral_color = "lightgray"
    else:  # active_scalar_source == "cells"
        cell_neutral_color = None  # Will be colored by scalars

    ### Create figure and axes if not provided
    if ax is None:
        if mesh.n_spatial_dims == 3:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()

    ### Determine scalar colormap normalization
    if active_scalar_source == "points" and point_scalar_values is not None:
        scalar_values_for_norm = point_scalar_values.cpu().numpy()
    elif active_scalar_source == "cells" and cell_scalar_values is not None:
        scalar_values_for_norm = cell_scalar_values.cpu().numpy()
    else:
        scalar_values_for_norm = None

    if scalar_values_for_norm is not None:
        norm = Normalize(
            vmin=vmin if vmin is not None else scalar_values_for_norm.min(),
            vmax=vmax if vmax is not None else scalar_values_for_norm.max(),
        )
        scalar_mapper = ScalarMappable(norm=norm, cmap=cmap)
    else:
        norm = None
        scalar_mapper = None

    ### Draw based on spatial dimensionality
    if mesh.n_spatial_dims == 0:
        _draw_0d(
            ax,
            points_np,
            point_scalar_values,
            active_scalar_source,
            scalar_mapper,
            point_neutral_color,
            alpha_points,
        )
    elif mesh.n_spatial_dims == 1:
        _draw_1d(
            ax,
            points_np,
            cells_np,
            point_scalar_values,
            cell_scalar_values,
            active_scalar_source,
            scalar_mapper,
            point_neutral_color,
            cell_neutral_color,
            alpha_points,
            alpha_cells,
        )
    elif mesh.n_spatial_dims == 2:
        _draw_2d(
            ax,
            points_np,
            cells_np,
            point_scalar_values,
            cell_scalar_values,
            active_scalar_source,
            scalar_mapper,
            point_neutral_color,
            cell_neutral_color,
            alpha_points,
            alpha_cells,
            alpha_edges,
            show_edges,
        )
    elif mesh.n_spatial_dims == 3:
        _draw_3d(
            ax,
            points_np,
            cells_np,
            point_scalar_values,
            cell_scalar_values,
            active_scalar_source,
            scalar_mapper,
            point_neutral_color,
            cell_neutral_color,
            alpha_points,
            alpha_cells,
            alpha_edges,
            show_edges,
        )
    else:
        raise ValueError(
            f"Cannot visualize mesh with {mesh.n_spatial_dims=}.\n"
            f"Supported spatial dimensions: 0, 1, 2, 3."
        )

    ### Add colorbar if we have active scalars
    if scalar_mapper is not None:
        plt.colorbar(scalar_mapper, ax=ax, label=scalar_label or "")

    ### Set labels and make axes equal
    if mesh.n_spatial_dims == 1:
        ax.set_xlabel("x")
        ax.set_aspect("equal", adjustable="box")
    elif mesh.n_spatial_dims == 2:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
    elif mesh.n_spatial_dims == 3:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")  # ty: ignore[possibly-missing-attribute]

        ### Make 3D axes equal by adjusting limits to have same range
        ax.set_box_aspect((1, 1, 1))  # ty: ignore[invalid-argument-type]

        xlim = ax.get_xlim3d()  # ty: ignore[possibly-missing-attribute]
        ylim = ax.get_ylim3d()  # ty: ignore[possibly-missing-attribute]
        zlim = ax.get_zlim3d()  # ty: ignore[possibly-missing-attribute]

        x_range = abs(xlim[1] - xlim[0])
        x_middle = np.mean(xlim)
        y_range = abs(ylim[1] - ylim[0])
        y_middle = np.mean(ylim)
        z_range = abs(zlim[1] - zlim[0])
        z_middle = np.mean(zlim)

        # Use the maximum range to ensure all axes have equal scale
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])  # ty: ignore[possibly-missing-attribute]
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])  # ty: ignore[possibly-missing-attribute]
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])  # ty: ignore[possibly-missing-attribute]

    if show:
        plt.show()

    return ax


def _draw_0d(
    ax,
    points_np,
    point_scalar_values,
    active_scalar_source,
    scalar_mapper,
    point_neutral_color,
    alpha_points,
):
    """Draw 0D manifold (point cloud) in 0D space."""
    # For 0D spatial dimensions, all points are at the origin
    # We can represent them as points at x=0
    n_points = len(points_np)

    if active_scalar_source == "points" and point_scalar_values is not None:
        colors = scalar_mapper.to_rgba(point_scalar_values.cpu().numpy())
    else:
        colors = point_neutral_color

    # Draw points at the origin
    ax.scatter(
        np.zeros(n_points), np.zeros(n_points), c=colors, alpha=alpha_points, s=5
    )
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.5, 0.5)


def _draw_1d(
    ax,
    points_np,
    cells_np,
    point_scalar_values,
    cell_scalar_values,
    active_scalar_source,
    scalar_mapper,
    point_neutral_color,
    cell_neutral_color,
    alpha_points,
    alpha_cells,
):
    """Draw 1D manifold (edges) in 1D or 2D space."""
    # Points are 1D, so plot along x-axis (or in 2D if embedded in 2D)
    if points_np.shape[1] == 1:
        # Truly 1D: plot on x-axis
        x = points_np[:, 0]
    else:
        # Should not happen for n_spatial_dims=1, but handle gracefully
        raise ValueError(
            f"Expected 1D points for 1D spatial dimension, got shape {points_np.shape}"
        )

    ### Draw cells (line segments)
    if cells_np.shape[0] > 0 and alpha_cells > 0:
        segments = points_np[cells_np[:, :2]]  # Shape: (n_cells, 2, 1)
        segments = np.stack(
            [segments[:, :, 0], np.zeros((len(segments), 2))], axis=-1
        )  # Add y=0

        if active_scalar_source == "cells" and cell_scalar_values is not None:
            colors = scalar_mapper.to_rgba(cell_scalar_values.cpu().numpy())
        else:
            colors = cell_neutral_color

        lc = LineCollection(
            segments, colors=colors, alpha=alpha_cells, linewidths=2, zorder=1
        )
        ax.add_collection(lc)

    ### Draw points
    if alpha_points > 0:
        if active_scalar_source == "points" and point_scalar_values is not None:
            colors = scalar_mapper.to_rgba(point_scalar_values.cpu().numpy())
        else:
            colors = point_neutral_color

        ax.scatter(x, np.zeros_like(x), c=colors, alpha=alpha_points, s=5, zorder=2)


def _draw_2d(
    ax,
    points_np,
    cells_np,
    point_scalar_values,
    cell_scalar_values,
    active_scalar_source,
    scalar_mapper,
    point_neutral_color,
    cell_neutral_color,
    alpha_points,
    alpha_cells,
    alpha_edges,
    show_edges,
):
    """Draw 2D manifold (triangles) in 2D space."""
    ### Draw cells (filled polygons)
    if cells_np.shape[0] > 0 and alpha_cells > 0:
        # Create polygons from cells
        verts = points_np[cells_np]  # Shape: (n_cells, n_vertices_per_cell, 2)

        if active_scalar_source == "cells" and cell_scalar_values is not None:
            facecolors = scalar_mapper.to_rgba(cell_scalar_values.cpu().numpy())
        elif active_scalar_source == "points" and point_scalar_values is not None:
            # Map per-vertex scalars to per-face colors by averaging each
            # face's vertex RGBA values (same approach as _draw_3d).
            vertex_colors = scalar_mapper.to_rgba(
                point_scalar_values.cpu().numpy()
            )  # (n_points, 4)
            facecolors = vertex_colors[cells_np].mean(axis=1)  # (n_faces, 4)
        else:
            facecolors = cell_neutral_color

        if show_edges and alpha_edges > 0:
            edgecolors = "black"
            linewidths = 0.25
        else:
            edgecolors = "none"
            linewidths = 0

        pc = PolyCollection(
            verts,
            facecolors=facecolors,
            edgecolors=edgecolors,
            linewidths=linewidths,
            alpha=alpha_cells,
            zorder=1,
        )
        # Set edge alpha separately if needed
        if show_edges and alpha_edges > 0 and alpha_edges != alpha_cells:
            pc.set_edgecolor([(0, 0, 0, alpha_edges)] * len(verts))

        ax.add_collection(pc)

    ### Draw points
    # When point scalars have been mapped onto face colors (above), suppress the
    # scatter overlay - the faces already carry the vertex color information and
    # overlaid dots would be redundant. This matches _draw_3d behavior.
    has_colored_surface = (
        cells_np.shape[0] > 0
        and alpha_cells > 0
        and active_scalar_source == "points"
        and point_scalar_values is not None
    )

    if alpha_points > 0 and not has_colored_surface:
        if active_scalar_source == "points" and point_scalar_values is not None:
            colors = scalar_mapper.to_rgba(point_scalar_values.cpu().numpy())
        else:
            colors = point_neutral_color

        ax.scatter(
            points_np[:, 0],
            points_np[:, 1],
            c=colors,
            alpha=alpha_points,
            s=5,
            zorder=2,
        )

    ### Set axis limits based on data
    if len(points_np) > 0:
        margin = 0.05 * (points_np.max() - points_np.min())
        ax.set_xlim(points_np[:, 0].min() - margin, points_np[:, 0].max() + margin)
        ax.set_ylim(points_np[:, 1].min() - margin, points_np[:, 1].max() + margin)


def _draw_3d(
    ax,
    points_np,
    cells_np,
    point_scalar_values,
    cell_scalar_values,
    active_scalar_source,
    scalar_mapper,
    point_neutral_color,
    cell_neutral_color,
    alpha_points,
    alpha_cells,
    alpha_edges,
    show_edges,
):
    """Draw mesh in 3D space using mpl_toolkits.mplot3d."""
    _art3d = importlib.import_module("mpl_toolkits.mplot3d.art3d")
    Line3DCollection = _art3d.Line3DCollection
    Poly3DCollection = _art3d.Poly3DCollection

    ### Draw cells based on manifold dimension
    if cells_np.shape[0] > 0 and alpha_cells > 0:
        n_manifold_dims = cells_np.shape[1] - 1

        if n_manifold_dims == 0:
            # 0D manifold in 3D: just points (handled below)
            pass

        elif n_manifold_dims == 1:
            # 1D manifold (edges) in 3D: use Line3DCollection
            segments = points_np[cells_np[:, :2]]  # Shape: (n_cells, 2, 3)

            if active_scalar_source == "cells" and cell_scalar_values is not None:
                colors = scalar_mapper.to_rgba(cell_scalar_values.cpu().numpy())
            else:
                colors = cell_neutral_color

            lc = Line3DCollection(
                segments, colors=colors, alpha=alpha_cells, linewidths=2, zorder=1
            )
            ax.add_collection3d(lc)

        elif n_manifold_dims == 2:
            # 2D manifold (triangles) in 3D: use Poly3DCollection
            verts = points_np[cells_np]  # Shape: (n_cells, 3, 3)

            if active_scalar_source == "cells" and cell_scalar_values is not None:
                facecolors = scalar_mapper.to_rgba(cell_scalar_values.cpu().numpy())
            elif active_scalar_source == "points" and point_scalar_values is not None:
                # Map per-vertex scalars to per-face colors by averaging each
                # face's vertex RGBA values.  This avoids a separate scatter
                # overlay that matplotlib cannot correctly depth-sort against
                # Poly3DCollection faces (painter's algorithm limitation).
                vertex_colors = scalar_mapper.to_rgba(
                    point_scalar_values.cpu().numpy()
                )  # (n_points, 4)
                facecolors = vertex_colors[cells_np].mean(axis=1)  # (n_faces, 4)
            else:
                facecolors = cell_neutral_color

            if show_edges and alpha_edges > 0:
                edgecolors = [(0, 0, 0, alpha_edges)] * len(verts)
                linewidths = 0.25
            else:
                edgecolors = None
                linewidths = 0

            pc = Poly3DCollection(
                verts,
                facecolors=facecolors,
                edgecolors=edgecolors,
                linewidths=linewidths,
                alpha=alpha_cells,
                shade=True,
                zorder=1,
            )
            ax.add_collection3d(pc)

        else:
            # Volume meshes (3D+ manifold) are reduced to surface meshes in
            # draw_mesh_matplotlib() before reaching this function.
            raise ValueError(
                f"Cannot render {n_manifold_dims}D cells directly in matplotlib. "
                f"Volume meshes should be converted to surface meshes via "
                f"get_facet_mesh() before calling _draw_3d."
            )

    ### Draw points
    # For 3D surface meshes, skip the scatter overlay: matplotlib cannot
    # depth-sort scatter points against Poly3DCollection faces (painter's
    # algorithm limitation).  Vertices are visible at face corners via edges,
    # and point scalars are mapped onto face colors above.
    has_opaque_surface = (
        cells_np.shape[0] > 0 and cells_np.shape[1] - 1 == 2 and alpha_cells > 0
    )

    if alpha_points > 0 and not has_opaque_surface:
        if active_scalar_source == "points" and point_scalar_values is not None:
            colors = scalar_mapper.to_rgba(point_scalar_values.cpu().numpy())
        else:
            colors = point_neutral_color

        ax.scatter(
            points_np[:, 0],
            points_np[:, 1],
            points_np[:, 2],
            c=colors,
            alpha=alpha_points,
            s=5,
            zorder=2,
        )

    ### Set axis limits based on data
    if len(points_np) > 0:
        margin = 0.01 * (points_np.max() - points_np.min())
        ax.set_xlim(points_np[:, 0].min() - margin, points_np[:, 0].max() + margin)
        ax.set_ylim(points_np[:, 1].min() - margin, points_np[:, 1].max() + margin)
        ax.set_zlim(points_np[:, 2].min() - margin, points_np[:, 2].max() + margin)
