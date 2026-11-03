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

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch
from jaxtyping import Float

from physicsnemo.core.version_check import OptionalImport

### Optional dependencies. Each ``OptionalImport`` is a lazy proxy:
### construction does not import the package; the friendly ``ImportError``
### (with the ``[mesh-extras]`` install hint) fires only on first attribute
### access. We therefore avoid extracting individual classes at module level
### (e.g. ``ScalarMappable = _cm.ScalarMappable`` would import matplotlib at
### load time) and reach through the proxy at the use sites instead.
###
### This module is only loaded by ``draw_mesh.draw_mesh(..., backend="matplotlib")``,
### so that first attribute access already implies the user opted in.
if TYPE_CHECKING:
    import matplotlib.cm as mpl_cm
    import matplotlib.collections as mpl_collections
    import matplotlib.colors as mpl_colors
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d.art3d as mpl_art3d
    from matplotlib import axes as mpl_axes
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    from physicsnemo.mesh import Mesh
else:
    plt = OptionalImport("matplotlib.pyplot")
    mpl_cm = OptionalImport("matplotlib.cm")
    mpl_collections = OptionalImport("matplotlib.collections")
    mpl_colors = OptionalImport("matplotlib.colors")
    mpl_axes = OptionalImport("matplotlib.axes")
    mpl_art3d = OptionalImport("mpl_toolkits.mplot3d.art3d")


def draw_mesh_matplotlib(
    mesh: "Mesh",
    point_scalar_values: Float[torch.Tensor, " n_points"] | None,
    cell_scalar_values: Float[torch.Tensor, " n_cells"] | None,
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
    ax: Any = None,
) -> Any:
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
        Existing matplotlib Axes to draw on. If ``None``, a new figure is
        created. Use this to overlay multiple meshes on the same axes.

    Returns
    -------
    matplotlib.axes.Axes
        Matplotlib axes object.
    """
    ### Validate ax type
    if ax is not None and not isinstance(ax, mpl_axes.Axes):
        raise ValueError(
            f"Expected a matplotlib Axes for the 'ax' parameter, "
            f"got {type(ax).__name__}. "
            f"PyVista Plotters are only supported for pyvista backend."
        )

    ### For volume meshes (3D+ manifold), reduce to a surface mesh.
    ### Matplotlib can only render 2D facets (polygons), not volumetric cells
    ### like tetrahedra. Extract boundary facets for clean surface visualization.
    if mesh.n_manifold_dims >= 3:
        _VIZ_KEY = "_viz_cell_scalars"

        ### If cell scalars are active, inject them into a cloned cell_data so
        ### get_facet_mesh can propagate them to boundary facets via averaging.
        ### We clone to avoid mutating the caller's mesh.
        if cell_scalar_values is not None:
            augmented_cell_data = mesh.cell_data.clone()
            augmented_cell_data[_VIZ_KEY] = cell_scalar_values
            mesh = mesh.with_data(cell_data=augmented_cell_data)

        mesh = mesh.get_facet_mesh(
            manifold_codimension=mesh.n_manifold_dims - 2,
            data_source="cells",
            data_aggregation="mean",
            target_counts="boundary",
        )

        ### Extract propagated cell scalars from the facet mesh
        if cell_scalar_values is not None:
            cell_scalar_values = mesh.cell_data[_VIZ_KEY]

    ### Ensure scalar tensors are NumPy-compatible (BFloat16 has no NumPy equivalent)
    if point_scalar_values is not None:
        point_scalar_values = point_scalar_values.float()
    if cell_scalar_values is not None:
        cell_scalar_values = cell_scalar_values.float()

    ### Convert mesh data to numpy
    points_np = mesh.points.float().cpu().detach().numpy()
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
        norm = mpl_colors.Normalize(
            vmin=vmin if vmin is not None else scalar_values_for_norm.min(),
            vmax=vmax if vmax is not None else scalar_values_for_norm.max(),
        )
        scalar_mapper = mpl_cm.ScalarMappable(norm=norm, cmap=cmap)
    else:
        norm = None
        scalar_mapper = None

    ### Cell colors: mapped from the cell scalars if those are active, else
    ### the neutral cell color
    if active_scalar_source == "cells" and cell_scalar_values is not None:
        cell_colors = scalar_mapper.to_rgba(cell_scalar_values.cpu().numpy())
    else:
        cell_colors = cell_neutral_color

    ### Draw based on spatial dimensionality
    if mesh.n_spatial_dims == 0:
        _draw_0d(
            ax,
            points_np,
            cells_np,
            point_scalar_values,
            active_scalar_source,
            scalar_mapper,
            point_neutral_color,
            cell_colors,
            alpha_points,
            alpha_cells,
        )
    elif mesh.n_spatial_dims == 1:
        _draw_1d(
            ax,
            points_np,
            cells_np,
            point_scalar_values,
            active_scalar_source,
            scalar_mapper,
            point_neutral_color,
            cell_colors,
            alpha_points,
            alpha_cells,
        )
    elif mesh.n_spatial_dims == 2:
        _draw_2d(
            ax,
            points_np,
            cells_np,
            point_scalar_values,
            active_scalar_source,
            scalar_mapper,
            point_neutral_color,
            cell_colors,
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
            active_scalar_source,
            scalar_mapper,
            point_neutral_color,
            cell_colors,
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
        # ax was created with projection="3d" upstream when n_spatial_dims=3
        # (see the figure setup earlier), or the caller provided a 3D-projected
        # Axes. The 3D-only methods (set_zlabel, get_*lim3d, set_*lim3d) live on
        # Axes3D, not the 2D base Axes - so narrow the static type once here so
        # they're visible without per-line suppressions.
        ax_3d = cast("Axes3D", ax)

        ax_3d.set_xlabel("x")
        ax_3d.set_ylabel("y")
        ax_3d.set_zlabel("z")

        ### Make 3D axes equal by adjusting limits to have same range
        ax_3d.set_box_aspect((1, 1, 1))

        xlim = ax_3d.get_xlim3d()
        ylim = ax_3d.get_ylim3d()
        zlim = ax_3d.get_zlim3d()

        x_range = abs(xlim[1] - xlim[0])
        x_middle = np.mean(xlim)
        y_range = abs(ylim[1] - ylim[0])
        y_middle = np.mean(ylim)
        z_range = abs(zlim[1] - zlim[0])
        z_middle = np.mean(zlim)

        # Use the maximum range to ensure all axes have equal scale
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax_3d.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax_3d.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax_3d.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    if show:
        plt.show()

    return ax


def _draw_vertex_cells(ax, cell_points_np, cell_colors, alpha_cells):
    """Draw vertex cells as markers at their points (2D or 3D coordinates).

    The markers are larger than the point markers, so a point drawn over its
    vertex cell leaves a colored ring.
    """
    ax.scatter(*cell_points_np.T, c=cell_colors, alpha=alpha_cells, s=20, zorder=1)


def _draw_0d(
    ax,
    points_np,
    cells_np,
    point_scalar_values,
    active_scalar_source,
    scalar_mapper,
    point_neutral_color,
    cell_colors,
    alpha_points,
    alpha_cells,
):
    """Draw 0D manifold (point cloud) in 0D space."""
    # For 0D spatial dimensions, all points are at the origin
    # We can represent them as points at x=0
    n_points = len(points_np)

    ### Draw vertex cells, also at the origin
    if cells_np.shape[0] > 0 and alpha_cells > 0:
        _draw_vertex_cells(ax, np.zeros((len(cells_np), 2)), cell_colors, alpha_cells)

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
    active_scalar_source,
    scalar_mapper,
    point_neutral_color,
    cell_colors,
    alpha_points,
    alpha_cells,
):
    """Draw vertex cells or edges in 1D space."""
    # Points are 1D, so plot along x-axis (or in 2D if embedded in 2D)
    if points_np.shape[1] == 1:
        # Truly 1D: plot on x-axis
        x = points_np[:, 0]
    else:
        # Should not happen for n_spatial_dims=1, but handle gracefully
        raise ValueError(
            f"Expected 1D points for 1D spatial dimension, got shape {points_np.shape}"
        )

    ### Draw cells: markers for vertex cells, else line segments
    if cells_np.shape[0] > 0 and alpha_cells > 0:
        # Plot along the x-axis, at y=0
        cell_points_np = np.stack(
            [x[cells_np], np.zeros(cells_np.shape)], axis=-1
        )  # Shape: (n_cells, n_vertices_per_cell, 2)
        if cells_np.shape[1] == 1:
            _draw_vertex_cells(ax, cell_points_np[:, 0], cell_colors, alpha_cells)
        else:
            lc = mpl_collections.LineCollection(
                cell_points_np,
                colors=cell_colors,
                alpha=alpha_cells,
                linewidths=2,
                zorder=1,
            )
            ax.add_collection(lc)

    ### Draw points
    if alpha_points > 0:
        if active_scalar_source == "points" and point_scalar_values is not None:
            colors = scalar_mapper.to_rgba(point_scalar_values.cpu().numpy())
        else:
            colors = point_neutral_color

        ax.scatter(x, np.zeros_like(x), c=colors, alpha=alpha_points, s=5, zorder=2)

    ### add_collection does not rescale the view, so fit it to the lines too.
    ### A no-op on axes whose limits the caller fixed.
    ax.autoscale_view()


def _draw_2d(
    ax,
    points_np,
    cells_np,
    point_scalar_values,
    active_scalar_source,
    scalar_mapper,
    point_neutral_color,
    cell_colors,
    alpha_points,
    alpha_cells,
    alpha_edges,
    show_edges,
):
    """Draw mesh in 2D space: markers, line segments or polygons for cells."""
    ### Draw cells: markers for vertex cells
    if cells_np.shape[0] > 0 and cells_np.shape[1] == 1 and alpha_cells > 0:
        _draw_vertex_cells(ax, points_np[cells_np[:, 0]], cell_colors, alpha_cells)

    ### Draw cells: line segments for 1D manifolds (edges)
    elif cells_np.shape[0] > 0 and cells_np.shape[1] == 2 and alpha_cells > 0:
        lc = mpl_collections.LineCollection(
            points_np[cells_np],
            colors=cell_colors,
            alpha=alpha_cells,
            linewidths=2,
            zorder=1,
        )
        ax.add_collection(lc)

    ### Draw cells (filled polygons)
    elif cells_np.shape[0] > 0 and alpha_cells > 0:
        # Create polygons from cells
        verts = points_np[cells_np]  # Shape: (n_cells, n_vertices_per_cell, 2)

        if active_scalar_source == "points" and point_scalar_values is not None:
            # Map per-vertex scalars to per-face colors by averaging each
            # face's vertex RGBA values (same approach as _draw_3d).
            vertex_colors = scalar_mapper.to_rgba(
                point_scalar_values.cpu().numpy()
            )  # (n_points, 4)
            facecolors = vertex_colors[cells_np].mean(axis=1)  # (n_faces, 4)
        else:
            facecolors = cell_colors

        if show_edges and alpha_edges > 0:
            edgecolors = (0, 0, 0, alpha_edges)
            linewidths = 0.25
        else:
            edgecolors = "none"
            linewidths = 0

        # Bake the opacities into the RGBA colors: a collection-wide alpha
        # would override alpha_edges.
        pc = mpl_collections.PolyCollection(
            verts,
            facecolors=mpl_colors.to_rgba_array(facecolors, alpha_cells),
            edgecolors=edgecolors,
            linewidths=linewidths,
            zorder=1,
        )
        ax.add_collection(pc)

    ### Draw points
    # When point scalars have been mapped onto face colors (above), suppress the
    # scatter overlay - the faces already carry the vertex color information and
    # overlaid dots would be redundant. This matches _draw_3d behavior.
    has_colored_surface = (
        cells_np.shape[0] > 0
        and cells_np.shape[1] >= 3
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
    active_scalar_source,
    scalar_mapper,
    point_neutral_color,
    cell_colors,
    alpha_points,
    alpha_cells,
    alpha_edges,
    show_edges,
):
    """Draw mesh in 3D space using mpl_toolkits.mplot3d."""
    ### Draw cells based on manifold dimension
    if cells_np.shape[0] > 0 and alpha_cells > 0:
        n_manifold_dims = cells_np.shape[1] - 1

        if n_manifold_dims == 0:
            # 0D manifold (vertex cells) in 3D: markers
            _draw_vertex_cells(ax, points_np[cells_np[:, 0]], cell_colors, alpha_cells)

        elif n_manifold_dims == 1:
            # 1D manifold (edges) in 3D: use Line3DCollection
            segments = points_np[cells_np[:, :2]]  # Shape: (n_cells, 2, 3)

            lc = mpl_art3d.Line3DCollection(
                segments, colors=cell_colors, alpha=alpha_cells, linewidths=2, zorder=1
            )
            ax.add_collection3d(lc)

        elif n_manifold_dims == 2:
            # 2D manifold (triangles) in 3D: use Poly3DCollection
            verts = points_np[cells_np]  # Shape: (n_cells, 3, 3)

            if active_scalar_source == "points" and point_scalar_values is not None:
                # Map per-vertex scalars to per-face colors by averaging each
                # face's vertex RGBA values.  This avoids a separate scatter
                # overlay that matplotlib cannot correctly depth-sort against
                # Poly3DCollection faces (painter's algorithm limitation).
                vertex_colors = scalar_mapper.to_rgba(
                    point_scalar_values.cpu().numpy()
                )  # (n_points, 4)
                facecolors = vertex_colors[cells_np].mean(axis=1)  # (n_faces, 4)
            else:
                facecolors = cell_colors

            if show_edges and alpha_edges > 0:
                edgecolors = [(0, 0, 0, alpha_edges)] * len(verts)
                linewidths = 0.25
            else:
                ### Per-face transparent RGBA, NOT the string "none":
                ### Poly3DCollection with shade=True calls _shade_colors on
                ### edgecolors, which crashes on the empty array that "none"
                ### resolves to (matplotlib >= 3.7).  A transparent RGBA list
                ### of the right length renders identically (invisible) and
                ### threads through the shading code without breaking.
                edgecolors = [(0, 0, 0, 0)] * len(verts)
                linewidths = 0

            pc = mpl_art3d.Poly3DCollection(
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
