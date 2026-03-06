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

"""PyVista backend for mesh visualization."""

import importlib
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from physicsnemo.mesh import Mesh

# Dynamic import for optional pyvista dependency (invisible to static analysis)
pv = importlib.import_module("pyvista")


def draw_mesh_pyvista(
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
    show_edges: bool,
    **kwargs,
):
    """Draw mesh using PyVista backend.

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
        Human-readable label for the colorbar title.
    show : bool
        Whether to call plotter.show().
    cmap : str
        Colormap name.
    vmin : float or None
        Minimum value for colormap normalization (clim).
    vmax : float or None
        Maximum value for colormap normalization (clim).
    alpha_points : float
        Opacity for points (0-1).
    alpha_cells : float
        Opacity for cells (0-1).
    show_edges : bool
        Whether to draw cell edges.
    **kwargs : dict
        Additional backend-specific arguments passed to PyVista.

    Returns
    -------
    pyvista.Plotter
        PyVista plotter object (even if show=True, returns before calling .show()).
    """
    ### Convert mesh to PyVista format
    from physicsnemo.mesh.io.io_pyvista import to_pyvista

    pv_mesh = to_pyvista(mesh)

    ### Add scalar data to PyVista mesh based on active_scalar_source
    scalar_name = None
    if active_scalar_source == "points" and point_scalar_values is not None:
        pv_mesh.point_data["_viz_scalars"] = point_scalar_values.cpu().numpy()
        scalar_name = "_viz_scalars"
    elif active_scalar_source == "cells" and cell_scalar_values is not None:
        pv_mesh.cell_data["_viz_scalars"] = cell_scalar_values.cpu().numpy()
        scalar_name = "_viz_scalars"

    ### Create plotter
    plotter = pv.Plotter()

    ### Determine colors based on active_scalar_source
    if active_scalar_source is None:
        # No scalars: use neutral colors
        color = "lightblue"
        scalars = None
    elif active_scalar_source == "points":
        # Point scalars active
        color = None
        scalars = scalar_name
        # Cell neutral color will be handled by render_points_as_spheres=False
    elif active_scalar_source == "cells":
        # Cell scalars active
        color = None
        scalars = scalar_name
    else:
        color = "lightblue"
        scalars = None

    ### Determine clim (color limits) if scalars are present
    if scalars is not None:
        if vmin is not None or vmax is not None:
            clim = [
                vmin
                if vmin is not None
                else (
                    point_scalar_values.min().item()
                    if point_scalar_values is not None
                    else cell_scalar_values.min().item()
                ),
                vmax
                if vmax is not None
                else (
                    point_scalar_values.max().item()
                    if point_scalar_values is not None
                    else cell_scalar_values.max().item()
                ),
            ]
        else:
            clim = None
    else:
        clim = None

    ### Add mesh to plotter
    # PyVista's add_mesh handles different mesh types appropriately
    scalar_bar_args = {"title": scalar_label} if scalar_label else None
    plotter.add_mesh(
        pv_mesh,
        scalars=scalars,
        cmap=cmap,
        color=color,
        opacity=alpha_cells,
        show_edges=show_edges,
        edge_color="black",
        line_width=1.0 if show_edges else 0,
        clim=clim,
        scalar_bar_args=scalar_bar_args,
        **kwargs,
    )

    ### Add points as a separate actor if needed
    # PyVista's point rendering can be controlled via render_points_as_spheres
    # For now, we'll use a simple approach: if alpha_points > 0, show points
    if alpha_points > 0 and mesh.n_points > 0:
        # Create a point cloud from the PyVista mesh points (already 3D-padded)
        point_cloud = pv.PolyData(pv_mesh.points)

        # Add point scalar data if present
        if active_scalar_source == "points" and point_scalar_values is not None:
            point_cloud.point_data["_viz_scalars"] = point_scalar_values.cpu().numpy()
            point_scalars = "_viz_scalars"
            point_color = None
        else:
            point_scalars = None
            point_color = "black"

        plotter.add_mesh(
            point_cloud,
            scalars=point_scalars,
            cmap=cmap if point_scalars else None,
            color=point_color,
            point_size=5.0,
            render_points_as_spheres=True,
            opacity=alpha_points,
            clim=clim if point_scalars else None,
            scalar_bar_args=scalar_bar_args if point_scalars else None,
        )

    ### Show plotter if requested
    if show:
        plotter.show()

    return plotter
