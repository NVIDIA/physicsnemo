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

"""Main entry point for mesh visualization with backend selection."""

from typing import TYPE_CHECKING, Any, Literal

import torch

from physicsnemo.core.version_check import check_version_spec

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh

# Check availability at module load (add new backends here)
BACKENDS_INSTALLED: dict[str, bool] = {
    name: check_version_spec(name) for name in ["matplotlib", "pyvista"]
}


def draw_mesh(
    mesh: "Mesh",
    backend: Literal["matplotlib", "pyvista", "auto"] = "auto",
    show: bool = True,
    point_scalars: None | torch.Tensor | str | tuple[str, ...] = None,
    cell_scalars: None | torch.Tensor | str | tuple[str, ...] = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    alpha_points: float = 1.0,
    alpha_cells: float = 1.0,
    alpha_edges: float = 1.0,
    show_edges: bool = True,
    ax=None,
    backend_options: dict[str, Any] | None = None,
):
    """Draw a mesh using matplotlib or PyVista backend.

    This is the main visualization function for Mesh objects. It automatically
    selects the appropriate backend based on spatial dimensions, or allows
    explicit backend specification.

    Parameters
    ----------
    mesh : Mesh
        Mesh object to visualize.
    backend : {"auto", "matplotlib", "pyvista"}
        Visualization backend to use:
        - "auto": Automatically select based on n_spatial_dims
          (matplotlib for 0D/1D/2D, PyVista for 3D)
        - "matplotlib": Force matplotlib backend (supports 3D via mplot3d)
        - "pyvista": Force PyVista backend (requires n_spatial_dims <= 3)
    show : bool
        Whether to display the plot immediately (calls plt.show() or
        plotter.show()). If False, returns the plotter/axes for further
        customization before display.
    point_scalars : torch.Tensor or str or tuple[str, ...] or None, optional
        Scalar data to color points. Mutually exclusive with
        cell_scalars. Can be:
        - None: Points use neutral color (black)
        - torch.Tensor: Direct scalar values, shape (n_points,) or
          (n_points, ...) where trailing dimensions are L2-normed
        - str or tuple[str, ...]: Key to lookup in mesh.point_data
    cell_scalars : torch.Tensor or str or tuple[str, ...] or None, optional
        Scalar data to color cells. Mutually exclusive with
        point_scalars. Can be:
        - None: Cells use neutral color (lightblue if no scalars,
          lightgray if point_scalars active)
        - torch.Tensor: Direct scalar values, shape (n_cells,) or
          (n_cells, ...) where trailing dimensions are L2-normed
        - str or tuple[str, ...]: Key to lookup in mesh.cell_data
    cmap : str
        Colormap name for scalar visualization.
    vmin : float or None, optional
        Minimum value for colormap normalization. If None, uses data min.
    vmax : float or None, optional
        Maximum value for colormap normalization. If None, uses data max.
    alpha_points : float
        Opacity for points, range [0, 1].
    alpha_cells : float
        Opacity for cells/faces, range [0, 1].
    alpha_edges : float
        Opacity for cell edges, range [0, 1].
    show_edges : bool
        Whether to draw cell edges.
    ax : matplotlib.axes.Axes, optional
        (matplotlib only) Existing matplotlib axes to plot on. If None,
        creates new figure and axes.
    backend_options : dict[str, Any], optional
        Additional keyword arguments forwarded to the underlying
        visualization backend (e.g. PyVista's ``plotter.add_mesh()``).

    Returns
    -------
    matplotlib.axes.Axes or pyvista.Plotter
        - matplotlib backend: matplotlib.axes.Axes object
        - PyVista backend: pyvista.Plotter object

    Raises
    ------
    ValueError
        If both point_scalars and cell_scalars are specified,
        or if n_spatial_dims is not supported by the chosen backend,
        or if backend selection fails.
        or if `ax` is provided for PyVista backend.
    ImportError
        If the requested backend is not installed.

    Examples
    --------
    >>> # Draw mesh with automatic backend selection
    >>> mesh.draw()  # doctest: +SKIP
    >>>
    >>> # Color cells by pressure data
    >>> mesh.draw(cell_scalars="pressure", cmap="coolwarm")  # doctest: +SKIP
    >>>
    >>> # Color points by velocity magnitude (computing norm of vector field)
    >>> mesh.draw(point_scalars="velocity")  # velocity is (n_points, 3)  # doctest: +SKIP
    >>>
    >>> # Use nested TensorDict key
    >>> mesh.draw(cell_scalars=("flow", "temperature"))  # doctest: +SKIP
    >>>
    >>> # Customize and display later
    >>> ax = mesh.draw(show=False, backend="matplotlib")  # doctest: +SKIP
    >>> ax.set_title("My Mesh")  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP
    """
    ### Validate and process scalar data
    from physicsnemo.mesh.visualization._scalar_utils import (
        validate_and_process_scalars,
    )

    point_scalar_values, cell_scalar_values, active_scalar_source, scalar_label = (
        validate_and_process_scalars(
            point_scalars=point_scalars,
            cell_scalars=cell_scalars,
            point_data=mesh.point_data,
            cell_data=mesh.cell_data,
            n_points=mesh.n_points,
            n_cells=mesh.n_cells,
        )
    )

    ### Validate spatial dimensions
    if mesh.n_spatial_dims > 3:
        raise ValueError(
            f"Visualization does not support {mesh.n_spatial_dims=}.\n"
            "Maximum spatial dimensions: 3."
        )

    ### Determine and validate backend
    if backend == "auto":
        # Check that at least one backend is available
        if not any(BACKENDS_INSTALLED.values()):
            options = ", ".join(BACKENDS_INSTALLED)
            raise ImportError(
                f"No visualization backend available. Install one of: {options}"
            )

        # Auto-select based on spatial dimensions with fallback
        if mesh.n_spatial_dims <= 2:
            # Prefer matplotlib for 0D/1D/2D
            backend = "matplotlib" if BACKENDS_INSTALLED["matplotlib"] else "pyvista"
        else:
            # Prefer pyvista for 3D
            backend = "pyvista" if BACKENDS_INSTALLED["pyvista"] else "matplotlib"

    elif backend in BACKENDS_INSTALLED:
        if not BACKENDS_INSTALLED[backend]:
            alternatives = [
                n for n, ok in BACKENDS_INSTALLED.items() if ok and n != backend
            ]
            alt_hint = f" ({', '.join(alternatives)} available)" if alternatives else ""
            raise ImportError(f"{backend} is not installed{alt_hint}.")

    else:
        supported = ", ".join(repr(b) for b in BACKENDS_INSTALLED)
        raise ValueError(
            f"Unknown {backend=!r}. Supported backends: {supported}, 'auto'."
        )

    # Track the resolved backend for warning checks below
    resolved_backend = backend

    ### Warn about unsupported options
    if backend_options and resolved_backend == "matplotlib":
        import warnings

        warnings.warn(
            "backend_options are only supported with the 'pyvista' backend and will be ignored.",
            stacklevel=2,
        )

    if alpha_edges != 1.0 and resolved_backend == "pyvista":
        import warnings

        warnings.warn(
            "alpha_edges is not supported by the 'pyvista' backend and will be ignored.",
            stacklevel=2,
        )

    ### Dispatch to backend
    if backend == "matplotlib":
        from physicsnemo.mesh.visualization._matplotlib_impl import draw_mesh_matplotlib

        return draw_mesh_matplotlib(
            mesh=mesh,
            point_scalar_values=point_scalar_values,
            cell_scalar_values=cell_scalar_values,
            active_scalar_source=active_scalar_source,
            scalar_label=scalar_label,
            show=show,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha_points=alpha_points,
            alpha_cells=alpha_cells,
            alpha_edges=alpha_edges,
            show_edges=show_edges,
            ax=ax,
        )

    elif backend == "pyvista":
        from physicsnemo.mesh.visualization._pyvista_impl import draw_mesh_pyvista

        if ax is not None:
            raise ValueError(
                "The 'ax' parameter is only supported for matplotlib backend.\n"
                "PyVista backend creates its own plotter."
            )

        return draw_mesh_pyvista(
            mesh=mesh,
            point_scalar_values=point_scalar_values,
            cell_scalar_values=cell_scalar_values,
            active_scalar_source=active_scalar_source,
            scalar_label=scalar_label,
            show=show,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha_points=alpha_points,
            alpha_cells=alpha_cells,
            show_edges=show_edges,
            **(backend_options or {}),
        )

    else:
        raise AssertionError(
            f"Unreachable: {backend=!r} passed validation but has no dispatch."
        )
