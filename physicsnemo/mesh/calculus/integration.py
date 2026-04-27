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

r"""Integration of scalar, vector, and tensor fields over simplicial meshes.

Provides quadrature rules for integrating fields discretized on simplicial
meshes of any manifold dimension.  The manifold dimension determines the
measure automatically: arc length for 1-manifolds, surface area for
2-manifolds, volume for 3-manifolds, etc.

Two data sources are supported:

**Cell data (P0)** - piecewise-constant fields:

.. math::
    \int_\Omega f\,d\Omega = \sum_c f_c \,|\sigma_c|

**Point data (P1)** - vertex-centered fields treated as nodal values of a
piecewise-linear field interpolated via barycentric coordinates.  The
integral of a linear function over an n-simplex equals the volume times the
arithmetic mean of vertex values:

.. math::
    \int_\Omega f\,d\Omega
    = \sum_c |\sigma_c| \cdot \frac{1}{n_v} \sum_{v \in c} f(v)

This is exact for P1 fields and second-order accurate for smooth fields.
"""

from typing import TYPE_CHECKING, Literal

import torch
from jaxtyping import Float

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def _resolve_field(
    mesh: "Mesh",
    field: str | tuple[str, ...] | Float[torch.Tensor, "n ..."],
    data_source: Literal["cells", "points"],
) -> Float[torch.Tensor, "n ..."]:
    r"""Resolve a field specification to a concrete tensor.

    Parameters
    ----------
    mesh : Mesh
        Source mesh.
    field : str, tuple, or torch.Tensor
        A string or tuple is looked up in ``cell_data`` or ``point_data``
        depending on ``data_source``.  A tensor is returned as-is.
    data_source : {"cells", "points"}
        Which data dictionary to use for string key lookups.

    Returns
    -------
    torch.Tensor
        The resolved field tensor.
    """
    if isinstance(field, torch.Tensor):
        return field
    match data_source:
        case "cells":
            data, attr_name = mesh.cell_data, "cell_data"
        case "points":
            data, attr_name = mesh.point_data, "point_data"
        case _:
            raise ValueError(f"Invalid {data_source=!r}. Must be 'cells' or 'points'.")
    try:
        return data[field]
    except KeyError:
        available = sorted(data.keys())
        raise KeyError(
            f"Field {field!r} not found in {attr_name}. Available keys: {available}"
        ) from None


def integrate_cell_data(
    mesh: "Mesh",
    field: Float[torch.Tensor, "n_cells ..."],
) -> Float[torch.Tensor, " ..."]:
    r"""Integrate a cell-centered (P0) field over the mesh.

    Computes the exact integral of a piecewise-constant field:

    .. math::
        \int_\Omega f\,d\Omega = \sum_c f_c \,|\sigma_c|

    NaN values in ``field`` are excluded from the sum (treated as zero
    contribution), which is appropriate for fields with patched-out
    regions (e.g. non-physical points in CFD solutions).

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh with at least one cell.
    field : torch.Tensor
        Cell-centered values, shape ``(n_cells, ...)``.
        Trailing dimensions are preserved in the output.

    Returns
    -------
    torch.Tensor
        Integral value.  Shape matches ``field.shape[1:]`` (the trailing
        dimensions).  A scalar field ``(n_cells,)`` produces a 0-d tensor.

    Raises
    ------
    ValueError
        If ``field.shape[0]`` does not equal ``mesh.n_cells``.
    """
    if not torch.compiler.is_compiling():
        if field.shape[0] != mesh.n_cells:
            raise ValueError(
                f"Field leading dimension ({field.shape[0]}) must equal "
                f"n_cells ({mesh.n_cells})."
            )

    cell_areas = mesh.cell_areas  # (n_cells,)

    ### Reshape cell_areas for broadcasting with arbitrary trailing dims
    weights = cell_areas.reshape(-1, *([1] * (field.ndim - 1)))

    return torch.nansum(field * weights, dim=0)


def integrate_point_data(
    mesh: "Mesh",
    field: Float[torch.Tensor, "n_points ..."],
) -> Float[torch.Tensor, " ..."]:
    r"""Integrate a vertex-centered (P1) field over the mesh.

    Treats vertex values as nodal values of a piecewise-linear field
    and integrates analytically per simplex using the vertex-averaging
    rule (second-order accurate for smooth fields).

    If any vertex of a cell has NaN, that cell's contribution is NaN and
    is excluded by ``nansum`` (the P1 interpolant is undefined on that cell).

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh with at least one cell.
    field : torch.Tensor
        Vertex-centered values, shape ``(n_points, ...)``.
        Trailing dimensions are preserved in the output.

    Returns
    -------
    torch.Tensor
        Integral value with shape ``field.shape[1:]``.

    Raises
    ------
    ValueError
        If ``field.shape[0]`` does not equal ``mesh.n_points``.
    """
    if not torch.compiler.is_compiling():
        if field.shape[0] != mesh.n_points:
            raise ValueError(
                f"Field leading dimension ({field.shape[0]}) must equal "
                f"n_points ({mesh.n_points})."
            )

    cell_areas = mesh.cell_areas  # (n_cells,)

    ### Gather vertex values for each cell: (n_cells, n_verts_per_cell, ...)
    cell_vertex_values = field[mesh.cells]

    ### Mean over vertices within each cell: (n_cells, ...)
    cell_means = cell_vertex_values.mean(dim=1)

    ### Weight by cell area and sum
    weights = cell_areas.reshape(-1, *([1] * (cell_means.ndim - 1)))
    return torch.nansum(cell_means * weights, dim=0)


def integrate(
    mesh: "Mesh",
    field: str | tuple[str, ...] | Float[torch.Tensor, "n_cells_or_points ..."],
    data_source: Literal["cells", "points"] = "cells",
) -> Float[torch.Tensor, " ..."]:
    r"""Integrate a field over the mesh domain.

    This is the unified entry point for mesh integration.  It dispatches to
    :func:`integrate_cell_data` or :func:`integrate_point_data` based on
    ``data_source``, and resolves ``field`` from a string key or tensor.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh.
    field : str, tuple[str, ...], or torch.Tensor
        Field to integrate.

        - ``str`` or ``tuple``: looked up in ``cell_data`` or ``point_data``
          according to ``data_source``.
        - ``torch.Tensor``: used directly.
    data_source : {"cells", "points"}
        Whether ``field`` is cell-centered (P0) or vertex-centered (P1).

    Returns
    -------
    torch.Tensor
        Integral value.  Shape matches the trailing dimensions of the field
        (scalar field -> 0-d tensor, vector field -> 1-d tensor, etc.).

    Raises
    ------
    KeyError
        If ``field`` is a string key not present in the specified data source.
    ValueError
        If the mesh has no cells, or if a raw tensor has the wrong leading
        dimension for the specified ``data_source``.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh import Mesh
    >>> pts = torch.tensor([[0., 0.], [1., 0.], [0.5, 1.]])
    >>> cells = torch.tensor([[0, 1, 2]])
    >>> mesh = Mesh(points=pts, cells=cells)
    >>> mesh.cell_data["p"] = torch.tensor([3.0])
    >>> mesh.integrate("p")  # integrate cell-centered pressure
    tensor(1.5000)
    >>> mesh.point_data["T"] = torch.tensor([1.0, 2.0, 3.0])
    >>> mesh.integrate("T", data_source="points")  # P1 integral
    tensor(1.)
    """
    if not torch.compiler.is_compiling():
        if mesh.n_cells == 0:
            raise ValueError(
                "Cannot integrate over a mesh with no cells. "
                "Integration requires simplicial connectivity."
            )

    resolved = _resolve_field(mesh, field, data_source)

    match data_source:
        case "cells":
            return integrate_cell_data(mesh, resolved)
        case "points":
            return integrate_point_data(mesh, resolved)
        case _:
            raise ValueError(f"Invalid {data_source=!r}. Must be 'cells' or 'points'.")


def integrate_flux(
    mesh: "Mesh",
    field: str
    | tuple[str, ...]
    | Float[torch.Tensor, "n_cells_or_points n_spatial_dims"],
    data_source: Literal["cells", "points"] = "cells",
) -> Float[torch.Tensor, ""]:
    r"""Compute the surface flux integral for codimension-1 meshes.

    Computes the oriented flux of a vector field through the mesh surface:

    .. math::
        \int_\Gamma \mathbf{F} \cdot \mathbf{n}\,d\Gamma

    This is only defined for codimension-1 meshes (surfaces in 3D, curves
    in 2D) where unique cell normals exist.

    For cell data, the flux is:

    .. math::
        \int_\Gamma \mathbf{F} \cdot \mathbf{n}\,d\Gamma
        = \sum_c (\mathbf{F}_c \cdot \mathbf{n}_c)\,|\sigma_c|

    For point data, the P1 vertex-averaged field is dotted with the cell
    normal (which is constant per cell):

    .. math::
        \int_\Gamma \mathbf{F} \cdot \mathbf{n}\,d\Gamma
        = \sum_c \Bigl(\frac{1}{n_v}\sum_{v \in c} \mathbf{F}(v)\Bigr)
          \cdot \mathbf{n}_c\,|\sigma_c|

    Parameters
    ----------
    mesh : Mesh
        Codimension-1 simplicial mesh (i.e. ``n_manifold_dims ==
        n_spatial_dims - 1``).
    field : str, tuple[str, ...], or torch.Tensor
        Vector field to integrate.  Must have last dimension equal to
        ``n_spatial_dims``.
    data_source : {"cells", "points"}
        Whether ``field`` is cell-centered or vertex-centered.

    Returns
    -------
    torch.Tensor
        Scalar flux value (0-d tensor).

    Raises
    ------
    KeyError
        If ``field`` is a string key not present in the specified data source.
    ValueError
        If the mesh is not codimension-1, if the field leading dimension
        does not match the expected entity count, or if the field does
        not have the correct trailing dimension.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh import Mesh
    >>> # Unit square boundary in 2D (4 edges forming a closed loop)
    >>> pts = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    >>> cells = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
    >>> mesh = Mesh(points=pts, cells=cells)
    >>> # Constant outward velocity field - flux through closed boundary
    >>> mesh.cell_data["v"] = torch.zeros(4, 2)
    >>> mesh.integrate_flux("v")
    tensor(0.)
    """
    if not torch.compiler.is_compiling():
        if mesh.codimension != 1:
            raise ValueError(
                f"integrate_flux requires a codimension-1 mesh "
                f"(n_manifold_dims == n_spatial_dims - 1), but got "
                f"{mesh.n_manifold_dims=}, {mesh.n_spatial_dims=} "
                f"(codimension={mesh.codimension})."
            )

    resolved = _resolve_field(mesh, field, data_source)

    if not torch.compiler.is_compiling():
        expected_leading = mesh.n_cells if data_source == "cells" else mesh.n_points
        if resolved.shape[0] != expected_leading:
            entity = "n_cells" if data_source == "cells" else "n_points"
            raise ValueError(
                f"Field leading dimension ({resolved.shape[0]}) must equal "
                f"{entity} ({expected_leading})."
            )
        if resolved.shape[-1] != mesh.n_spatial_dims:
            raise ValueError(
                f"Field last dimension ({resolved.shape[-1]}) must match "
                f"n_spatial_dims ({mesh.n_spatial_dims}) for flux integration."
            )

    cell_normals = mesh.cell_normals  # (n_cells, n_spatial_dims)
    cell_areas = mesh.cell_areas  # (n_cells,)

    ### Resolve per-cell vector field
    match data_source:
        case "cells":
            cell_field = resolved
        case "points":
            cell_field = resolved[mesh.cells].mean(dim=1)  # P1 average
        case _:
            raise ValueError(f"Invalid {data_source=!r}. Must be 'cells' or 'points'.")

    f_dot_n = (cell_field * cell_normals).sum(dim=-1)  # (n_cells,)
    return torch.nansum(f_dot_n * cell_areas, dim=0)
