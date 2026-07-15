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

**Measure weights.**  All integrators use the effective cell measure
``cell_areas * measure_weights`` rather than the bare geometric areas (see
:mod:`physicsnemo.mesh.calculus.measure`); e.g. for cell-subsampled meshes
carrying Horvitz-Thompson weights, the integral is an unbiased estimate of
the corresponding full-mesh integral.  For meshes without recorded weights
this reduces exactly to the geometric measure.
"""

import math
import warnings
from typing import TYPE_CHECKING, Literal

import torch
from jaxtyping import Float

from physicsnemo.core.warnings import LegacyFeatureWarning
from physicsnemo.mesh.calculus.measure import cell_measures

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


NanPolicy = Literal["omit", "propagate"]


def _sum_with_nan_policy(
    values: torch.Tensor,
    *,
    dim: int,
    nan_policy: NanPolicy,
) -> torch.Tensor:
    """Reduce ``values`` according to the public integration NaN policy."""
    match nan_policy:
        case "omit":
            return torch.nansum(values, dim=dim)
        case "propagate":
            return torch.sum(values, dim=dim)
        case _:
            raise ValueError(f"Invalid {nan_policy=}. Must be 'omit' or 'propagate'.")


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


def _integrate_cell_data(
    mesh: "Mesh",
    field: Float[torch.Tensor, "n_cells ..."],
    *,
    nan_policy: NanPolicy = "omit",
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
    nan_policy : {"omit", "propagate"}, default "omit"
        ``"omit"`` treats NaN entries as zero contributions, preserving the
        historical behavior for deliberately masked CFD data. ``"propagate"``
        uses an ordinary sum so any NaN contribution remains visible.

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

    measures = cell_measures(mesh)  # (n_cells,)

    ### Reshape for broadcasting with arbitrary trailing dims
    weights = measures.reshape(-1, *([1] * (field.ndim - 1)))

    return _sum_with_nan_policy(
        field * weights,
        dim=0,
        nan_policy=nan_policy,
    )


def _integrate_point_data(
    mesh: "Mesh",
    field: Float[torch.Tensor, "n_points ..."],
    *,
    nan_policy: NanPolicy = "omit",
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
    nan_policy : {"omit", "propagate"}, default "omit"
        ``"omit"`` excludes cells whose P1 cell average is NaN, preserving
        the historical behavior for deliberately masked data.
        ``"propagate"`` uses an ordinary sum so affected output components
        remain NaN.

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

    measures = cell_measures(mesh)  # (n_cells,)

    ### Gather vertex values for each cell: (n_cells, n_verts_per_cell, ...)
    cell_vertex_values = field[mesh.cells]

    ### Mean over vertices within each cell: (n_cells, ...)
    cell_means = cell_vertex_values.mean(dim=1)

    ### Weight by effective cell measure and sum
    weights = measures.reshape(-1, *([1] * (cell_means.ndim - 1)))
    return _sum_with_nan_policy(
        cell_means * weights,
        dim=0,
        nan_policy=nan_policy,
    )


def integrate(
    mesh: "Mesh",
    field: str | tuple[str, ...] | Float[torch.Tensor, "n_cells_or_points ..."],
    data_source: Literal["cells", "points"] = "cells",
    *,
    nan_policy: NanPolicy = "omit",
) -> Float[torch.Tensor, " ..."]:
    r"""Integrate a field over the mesh domain.

    This is the public entry point for ordinary mesh-field integration. It
    selects P0 cell quadrature or P1 point quadrature from ``data_source`` and
    resolves ``field`` from a string key or tensor.

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
    nan_policy : {"omit", "propagate"}, default "omit"
        NaN reduction behavior. ``"omit"`` preserves the historical
        masked-data behavior; ``"propagate"`` is appropriate when NaNs
        should remain visible, such as inside neural-operator reductions.

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
            return _integrate_cell_data(mesh, resolved, nan_policy=nan_policy)
        case "points":
            return _integrate_point_data(mesh, resolved, nan_policy=nan_policy)
        case _:
            raise ValueError(f"Invalid {data_source=!r}. Must be 'cells' or 'points'.")


def integrate_cell_data(
    mesh: "Mesh",
    field: Float[torch.Tensor, "n_cells ..."],
    *,
    nan_policy: NanPolicy = "omit",
) -> Float[torch.Tensor, " ..."]:
    r"""Deprecated compatibility wrapper for cell-centered integration.

    Use :func:`integrate` with ``data_source="cells"`` instead.
    """
    warnings.warn(
        "`integrate_cell_data` is deprecated and will be removed in a future "
        "release. Use `integrate(mesh, field, data_source='cells')` instead.",
        LegacyFeatureWarning,
        stacklevel=2,
    )
    return integrate(
        mesh,
        field,
        data_source="cells",
        nan_policy=nan_policy,
    )


def integrate_point_data(
    mesh: "Mesh",
    field: Float[torch.Tensor, "n_points ..."],
    *,
    nan_policy: NanPolicy = "omit",
) -> Float[torch.Tensor, " ..."]:
    r"""Deprecated compatibility wrapper for point-centered integration.

    Use :func:`integrate` with ``data_source="points"`` instead.
    """
    warnings.warn(
        "`integrate_point_data` is deprecated and will be removed in a future "
        "release. Use `integrate(mesh, field, data_source='points')` instead.",
        LegacyFeatureWarning,
        stacklevel=2,
    )
    return integrate(
        mesh,
        field,
        data_source="points",
        nan_policy=nan_policy,
    )


def _integrate_weighted_moment(
    left: torch.Tensor,
    right: torch.Tensor,
    weights: torch.Tensor,
    *,
    aligned_dims: int,
    accumulation_dtype: torch.dtype | None,
    nan_policy: NanPolicy,
) -> torch.Tensor:
    r"""Core weighted grouped moment used by Mesh and streamed operators.

    Computes :math:`\sum_i w_i \, a_i \otimes b_i` over the leading entity
    axis, where an optional block of aligned (group) dimensions immediately
    after the entity axis appears once in the output instead of
    participating in the outer product.

    Parameters
    ----------
    left, right : torch.Tensor
        Field values shaped ``(n_entities, *aligned_shape, *event_shape)``.
        The leading axis indexes integration entities (e.g. cells).  The
        ``aligned_dims`` axes after it must match between the two tensors;
        the remaining trailing (event) axes may differ.
    weights : torch.Tensor
        Quadrature weights shaped ``(n_entities,)`` (e.g. cell measures).
        Callers integrating over a possibly-subsampled mesh should pass the
        effective measure ``cell_areas * measure_weights`` (what
        :func:`integrate_moment` does via
        :func:`physicsnemo.mesh.calculus.measure.cell_measures`), not
        the bare geometric areas.
    aligned_dims : int
        Number of dimensions immediately after the entity axis treated as
        aligned batch/group axes shared by ``left`` and ``right``.
    accumulation_dtype : torch.dtype or None
        Minimum dtype for the weighted matrix product.  The compute dtype
        is the promotion of ``left``, ``right``, ``weights``, and this
        dtype; ``None`` applies ordinary input promotion with no
        additional precision floor.
    nan_policy : {"omit", "propagate"}
        ``"omit"`` zeroes NaN entries in ``left`` and ``right`` before the
        product; ``"propagate"`` leaves them untouched.

    Returns
    -------
    torch.Tensor
        Moment with shape ``aligned_shape + left_event_shape +
        right_event_shape`` in the promoted compute dtype.
    """
    if not torch.compiler.is_compiling():
        if left.ndim < 1 or right.ndim < 1 or weights.ndim != 1:
            raise ValueError("left, right, and weights must have a leading entity axis")
        if left.shape[0] != right.shape[0] or left.shape[0] != weights.shape[0]:
            raise ValueError("left, right, and weights must have equal entity counts")
        if left.device != right.device or left.device != weights.device:
            raise ValueError("left, right, and weights must share a device")
        if accumulation_dtype is not None and not (
            accumulation_dtype.is_floating_point or accumulation_dtype.is_complex
        ):
            raise TypeError(
                "accumulation_dtype must be a floating-point or complex dtype, "
                f"got {accumulation_dtype}."
            )
        if isinstance(aligned_dims, bool) or not isinstance(aligned_dims, int):
            raise TypeError(f"aligned_dims must be an integer, got {aligned_dims!r}.")
        max_aligned = min(left.ndim, right.ndim) - 1
        if aligned_dims < 0 or aligned_dims > max_aligned:
            raise ValueError(
                f"aligned_dims must be between 0 and {max_aligned}, got {aligned_dims}."
            )
        left_aligned = left.shape[1 : 1 + aligned_dims]
        right_aligned = right.shape[1 : 1 + aligned_dims]
        if left_aligned != right_aligned:
            raise ValueError(
                "Aligned field dimensions must match, got "
                f"{tuple(left_aligned)} and {tuple(right_aligned)}."
            )
    if nan_policy not in ("omit", "propagate"):
        raise ValueError(f"Invalid {nan_policy=}. Must be 'omit' or 'propagate'.")

    compute_dtype = torch.promote_types(left.dtype, right.dtype)
    compute_dtype = torch.promote_types(compute_dtype, weights.dtype)
    if accumulation_dtype is not None:
        compute_dtype = torch.promote_types(compute_dtype, accumulation_dtype)

    n_entities = left.shape[0]
    aligned_shape = left.shape[1 : 1 + aligned_dims]
    left_event_shape = left.shape[1 + aligned_dims :]
    right_event_shape = right.shape[1 + aligned_dims :]
    n_groups = math.prod(aligned_shape)
    left_event_size = math.prod(left_event_shape)
    right_event_size = math.prod(right_event_shape)
    left_flat = left.reshape(n_entities, n_groups, left_event_size).to(
        dtype=compute_dtype
    )
    right_flat = right.reshape(n_entities, n_groups, right_event_size).to(
        dtype=compute_dtype
    )
    weights_flat = weights.reshape(1, n_entities, 1).to(dtype=compute_dtype)

    if nan_policy == "omit":
        if left_flat.is_floating_point() or left_flat.is_complex():
            left_flat = torch.where(
                torch.isnan(left_flat), torch.zeros_like(left_flat), left_flat
            )
        if right_flat.is_floating_point() or right_flat.is_complex():
            right_flat = torch.where(
                torch.isnan(right_flat), torch.zeros_like(right_flat), right_flat
            )

    # Explicitly disable autocast so ``accumulation_dtype`` remains a real
    # accumulation guarantee rather than merely an input cast before an
    # autocast-lowered matrix multiplication.
    with torch.autocast(device_type=left_flat.device.type, enabled=False):
        result_flat = torch.bmm(
            left_flat.permute(1, 2, 0),
            right_flat.permute(1, 0, 2) * weights_flat,
        )

    result_shape = aligned_shape + left_event_shape + right_event_shape
    return result_flat.reshape(result_shape)


def integrate_moment(
    mesh: "Mesh",
    left: str | tuple[str, ...] | Float[torch.Tensor, "n_cells ..."],
    right: str | tuple[str, ...] | Float[torch.Tensor, "n_cells ..."],
    *,
    aligned_dims: int = 0,
    accumulation_dtype: torch.dtype | None = torch.float32,
    nan_policy: NanPolicy = "omit",
) -> torch.Tensor:
    r"""Integrate the outer product of two cell-centered fields.

    Computes the P0 quadrature moment

    .. math::
        M = \sum_c |\sigma_c|\, a_c \otimes b_c,

    where ``a`` is ``left``, ``b`` is ``right``, and :math:`|\sigma_c|` is
    the cell's effective measure (its geometric area times any
    recorded measure weight; see :mod:`physicsnemo.mesh.calculus.measure`). By
    default the result has
    shape ``left.shape[1:] + right.shape[1:]``. ``aligned_dims`` may designate
    a common leading subset of the trailing dimensions as independent groups;
    those axes appear only once in the output rather than participating in the
    outer product. The implementation evaluates a batched weighted matrix
    product and never materializes the per-cell outer product.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh with at least one cell.
    left, right : str, tuple[str, ...], or torch.Tensor
        Cell-centered fields. String and tuple keys are resolved from
        ``mesh.cell_data``. Their leading dimensions must equal
        ``mesh.n_cells``; arbitrary trailing dimensions are supported.
    aligned_dims : int, default=0
        Number of leading trailing dimensions shared by ``left`` and ``right``
        and treated as aligned batch/group axes. For example, inputs shaped
        ``(N, H, A)`` and ``(N, H, B)`` with ``aligned_dims=1`` produce
        ``(H, A, B)`` instead of ``(H, A, H, B)``. The aligned shapes must
        match exactly.
    accumulation_dtype : torch.dtype or None, default torch.float32
        Minimum dtype used by the weighted matrix product. The actual compute
        dtype is the promotion of both inputs, the cell measures, and
        this dtype, so the default accumulates reduced-precision inputs in at least FP32
        without downcasting FP64 inputs. Pass ``None`` to use ordinary input
        promotion with no additional precision floor, or ``torch.float64`` to
        request at least FP64 accumulation.
    nan_policy : {"omit", "propagate"}, default "omit"
        ``"omit"`` replaces NaN field contributions with zero before the
        matrix product. ``"propagate"`` leaves them untouched.

    Returns
    -------
    torch.Tensor
        Weighted outer-product moment with shape ``aligned_shape +
        left_event_shape + right_event_shape`` and the accumulation dtype.

    Raises
    ------
    KeyError
        If a named field is absent from ``mesh.cell_data``.
    TypeError
        If ``aligned_dims`` is not an integer or ``accumulation_dtype`` is not
        floating-point or complex.
    ValueError
        If the mesh is empty, a leading dimension is wrong, aligned dimensions
        are invalid, the fields are on different devices, or ``nan_policy`` is
        invalid.

    Notes
    -----
    ``nan_policy="omit"`` is intended for finite data containing NaN masks.
    As with any matrix product, indeterminate expressions involving infinities
    (for example ``0 * inf``) are not treated as missing data.
    """
    if not torch.compiler.is_compiling() and mesh.n_cells == 0:
        raise ValueError(
            "Cannot integrate a moment over a mesh with no cells. "
            "Integration requires simplicial connectivity."
        )

    left_tensor = _resolve_field(mesh, left, "cells")
    right_tensor = _resolve_field(mesh, right, "cells")

    ### Only mesh-coupling checks live here; everything expressible on the
    ### bare tensors (aligned dims, dtypes, device agreement) is validated
    ### once in `_integrate_weighted_moment`.
    if not torch.compiler.is_compiling():
        for name, tensor in (("left", left_tensor), ("right", right_tensor)):
            if tensor.ndim < 1 or tensor.shape[0] != mesh.n_cells:
                leading = tensor.shape[0] if tensor.ndim > 0 else None
                raise ValueError(
                    f"{name} field leading dimension ({leading}) must equal "
                    f"n_cells ({mesh.n_cells})."
                )
            if tensor.device != mesh.points.device:
                raise ValueError(
                    f"{name} field and mesh must be on the same device, got "
                    f"{tensor.device} and {mesh.points.device}."
                )

    return _integrate_weighted_moment(
        left_tensor,
        right_tensor,
        cell_measures(mesh),
        aligned_dims=aligned_dims,
        accumulation_dtype=accumulation_dtype,
        nan_policy=nan_policy,
    )


def integrate_flux(
    mesh: "Mesh",
    field: str
    | tuple[str, ...]
    | Float[torch.Tensor, "n_cells_or_points n_spatial_dims"],
    data_source: Literal["cells", "points"] = "cells",
    *,
    nan_policy: NanPolicy = "omit",
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
    nan_policy : {"omit", "propagate"}, default "omit"
        ``"omit"`` excludes cells whose normal flux is NaN. ``"propagate"``
        uses an ordinary sum so any NaN normal-flux contribution remains
        visible.

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
    measures = cell_measures(mesh)  # (n_cells,)

    ### Resolve per-cell vector field
    match data_source:
        case "cells":
            cell_field = resolved
        case "points":
            cell_field = resolved[mesh.cells].mean(dim=1)  # P1 average
        case _:
            raise ValueError(f"Invalid {data_source=!r}. Must be 'cells' or 'points'.")

    f_dot_n = (cell_field * cell_normals).sum(dim=-1)  # (n_cells,)
    return _sum_with_nan_policy(
        f_dot_n * measures,
        dim=0,
        nan_policy=nan_policy,
    )
