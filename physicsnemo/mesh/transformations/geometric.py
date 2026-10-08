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

"""Linear and affine transformations for simplicial meshes.

This module implements geometric point transformations with intelligent cache
handling. Topology caches survive coordinate-only changes; geometry caches are
invalidated unless a transformation explicitly preserves or updates them.

Cached fields handled:
- areas: point and cell cache categories
- normals: point and cell cache categories
- centroids: cell cache category only

"""

import math
import numbers
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float
from tensordict import TensorDict

from physicsnemo.mesh.utilities._device import to_device
from physicsnemo.nn.functional import safe_normalize

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


### User Data Transformation ###


def _transform_tensordict(
    data: TensorDict,
    matrix: Float[torch.Tensor, "new_n_spatial_dims n_spatial_dims"],
    n_spatial_dims: int,
    field_type: str,
    mask: TensorDict | None = None,
) -> TensorDict:
    """Transform vector/tensor fields in a TensorDict.

    When ``mask`` is ``None``, all fields with compatible shapes are
    transformed. When ``mask`` is a ``TensorDict`` of scalar bool leaves,
    only fields whose corresponding mask value is ``True`` are transformed;
    fields absent from the mask are left unchanged.

    Parameters
    ----------
    data : TensorDict
        TensorDict with cache already stripped.
    matrix : Float[torch.Tensor, "new_n_spatial_dims n_spatial_dims"]
        Transformation matrix, shape :math:`(S', S)`.
    n_spatial_dims : int
        Expected spatial dimensionality.
    field_type : str
        Description for error messages (e.g., ``"point_data"``, ``"global_data"``).
    mask : TensorDict or None, optional
        Parallel TensorDict with scalar ``bool`` tensor leaves. When
        provided, only keys whose mask is ``True`` are transformed.
        Keys absent from the mask default to ``False`` (not transformed).

    Returns
    -------
    TensorDict
        TensorDict with transformed fields (modified in place).
    """
    batch_size = data.batch_size
    has_batch_dim = len(batch_size) > 0

    def transform_field(key: str, value: torch.Tensor) -> torch.Tensor:
        """Transform a single vector or tensor field."""
        shape = value.shape[len(batch_size) :]

        ### Scalars are invariant under linear transformations
        if len(shape) == 0:
            return value

        ### Validate spatial dimension compatibility
        if shape[0] != n_spatial_dims:
            raise ValueError(
                f"Cannot transform {field_type} field {key!r} with shape {value.shape}. "
                f"First spatial dimension must be {n_spatial_dims}, but got {shape[0]}. "
                f"Use a dict to select specific fields, e.g. "
                f'transform_{field_type}={{"field_name": True}}.'
            )

        ### Vector field: v' = v @ M^T
        if len(shape) == 1:
            return value @ matrix.T

        ### Rank-2 tensor field: T' = M @ T @ M^T (e.g., stress tensors)
        if shape == (n_spatial_dims, n_spatial_dims):
            if has_batch_dim:
                return torch.einsum("ij,bjk,lk->bil", matrix, value, matrix)
            else:
                return torch.einsum("ij,jk,lk->il", matrix, value, matrix)

        ### Higher-rank tensor field: apply transformation to each spatial index
        if all(s == n_spatial_dims for s in shape):
            result = value
            # Index chars for einsum (skip 'b' for batch and 'z' for contraction)
            chars = "acdefghijklmnopqrstuvwxy"
            batch_prefix = "b" if has_batch_dim else ""

            for dim_idx in range(len(shape)):
                input_indices = "".join(
                    chars[i].upper()
                    if i < dim_idx
                    else "z"
                    if i == dim_idx
                    else chars[i]
                    for i in range(len(shape))
                )
                output_indices = "".join(
                    chars[i].upper() if i <= dim_idx else chars[i]
                    for i in range(len(shape))
                )
                einsum_str = f"{chars[dim_idx].upper()}z,{batch_prefix}{input_indices}->{batch_prefix}{output_indices}"
                result = torch.einsum(einsum_str, matrix, result)

            return result

        raise ValueError(
            f"Cannot transform {field_type} field {key!r} with shape {value.shape}. "
            f"Expected all spatial dimensions to be {n_spatial_dims}, but got {shape}"
        )

    if mask is None:
        transformed = data.named_apply(transform_field, batch_size=batch_size)
    else:

        def selective_transform(
            key: str, value: torch.Tensor, should_transform: torch.Tensor
        ) -> torch.Tensor:
            if not should_transform.item():
                return value
            return transform_field(key, value)

        transformed = data.named_apply(
            selective_transform,
            mask,
            default=torch.tensor(False),
            batch_size=batch_size,
        )

    data.update(transformed)
    return data


### Rotation Matrix Construction ###


def _build_rotation_matrix(
    angle: float | Float[torch.Tensor, ""],
    axis: Float[torch.Tensor, " n_spatial_dims"] | None,
    device: torch.device,
    assume_valid_axis: bool = False,
) -> Float[torch.Tensor, "n_spatial_dims n_spatial_dims"]:
    """Build rotation matrix for 2D or 3D.

    Parameters
    ----------
    angle : float or Float[torch.Tensor, ""]
        Rotation angle in radians.
    axis : Float[torch.Tensor, " n_spatial_dims"] or None
        Rotation axis vector. None for 2D, shape :math:`(3,)` for 3D. A host
        axis must be one built from Python values by
        :func:`_resolve_rotation_axis`.
    device : device
        Target device for the output matrix.
    assume_valid_axis : bool
        Skip the check that ``axis`` has non-zero length. See :func:`rotate`.

    Returns
    -------
    Float[torch.Tensor, "n_spatial_dims n_spatial_dims"]
        Rotation matrix: :math:`(2, 2)` if axis is None,
        :math:`(3, 3)` if axis has shape :math:`(3,)`.
    """
    angle = to_device(angle, device)
    c, s = torch.cos(angle), torch.sin(angle)

    if axis is None:
        ### 2D rotation matrix: [[c, -s], [s, c]]
        return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])

    ### 3D rotation using Rodrigues' formula: R = cI + s[u]_× + (1-c)(u⊗u)
    # An axis given as Python values arrives as a host tensor, so validating it
    # reads no device memory; it moves to the device afterwards.
    axis = axis.to(dtype=angle.dtype)
    if axis.shape != (3,):
        raise NotImplementedError(
            f"Rotation only supported for 2D (axis=None) or 3D (axis shape (3,)). "
            f"Got axis with shape {axis.shape}."
        )
    if not assume_valid_axis and axis.norm() < 1e-10:
        raise ValueError(f"Axis vector has near-zero length: {axis.norm()=}")

    # A host axis is fresh pageable memory, staged by CUDA before the copy returns
    u = F.normalize(axis.to(device, non_blocking=True), dim=0, eps=0.0)
    ux, uy, uz = u
    zero = torch.zeros((), device=device, dtype=u.dtype)

    # Skew-symmetric cross-product matrix [u]_×
    u_cross = torch.stack(
        [
            torch.stack([zero, -uz, uy]),
            torch.stack([uz, zero, -ux]),
            torch.stack([-uy, ux, zero]),
        ]
    )

    identity = torch.eye(3, device=device, dtype=u.dtype)
    return c * identity + s * u_cross + (1 - c) * u.outer(u)


### Axis Resolution ###


def _resolve_rotation_axis(
    axis: Float[torch.Tensor, " n_spatial_dims"]
    | Sequence[float]
    | Literal["x", "y", "z"]
    | None,
    n_spatial_dims: int,
    device: torch.device,
) -> Float[torch.Tensor, " n_spatial_dims"] | None:
    """Normalize an axis specification into a tensor or None.

    Parameters
    ----------
    axis : Float[torch.Tensor, " n_spatial_dims"] or Sequence[float] or {"x", "y", "z"} or None
        Rotation axis. ``None`` for 2D, tensor/sequence/string for 3D.
    n_spatial_dims : int
        Number of spatial dimensions (used for validation).
    device : torch.device
        Target device for a tensor axis. An axis given as a string or as
        Python values is returned on the host, so that it can be validated
        there without a device synchronization.

    Returns
    -------
    Float[torch.Tensor, " n_spatial_dims"] or None
        Normalized axis tensor with shape :math:`(3,)` and dtype
        ``float32``, or ``None`` for 2D rotation.
    """
    if isinstance(axis, str):
        axis_map = {"x": 0, "y": 1, "z": 2}
        if axis not in axis_map:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
        idx = axis_map[axis]
        if idx >= n_spatial_dims:
            raise ValueError(
                f"axis={axis!r} is invalid for mesh with "
                f"n_spatial_dims={n_spatial_dims}"
            )
        return torch.eye(n_spatial_dims, device="cpu")[idx]

    if axis is not None:
        axis = torch.as_tensor(
            axis,
            device=device if isinstance(axis, torch.Tensor) else "cpu",
            dtype=torch.float32,
        )

    expected_dims = 2 if axis is None else 3
    if n_spatial_dims != expected_dims:
        raise ValueError(
            f"axis={'None' if axis is None else 'provided'} implies "
            f"{expected_dims}D rotation, but mesh has "
            f"n_spatial_dims={n_spatial_dims}"
        )
    return axis


### Matrix Construction Helpers ###


def rotation_matrix(
    angle: float,
    axis: Float[torch.Tensor, " n_spatial_dims"]
    | Sequence[float]
    | Literal["x", "y", "z"]
    | None,
    n_spatial_dims: int,
    device: torch.device,
    dtype: torch.dtype,
    assume_valid_axis: bool = False,
) -> Float[torch.Tensor, "n_spatial_dims n_spatial_dims"]:
    """Build a rotation matrix from angle and axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians (counterclockwise, right-hand rule).
    axis : Float[torch.Tensor, " n_spatial_dims"] or Sequence[float] or {"x", "y", "z"} or None
        Rotation axis. ``None`` for 2D, tensor/sequence/string for 3D.
    n_spatial_dims : int
        Number of spatial dimensions.
    device : torch.device
        Target device for the output matrix.
    dtype : torch.dtype
        Target dtype for the output matrix.
    assume_valid_axis : bool
        Skip the check that ``axis`` has non-zero length. See :func:`rotate`.

    Returns
    -------
    Float[torch.Tensor, "n_spatial_dims n_spatial_dims"]
        Rotation matrix, shape :math:`(S, S)`.
    """
    resolved = _resolve_rotation_axis(axis, n_spatial_dims, device)
    return _build_rotation_matrix(
        angle=angle,
        axis=resolved,
        device=device,
        assume_valid_axis=assume_valid_axis,
    ).to(dtype=dtype)


def scale_matrix(
    factor: float | Float[torch.Tensor, " n_spatial_dims"] | Sequence[float],
    n_spatial_dims: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Float[torch.Tensor, "n_spatial_dims n_spatial_dims"]:
    """Build a diagonal scale matrix from a factor specification.

    Parameters
    ----------
    factor : float or Float[torch.Tensor, " n_spatial_dims"] or Sequence[float]
        Scale factor(s). Scalar for uniform, vector for non-uniform.
    n_spatial_dims : int
        Number of spatial dimensions.
    device : torch.device
        Target device for the output matrix.
    dtype : torch.dtype
        Target dtype for the output matrix.

    Returns
    -------
    Float[torch.Tensor, "n_spatial_dims n_spatial_dims"]
        Diagonal scale matrix, shape :math:`(S, S)`.

    Raises
    ------
    ValueError
        If ``factor`` is a vector whose length does not match
        ``n_spatial_dims``.
    """
    factor_t = to_device(factor, device, dtype)
    if factor_t.ndim == 0:
        factor_t = factor_t.expand(n_spatial_dims)
    elif not torch.compiler.is_compiling() and factor_t.shape[-1] != n_spatial_dims:
        raise ValueError(
            f"factor must be scalar or shape ({n_spatial_dims},), got {factor_t.shape}"
        )
    return torch.diag(factor_t)


### Transform Mask Normalization ###


def _normalize_transform_mask(
    spec: bool | dict | TensorDict,
) -> TensorDict | None:
    """Convert a transform spec to a TensorDict mask, or None.

    Parameters
    ----------
    spec : bool or dict or TensorDict
        Transform specification. ``True`` returns ``None`` (transform all).
        A ``dict`` of bools is recursively converted to a ``TensorDict``
        with scalar ``bool`` tensor leaves. A ``TensorDict`` is used
        directly.

    Returns
    -------
    TensorDict or None
        Mask TensorDict with scalar bool leaves, or ``None`` to
        transform all fields.
    """
    if spec is True:
        return None
    if isinstance(spec, TensorDict):
        return spec
    if isinstance(spec, dict):
        return TensorDict(
            {
                k: torch.tensor(v)
                if isinstance(v, bool)
                else _normalize_transform_mask(v)
                for k, v in spec.items()
            },
            batch_size=[],
        )
    raise TypeError(f"Expected bool, dict, or TensorDict, got {type(spec)!r}")


def _maybe_transform_data(
    data: TensorDict,
    spec: bool | TensorDict,
    matrix: torch.Tensor,
    n_spatial_dims: int,
    label: str,
) -> TensorDict:
    """Clone and transform a data TensorDict if spec is not False."""
    if spec is False:
        return data
    cloned = data.clone()
    _transform_tensordict(
        cloned,
        matrix,
        n_spatial_dims,
        label,
        mask=_normalize_transform_mask(spec),
    )
    return cloned


def _is_similarity_transform(matrix: torch.Tensor, atol: float = 1e-6) -> bool:
    r"""Whether ``matrix`` is orthogonal up to a uniform scale (:math:`M^\top M = cI`).

    Such maps -- rotations, reflections, isotropic scales, and their compositions
    -- preserve angles. Angle-based vertex-normal weighting is therefore invariant
    under them, so the inverse-transpose cache propagation of point normals is exact.
    Shears and non-uniform scales fail this test.
    """
    n = matrix.shape[-1]
    gram = matrix.T @ matrix
    scale = gram.diagonal(dim1=-2, dim2=-1).mean()
    identity = torch.eye(n, device=matrix.device, dtype=matrix.dtype)
    return bool(torch.allclose(gram, scale * identity, atol=atol, rtol=1e-5))


def _scale_assumptions(
    factor: float | Float[torch.Tensor, " n_spatial_dims"] | Sequence[float],
    n_spatial_dims: int,
) -> tuple[bool | None, bool | None]:
    """Whether a scale is invertible and a similarity, where knowable on the host.

    :func:`transform` otherwise tests both at runtime, and each test reads a
    device scalar back to the host. For Python factors, this applies the same
    tests to ``diag(factor)`` in Python arithmetic: ``|det| > 1e-10``, and the
    ``M.T @ M == c * I`` test of :func:`_is_similarity_transform`. A scalar
    factor is an isotropic scale, hence a similarity, whatever its value.
    Products use ``*``, which overflows to ``inf`` as tensor arithmetic does,
    where Python's ``**`` raises :class:`OverflowError`.

    Returns
    -------
    tuple[bool or None, bool or None]
        ``(assume_invertible, assume_similarity)`` for :func:`transform`,
        with ``None`` where only the device values can tell.
    """
    if isinstance(factor, torch.Tensor):
        return None, (True if factor.ndim == 0 else None)
    if isinstance(factor, numbers.Real):
        return abs(math.prod([factor] * n_spatial_dims)) > 1e-10, True
    values = [float(f) for f in factor]
    squares = [v * v for v in values]
    mean = sum(squares) / len(squares)
    is_similarity = all(abs(s - mean) <= 1e-6 + 1e-5 * mean for s in squares)
    return abs(math.prod(values)) > 1e-10, is_similarity


### Public API ###


def transform(
    mesh: "Mesh",
    matrix: Float[torch.Tensor, "new_n_spatial_dims n_spatial_dims"],
    transform_point_data: bool | TensorDict = False,
    transform_cell_data: bool | TensorDict = False,
    transform_global_data: bool | TensorDict = False,
    assume_invertible: bool | None = None,
    assume_similarity: bool | None = None,
) -> "Mesh":
    """Apply a linear transformation to the mesh.

    Call it as ``transform(mesh, ...)`` or as ``mesh.transform(...)``. The
    bound method supplies ``mesh`` automatically.

    Parameters
    ----------
    mesh : Mesh
        Input mesh to transform.
    matrix : Float[torch.Tensor, "new_n_spatial_dims n_spatial_dims"]
        Transformation matrix, shape :math:`(S', S)`.
    transform_point_data : bool or TensorDict
        Controls transformation of ``point_data`` fields. ``True``
        transforms all compatible fields; ``False`` transforms none;
        a ``TensorDict`` (or ``dict``) with scalar bool leaves
        selectively transforms only the named fields.
    transform_cell_data : bool or TensorDict
        Same semantics as ``transform_point_data``, for ``cell_data``.
    transform_global_data : bool or TensorDict
        Same semantics as ``transform_point_data``, for ``global_data``.
    assume_invertible : bool or None
        Controls cache propagation for square matrices:

        - ``True``: assume ``matrix`` is invertible and propagate caches
          (compile-safe). This is a promise, not a check. If ``matrix`` is in
          fact singular, the inverse-transpose step silently yields non-finite
          values instead of raising, so the propagated ``normals`` and ``areas``
          caches -- and anything derived from them, such as the sum of
          ``cell_areas`` -- come back as NaN. Use ``False`` or ``None`` unless
          you know the matrix is non-singular.
        - ``False``: assume ``matrix`` is singular and skip cache propagation
          (compile-safe). Caches are dropped and recomputed lazily on demand,
          which is always correct, just slower.
        - ``None`` (default): test ``abs(det(matrix)) > 1e-10`` at runtime and
          take one of the branches above. Safe for singular input, but the test
          reads a device scalar back to the host, which synchronizes on CUDA and
          may cause graph breaks under ``torch.compile``.
    assume_similarity : bool or None
        Whether ``matrix`` is a similarity, orthogonal up to a uniform scale
        (rotations, reflections, isotropic scales, and their compositions).
        Similarities preserve angles, so they exactly carry over the cached
        angle-weighted point normals of 2+ manifolds and the point measures of
        lower-dimensional quadrature:

        - ``True``: assume a similarity (compile-safe). This is a promise, not
          a check: for any other matrix the propagated point normals are
          silently inexact, and point measures scale incorrectly.
        - ``False``: assume not a similarity (compile-safe). The point-normal
          cache is dropped and recomputed lazily on demand.
        - ``None`` (default): test ``M.T @ M == c * I`` at runtime when the
          answer is needed. The test reads a device scalar back to the host,
          which synchronizes on CUDA; under ``torch.compile`` it is skipped and
          the point-normal cache is dropped.

    Returns
    -------
    Mesh
        New Mesh with transformed geometry and appropriately updated caches.

    Notes
    -----
    Explicit effective measures follow geometry independently of the ordinary
    field-transformation flags. Cell measures follow geometric measure ratios.
    Point measures use their represented dimension for similarities and the
    determinant for full-dimensional square maps. Other point-measure maps
    require support geometry and raise ValueError. To retain reference measures
    explicitly, use ``mesh.with_points(new_points, preserve_measures=True)``.

    Cache Handling:

        - areas: For square invertible matrices:

            - Full-dimensional meshes: scaled by ``|det|``
            - Codimension-1 manifolds: per-element scaling using ``|det| * ||M^{-T} n||``
            - Higher codimension: invalidated
        - centroids: Always transformed
        - normals: For square invertible matrices, transformed by inverse-transpose
    """
    if not torch.compiler.is_compiling():
        if matrix.ndim != 2:
            raise ValueError(f"matrix must be 2D, got shape {matrix.shape}")
        if matrix.shape[1] != mesh.n_spatial_dims:
            raise ValueError(
                f"matrix shape[1] must equal mesh.n_spatial_dims.\n"
                f"Got matrix.shape={matrix.shape}, mesh.n_spatial_dims={mesh.n_spatial_dims}"
            )

    new_points = mesh.points @ matrix.T

    ### Start from the cache policy for coordinate replacement: retain topology,
    # invalidate geometry, then opt individual transformed values back in below.
    transformed_mesh = mesh.with_points(new_points, preserve_measures=True)
    new_cache = transformed_mesh._cache

    ### Opt-in: areas and normals (only for square invertible matrices)
    if matrix.shape[0] == matrix.shape[1]:
        det = matrix.det()

        ### The runtime det test syncs (host readback of a cuda tensor).
        if assume_invertible is not None:
            is_invertible = assume_invertible
        else:
            is_invertible = bool(det.abs() > 1e-10)

        if is_invertible:
            det_sign = det.sign()
            det_abs = det.abs()

            ### Full-dimensional meshes: global area scaling
            if mesh.n_manifold_dims == mesh.n_spatial_dims:
                if (v := mesh._cache.get(("cell", "areas"), None)) is not None:
                    new_cache["cell", "areas"] = v * det_abs

            ### Codimension-1 manifolds: per-element area scaling via normals
            # Formula: area' = area * |det(M)| * ||M^{-T} n||
            elif mesh.codimension == 1:
                # Normals map by the inverse transpose: as rows, n' = n @ M^-1. One
                # small inverse and a matmul; on CUDA, solve_ex with millions of
                # right-hand sides is ~200x slower whenever its LU factorization pivots.
                if any(
                    mesh._cache.get((association, "normals"), None) is not None
                    for association in ("cell", "point")
                ):
                    inverse = torch.linalg.inv_ex(matrix, check_errors=False).inverse

                ### Cell (face) normals: the inverse-transpose law is exact per face.
                if (v := mesh._cache.get(("cell", "normals"), None)) is not None:
                    transformed = v @ inverse
                    norm_scale = transformed.norm(dim=-1)
                    if (areas := mesh._cache.get(("cell", "areas"), None)) is not None:
                        new_cache["cell", "areas"] = areas * det_abs * norm_scale
                    new_cache["cell", "normals"] = det_sign * safe_normalize(
                        transformed, dim=-1
                    )

                ### Vertex (point) normals are a *weighted average* of incident
                # cell normals, so the inverse-transpose law applies to the average
                # only when M preserves the averaging weights. Area weighting (used
                # for 1-manifolds) is preserved under any invertible M, but the
                # angle / angle_area weighting used for 2+ manifolds is NOT preserved
                # by anisotropic maps (interior angles change). Only propagate when
                # the weighting is area-based (n_manifold_dims < 2) or M is a
                # similarity; otherwise drop the cache so point_normals recomputes
                # lazily and correctly. (Under torch.compile we conservatively skip
                # the similarity check -- a host sync -- and drop the cache to avoid
                # a graph break.)
                if (v := mesh._cache.get(("point", "normals"), None)) is not None and (
                    mesh.n_manifold_dims < 2
                    or (
                        assume_similarity
                        if assume_similarity is not None
                        else (
                            not torch.compiler.is_compiling()
                            and _is_similarity_transform(matrix)
                        )
                    )
                ):
                    transformed = v @ inverse
                    new_cache["point", "normals"] = det_sign * safe_normalize(
                        transformed, dim=-1
                    )

    ### Opt-in: centroids
    if (v := mesh._cache.get(("cell", "centroids"), None)) is not None:
        new_cache["cell", "centroids"] = v @ matrix.T

    ### Transform user data if requested
    new_point_data = _maybe_transform_data(
        mesh.point_data, transform_point_data, matrix, mesh.n_spatial_dims, "point_data"
    )
    new_cell_data = _maybe_transform_data(
        mesh.cell_data, transform_cell_data, matrix, mesh.n_spatial_dims, "cell_data"
    )
    new_global_data = _maybe_transform_data(
        mesh.global_data,
        transform_global_data,
        matrix,
        mesh.n_spatial_dims,
        "global_data",
    )

    if (
        transform_point_data is not False
        or transform_cell_data is not False
        or transform_global_data is not False
    ):
        transformed_mesh = transformed_mesh.with_data(
            point_data=(new_point_data if transform_point_data is not False else None),
            cell_data=new_cell_data if transform_cell_data is not False else None,
            global_data=(
                new_global_data if transform_global_data is not False else None
            ),
        )

    from physicsnemo.mesh.calculus.measure import (
        _transfer_cell_measures,
        _transform_point_measures,
    )

    _transfer_cell_measures(mesh, transformed_mesh)
    _transform_point_measures(mesh, transformed_mesh, matrix, assume_similarity)
    return transformed_mesh


def translate(
    mesh: "Mesh",
    offset: Float[torch.Tensor, " n_spatial_dims"] | Sequence[float],
) -> "Mesh":
    """Apply a translation to the mesh.

    Translation only affects point positions and centroids. Vector/tensor fields
    are unchanged by translation (they represent directions, not positions).

    Call it as ``translate(mesh, ...)`` or as ``mesh.translate(...)``. The
    bound method supplies ``mesh`` automatically.

    Parameters
    ----------
    mesh : Mesh
        Input mesh to translate.
    offset : Float[torch.Tensor, " n_spatial_dims"] or Sequence[float]
        Translation vector, shape :math:`(S,)`.

    Returns
    -------
    Mesh
        New Mesh with translated geometry.

    Notes
    -----
    Cache Handling:

        - areas: Unchanged
        - centroids: Translated
        - normals: Unchanged
    """
    offset = to_device(offset, mesh.points.device, mesh.points.dtype)

    if not torch.compiler.is_compiling():
        if offset.shape[-1] != mesh.n_spatial_dims:
            raise ValueError(
                f"offset must have shape ({mesh.n_spatial_dims},), got {offset.shape}"
            )

    new_points = mesh.points + offset
    translated_mesh = mesh.with_points(
        new_points,
        preserve_measures=True,
        keep=(
            "topology",
            ("cell", "areas"),
            ("cell", "centroids"),
            ("cell", "normals"),
            ("point", "areas"),
            ("point", "normals"),
        ),
    )

    ### Centroids are translated
    if (v := mesh._cache.get(("cell", "centroids"), None)) is not None:
        translated_mesh._cache["cell", "centroids"] = v + offset

    return translated_mesh


def rotate(
    mesh: "Mesh",
    angle: float,
    axis: Float[torch.Tensor, " n_spatial_dims"]
    | Sequence[float]
    | Literal["x", "y", "z"]
    | None = None,
    center: Float[torch.Tensor, " n_spatial_dims"] | Sequence[float] | None = None,
    transform_point_data: bool | TensorDict = False,
    transform_cell_data: bool | TensorDict = False,
    transform_global_data: bool | TensorDict = False,
    assume_valid_axis: bool = False,
) -> "Mesh":
    """Rotate the mesh about an axis by a specified angle.

    Call it as ``rotate(mesh, ...)`` or as ``mesh.rotate(...)``. The bound
    method supplies ``mesh`` automatically.

    Parameters
    ----------
    mesh : Mesh
        Input mesh to rotate.
    angle : float
        Rotation angle in radians (counterclockwise, right-hand rule).
    axis : Float[torch.Tensor, " n_spatial_dims"] or Sequence[float] or {"x", "y", "z"} or None
        Rotation axis vector. ``None`` for 2D, shape :math:`(3,)` for 3D.
        String literals ``"x"``, ``"y"``, ``"z"`` are converted to unit
        vectors ``(1,0,0)``, ``(0,1,0)``, ``(0,0,1)`` respectively.
    center : Float[torch.Tensor, " n_spatial_dims"] or Sequence[float] or None
        Center point for rotation. If ``None``, rotates about the origin.
    transform_point_data : bool or TensorDict
        Controls transformation of ``point_data`` fields. See
        :func:`transform` for full semantics.
    transform_cell_data : bool or TensorDict
        Same semantics as ``transform_point_data``, for ``cell_data``.
    transform_global_data : bool or TensorDict
        Same semantics as ``transform_point_data``, for ``global_data``.
    assume_valid_axis : bool
        Skip the check that ``axis`` has non-zero length (compile-safe). Only
        an axis given as a device tensor benefits: checking it reads a device
        scalar back to the host, which synchronizes on CUDA, whereas an axis
        given as Python values or a string is checked on the host. This is a
        promise, not a check: a zero-length axis then gives a NaN rotation.

    Returns
    -------
    Mesh
        New Mesh with rotated geometry.

    Notes
    -----
    Cache Handling:

        - areas: Unchanged (rotation preserves volumes)
        - centroids: Rotated
        - normals: Rotated
    """
    R = rotation_matrix(
        angle=angle,
        axis=axis,
        n_spatial_dims=mesh.n_spatial_dims,
        device=mesh.points.device,
        dtype=mesh.points.dtype,
        assume_valid_axis=assume_valid_axis,
    )

    ### Handle center by translate-rotate-translate
    if center is not None:
        center = to_device(center, mesh.points.device, mesh.points.dtype)
        return translate(
            rotate(
                translate(mesh, -center),
                angle,
                axis,
                center=None,
                transform_point_data=transform_point_data,
                transform_cell_data=transform_cell_data,
                transform_global_data=transform_global_data,
                assume_valid_axis=assume_valid_axis,
            ),
            center,
        )

    return transform(
        mesh,
        matrix=R,
        transform_point_data=transform_point_data,
        transform_cell_data=transform_cell_data,
        transform_global_data=transform_global_data,
        assume_invertible=True,
        assume_similarity=True,
    )


def scale(
    mesh: "Mesh",
    factor: float | Float[torch.Tensor, " n_spatial_dims"] | Sequence[float],
    center: Float[torch.Tensor, " n_spatial_dims"] | Sequence[float] | None = None,
    transform_point_data: bool | TensorDict = False,
    transform_cell_data: bool | TensorDict = False,
    transform_global_data: bool | TensorDict = False,
    assume_invertible: bool | None = None,
) -> "Mesh":
    """Scale the mesh by specified factor(s).

    Call it as ``scale(mesh, ...)`` or as ``mesh.scale(...)``. The bound method
    supplies ``mesh`` automatically.

    Parameters
    ----------
    mesh : Mesh
        Input mesh to scale.
    factor : float or Float[torch.Tensor, " n_spatial_dims"] or Sequence[float]
        Scale factor(s). Scalar for uniform, vector for non-uniform.
    center : Float[torch.Tensor, " n_spatial_dims"] or Sequence[float] or None
        Center point for scaling. If ``None``, scales about the origin.
    transform_point_data : bool or TensorDict
        Controls transformation of ``point_data`` fields. See
        :func:`transform` for full semantics.
    transform_cell_data : bool or TensorDict
        Same semantics as ``transform_point_data``, for ``cell_data``.
    transform_global_data : bool or TensorDict
        Same semantics as ``transform_point_data``, for ``global_data``.
    assume_invertible : bool or None
        Controls cache propagation:

        - True: Assume all factors are non-zero, propagate caches (compile-safe)
        - False: Assume some factor is zero, skip cache propagation (compile-safe)
        - None: Check the determinant; on the host for Python factors, else at
          runtime (may cause graph breaks under torch.compile)

    Returns
    -------
    Mesh
        New Mesh with scaled geometry.

    Notes
    -----
    Cache Handling:

        - areas: Scaled correctly. For non-isotropic transforms of codimension-1
                 embedded manifolds, per-element scaling is computed using normals.
        - centroids: Scaled
        - normals: Transformed by inverse-transpose (direction adjusted, magnitude normalized)
    """
    M = scale_matrix(
        factor=factor,
        n_spatial_dims=mesh.n_spatial_dims,
        device=mesh.points.device,
        dtype=mesh.points.dtype,
    )

    ### Handle center by translate-scale-translate
    if center is not None:
        center = to_device(center, mesh.points.device, mesh.points.dtype)
        return translate(
            scale(
                translate(mesh, -center),
                factor,
                center=None,
                transform_point_data=transform_point_data,
                transform_cell_data=transform_cell_data,
                transform_global_data=transform_global_data,
                assume_invertible=assume_invertible,
            ),
            center,
        )

    is_invertible, is_similarity = _scale_assumptions(factor, mesh.n_spatial_dims)
    return transform(
        mesh,
        matrix=M,
        transform_point_data=transform_point_data,
        transform_cell_data=transform_cell_data,
        transform_global_data=transform_global_data,
        assume_invertible=is_invertible
        if assume_invertible is None
        else assume_invertible,
        assume_similarity=is_similarity,
    )
