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

"""Geometric transformations for simplicial meshes.

This module implements linear and affine transformations with intelligent
cache handling. By default, all caches are invalidated; transformations
explicitly opt-in to preserve/transform specific cache fields.

Cached fields handled:
- areas: point_data and cell_data
- normals: point_data and cell_data
- centroids: cell_data only
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float
from tensordict import TensorDict

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
) -> Float[torch.Tensor, "n_spatial_dims n_spatial_dims"]:
    """Build rotation matrix for 2D or 3D.

    Parameters
    ----------
    angle : float or Float[torch.Tensor, ""]
        Rotation angle in radians.
    axis : Float[torch.Tensor, " n_spatial_dims"] or None
        Rotation axis vector. None for 2D, shape :math:`(3,)` for 3D.
    device : device
        Target device for the output matrix.

    Returns
    -------
    Float[torch.Tensor, "n_spatial_dims n_spatial_dims"]
        Rotation matrix: :math:`(2, 2)` if axis is None,
        :math:`(3, 3)` if axis has shape :math:`(3,)`.
    """
    angle = torch.as_tensor(angle, device=device)
    c, s = torch.cos(angle), torch.sin(angle)

    if axis is None:
        ### 2D rotation matrix: [[c, -s], [s, c]]
        return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])

    ### 3D rotation using Rodrigues' formula: R = cI + s[u]_× + (1-c)(u⊗u)
    axis = torch.as_tensor(axis, device=device, dtype=angle.dtype)
    if axis.shape != (3,):
        raise NotImplementedError(
            f"Rotation only supported for 2D (axis=None) or 3D (axis shape (3,)). "
            f"Got axis with shape {axis.shape}."
        )
    if axis.norm() < 1e-10:
        raise ValueError(f"Axis vector has near-zero length: {axis.norm()=}")

    u = F.normalize(axis, dim=0, eps=0.0)
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
        Target device for the output tensor.

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
        resolved = torch.zeros(n_spatial_dims, device=device)
        resolved[idx] = 1.0
        return resolved

    if axis is not None:
        axis = torch.as_tensor(axis, device=device, dtype=torch.float32)

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
) -> Float[torch.Tensor, "n_spatial_dims n_spatial_dims"]:
    r"""Build a rotation matrix from angle and axis.

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

    Returns
    -------
    Float[torch.Tensor, "n_spatial_dims n_spatial_dims"]
        Rotation matrix, shape :math:`(S, S)`.
    """
    resolved = _resolve_rotation_axis(axis, n_spatial_dims, device)
    return _build_rotation_matrix(angle=angle, axis=resolved, device=device).to(
        dtype=dtype
    )


def scale_matrix(
    factor: float | Float[torch.Tensor, " n_spatial_dims"] | Sequence[float],
    n_spatial_dims: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Float[torch.Tensor, "n_spatial_dims n_spatial_dims"]:
    r"""Build a diagonal scale matrix from a factor specification.

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
    factor_t = torch.as_tensor(factor, device=device, dtype=dtype)
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


### Public API ###


def transform(
    mesh: "Mesh",
    matrix: Float[torch.Tensor, "new_n_spatial_dims n_spatial_dims"],
    transform_point_data: bool | TensorDict = False,
    transform_cell_data: bool | TensorDict = False,
    transform_global_data: bool | TensorDict = False,
    assume_invertible: bool | None = None,
) -> "Mesh":
    """Apply a linear transformation to the mesh.

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
        - True: Assume matrix is invertible, propagate caches (compile-safe)
        - False: Assume matrix is singular, skip cache propagation (compile-safe)
        - None: Check determinant at runtime (may cause graph breaks under torch.compile)

    Returns
    -------
    Mesh
        New Mesh with transformed geometry and appropriately updated caches.

    Notes
    -----
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
    device = mesh.points.device
    new_cache = TensorDict(
        {
            "cell": TensorDict({}, batch_size=[mesh.n_cells]),
            "point": TensorDict({}, batch_size=[mesh.n_points]),
            "topology": mesh._cache.get("topology", TensorDict({})),
        },
        device=device,
    )

    ### Opt-in: areas and normals (only for square invertible matrices)
    if matrix.shape[0] == matrix.shape[1]:
        det = matrix.det()

        if assume_invertible is not None:
            is_invertible = assume_invertible
        else:
            is_invertible = det.abs() > 1e-10

        if is_invertible:
            det_sign = det.sign()
            det_abs = det.abs()

            ### Full-dimensional meshes: global area scaling
            if mesh.n_manifold_dims == mesh.n_spatial_dims:
                if (v := mesh._cache.get(("point", "areas"), None)) is not None:
                    new_cache["point", "areas"] = v * det_abs
                if (v := mesh._cache.get(("cell", "areas"), None)) is not None:
                    new_cache["cell", "areas"] = v * det_abs

            ### Codimension-1 manifolds: per-element area scaling via normals
            # Formula: area' = area * |det(M)| * ||M^{-T} n||
            elif mesh.codimension == 1:
                if (v := mesh._cache.get(("point", "normals"), None)) is not None:
                    transformed = torch.linalg.solve(matrix.T, v.T).T
                    norm_scale = transformed.norm(dim=-1)
                    if (areas := mesh._cache.get(("point", "areas"), None)) is not None:
                        new_cache["point", "areas"] = areas * det_abs * norm_scale
                    new_cache["point", "normals"] = det_sign * F.normalize(
                        transformed, dim=-1
                    )

                if (v := mesh._cache.get(("cell", "normals"), None)) is not None:
                    transformed = torch.linalg.solve(matrix.T, v.T).T
                    norm_scale = transformed.norm(dim=-1)
                    if (areas := mesh._cache.get(("cell", "areas"), None)) is not None:
                        new_cache["cell", "areas"] = areas * det_abs * norm_scale
                    new_cache["cell", "normals"] = det_sign * F.normalize(
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

    from physicsnemo.mesh.mesh import Mesh

    return Mesh(
        points=new_points,
        cells=mesh.cells,
        point_data=new_point_data,
        cell_data=new_cell_data,
        global_data=new_global_data,
        _cache=new_cache,
    )


def translate(
    mesh: "Mesh",
    offset: Float[torch.Tensor, " n_spatial_dims"] | Sequence[float],
) -> "Mesh":
    """Apply a translation to the mesh.

    Translation only affects point positions and centroids. Vector/tensor fields
    are unchanged by translation (they represent directions, not positions).

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
    offset = torch.as_tensor(offset, device=mesh.points.device, dtype=mesh.points.dtype)

    if not torch.compiler.is_compiling():
        if offset.shape[-1] != mesh.n_spatial_dims:
            raise ValueError(
                f"offset must have shape ({mesh.n_spatial_dims},), got {offset.shape}"
            )

    new_points = mesh.points + offset
    device = mesh.points.device
    new_cache = TensorDict(
        {
            "cell": TensorDict({}, batch_size=[mesh.n_cells]),
            "point": TensorDict({}, batch_size=[mesh.n_points]),
            "topology": mesh._cache.get("topology", TensorDict({})),
        },
        device=device,
    )

    ### Areas and normals are unchanged by translation
    for category in ("cell", "point"):
        for key in ("areas", "normals"):
            if (v := mesh._cache.get((category, key), None)) is not None:
                new_cache[category, key] = v

    ### Centroids are translated
    if (v := mesh._cache.get(("cell", "centroids"), None)) is not None:
        new_cache["cell", "centroids"] = v + offset

    from physicsnemo.mesh.mesh import Mesh

    return Mesh(
        points=new_points,
        cells=mesh.cells,
        point_data=mesh.point_data,
        cell_data=mesh.cell_data,
        global_data=mesh.global_data,
        _cache=new_cache,
    )


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
) -> "Mesh":
    """Rotate the mesh about an axis by a specified angle.

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
    )

    ### Handle center by translate-rotate-translate
    if center is not None:
        center = torch.as_tensor(
            center, device=mesh.points.device, dtype=mesh.points.dtype
        )
        return translate(
            rotate(
                translate(mesh, -center),
                angle,
                axis,
                center=None,
                transform_point_data=transform_point_data,
                transform_cell_data=transform_cell_data,
                transform_global_data=transform_global_data,
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
        - None: Check determinant at runtime (may cause graph breaks under torch.compile)

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
        center = torch.as_tensor(
            center, device=mesh.points.device, dtype=mesh.points.dtype
        )
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

    return transform(
        mesh,
        matrix=M,
        transform_point_data=transform_point_data,
        transform_cell_data=transform_cell_data,
        transform_global_data=transform_global_data,
        assume_invertible=assume_invertible,
    )
