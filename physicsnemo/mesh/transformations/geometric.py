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

from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from tensordict import TensorDict

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


### User Data Transformation ###


def _transform_tensordict(
    data: TensorDict,
    matrix: torch.Tensor,
    n_spatial_dims: int,
    field_type: str,
) -> TensorDict:
    """Transform all vector/tensor fields in a TensorDict.

    Parameters
    ----------
    data : TensorDict
        TensorDict with cache already stripped.
    matrix : torch.Tensor
        Transformation matrix.
    n_spatial_dims : int
        Expected spatial dimensionality.
    field_type : str
        Description for error messages (e.g., "point_data", "global_data").

    Returns
    -------
    TensorDict
        TensorDict with transformed fields.
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
                f"Set the corresponding transform_*_data=False to skip this field."
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

    transformed = data.named_apply(transform_field, batch_size=batch_size)
    data.update(transformed)
    return data


### Rotation Matrix Construction ###


def _build_rotation_matrix(
    angle: float | torch.Tensor,
    axis: torch.Tensor | None,
    device,
) -> torch.Tensor:
    """Build rotation matrix for 2D or 3D.

    Parameters
    ----------
    angle : float or torch.Tensor
        Rotation angle in radians.
    axis : torch.Tensor or None
        Rotation axis vector. None for 2D, shape (3,) for 3D.
    device : device
        Target device for the output matrix.

    Returns
    -------
    torch.Tensor
        Rotation matrix: 2×2 if axis is None, 3×3 if axis has shape (3,).
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


### Public API ###


def transform(
    mesh: "Mesh",
    matrix: torch.Tensor,
    transform_point_data: bool = False,
    transform_cell_data: bool = False,
    transform_global_data: bool = False,
    assume_invertible: bool | None = None,
) -> "Mesh":
    """Apply a linear transformation to the mesh.

    Parameters
    ----------
    mesh : Mesh
        Input mesh to transform.
    matrix : torch.Tensor
        Transformation matrix, shape (new_n_spatial_dims, n_spatial_dims).
    transform_point_data : bool
        If True, transform vector/tensor fields in point_data.
    transform_cell_data : bool
        If True, transform vector/tensor fields in cell_data.
    transform_global_data : bool
        If True, transform vector/tensor fields in global_data.
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
            - Full-dimensional meshes: scaled by |det|
            - Codimension-1 manifolds: per-element scaling using |det| × ||M^{-T} n||
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
            "cell": TensorDict({}, batch_size=[mesh.n_cells], device=device),
            "point": TensorDict({}, batch_size=[mesh.n_points], device=device),
        },
        batch_size=[],
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
    new_point_data = mesh.point_data
    new_cell_data = mesh.cell_data
    if transform_point_data:
        new_point_data = mesh.point_data.clone()
        _transform_tensordict(new_point_data, matrix, mesh.n_spatial_dims, "point_data")
    if transform_cell_data:
        new_cell_data = mesh.cell_data.clone()
        _transform_tensordict(new_cell_data, matrix, mesh.n_spatial_dims, "cell_data")
    new_global_data = mesh.global_data
    if transform_global_data:
        new_global_data = mesh.global_data.clone()
        _transform_tensordict(
            new_global_data, matrix, mesh.n_spatial_dims, "global_data"
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
    offset: torch.Tensor | list | tuple,
) -> "Mesh":
    """Apply a translation to the mesh.

    Translation only affects point positions and centroids. Vector/tensor fields
    are unchanged by translation (they represent directions, not positions).

    Parameters
    ----------
    mesh : Mesh
        Input mesh to translate.
    offset : torch.Tensor or list or tuple
        Translation vector, shape (n_spatial_dims,).

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
            "cell": TensorDict({}, batch_size=[mesh.n_cells], device=device),
            "point": TensorDict({}, batch_size=[mesh.n_points], device=device),
        },
        batch_size=[],
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
    axis: torch.Tensor | list | tuple | Literal["x", "y", "z"] | None = None,
    center: torch.Tensor | list | tuple | None = None,
    transform_point_data: bool = False,
    transform_cell_data: bool = False,
    transform_global_data: bool = False,
) -> "Mesh":
    """Rotate the mesh about an axis by a specified angle.

    Parameters
    ----------
    mesh : Mesh
        Input mesh to rotate.
    angle : float
        Rotation angle in radians (counterclockwise, right-hand rule).
    axis : torch.Tensor or list or tuple or {"x", "y", "z"} or None
        Rotation axis vector. None for 2D, shape (3,) for 3D.
        String literals "x", "y", "z" are converted to unit vectors
        (1,0,0), (0,1,0), (0,0,1) respectively.
    center : torch.Tensor or list or tuple or None
        Center point for rotation. If None, rotates about the origin.
    transform_point_data : bool
        If True, rotate vector/tensor fields in point_data.
    transform_cell_data : bool
        If True, rotate vector/tensor fields in cell_data.
    transform_global_data : bool
        If True, rotate vector/tensor fields in global_data.

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
    ### Convert string axis to one-hot tensor
    if isinstance(axis, str):
        axis_map = {"x": 0, "y": 1, "z": 2}
        if axis not in axis_map:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
        idx = axis_map[axis]
        if idx >= mesh.n_spatial_dims:
            raise ValueError(
                f"axis={axis!r} is invalid for mesh with "
                f"n_spatial_dims={mesh.n_spatial_dims}"
            )
        axis = torch.zeros(mesh.n_spatial_dims, device=mesh.points.device)
        axis[idx] = 1.0

    if axis is not None:
        axis = torch.as_tensor(axis, device=mesh.points.device, dtype=torch.float32)

    ### Validate axis matches mesh dimensionality
    expected_dims = 2 if axis is None else 3
    if mesh.n_spatial_dims != expected_dims:
        raise ValueError(
            f"axis={'None' if axis is None else 'provided'} implies {expected_dims}D rotation, "
            f"but mesh has n_spatial_dims={mesh.n_spatial_dims}"
        )

    rotation_matrix = _build_rotation_matrix(angle, axis, mesh.points.device)
    rotation_matrix = rotation_matrix.to(dtype=mesh.points.dtype)

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

    ### Apply transformation (handles points, areas, centroids, normals, user data)
    ### For rotation: det=±1, always invertible, so we can skip the runtime check
    return transform(
        mesh,
        rotation_matrix,
        transform_point_data=transform_point_data,
        transform_cell_data=transform_cell_data,
        transform_global_data=transform_global_data,
        assume_invertible=True,
    )


def scale(
    mesh: "Mesh",
    factor: float | torch.Tensor | list | tuple,
    center: torch.Tensor | list | tuple | None = None,
    transform_point_data: bool = False,
    transform_cell_data: bool = False,
    transform_global_data: bool = False,
    assume_invertible: bool | None = None,
) -> "Mesh":
    """Scale the mesh by specified factor(s).

    Parameters
    ----------
    mesh : Mesh
        Input mesh to scale.
    factor : float or torch.Tensor or list or tuple
        Scale factor(s). Scalar for uniform, vector for non-uniform.
    center : torch.Tensor or list or tuple or None
        Center point for scaling. If None, scales about the origin.
    transform_point_data : bool
        If True, scale vector/tensor fields in point_data.
    transform_cell_data : bool
        If True, scale vector/tensor fields in cell_data.
    transform_global_data : bool
        If True, scale vector/tensor fields in global_data.
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
    ### Parse factor and build scale matrix
    factor_tensor = torch.as_tensor(
        factor, device=mesh.points.device, dtype=mesh.points.dtype
    )
    if factor_tensor.ndim == 0:
        factor_tensor = factor_tensor.expand(mesh.n_spatial_dims)
    elif (
        not torch.compiler.is_compiling()
        and factor_tensor.shape[-1] != mesh.n_spatial_dims
    ):
        raise ValueError(
            f"factor must be scalar or shape ({mesh.n_spatial_dims},), "
            f"got {factor_tensor.shape}"
        )

    scale_matrix = torch.diag(factor_tensor)

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

    ### Apply transformation (handles points, areas, centroids, normals, user data)
    return transform(
        mesh,
        scale_matrix,
        transform_point_data=transform_point_data,
        transform_cell_data=transform_cell_data,
        transform_global_data=transform_global_data,
        assume_invertible=assume_invertible,
    )
