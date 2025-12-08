"""Geometric transformations for simplicial meshes.

This module implements linear and affine transformations with intelligent
cache handling. By default, all caches are invalidated; transformations
explicitly opt-in to preserve/transform specific cache fields.

Cached fields handled:
- areas: point_data and cell_data
- normals: point_data and cell_data
- centroids: cell_data only
"""

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from physicsnemo.mesh.utilities import get_cached, set_cached

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


### Cache Handling ###


def _strip_all_caches(mesh: "Mesh") -> tuple[TensorDict, TensorDict, TensorDict]:
    """Strip _cache from all data containers. Safe default for transformations.

    Returns:
        Tuple of (point_data, cell_data, global_data) with _cache excluded from each.
    """
    return (
        mesh.point_data.exclude("_cache"),
        mesh.cell_data.exclude("_cache"),
        mesh.global_data.exclude("_cache"),
    )


### User Data Transformation ###


def _transform_tensordict(
    data: TensorDict,
    matrix: torch.Tensor,
    n_spatial_dims: int,
    field_type: str,
    has_batch_dim: bool,
    batch_size: torch.Size,
) -> TensorDict:
    """Transform all vector/tensor fields in a TensorDict.

    Args:
        data: TensorDict with cache already stripped
        matrix: Transformation matrix
        n_spatial_dims: Expected spatial dimensionality
        field_type: Description for error messages
        has_batch_dim: Whether tensors have a leading batch dimension
        batch_size: Batch size for the returned TensorDict

    Returns:
        New TensorDict with transformed fields
    """

    def transform_data_field(
        key: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Transform a single vector or tensor field by a linear transformation matrix.

        Args:
            key: Field name (for error messages)
            value: Field tensor. Shape depends on has_batch_dim:
                - has_batch_dim=True: (batch, ...) where batch is n_points or n_cells
                - has_batch_dim=False: (...) for global_data
                The remaining dimensions define the field type:
                - () = scalar (unchanged)
                - (n_spatial_dims,) = vector
                - (n_spatial_dims, n_spatial_dims) = rank-2 tensor
                - (n_spatial_dims, ..., n_spatial_dims) = higher-rank tensor
            matrix: Transformation matrix, shape (new_n_spatial_dims, n_spatial_dims)
            n_spatial_dims: Expected spatial dimensionality of the field
            field_type: Description for error messages (e.g., "point_data", "global_data")
            has_batch_dim: Whether the tensor has a leading batch dimension

        Returns:
            Transformed field tensor

        Raises:
            ValueError: If field shape is incompatible with transformation
        """
        shape = value.shape[1:] if has_batch_dim else value.shape

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

    transformed = data.exclude("_cache").named_apply(
        transform_data_field, batch_size=batch_size
    )
    data.update(transformed)
    return data


### Rotation Matrix Construction ###


def _build_3d_rotation(
    u: torch.Tensor, angle: torch.Tensor | float, device
) -> torch.Tensor:
    """Build 3D rotation matrix using Rodrigues' formula.

    Implements: R = cos(θ) I + sin(θ) [u]_× + (1 - cos(θ)) (u ⊗ u)

    where:
    - u is a unit vector (axis of rotation)
    - θ is the rotation angle
    - [u]_× is the skew-symmetric cross product matrix
    - u ⊗ u is the outer product u u^T

    Args:
        u: Unit vector axis of rotation, shape (3,)
        angle: Rotation angle in radians (scalar or 0-d tensor)
        device: Target device for the output matrix

    Returns:
        3×3 rotation matrix
    """
    angle = torch.as_tensor(angle, device=device)
    c, s = torch.cos(angle), torch.sin(angle)

    ### Skew-symmetric cross-product matrix [u]_×
    ux, uy, uz = u
    zero = torch.zeros((), device=device, dtype=u.dtype)
    u_cross = torch.stack(
        [
            torch.stack([zero, -uz, uy]),
            torch.stack([uz, zero, -ux]),
            torch.stack([-uy, ux, zero]),
        ]
    )

    ### Rodrigues' rotation formula
    identity = torch.eye(3, device=device, dtype=u.dtype)
    return c * identity + s * u_cross + (1 - c) * u.outer(u)


def _build_rotation_matrix(
    axis: torch.Tensor | None,
    angle: float,
    n_spatial_dims: int,
    device,
) -> torch.Tensor:
    """Build rotation matrix for arbitrary spatial dimensions.

    Args:
        axis: Rotation axis vector. For 2D, this is ignored. For 3D, must be a
            3D vector (will be normalized automatically).
        angle: Rotation angle in radians
        n_spatial_dims: Spatial dimensionality (2 or 3)
        device: Target device for the output matrix

    Returns:
        Rotation matrix of shape (n_spatial_dims, n_spatial_dims)
    """
    if n_spatial_dims == 2:
        u = torch.tensor([0.0, 0.0, 1.0], device=device)
        R_3d = _build_3d_rotation(u, angle, device)
        return R_3d[:2, :2]

    elif n_spatial_dims == 3:
        if axis is None:
            raise ValueError("axis must be provided for 3D rotation")

        axis = torch.as_tensor(axis, device=device, dtype=torch.float32)
        if axis.shape != (3,):
            raise ValueError(
                f"For 3D rotation, axis must have shape (3,), got {axis.shape}"
            )

        if axis.norm() < 1e-10:
            raise ValueError(f"Axis vector has near-zero length: {axis.norm()=}")
        u = F.normalize(axis, dim=0, eps=0.0)

        return _build_3d_rotation(u, angle, device)

    else:
        raise NotImplementedError(
            f"Axis-angle rotation not supported for {n_spatial_dims}D spaces. "
            f"For dimensions > 3, use transform() with an explicit rotation matrix."
        )


### Public API ###


def transform(
    mesh: "Mesh",
    matrix: torch.Tensor,
    transform_point_data: bool = False,
    transform_cell_data: bool = False,
    transform_global_data: bool = False,
) -> "Mesh":
    """Apply a linear transformation to the mesh.

    Args:
        mesh: Input mesh to transform
        matrix: Transformation matrix, shape (new_n_spatial_dims, n_spatial_dims)
        transform_point_data: If True, transform vector/tensor fields in point_data
        transform_cell_data: If True, transform vector/tensor fields in cell_data
        transform_global_data: If True, transform vector/tensor fields in global_data

    Returns:
        New Mesh with transformed geometry and appropriately updated caches.

    Cache Handling:
        - areas: For square matrices, scaled by |det|^(n_manifold_dims/n_spatial_dims)
        - centroids: Always transformed
        - normals: Invalidated (directions change for non-orthogonal transforms)
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
    new_point_data, new_cell_data, new_global_data = _strip_all_caches(mesh)

    ### Opt-in: areas (only for square matrices)
    if matrix.shape[0] == matrix.shape[1]:
        scale_factor = matrix.det().abs() ** (mesh.n_manifold_dims / mesh.n_spatial_dims)
        if (v := get_cached(mesh.point_data, "areas")) is not None:
            set_cached(new_point_data, "areas", v * scale_factor)
        if (v := get_cached(mesh.cell_data, "areas")) is not None:
            set_cached(new_cell_data, "areas", v * scale_factor)

    ### Opt-in: centroids
    if (v := get_cached(mesh.cell_data, "centroids")) is not None:
        set_cached(new_cell_data, "centroids", v @ matrix.T)

    ### Transform user data if requested
    if transform_point_data:
        _transform_tensordict(
            new_point_data,
            matrix,
            mesh.n_spatial_dims,
            "point_data",
            has_batch_dim=True,
            batch_size=torch.Size([mesh.n_points]),
        )
    if transform_cell_data:
        _transform_tensordict(
            new_cell_data,
            matrix,
            mesh.n_spatial_dims,
            "cell_data",
            has_batch_dim=True,
            batch_size=torch.Size([mesh.n_cells]),
        )
    if transform_global_data:
        _transform_tensordict(
            new_global_data,
            matrix,
            mesh.n_spatial_dims,
            "global_data",
            has_batch_dim=False,
            batch_size=torch.Size([]),
        )

    from physicsnemo.mesh.mesh import Mesh

    return Mesh(
        points=new_points,
        cells=mesh.cells,
        point_data=new_point_data,
        cell_data=new_cell_data,
        global_data=new_global_data,
    )


def translate(
    mesh: "Mesh",
    offset: torch.Tensor | list | tuple,
) -> "Mesh":
    """Apply a translation to the mesh.

    Translation only affects point positions and centroids. Vector/tensor fields
    are unchanged by translation (they represent directions, not positions).

    Args:
        mesh: Input mesh to translate
        offset: Translation vector, shape (n_spatial_dims,)

    Returns:
        New Mesh with translated geometry.

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
    new_point_data, new_cell_data, new_global_data = _strip_all_caches(mesh)

    ### Opt-in: areas (unchanged)
    if (v := get_cached(mesh.point_data, "areas")) is not None:
        set_cached(new_point_data, "areas", v)
    if (v := get_cached(mesh.cell_data, "areas")) is not None:
        set_cached(new_cell_data, "areas", v)

    ### Opt-in: normals (unchanged)
    if (v := get_cached(mesh.point_data, "normals")) is not None:
        set_cached(new_point_data, "normals", v)
    if (v := get_cached(mesh.cell_data, "normals")) is not None:
        set_cached(new_cell_data, "normals", v)

    ### Opt-in: centroids (translate)
    if (v := get_cached(mesh.cell_data, "centroids")) is not None:
        set_cached(new_cell_data, "centroids", v + offset)

    from physicsnemo.mesh.mesh import Mesh

    return Mesh(
        points=new_points,
        cells=mesh.cells,
        point_data=new_point_data,
        cell_data=new_cell_data,
        global_data=new_global_data,
    )


def rotate(
    mesh: "Mesh",
    axis: torch.Tensor | list | tuple | None,
    angle: float,
    center: torch.Tensor | list | tuple | None = None,
    transform_point_data: bool = False,
    transform_cell_data: bool = False,
    transform_global_data: bool = False,
) -> "Mesh":
    """Rotate the mesh about an axis by a specified angle.

    Args:
        mesh: Input mesh to rotate
        axis: Rotation axis vector. For 2D meshes, this is ignored.
            For 3D meshes, must be a 3D vector (will be normalized).
        angle: Rotation angle in radians (counterclockwise, right-hand rule)
        center: Center point for rotation. If None, rotates about the origin.
        transform_point_data: If True, rotate vector/tensor fields in point_data
        transform_cell_data: If True, rotate vector/tensor fields in cell_data
        transform_global_data: If True, rotate vector/tensor fields in global_data

    Returns:
        New Mesh with rotated geometry.

    Cache Handling:
        - areas: Unchanged (rotation preserves volumes)
        - centroids: Rotated
        - normals: Rotated
    """
    if axis is not None:
        axis = torch.as_tensor(axis, device=mesh.points.device, dtype=torch.float32)

    rotation_matrix = _build_rotation_matrix(
        axis=axis,
        angle=angle,
        n_spatial_dims=mesh.n_spatial_dims,
        device=mesh.points.device,
    )

    if center is not None:
        center = torch.as_tensor(
            center, device=mesh.points.device, dtype=mesh.points.dtype
        )
        mesh_centered = translate(mesh, -center)
        mesh_rotated = rotate(
            mesh_centered,
            axis,
            angle,
            center=None,
            transform_point_data=transform_point_data,
            transform_cell_data=transform_cell_data,
            transform_global_data=transform_global_data,
        )
        return translate(mesh_rotated, center)

    new_points = mesh.points @ rotation_matrix.T
    new_point_data, new_cell_data, new_global_data = _strip_all_caches(mesh)

    ### Opt-in: areas (unchanged)
    if (v := get_cached(mesh.point_data, "areas")) is not None:
        set_cached(new_point_data, "areas", v)
    if (v := get_cached(mesh.cell_data, "areas")) is not None:
        set_cached(new_cell_data, "areas", v)

    ### Opt-in: normals (rotate)
    if (v := get_cached(mesh.point_data, "normals")) is not None:
        set_cached(new_point_data, "normals", v @ rotation_matrix.T)
    if (v := get_cached(mesh.cell_data, "normals")) is not None:
        set_cached(new_cell_data, "normals", v @ rotation_matrix.T)

    ### Opt-in: centroids (rotate)
    if (v := get_cached(mesh.cell_data, "centroids")) is not None:
        set_cached(new_cell_data, "centroids", v @ rotation_matrix.T)

    ### Transform user data if requested
    if transform_point_data:
        _transform_tensordict(
            new_point_data,
            rotation_matrix,
            mesh.n_spatial_dims,
            "point_data",
            has_batch_dim=True,
            batch_size=torch.Size([mesh.n_points]),
        )
    if transform_cell_data:
        _transform_tensordict(
            new_cell_data,
            rotation_matrix,
            mesh.n_spatial_dims,
            "cell_data",
            has_batch_dim=True,
            batch_size=torch.Size([mesh.n_cells]),
        )
    if transform_global_data:
        _transform_tensordict(
            new_global_data,
            rotation_matrix,
            mesh.n_spatial_dims,
            "global_data",
            has_batch_dim=False,
            batch_size=torch.Size([]),
        )

    from physicsnemo.mesh.mesh import Mesh

    return Mesh(
        points=new_points,
        cells=mesh.cells,
        point_data=new_point_data,
        cell_data=new_cell_data,
        global_data=new_global_data,
    )


def scale(
    mesh: "Mesh",
    factor: float | torch.Tensor | list | tuple,
    center: torch.Tensor | list | tuple | None = None,
    transform_point_data: bool = False,
    transform_cell_data: bool = False,
    transform_global_data: bool = False,
) -> "Mesh":
    """Scale the mesh by specified factor(s).

    Args:
        mesh: Input mesh to scale
        factor: Scale factor(s). Scalar for uniform, vector for non-uniform.
        center: Center point for scaling. If None, scales about the origin.
        transform_point_data: If True, scale vector/tensor fields in point_data
        transform_cell_data: If True, scale vector/tensor fields in cell_data
        transform_global_data: If True, scale vector/tensor fields in global_data

    Returns:
        New Mesh with scaled geometry.

    Cache Handling (uniform scaling):
        - areas: Multiplied by |factor|^n_manifold_dims
        - centroids: Scaled
        - normals: Flipped if factor < 0 and n_manifold_dims is odd

    Cache Handling (non-uniform scaling):
        - areas: Invalidated
        - centroids: Scaled component-wise
        - normals: Invalidated
    """
    ### Parse factor
    if isinstance(factor, (int, float)):
        is_uniform = True
        factor_scalar = float(factor)
        scale_matrix = (
            torch.eye(mesh.n_spatial_dims, device=mesh.points.device) * factor_scalar
        )
    else:
        factor_tensor = torch.as_tensor(
            factor, device=mesh.points.device, dtype=mesh.points.dtype
        )
        if factor_tensor.ndim == 0:
            is_uniform = True
            factor_scalar = factor_tensor.item()
            scale_matrix = (
                torch.eye(mesh.n_spatial_dims, device=mesh.points.device)
                * factor_scalar
            )
        else:
            if not torch.compiler.is_compiling():
                if factor_tensor.shape[-1] != mesh.n_spatial_dims:
                    raise ValueError(
                        f"factor must be scalar or shape ({mesh.n_spatial_dims},), "
                        f"got {factor_tensor.shape}"
                    )
            is_uniform = False
            scale_matrix = torch.diag(factor_tensor)

    if center is not None:
        center = torch.as_tensor(
            center, device=mesh.points.device, dtype=mesh.points.dtype
        )
        mesh_centered = translate(mesh, -center)
        mesh_scaled = scale(
            mesh_centered,
            factor,
            center=None,
            transform_point_data=transform_point_data,
            transform_cell_data=transform_cell_data,
            transform_global_data=transform_global_data,
        )
        return translate(mesh_scaled, center)

    new_points = mesh.points @ scale_matrix.T
    new_point_data, new_cell_data, new_global_data = _strip_all_caches(mesh)

    if is_uniform:
        # Areas: scale by |factor|^n_manifold_dims
        area_scale = abs(factor_scalar) ** mesh.n_manifold_dims
        if (v := get_cached(mesh.point_data, "areas")) is not None:
            set_cached(new_point_data, "areas", v * area_scale)
        if (v := get_cached(mesh.cell_data, "areas")) is not None:
            set_cached(new_cell_data, "areas", v * area_scale)

        # Normals: flip if factor < 0 AND n_manifold_dims is odd
        sign = -1 if (factor_scalar < 0 and mesh.n_manifold_dims % 2 == 1) else 1
        if (v := get_cached(mesh.point_data, "normals")) is not None:
            set_cached(new_point_data, "normals", v * sign)
        if (v := get_cached(mesh.cell_data, "normals")) is not None:
            set_cached(new_cell_data, "normals", v * sign)

        # Centroids: scale
        if (v := get_cached(mesh.cell_data, "centroids")) is not None:
            set_cached(new_cell_data, "centroids", v * factor_scalar)
    else:
        # Non-uniform: only centroids preserved
        if (v := get_cached(mesh.cell_data, "centroids")) is not None:
            set_cached(new_cell_data, "centroids", v @ scale_matrix.T)

    ### Transform user data if requested
    if transform_point_data:
        _transform_tensordict(
            new_point_data,
            scale_matrix,
            mesh.n_spatial_dims,
            "point_data",
            has_batch_dim=True,
            batch_size=torch.Size([mesh.n_points]),
        )
    if transform_cell_data:
        _transform_tensordict(
            new_cell_data,
            scale_matrix,
            mesh.n_spatial_dims,
            "cell_data",
            has_batch_dim=True,
            batch_size=torch.Size([mesh.n_cells]),
        )
    if transform_global_data:
        _transform_tensordict(
            new_global_data,
            scale_matrix,
            mesh.n_spatial_dims,
            "global_data",
            has_batch_dim=False,
            batch_size=torch.Size([]),
        )

    from physicsnemo.mesh.mesh import Mesh

    return Mesh(
        points=new_points,
        cells=mesh.cells,
        point_data=new_point_data,
        cell_data=new_cell_data,
        global_data=new_global_data,
    )
