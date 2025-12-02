"""Geometric transformations for simplicial meshes.

This module implements linear and affine transformations with intelligent
cache handling for geometric properties stored in cell_data[("_cache", "areas")],
cell_data[("_cache", "centroids")], and cell_data[("_cache", "normals")].
"""

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from physicsnemo.mesh.utilities import get_cached, set_cached

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


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
    c = torch.cos(torch.as_tensor(angle, device=device))
    s = torch.sin(torch.as_tensor(angle, device=device))

    ### Identity matrix
    identity = torch.eye(3, device=device, dtype=u.dtype)

    ### Outer product: u ⊗ u = u u^T
    u_outer = u.unsqueeze(1) @ u.unsqueeze(0)

    ### Cross product matrix [u]_×
    # This is a skew-symmetric matrix such that [u]_× v = u × v for any vector v
    u_cross = torch.zeros(3, 3, device=device, dtype=u.dtype)
    u_cross[0, 1] = -u[2]
    u_cross[0, 2] = u[1]
    u_cross[1, 0] = u[2]
    u_cross[1, 2] = -u[0]
    u_cross[2, 0] = -u[1]
    u_cross[2, 1] = u[0]

    ### Rodrigues' rotation formula
    R = c * identity + s * u_cross + (1 - c) * u_outer

    return R


def _build_rotation_matrix(
    axis: torch.Tensor | None,
    angle: float,
    n_spatial_dims: int,
    device,
) -> torch.Tensor:
    """Build rotation matrix for arbitrary spatial dimensions.

    Uses generalized Rodrigues' formula for rotations. For 2D, treats rotation
    as a 3D rotation about the z-axis and extracts the 2×2 submatrix.

    Args:
        axis: Rotation axis vector. For 2D, this is ignored. For 3D, must be a
            3D vector (will be normalized automatically).
        angle: Rotation angle in radians
        n_spatial_dims: Spatial dimensionality (2 or 3)
        device: Target device for the output matrix

    Returns:
        Rotation matrix of shape (n_spatial_dims, n_spatial_dims)

    Raises:
        NotImplementedError: If n_spatial_dims > 3 (axis-angle representation
            is insufficient for higher dimensions)
    """
    if n_spatial_dims == 2:
        ### 2D rotation: treat as 3D rotation about z-axis, extract 2×2 submatrix
        u = torch.tensor([0.0, 0.0, 1.0], device=device)
        R_3d = _build_3d_rotation(u, angle, device)
        return R_3d[:2, :2]

    elif n_spatial_dims == 3:
        ### 3D rotation: normalize axis and apply Rodrigues' formula
        if axis is None:
            raise ValueError("axis must be provided for 3D rotation")

        axis = torch.as_tensor(axis, device=device, dtype=torch.float32)
        if axis.shape != (3,):
            raise ValueError(
                f"For 3D rotation, axis must have shape (3,), got {axis.shape}"
            )

        # Normalize axis to unit length
        axis_norm = torch.norm(axis)
        if axis_norm < 1e-10:
            raise ValueError(f"Axis vector has near-zero length: {axis_norm=}")
        u = axis / axis_norm

        return _build_3d_rotation(u, angle, device)

    else:
        raise NotImplementedError(
            f"Axis-angle rotation not supported for {n_spatial_dims}D spaces. "
            f"For dimensions > 3, use transform() with an explicit rotation matrix instead."
        )


def _transform_higher_order_tensor(
    tensor: torch.Tensor,
    matrix: torch.Tensor,
) -> torch.Tensor:
    """Transform higher-order tensors under a linear transformation.

    For rank-2 tensors (e.g., stress tensors), applies the transformation:
        T' = M @ T @ M^T

    For higher-rank tensors where all non-batch dimensions equal n_spatial_dims,
    applies the transformation to each spatial index using Einstein summation.

    For rank-N tensors, the transformation is:
        T'_{i1,i2,...,iN} = M_{i1,j1} * M_{i2,j2} * ... * M_{iN,jN} * T_{j1,j2,...,jN}

    Args:
        tensor: Tensor to transform, shape (batch, n_spatial_dims, n_spatial_dims, ...)
            where all dimensions after the batch dimension must equal n_spatial_dims
        matrix: Transformation matrix, shape (new_n_spatial_dims, n_spatial_dims)

    Returns:
        Transformed tensor, shape (batch, new_n_spatial_dims, new_n_spatial_dims, ...)
    """
    n_dims = len(tensor.shape) - 1  # Number of spatial dimensions (excluding batch)

    if n_dims == 2:
        ### Rank-2 tensor: T' = M @ T @ M^T
        # Use batched matrix multiplication for efficiency
        # tensor shape: (batch, n, n)
        # matrix shape: (m, n)
        # Result: (batch, m, m)
        result = torch.einsum("ij,bjk,lk->bil", matrix, tensor, matrix)
        return result

    else:
        ### Higher-rank tensor: apply transformation to each spatial index
        # For rank-N tensor with shape (batch, n, n, ..., n),
        # we need to contract the transformation matrix along each spatial dimension

        # Strategy: maintain consistent index labels throughout the loop.
        # We use lowercase letters starting from 'a' for the current tensor indices,
        # and 'z' for the dimension being contracted, transforming to uppercase letter.

        # Start with indices: batch is always 'b', spatial dims are a,c,d,e,f,...
        # (skip 'b' since it's batch, use 'a' and then letters after 'b')
        result = tensor

        # Generate index characters, skipping 'b' (batch) and 'z' (contraction var)
        available_chars = [
            chr(i) for i in range(ord("a"), ord("z") + 1) if chr(i) not in ["b", "z"]
        ]

        for dim_idx in range(n_dims):
            # Build einsum to contract one dimension at a time
            # Current result has indices: b, [transformed dims], [untransformed dims]
            # We'll transform dimension at position dim_idx

            # Build the index string for the current result
            input_indices = []
            output_indices = []

            for i in range(n_dims):
                if i < dim_idx:
                    # Already transformed - use uppercase to track
                    char = available_chars[i].upper()
                    input_indices.append(char)
                    output_indices.append(char)
                elif i == dim_idx:
                    # Currently transforming - contract with 'z', output as uppercase
                    input_indices.append("z")
                    output_indices.append(available_chars[i].upper())
                else:
                    # Not yet transformed - use lowercase
                    char = available_chars[i]
                    input_indices.append(char)
                    output_indices.append(char)

            input_str = "b" + "".join(input_indices)
            output_str = "b" + "".join(output_indices)

            # Matrix contracts index position of original tensor dimension
            # matrix indices: output_dim, input_dim (i.e., uppercase[dim_idx], 'z')
            matrix_indices = f"{available_chars[dim_idx].upper()}z"

            einsum_str = f"{matrix_indices},{input_str}->{output_str}"
            result = torch.einsum(einsum_str, matrix, result)

        return result


def _handle_caches_for_transform(
    mesh: "Mesh",
    matrix: torch.Tensor,
) -> TensorDict:
    """Handle cache updates for general linear transformations.

    Args:
        mesh: Input mesh
        matrix: Transformation matrix, shape (new_n_spatial_dims, n_spatial_dims)

    Returns:
        Updated cell_data TensorDict with transformed/invalidated caches
    """
    # Start with non-cache data
    new_cell_data = mesh.cell_data.exclude("_cache")

    ### Areas: scale by determinant formula (square matrices only)
    if matrix.shape[0] == matrix.shape[1]:  # Square matrix
        cached_areas = get_cached(mesh.cell_data, "areas")
        if cached_areas is not None:
            det = torch.det(matrix)
            # Areas scale by |det|^(n_manifold_dims / n_spatial_dims)
            scale_factor = abs(det) ** (mesh.n_manifold_dims / mesh.n_spatial_dims)
            set_cached(new_cell_data, "areas", cached_areas * scale_factor)
    # else: Non-square matrix: projection changes dimensionality, don't add areas to cache

    ### Centroids: always transform (geometric property)
    cached_centroids = get_cached(mesh.cell_data, "centroids")
    if cached_centroids is not None:
        set_cached(new_cell_data, "centroids", cached_centroids @ matrix.T)

    ### Normals: invalidate for general transforms (not added back to cache)

    return new_cell_data


def _handle_caches_for_rotation(
    mesh: "Mesh",
    rotation_matrix: torch.Tensor,
) -> TensorDict:
    """Handle cache updates for rotations.

    Rotations are orthogonal transformations with det(R) = 1, so areas are preserved.
    Both centroids and normals are rotated.

    Args:
        mesh: Input mesh
        rotation_matrix: Rotation matrix, shape (n_spatial_dims, n_spatial_dims)

    Returns:
        Updated cell_data TensorDict with rotated caches
    """
    new_cell_data = mesh.cell_data.exclude("_cache")

    ### Areas: unchanged (det(R) = 1 for rotations), keep if present
    cached_areas = get_cached(mesh.cell_data, "areas")
    if cached_areas is not None:
        set_cached(new_cell_data, "areas", cached_areas)

    ### Centroids: rotate (geometric property, always transform)
    cached_centroids = get_cached(mesh.cell_data, "centroids")
    if cached_centroids is not None:
        set_cached(new_cell_data, "centroids", cached_centroids @ rotation_matrix.T)

    ### Normals: rotate (geometric property, always transform)
    cached_normals = get_cached(mesh.cell_data, "normals")
    if cached_normals is not None:
        set_cached(new_cell_data, "normals", cached_normals @ rotation_matrix.T)

    return new_cell_data


def _handle_caches_for_uniform_scale(
    mesh: "Mesh",
    factor: float,
) -> TensorDict:
    """Handle cache updates for uniform scaling.

    Args:
        mesh: Input mesh
        factor: Uniform scale factor (can be negative for reflection)

    Returns:
        Updated cell_data TensorDict with scaled caches
    """
    new_cell_data = mesh.cell_data.exclude("_cache")

    ### Areas: scale by |factor|^n_manifold_dims
    cached_areas = get_cached(mesh.cell_data, "areas")
    if cached_areas is not None:
        set_cached(
            new_cell_data, "areas", cached_areas * (abs(factor) ** mesh.n_manifold_dims)
        )

    ### Centroids: scale
    cached_centroids = get_cached(mesh.cell_data, "centroids")
    if cached_centroids is not None:
        set_cached(new_cell_data, "centroids", cached_centroids * factor)

    ### Normals: keep if positive scaling, invalidate if negative (winding order changes)
    if factor >= 0:
        cached_normals = get_cached(mesh.cell_data, "normals")
        if cached_normals is not None:
            set_cached(new_cell_data, "normals", cached_normals)
    # else: negative scaling invalidates normals (not added back to cache)

    return new_cell_data


def transform(
    mesh: "Mesh",
    matrix: torch.Tensor,
    transform_data: bool = False,
) -> "Mesh":
    """Apply a linear transformation to the mesh.

    This is the core transformation primitive that applies an arbitrary linear
    transformation matrix to all mesh points. The matrix can be square (standard
    transformation), tall (projection to higher dimensions), or wide (projection
    to lower dimensions).

    Linear transformations preserve the origin and straight lines, but may change
    angles, lengths, and orientations. Examples include rotations, scaling, shearing,
    and projections.

    For affine transformations (which don't preserve the origin), use translate()
    in combination with this function.

    Args:
        mesh: Input mesh to transform
        matrix: Transformation matrix, shape (new_n_spatial_dims, n_spatial_dims).
            For square matrices, this performs standard linear transformations.
            For non-square matrices, this performs dimensional projections.
        transform_data: If True, also transform vector and tensor fields in
            point_data and cell_data. Scalar fields are always unchanged.
            Geometric caches (centroids, normals) are always transformed
            regardless of this flag.

    Returns:
        New Mesh with transformed geometry and appropriately updated caches.

    Cache Handling:
        - areas: For square matrices, multiplied by |det(matrix)|^(n_manifold_dims/n_spatial_dims).
                 For non-square matrices, invalidated.
        - centroids: Always transformed by the matrix (geometric property).
        - normals: Invalidated (directions change for non-orthogonal transforms).

    Example:
        >>> # 2D shear transformation
        >>> shear = torch.tensor([[1.0, 0.5], [0.0, 1.0]])
        >>> sheared_mesh = transform(mesh, shear)
        >>>
        >>> # 3D to 2D projection onto xy-plane
        >>> proj_xy = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> mesh_2d = transform(mesh_3d, proj_xy)
    """
    ### Validation
    if not torch.compiler.is_compiling():
        if matrix.ndim != 2:
            raise ValueError(f"matrix must be 2D, got shape {matrix.shape}")
        if matrix.shape[1] != mesh.n_spatial_dims:
            raise ValueError(
                f"matrix shape[1] must equal mesh.n_spatial_dims.\n"
                f"Got matrix.shape={matrix.shape}, mesh.n_spatial_dims={mesh.n_spatial_dims}"
            )

    ### Transform points
    new_points = mesh.points @ matrix.T

    ### Handle caches
    new_cell_data = _handle_caches_for_transform(mesh, matrix)

    ### Transform user data if requested
    if transform_data:
        # Transform point_data (excluding cache, which doesn't need transformation)
        def transform_point_data(key, value):
            shape = value.shape[1:]

            # Skip scalars (no spatial structure to transform)
            if len(shape) == 0:
                return value

            # Check if field is transformable
            if shape[0] != mesh.n_spatial_dims:
                raise ValueError(
                    f"Cannot transform data field {key!r} with shape {value.shape}. "
                    f"First dimension after batch must be {mesh.n_spatial_dims}, "
                    f"but got shape[1] = {shape[0]}. "
                    f"Set transform_data=False to skip data transformation."
                )

            if len(shape) == 1:  # Vector
                return value @ matrix.T
            elif all(s == mesh.n_spatial_dims for s in shape):  # Tensor
                return _transform_higher_order_tensor(value, matrix)
            else:
                raise ValueError(
                    f"Cannot transform data field {key!r} with shape {value.shape}. "
                    f"Expected all dimensions after batch to be {mesh.n_spatial_dims}, "
                    f"but got shape[1:] = {shape}"
                )

        new_point_data = mesh.point_data.exclude("_cache").named_apply(
            transform_point_data,
            batch_size=torch.Size([mesh.n_points]),
        )

        # Transform cell_data (cache already handled, only transform user data)
        def transform_cell_data(key, value):
            shape = value.shape[1:]

            # Skip scalars
            if len(shape) == 0:
                return value

            # Check if field is transformable
            if shape[0] != mesh.n_spatial_dims:
                raise ValueError(
                    f"Cannot transform cell_data field {key!r} with shape {value.shape}. "
                    f"First dimension after batch must be {mesh.n_spatial_dims}, "
                    f"but got shape[1] = {shape[0]}. "
                    f"Set transform_data=False to skip data transformation."
                )

            if len(shape) == 1:
                return value @ matrix.T
            elif all(s == mesh.n_spatial_dims for s in shape):
                return _transform_higher_order_tensor(value, matrix)
            else:
                raise ValueError(
                    f"Cannot transform cell_data field {key!r} with shape {value.shape}. "
                    f"Expected all dimensions after batch to be {mesh.n_spatial_dims}, "
                    f"but got shape[1:] = {shape}"
                )

        # Apply transformation and merge with cached data
        transformed_user_data = new_cell_data.exclude("_cache").named_apply(
            transform_cell_data,
            batch_size=torch.Size([mesh.n_cells]),
        )
        # Merge transformed user data with cache handling results
        new_cell_data.update(transformed_user_data)
    else:
        new_point_data = mesh.point_data

    ### Create new mesh
    from physicsnemo.mesh.mesh import Mesh

    return Mesh(
        points=new_points,
        cells=mesh.cells,
        point_data=new_point_data,
        cell_data=new_cell_data,
        global_data=mesh.global_data,
    )


def translate(mesh: "Mesh", offset: torch.Tensor | list | tuple) -> "Mesh":
    """Apply a translation (affine transformation) to the mesh.

    Translation moves all points by a constant offset vector. This is an affine
    transformation (not linear), meaning it doesn't preserve the origin but does
    preserve distances, angles, and orientations.

    Translation only affects point positions and centroids. It does not affect
    vectors (like normals or velocity fields) or areas, because these are
    properties that are independent of absolute position.

    Args:
        mesh: Input mesh to translate
        offset: Translation vector, shape (n_spatial_dims,) or broadcastable.
            Can be a tensor, list, or tuple.

    Returns:
        New Mesh with translated geometry.

    Cache Handling:
        - areas: Unchanged (translation doesn't affect sizes).
        - centroids: Translated by offset.
        - normals: Unchanged (translation doesn't affect directions).

    Note:
        This function does not have a transform_data parameter because translation
        does not affect vectors or tensors - only absolute positions change.

    Example:
        >>> # Translate mesh 2 units in x, 3 units in y
        >>> translated = translate(mesh, [2.0, 3.0])
        >>>
        >>> # Center mesh at origin by translating by negative centroid
        >>> centroid = mesh.cell_centroids.mean(dim=0)
        >>> centered = translate(mesh, -centroid)
    """
    ### Convert offset to tensor
    offset = torch.as_tensor(offset, device=mesh.points.device, dtype=mesh.points.dtype)

    ### Validation
    if not torch.compiler.is_compiling():
        if offset.shape[-1] != mesh.n_spatial_dims:
            raise ValueError(
                f"offset must be broadcastable to shape (..., {mesh.n_spatial_dims}), "
                f"got shape {offset.shape}"
            )

    ### Translate points
    new_points = mesh.points + offset

    ### Handle caches
    new_cell_data = mesh.cell_data.exclude("_cache")

    # Areas: unchanged, keep if present
    cached_areas = get_cached(mesh.cell_data, "areas")
    if cached_areas is not None:
        set_cached(new_cell_data, "areas", cached_areas)

    # Centroids: translate
    cached_centroids = get_cached(mesh.cell_data, "centroids")
    if cached_centroids is not None:
        set_cached(new_cell_data, "centroids", cached_centroids + offset)

    # Normals: unchanged, keep if present
    cached_normals = get_cached(mesh.cell_data, "normals")
    if cached_normals is not None:
        set_cached(new_cell_data, "normals", cached_normals)

    ### Create new mesh
    from physicsnemo.mesh.mesh import Mesh

    return Mesh(
        points=new_points,
        cells=mesh.cells,
        point_data=mesh.point_data,
        cell_data=new_cell_data,
        global_data=mesh.global_data,
    )


def rotate(
    mesh: "Mesh",
    axis: torch.Tensor | list | tuple | None,
    angle: float,
    center: torch.Tensor | list | tuple | None = None,
    transform_data: bool = False,
) -> "Mesh":
    """Rotate the mesh about an axis by a specified angle.

    Applies a rotation using the axis-angle representation. The rotation is
    performed about the specified axis (for 3D) or about the z-axis (for 2D)
    by the given angle in radians.

    Rotations are orthogonal transformations with determinant 1, which means
    they preserve distances, angles, and volumes/areas. Both centroids and
    normals are rotated along with the geometry.

    Args:
        mesh: Input mesh to rotate
        axis: Rotation axis vector. For 2D meshes, this is ignored (rotation
            is about the z-axis). For 3D meshes, must be a 3D vector (will be
            normalized automatically).
        angle: Rotation angle in radians. Positive angles rotate counterclockwise
            when looking along the axis direction (right-hand rule).
        center: Center point for rotation. If None, rotates about the origin.
            If provided, translates the mesh so that 'center' is at the origin,
            performs rotation, then translates back.
        transform_data: If True, also rotate vector and tensor fields in
            point_data and cell_data. Geometric caches (centroids, normals)
            are always rotated regardless of this flag.

    Returns:
        New Mesh with rotated geometry.

    Cache Handling:
        - areas: Unchanged (rotation preserves volumes).
        - centroids: Rotated (geometric property, always transformed).
        - normals: Rotated (geometric property, always transformed).

    Raises:
        ValueError: If axis is None for 3D mesh, or has incorrect shape.
        NotImplementedError: If mesh has spatial dimensions > 3.

    Example:
        >>> import numpy as np
        >>> # Rotate 2D mesh by 90 degrees
        >>> rotated_2d = rotate(mesh_2d, None, np.pi/2)
        >>>
        >>> # Rotate 3D mesh about z-axis by 45 degrees
        >>> rotated_3d = rotate(mesh_3d, [0, 0, 1], np.pi/4)
        >>>
        >>> # Rotate about arbitrary axis
        >>> axis = torch.tensor([1.0, 1.0, 1.0])  # Will be normalized
        >>> rotated = rotate(mesh_3d, axis, np.pi/3)
        >>>
        >>> # Rotate about mesh centroid
        >>> centroid = mesh.cell_centroids.mean(dim=0)
        >>> rotated = rotate(mesh, [0, 0, 1], np.pi/4, center=centroid)
    """
    ### Build rotation matrix
    # Convert axis to tensor if needed (for type checking)
    if axis is not None:
        axis = torch.as_tensor(axis, device=mesh.points.device, dtype=torch.float32)

    rotation_matrix = _build_rotation_matrix(
        axis=axis,
        angle=angle,
        n_spatial_dims=mesh.n_spatial_dims,
        device=mesh.points.device,
    )

    ### If center is provided, translate to origin, rotate, translate back
    if center is not None:
        center = torch.as_tensor(
            center, device=mesh.points.device, dtype=mesh.points.dtype
        )
        # Translate to origin
        mesh_centered = translate(mesh, -center)
        # Rotate
        mesh_rotated = rotate(
            mesh_centered, axis, angle, center=None, transform_data=transform_data
        )
        # Translate back
        return translate(mesh_rotated, center)

    ### Transform points with rotation matrix
    new_points = mesh.points @ rotation_matrix.T

    ### Handle caches with rotation-specific logic
    new_cell_data = _handle_caches_for_rotation(mesh, rotation_matrix)

    ### Transform user data if requested
    if transform_data:

        def transform_point_data(key, value):
            shape = value.shape[1:]

            # Skip scalars
            if len(shape) == 0:
                return value

            # Check first dimension matches
            if shape[0] != mesh.n_spatial_dims:
                raise ValueError(
                    f"Cannot transform point_data field {key!r} with shape {value.shape}. "
                    f"First dimension after batch must be {mesh.n_spatial_dims}, "
                    f"but got shape[1] = {shape[0]}. "
                    f"Set transform_data=False to skip data transformation."
                )

            if len(shape) == 1:
                return value @ rotation_matrix.T
            elif all(s == mesh.n_spatial_dims for s in shape):
                return _transform_higher_order_tensor(value, rotation_matrix)
            else:
                raise ValueError(
                    f"Cannot transform point_data field {key!r} with shape {value.shape}. "
                    f"Expected all dimensions after batch to be {mesh.n_spatial_dims}, "
                    f"but got shape[1:] = {shape}"
                )

        new_point_data = mesh.point_data.exclude("_cache").named_apply(
            transform_point_data,
            batch_size=torch.Size([mesh.n_points]),
        )

        def transform_cell_data(key, value):
            shape = value.shape[1:]

            # Skip scalars
            if len(shape) == 0:
                return value

            # Check first dimension matches
            if shape[0] != mesh.n_spatial_dims:
                raise ValueError(
                    f"Cannot transform cell_data field {key!r} with shape {value.shape}. "
                    f"First dimension after batch must be {mesh.n_spatial_dims}, "
                    f"but got shape[1] = {shape[0]}. "
                    f"Set transform_data=False to skip data transformation."
                )

            if len(shape) == 1:
                return value @ rotation_matrix.T
            elif all(s == mesh.n_spatial_dims for s in shape):
                return _transform_higher_order_tensor(value, rotation_matrix)
            else:
                raise ValueError(
                    f"Cannot transform cell_data field {key!r} with shape {value.shape}. "
                    f"Expected all dimensions after batch to be {mesh.n_spatial_dims}, "
                    f"but got shape[1:] = {shape}"
                )

        new_cell_data = new_cell_data.apply(
            transform_cell_data,
            batch_size=torch.Size([mesh.n_cells]),
            named=True,
        )
    else:
        new_point_data = mesh.point_data

    ### Create new mesh
    from physicsnemo.mesh.mesh import Mesh

    return Mesh(
        points=new_points,
        cells=mesh.cells,
        point_data=new_point_data,
        cell_data=new_cell_data,
        global_data=mesh.global_data,
    )


def scale(
    mesh: "Mesh",
    factor: float | torch.Tensor | list | tuple,
    center: torch.Tensor | list | tuple | None = None,
    transform_data: bool = False,
) -> "Mesh":
    """Scale the mesh by specified factor(s).

    Applies uniform or non-uniform scaling. Uniform scaling (single scalar factor)
    preserves angles and shape, while non-uniform scaling (different factors per
    dimension) can distort the mesh.

    Args:
        mesh: Input mesh to scale
        factor: Scale factor(s).
            - If scalar: uniform scaling by this factor
            - If tensor/list/tuple of shape (n_spatial_dims,): non-uniform scaling
              with different factors per dimension
            - Can be negative for reflection (uniform case only preserves normals)
        center: Center point for scaling. If None, scales about the origin.
            If provided, translates the mesh so that 'center' is at the origin,
            performs scaling, then translates back.
        transform_data: If True, also scale vector and tensor fields in
            point_data and cell_data. Geometric caches are handled separately.

    Returns:
        New Mesh with scaled geometry.

    Cache Handling (uniform scaling):
        - areas: Multiplied by |factor|^n_manifold_dims.
        - centroids: Scaled.
        - normals: Unchanged if factor > 0, flipped if factor < 0.

    Cache Handling (non-uniform scaling):
        - areas: Multiplied by |det(scale_matrix)|^(n_manifold_dims/n_spatial_dims).
        - centroids: Scaled (component-wise).
        - normals: Invalidated (directions change with non-uniform scaling).

    Example:
        >>> # Uniform scaling: double all dimensions
        >>> scaled = scale(mesh, 2.0)
        >>>
        >>> # Non-uniform scaling: stretch x by 2, compress y by 0.5
        >>> scaled = scale(mesh_2d, [2.0, 0.5])
        >>>
        >>> # Reflect across yz-plane (negate x)
        >>> reflected = scale(mesh_3d, [-1.0, 1.0, 1.0])
        >>>
        >>> # Scale about mesh centroid
        >>> centroid = mesh.cell_centroids.mean(dim=0)
        >>> scaled = scale(mesh, 1.5, center=centroid)
    """
    ### Convert factor to tensor and determine if uniform
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
            # Scalar tensor
            is_uniform = True
            factor_scalar = factor_tensor.item()
            scale_matrix = (
                torch.eye(mesh.n_spatial_dims, device=mesh.points.device)
                * factor_scalar
            )
        else:
            # Vector of scale factors
            if not torch.compiler.is_compiling():
                if factor_tensor.shape[-1] != mesh.n_spatial_dims:
                    raise ValueError(
                        f"factor must be scalar or have shape ({mesh.n_spatial_dims},), "
                        f"got shape {factor_tensor.shape}"
                    )
            is_uniform = False
            scale_matrix = torch.diag(factor_tensor)

    ### If center is provided, translate to origin, scale, translate back
    if center is not None:
        center = torch.as_tensor(
            center, device=mesh.points.device, dtype=mesh.points.dtype
        )
        mesh_centered = translate(mesh, -center)
        mesh_scaled = scale(
            mesh_centered, factor, center=None, transform_data=transform_data
        )
        return translate(mesh_scaled, center)

    ### Transform points with scale matrix
    new_points = mesh.points @ scale_matrix.T

    ### Handle caches with scale-specific logic
    if is_uniform:
        new_cell_data = _handle_caches_for_uniform_scale(mesh, factor_scalar)
    else:
        # Non-uniform scaling
        new_cell_data = mesh.cell_data.exclude("_cache")

        # Areas: invalidate (determinant formula doesn't work for embedded manifolds)
        # Not added back to cache

        # Centroids: scale component-wise
        cached_centroids = get_cached(mesh.cell_data, "centroids")
        if cached_centroids is not None:
            set_cached(new_cell_data, "centroids", cached_centroids @ scale_matrix.T)

        # Normals: invalidate (directions change with non-uniform scaling)
        # Not added back to cache

    ### Transform user data if requested
    if transform_data:

        def transform_point_data(key, value):
            shape = value.shape[1:]

            # Skip scalars
            if len(shape) == 0:
                return value

            # Check first dimension matches
            if shape[0] != mesh.n_spatial_dims:
                raise ValueError(
                    f"Cannot transform point_data field {key!r} with shape {value.shape}. "
                    f"First dimension after batch must be {mesh.n_spatial_dims}, "
                    f"but got shape[1] = {shape[0]}. "
                    f"Set transform_data=False to skip data transformation."
                )

            if len(shape) == 1:
                return value @ scale_matrix.T
            elif all(s == mesh.n_spatial_dims for s in shape):
                return _transform_higher_order_tensor(value, scale_matrix)
            else:
                raise ValueError(
                    f"Cannot transform point_data field {key!r} with shape {value.shape}. "
                    f"Expected all dimensions after batch to be {mesh.n_spatial_dims}, "
                    f"but got shape[1:] = {shape}"
                )

        new_point_data = mesh.point_data.exclude("_cache").named_apply(
            transform_point_data,
            batch_size=torch.Size([mesh.n_points]),
        )

        def transform_cell_data(key, value):
            shape = value.shape[1:]

            # Skip scalars
            if len(shape) == 0:
                return value

            # Check first dimension matches
            if shape[0] != mesh.n_spatial_dims:
                raise ValueError(
                    f"Cannot transform cell_data field {key!r} with shape {value.shape}. "
                    f"First dimension after batch must be {mesh.n_spatial_dims}, "
                    f"but got shape[1] = {shape[0]}. "
                    f"Set transform_data=False to skip data transformation."
                )

            if len(shape) == 1:
                return value @ scale_matrix.T
            elif all(s == mesh.n_spatial_dims for s in shape):
                return _transform_higher_order_tensor(value, scale_matrix)
            else:
                raise ValueError(
                    f"Cannot transform cell_data field {key!r} with shape {value.shape}. "
                    f"Expected all dimensions after batch to be {mesh.n_spatial_dims}, "
                    f"but got shape[1:] = {shape}"
                )

        # Apply transformation and merge with cached data
        transformed_user_data = new_cell_data.exclude("_cache").named_apply(
            transform_cell_data,
            batch_size=torch.Size([mesh.n_cells]),
        )
        new_cell_data.update(transformed_user_data)
    else:
        new_point_data = mesh.point_data

    ### Create new mesh
    from physicsnemo.mesh.mesh import Mesh

    return Mesh(
        points=new_points,
        cells=mesh.cells,
        point_data=new_point_data,
        cell_data=new_cell_data,
        global_data=mesh.global_data,
    )
