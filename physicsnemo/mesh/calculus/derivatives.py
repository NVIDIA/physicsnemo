# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Unified API for computing discrete derivatives on meshes.

Provides high-level interface for gradient, divergence, curl, and Laplacian
computations using both DEC and LSQ methods.
"""

from typing import TYPE_CHECKING, Literal, Sequence

from physicsnemo.mesh.utilities._cache import CACHE_KEY

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def compute_point_derivatives(
    mesh: "Mesh",
    keys: str | tuple[str, ...] | Sequence[str | tuple[str, ...]] | None = None,
    method: Literal["lsq", "dec"] = "lsq",
    gradient_type: Literal["intrinsic", "extrinsic", "both"] = "intrinsic",
) -> "Mesh":
    """Compute gradients of point_data fields.

    Computes discrete gradients using either DEC or LSQ methods, with support
    for both intrinsic (tangent space) and extrinsic (ambient space) derivatives.

    Args:
        mesh: Simplicial mesh
        keys: Fields to compute gradients of. Options:
            - None: All non-cached fields (excludes "_cache" subdictionary)
            - str: Single field name (e.g., "pressure")
            - tuple: Nested path (e.g., ("flow", "temperature"))
            - Sequence: List of above (e.g., ["pressure", ("flow", "velocity")])
        method: Discretization method:
            - "lsq": Weighted least-squares reconstruction (CFD standard)
            - "dec": Discrete Exterior Calculus (differential geometry)
        gradient_type: Type of gradient to compute:
            - "intrinsic": Project onto manifold tangent space
            - "extrinsic": Full ambient space gradient
            - "both": Compute and store both

    Returns:
        The input mesh with gradient fields added to point_data (modified in place).
        Field naming: "{field}_gradient" or "{field}_gradient_intrinsic/extrinsic"

    Example:
        >>> import torch
        >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
        >>> mesh = two_triangles_2d.load()
        >>> mesh.point_data["pressure"] = torch.randn(mesh.n_points)
        >>> # Compute gradient of pressure field
        >>> mesh_with_grad = compute_point_derivatives(mesh, keys="pressure")
        >>> grad_p = mesh_with_grad.point_data["pressure_gradient"]
    """
    from physicsnemo.mesh.calculus.gradient import (
        compute_gradient_points_dec,
        compute_gradient_points_lsq,
        project_to_tangent_space,
    )

    ### Parse keys: normalize to list of key paths
    if keys is None:
        # All non-cached fields
        key_list = list(
            mesh.point_data.exclude(CACHE_KEY).keys(
                include_nested=True, leaves_only=True
            )
        )
    elif isinstance(keys, (str, tuple)):
        key_list = [keys]
    elif isinstance(keys, Sequence):
        key_list = list(keys)
    else:
        raise TypeError(f"Invalid keys type: {type(keys)}")

    ### Compute gradients for each key (modify mesh.point_data in place)
    for key in key_list:
        # Get field values using native TensorDict indexing
        field_values = mesh.point_data[key]

        ### Compute gradient based on method and gradient_type
        if method == "lsq":
            if gradient_type == "intrinsic":
                grad_intrinsic = compute_gradient_points_lsq(
                    mesh, field_values, intrinsic=True
                )
                grad_extrinsic = None
            elif gradient_type == "extrinsic":
                grad_extrinsic = compute_gradient_points_lsq(
                    mesh, field_values, intrinsic=False
                )
                grad_intrinsic = None
            else:  # "both"
                grad_extrinsic = compute_gradient_points_lsq(
                    mesh, field_values, intrinsic=False
                )
                grad_intrinsic = compute_gradient_points_lsq(
                    mesh, field_values, intrinsic=True
                )
        elif method == "dec":
            # DEC always computes in ambient space initially
            grad_extrinsic = compute_gradient_points_dec(mesh, field_values)

            if gradient_type == "intrinsic":
                grad_intrinsic = project_to_tangent_space(
                    mesh, grad_extrinsic, "points"
                )
                grad_extrinsic = None
            elif gradient_type == "both":
                grad_intrinsic = project_to_tangent_space(
                    mesh, grad_extrinsic, "points"
                )
            else:  # extrinsic
                grad_intrinsic = None
        else:
            raise ValueError(f"Invalid {method=}. Must be 'lsq' or 'dec'.")

        ### Store gradients in mesh.point_data
        if gradient_type == "extrinsic":
            out_key = (
                f"{key}_gradient"
                if isinstance(key, str)
                else key[:-1] + (key[-1] + "_gradient",)
            )
            mesh.point_data[out_key] = grad_extrinsic

        elif gradient_type == "intrinsic":
            out_key = (
                f"{key}_gradient"
                if isinstance(key, str)
                else key[:-1] + (key[-1] + "_gradient",)
            )
            mesh.point_data[out_key] = grad_intrinsic

        elif gradient_type == "both":
            out_key_ext = (
                f"{key}_gradient_extrinsic"
                if isinstance(key, str)
                else key[:-1] + (key[-1] + "_gradient_extrinsic",)
            )
            out_key_int = (
                f"{key}_gradient_intrinsic"
                if isinstance(key, str)
                else key[:-1] + (key[-1] + "_gradient_intrinsic",)
            )
            mesh.point_data[out_key_ext] = grad_extrinsic
            mesh.point_data[out_key_int] = grad_intrinsic

        else:
            raise ValueError(f"Invalid {gradient_type=}")

    ### Return mesh for method chaining
    return mesh


def compute_cell_derivatives(
    mesh: "Mesh",
    keys: str | tuple[str, ...] | Sequence[str | tuple[str, ...]] | None = None,
    method: Literal["lsq", "dec"] = "lsq",
    gradient_type: Literal["intrinsic", "extrinsic", "both"] = "intrinsic",
) -> "Mesh":
    """Compute gradients of cell_data fields.

    Args:
        mesh: Simplicial mesh
        keys: Fields to compute gradients of (same format as compute_point_derivatives)
        method: "lsq" or "dec"
        gradient_type: "intrinsic", "extrinsic", or "both"

    Returns:
        The input mesh with gradient fields added to cell_data (modified in place)
    """
    from physicsnemo.mesh.calculus.gradient import (
        compute_gradient_cells_lsq,
        project_to_tangent_space,
    )

    ### Parse keys: normalize to list of key paths
    if keys is None:
        key_list = list(
            mesh.cell_data.exclude(CACHE_KEY).keys(
                include_nested=True, leaves_only=True
            )
        )
    elif isinstance(keys, (str, tuple)):
        key_list = [keys]
    elif isinstance(keys, Sequence):
        key_list = list(keys)
    else:
        raise TypeError(f"Invalid keys type: {type(keys)}")

    ### Compute gradients for each key (modify mesh.cell_data in place)
    for key in key_list:
        # Get field values using native TensorDict indexing
        field_values = mesh.cell_data[key]

        ### Compute extrinsic gradient
        if method == "lsq":
            grad_extrinsic = compute_gradient_cells_lsq(mesh, field_values)
        elif method == "dec":
            raise NotImplementedError(
                "DEC cell gradients not yet implemented. Use method='lsq'."
            )
        else:
            raise ValueError(f"Invalid {method=}")

        ### Store gradients in mesh.cell_data
        if gradient_type == "extrinsic":
            out_key = (
                f"{key}_gradient"
                if isinstance(key, str)
                else key[:-1] + (key[-1] + "_gradient",)
            )
            mesh.cell_data[out_key] = grad_extrinsic

        elif gradient_type == "intrinsic":
            grad_intrinsic = project_to_tangent_space(mesh, grad_extrinsic, "cells")
            out_key = (
                f"{key}_gradient"
                if isinstance(key, str)
                else key[:-1] + (key[-1] + "_gradient",)
            )
            mesh.cell_data[out_key] = grad_intrinsic

        elif gradient_type == "both":
            grad_intrinsic = project_to_tangent_space(mesh, grad_extrinsic, "cells")
            out_key_ext = (
                f"{key}_gradient_extrinsic"
                if isinstance(key, str)
                else key[:-1] + (key[-1] + "_gradient_extrinsic",)
            )
            out_key_int = (
                f"{key}_gradient_intrinsic"
                if isinstance(key, str)
                else key[:-1] + (key[-1] + "_gradient_intrinsic",)
            )
            mesh.cell_data[out_key_ext] = grad_extrinsic
            mesh.cell_data[out_key_int] = grad_intrinsic

        else:
            raise ValueError(f"Invalid {gradient_type=}")

    ### Return mesh for method chaining
    return mesh
