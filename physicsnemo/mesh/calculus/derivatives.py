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

"""Unified API for computing discrete derivatives on meshes.

Provides high-level interface for gradient, divergence, curl, and Laplacian
computations using both DEC and LSQ methods.
"""

from typing import TYPE_CHECKING, Literal, Sequence

from physicsnemo.mesh.utilities._cache import CACHE_KEY

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def _make_output_key(
    key: str | tuple[str, ...],
    suffix: str,
) -> str | tuple[str, ...]:
    """Build the output key name for a gradient field.

    Parameters
    ----------
    key : str or tuple[str, ...]
        Original field key (possibly a nested TensorDict path).
    suffix : str
        Suffix to append (e.g., ``"_gradient"``).

    Returns
    -------
    str or tuple[str, ...]
        Key with suffix appended to the leaf name.
    """
    if isinstance(key, str):
        return f"{key}{suffix}"
    return key[:-1] + (key[-1] + suffix,)


def compute_point_derivatives(
    mesh: "Mesh",
    keys: str | tuple[str, ...] | Sequence[str | tuple[str, ...]] | None = None,
    method: Literal["lsq", "dec"] = "lsq",
    gradient_type: Literal["intrinsic", "extrinsic", "both"] = "intrinsic",
) -> "Mesh":
    """Compute gradients of point_data fields.

    Computes discrete gradients using either DEC or LSQ methods, with support
    for both intrinsic (tangent space) and extrinsic (ambient space) derivatives.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh with point_data fields to differentiate.
    keys : str or tuple[str, ...] or Sequence or None
        Fields to compute gradients of. Options:

        - ``None``: All non-cached fields (excludes ``"_cache"`` subdictionary).
        - ``str``: Single field name (e.g., ``"pressure"``).
        - ``tuple``: Nested path (e.g., ``("flow", "temperature")``).
        - ``Sequence``: List of the above.
    method : {"lsq", "dec"}
        Discretization method:

        - ``"lsq"``: Weighted least-squares reconstruction (CFD standard).
        - ``"dec"``: Discrete Exterior Calculus (differential geometry).
    gradient_type : {"intrinsic", "extrinsic", "both"}
        Type of gradient to compute:

        - ``"intrinsic"``: Project onto manifold tangent space.
        - ``"extrinsic"``: Full ambient space gradient.
        - ``"both"``: Compute and store both.

    Returns
    -------
    Mesh
        A new Mesh with gradient fields added to ``point_data``. The original
        mesh is **not** modified. Field naming convention:

        - ``gradient_type="intrinsic"`` or ``"extrinsic"``:
          ``"{field}_gradient"``
        - ``gradient_type="both"``:
          ``"{field}_gradient_intrinsic"`` and ``"{field}_gradient_extrinsic"``

    Example
    -------
    >>> import torch
    >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
    >>> mesh = two_triangles_2d.load()
    >>> mesh.point_data["pressure"] = torch.randn(mesh.n_points)
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

    ### Clone point_data so we don't mutate the original mesh
    new_point_data = mesh.point_data.clone()

    ### Compute gradients for each key
    for key in key_list:
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
                grad_intrinsic = project_to_tangent_space(
                    mesh, grad_extrinsic, location="points"
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

        ### Store gradients in the cloned point_data
        if gradient_type in ("extrinsic", "intrinsic"):
            out_key = _make_output_key(key, "_gradient")
            value = grad_extrinsic if gradient_type == "extrinsic" else grad_intrinsic
            new_point_data[out_key] = value
        elif gradient_type == "both":
            new_point_data[_make_output_key(key, "_gradient_extrinsic")] = (
                grad_extrinsic
            )
            new_point_data[_make_output_key(key, "_gradient_intrinsic")] = (
                grad_intrinsic
            )
        else:
            raise ValueError(f"Invalid {gradient_type=}")

    ### Return a new Mesh with the augmented point_data
    from physicsnemo.mesh.mesh import Mesh

    return Mesh(
        points=mesh.points,
        cells=mesh.cells,
        point_data=new_point_data,
        cell_data=mesh.cell_data,
        global_data=mesh.global_data,
    )


def compute_cell_derivatives(
    mesh: "Mesh",
    keys: str | tuple[str, ...] | Sequence[str | tuple[str, ...]] | None = None,
    method: Literal["lsq", "dec"] = "lsq",
    gradient_type: Literal["intrinsic", "extrinsic", "both"] = "intrinsic",
) -> "Mesh":
    """Compute gradients of cell_data fields.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh with cell_data fields to differentiate.
    keys : str or tuple[str, ...] or Sequence or None
        Fields to compute gradients of (same format as
        :func:`compute_point_derivatives`).
    method : {"lsq"}
        Discretization method for cell-centered data. Currently only
        ``"lsq"`` (weighted least-squares) is implemented. DEC gradients
        for cell-centered data are not available because the standard DEC
        exterior derivative maps vertex 0-forms to edge 1-forms; there is
        no analogous cell-to-cell operator in the primal DEC complex.
    gradient_type : {"intrinsic", "extrinsic", "both"}
        Type of gradient to compute.

    Returns
    -------
    Mesh
        A new Mesh with gradient fields added to ``cell_data``. The original
        mesh is **not** modified.

    Raises
    ------
    NotImplementedError
        If ``method="dec"`` is requested.
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

    ### Clone cell_data so we don't mutate the original mesh
    new_cell_data = mesh.cell_data.clone()

    ### Compute gradients for each key
    for key in key_list:
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

        ### Store gradients in the cloned cell_data
        if gradient_type == "extrinsic":
            new_cell_data[_make_output_key(key, "_gradient")] = grad_extrinsic

        elif gradient_type == "intrinsic":
            grad_intrinsic = project_to_tangent_space(mesh, grad_extrinsic, "cells")
            new_cell_data[_make_output_key(key, "_gradient")] = grad_intrinsic

        elif gradient_type == "both":
            grad_intrinsic = project_to_tangent_space(mesh, grad_extrinsic, "cells")
            new_cell_data[_make_output_key(key, "_gradient_extrinsic")] = grad_extrinsic
            new_cell_data[_make_output_key(key, "_gradient_intrinsic")] = grad_intrinsic

        else:
            raise ValueError(f"Invalid {gradient_type=}")

    ### Return a new Mesh with the augmented cell_data
    from physicsnemo.mesh.mesh import Mesh

    return Mesh(
        points=mesh.points,
        cells=mesh.cells,
        point_data=mesh.point_data,
        cell_data=new_cell_data,
        global_data=mesh.global_data,
    )
