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

from __future__ import annotations

from typing import Any, Callable

import torch

from physicsnemo.domain_parallel import ShardTensor
from physicsnemo.nn.functional import signed_distance_field


def sharded_signed_distance_field(
    mesh_vertices: ShardTensor,
    mesh_indices: ShardTensor,
    input_points: ShardTensor,
    max_dist: float = 1e8,
    use_sign_winding_number: bool = False,
) -> tuple[ShardTensor, ShardTensor]:
    r"""Compute the signed distance field for a (possibly sharded) mesh.

    Parameters
    ----------
    mesh_vertices : ShardTensor
        Sharded tensor of mesh vertices.
    mesh_indices : ShardTensor
        Sharded tensor of mesh indices.
    input_points : ShardTensor
        Sharded tensor of input points.
    max_dist : float, default=1e8
        Maximum distance for the signed distance field.
    use_sign_winding_number : bool, default=False
        Whether to use sign winding number.

    Returns
    -------
    tuple[ShardTensor, ShardTensor]
        A tuple containing:

        - ``sharded_sdf_output`` : Signed distance field values
        - ``sharded_sdf_hit_point_output`` : Hit point coordinates
    """

    # We can not actually compute the signed distance function on a sharded mesh.
    # So, in this case, force the mesh to replicate placement if necessary:

    local_mesh_vertices = mesh_vertices.full_tensor()
    local_mesh_indices = mesh_indices.full_tensor()

    # For the input points, though, it doesn't matter - they can be sharded.
    # No communication is necessary

    local_input_points = input_points.to_local()

    local_sdf, local_sdf_hit_point = signed_distance_field(
        local_mesh_vertices,
        local_mesh_indices,
        local_input_points,
        max_dist,
        use_sign_winding_number,
    )

    # Then, construct the output shard tensors:

    if input_points._spec.placements[0].is_shard():
        # Compute the output sharding shapes

        # Output shape is always (N, 1), hit point is (N, 3)
        input_shard_shapes = input_points._spec.sharding_shapes()

        output_shard_shapes = {
            mesh_dim: tuple(torch.Size((s[0],)) for s in input_shard_shapes[mesh_dim])
            for mesh_dim in input_shard_shapes.keys()
        }

        sharded_sdf_output = ShardTensor.from_local(
            local_sdf,
            input_points._spec.mesh,
            input_points._spec.placements,
            sharding_shapes=output_shard_shapes,
        ).reshape(input_points.shape[:-1])

        sharded_sdf_hit_point_output = ShardTensor.from_local(
            local_sdf_hit_point,
            input_points._spec.mesh,
            input_points._spec.placements,
            sharding_shapes=input_shard_shapes,
        ).reshape(input_points.shape)

    else:
        # The input points were replicated, use that for output:
        sharded_sdf_output = ShardTensor.from_local(
            local_sdf,
            input_points._spec.mesh,
            input_points._spec.placements,
        )
        sharded_sdf_hit_point_output = ShardTensor.from_local(
            local_sdf_hit_point,
            input_points._spec.mesh,
            input_points._spec.placements,
        )

    return sharded_sdf_output, sharded_sdf_hit_point_output


def repackage_radius_search_wrapper_args(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    input_points: torch.Tensor,
    max_dist: float = 1e8,
    use_sign_winding_number: bool = False,
    *args,
    **kwargs,
) -> tuple[ShardTensor, ShardTensor, dict]:
    r"""Repackage signed distance field arguments into a standard format.

    Parameters
    ----------
    mesh_vertices : torch.Tensor
        Tensor of mesh vertices.
    mesh_indices : torch.Tensor
        Tensor of mesh indices.
    input_points : torch.Tensor
        Tensor of input points.
    max_dist : float, default=1e8
        Maximum distance for the signed distance field.
    use_sign_winding_number : bool, default=False
        Whether to use sign winding number.
    *args : Any
        Additional positional arguments (unused).
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    tuple[ShardTensor, ShardTensor, dict]
        Tuple containing (mesh_vertices, mesh_indices, input_points, kwargs_dict).
    """
    # Extract any additional parameters that might be in kwargs
    # or use defaults if not provided
    return_kwargs = {
        "max_dist": max_dist,
        "use_sign_winding_number": use_sign_winding_number,
    }

    # Add any explicitly passed parameters
    if kwargs:
        return_kwargs.update(kwargs)

    return mesh_vertices, mesh_indices, input_points, return_kwargs


def sharded_signed_distance_field_wrapper(
    func: Callable,
    types: tuple[Any, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[ShardTensor, ShardTensor]:
    r"""Wrapper for ``sharded_signed_distance_field`` to support sharded tensors.

    Parameters
    ----------
    func : Callable
        The original function (unused).
    types : tuple[Any, ...]
        The types of the inputs (unused).
    args : tuple
        Positional arguments to pass to ``sharded_signed_distance_field``.
    kwargs : dict
        Keyword arguments to pass to ``sharded_signed_distance_field``.

    Returns
    -------
    tuple[ShardTensor, ShardTensor]
        A tuple containing the signed distance field and hit point tensors.
    """

    return sharded_signed_distance_field(*args, **kwargs)


ShardTensor.register_named_function_handler(
    "physicsnemo.signed_distance_field.default", sharded_signed_distance_field_wrapper
)
