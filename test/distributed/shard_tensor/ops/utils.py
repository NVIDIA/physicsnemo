# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import copy
from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, distribute_module
from torch.distributed.tensor.device_mesh import DeviceMesh

from physicsnemo.distributed import ShardTensor


def unparallelize_module(module):
    """
    This is the inverse of "distribute_module".  Don't use this except in tests.

    (Why need this?  We're leveraging distribute_module to make sure all
    ranks have the same weights, if needed, instead of relying on random seeds.)
    """
    for name, param in list(module._parameters.items()):
        if isinstance(param, torch.nn.Parameter) and isinstance(param.data, DTensor):
            # gather to replicated then unwrap
            local_tensor = param.data.full_tensor()
            # replace with a normal Parameter
            module._parameters[name] = torch.nn.Parameter(
                local_tensor, requires_grad=param.requires_grad
            )
    # recurse into submodules
    for child in module.children():
        unparallelize_module(child)

    return module


def generate_image_like_data(
    batch_size: int,
    C_in: int,
    spatial_shape: tuple[int, ...],
    *,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Generate a random image-like tensor
    """
    return torch.randn(batch_size, C_in, *spatial_shape, device=device, dtype=dtype)


def sharded_to_local(container):
    """
    Convert a ShardTensor to a local tensor.

    In case the input is an iterable containing ShardTensors, this will convert
    each ShardTensor to a local tensor.
    """
    if isinstance(container, ShardTensor) or isinstance(container, DTensor):
        local_output = container.full_tensor()
        if container.requires_grad:
            local_output = local_output.detach().requires_grad_(True)
        return local_output
    elif isinstance(container, dict):
        return {key: sharded_to_local(value) for key, value in container.items()}
    elif isinstance(container, Iterable):
        return [sharded_to_local(item) for item in container]
    else:
        return container


def validate_shard_tensor_spec(shard_tensor):
    """
    Take a shard tensor and cross check on the dimensions and shapes.

    Basically, this is a consistency-check on sharding shapes.
    """

    # Take care about assertions here, since this is a collective

    # Check out shard shapes
    # The local shard shape needs to match the local tensor shape:
    sharding_shapes = shard_tensor._spec.sharding_shapes()
    mesh = shard_tensor._spec.mesh

    for mesh_dim in range(mesh.ndim):
        mesh_rank = mesh.get_local_rank(mesh_dim)
        mesh_size = dist.get_world_size(mesh.get_group(mesh_dim))

        # Is this axis sharded?
        this_placement = shard_tensor._spec.placements[mesh_dim]
        if this_placement.is_shard():
            # This axis is sharded.  the mesh dim should be in the shapes
            assert mesh_dim in sharding_shapes.keys()

            # The length of the sharding shapes should match the mesh size:
            assert len(sharding_shapes[mesh_dim]) == mesh_size

            # The local shape should match the listed shape for this rank:
            assert (
                sharding_shapes[mesh_dim][mesh_rank] == shard_tensor._local_tensor.shape
            )


def default_tensor_comparison(output, d_output, atol, rtol):
    if not isinstance(output, torch.Tensor):
        if isinstance(output, Iterable):
            return all(
                [
                    default_tensor_comparison(item, d_item, atol, rtol)
                    for item, d_item in zip(output, d_output)
                ]
            )

    if isinstance(d_output, ShardTensor):
        validate_shard_tensor_spec(d_output)

    local_output = sharded_to_local(d_output)

    # Check forward agreement:
    assert torch.allclose(output, local_output, atol=atol, rtol=rtol)

    return True


def default_loss_fn(output):
    return output.mean()


def numerical_shard_tensor_check(
    mesh: DeviceMesh,
    module: torch.nn.Module,
    input_args: Iterable,
    input_kwargs: dict,
    check_grads: bool = False,
    fwd_comparison_fn: callable = default_tensor_comparison,
    loss_fn: callable = default_loss_fn,
    atol: float = 1e-5,
    rtol: float = 1e-5,
):
    # Make sure the module's parameters all align on ever rank of the mesh:
    d_module = distribute_module(module, device_mesh=mesh)
    # (By default this replicates)

    # Then, get a local copy of the parameters
    module = copy.deepcopy(d_module)
    module = unparallelize_module(module)

    # Now, get the local version of the data:
    local_input_args = sharded_to_local(input_args)
    local_input_kwargs = sharded_to_local(input_kwargs)

    # Run the module on the local data:
    output = module(*local_input_args, **local_input_kwargs)

    # Run the distributed module on the distributed data:
    d_output = d_module(*input_args, **input_kwargs)

    fwd_comparison_fn(output, d_output, atol, rtol)

    if check_grads:
        # single device grads:
        default_loss_fn(output).backward()

        # distributed grads:
        default_loss_fn(d_output).backward()

        # compare the grads:
        for param, d_param in zip(module.parameters(), d_module.parameters()):
            default_tensor_comparison(param.grad, d_param.grad, atol=atol, rtol=rtol)

        # Check the input grads, if they are required:

        for input_arg, d_input_arg in zip(local_input_args, input_args):
            if d_input_arg.requires_grad:
                default_tensor_comparison(input_arg.grad, d_input_arg.grad, atol, rtol)

                # input gradients should have the same sharding and placements.
                # Check the spec:
                assert d_input_arg._spec == d_input_arg.grad._spec
