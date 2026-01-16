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

import copy
from collections.abc import Iterable
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, distribute_module
from torch.distributed.tensor.device_mesh import DeviceMesh

from physicsnemo.domain_parallel import ShardTensor


def collective_assert(
    condition: bool,
    msg: str = "Assertion failed",
    group: Optional[dist.ProcessGroup] = None,
) -> None:
    """
    Collective assertion that fails on all ranks if any rank fails.

    This prevents hangs in distributed tests where one rank might fail
    while others continue waiting for collective operations.

    Args:
        condition: The condition to check (True = pass, False = fail)
        msg: Error message to display if assertion fails
        group: Optional process group for the collective. If None, uses default group.
    """
    # Convert condition to tensor (1 = pass, 0 = fail)
    local_result = torch.tensor(
        [1 if condition else 0], dtype=torch.int32, device="cuda"
    )

    # Use all_reduce with MIN to find if any rank failed
    dist.all_reduce(local_result, op=dist.ReduceOp.MIN, group=group)

    # If any rank had condition=False, the min will be 0
    if local_result.item() == 0:
        rank = dist.get_rank(group)
        raise AssertionError(f"[Rank {rank}] Collective assertion failed: {msg}")


def collective_assert_close(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    msg: str = "Tensors not close",
    group: Optional[dist.ProcessGroup] = None,
) -> None:
    """
    Collective version of torch.allclose assertion.

    Fails on all ranks if any rank's tensors are not close.

    Args:
        tensor1: First tensor to compare
        tensor2: Second tensor to compare
        atol: Absolute tolerance
        rtol: Relative tolerance
        msg: Error message to display if assertion fails
        group: Optional process group for the collective
    """
    is_close = torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol)
    if not is_close:
        max_diff = (tensor1 - tensor2).abs().max().item()
        detailed_msg = f"{msg} (max diff: {max_diff}, atol: {atol}, rtol: {rtol})"
    else:
        detailed_msg = msg
    collective_assert(is_close, detailed_msg, group)


def collective_assert_equal(
    value1,
    value2,
    msg: str = "Values not equal",
    group: Optional[dist.ProcessGroup] = None,
) -> None:
    """
    Collective assertion for equality.

    Fails on all ranks if any rank's values are not equal.

    Args:
        value1: First value to compare
        value2: Second value to compare
        msg: Error message to display if assertion fails
        group: Optional process group for the collective
    """
    collective_assert(value1 == value2, f"{msg}: {value1} != {value2}", group)


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


def validate_shard_tensor_spec(shard_tensor, group: Optional[dist.ProcessGroup] = None):
    """
    Take a shard tensor and cross check on the dimensions and shapes.

    Basically, this is a consistency-check on sharding shapes.

    Args:
        shard_tensor: The ShardTensor to validate
        group: Optional process group for collective assertions
    """

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
            collective_assert(
                mesh_dim in sharding_shapes.keys(),
                f"mesh_dim {mesh_dim} not in sharding_shapes keys",
                group,
            )

            # The length of the sharding shapes should match the mesh size:
            collective_assert_equal(
                len(sharding_shapes[mesh_dim]),
                mesh_size,
                f"sharding_shapes length mismatch for mesh_dim {mesh_dim}",
                group,
            )

            # The local shape should match the listed shape for this rank:
            collective_assert_equal(
                sharding_shapes[mesh_dim][mesh_rank],
                shard_tensor._local_tensor.shape,
                f"local shape mismatch for mesh_dim {mesh_dim}, mesh_rank {mesh_rank}",
                group,
            )


def default_tensor_comparison(
    output,
    d_output,
    atol,
    rtol,
    group: Optional[dist.ProcessGroup] = None,
):
    if not isinstance(output, torch.Tensor):
        if isinstance(output, Iterable):
            return all(
                [
                    default_tensor_comparison(item, d_item, atol, rtol, group)
                    for item, d_item in zip(output, d_output)
                ]
            )

    if isinstance(d_output, ShardTensor):
        validate_shard_tensor_spec(d_output, group)

    local_output = sharded_to_local(d_output)

    # Check forward agreement:
    collective_assert_close(
        output,
        local_output,
        atol=atol,
        rtol=rtol,
        msg="Forward pass output mismatch",
        group=group,
    )

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
    group: Optional[dist.ProcessGroup] = None,
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

    fwd_comparison_fn(output, d_output, atol, rtol, group)

    if check_grads:
        # single device grads:
        default_loss_fn(output).backward()

        # distributed grads:
        default_loss_fn(d_output).backward()

        # compare the grads:
        for param, d_param in zip(module.parameters(), d_module.parameters()):
            default_tensor_comparison(
                param.grad, d_param.grad, atol=atol, rtol=rtol, group=group
            )

        # Check the input grads, if they are required:

        for input_arg, d_input_arg in zip(local_input_args, input_args):
            if d_input_arg.requires_grad:
                default_tensor_comparison(
                    input_arg.grad, d_input_arg.grad, atol, rtol, group
                )

                # input gradients should have the same sharding and placements.
                # Check the spec:
                collective_assert_equal(
                    d_input_arg._spec,
                    d_input_arg.grad._spec,
                    "Input gradient spec mismatch",
                    group,
                )
