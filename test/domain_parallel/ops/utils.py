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

r"""Utility functions for ShardTensor operation testing.

This module provides helper functions for testing ``ShardTensor`` operations,
including:

- Collective assertions that fail on all ranks if any rank fails
- Tensor comparison utilities for distributed testing
- Module de-parallelization for comparing distributed vs local results
- Numerical validation framework for ShardTensor operations
"""

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
    r"""Collective assertion that fails on all ranks if any rank fails.

    This prevents hangs in distributed tests where one rank might fail
    while others continue waiting for collective operations.

    Parameters
    ----------
    condition : bool
        The condition to check (``True`` = pass, ``False`` = fail).
    msg : str, default="Assertion failed"
        Error message to display if assertion fails.
    group : Optional[dist.ProcessGroup], optional
        Process group for the collective. If ``None``, uses default group.

    Raises
    ------
    AssertionError
        If any rank in the group has ``condition=False``.
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
    r"""Collective version of ``torch.allclose`` assertion.

    Fails on all ranks if any rank's tensors are not close.

    Parameters
    ----------
    tensor1 : torch.Tensor
        First tensor to compare.
    tensor2 : torch.Tensor
        Second tensor to compare.
    atol : float, default=1e-5
        Absolute tolerance.
    rtol : float, default=1e-5
        Relative tolerance.
    msg : str, default="Tensors not close"
        Error message to display if assertion fails.
    group : Optional[dist.ProcessGroup], optional
        Process group for the collective.

    Raises
    ------
    AssertionError
        If any rank's tensors fail the ``allclose`` check.
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
    r"""Collective assertion for equality.

    Fails on all ranks if any rank's values are not equal.

    Parameters
    ----------
    value1 : Any
        First value to compare.
    value2 : Any
        Second value to compare.
    msg : str, default="Values not equal"
        Error message to display if assertion fails.
    group : Optional[dist.ProcessGroup], optional
        Process group for the collective.

    Raises
    ------
    AssertionError
        If any rank's values are not equal.
    """
    collective_assert(value1 == value2, f"{msg}: {value1} != {value2}", group)


def unparallelize_module(module):
    r"""Convert a distributed module back to a regular module.

    This is the inverse of ``distribute_module``. Should only be used in tests.

    We use this because we leverage ``distribute_module`` to ensure all ranks
    have the same weights (instead of relying on random seeds), and then need
    to convert back to compare distributed vs local results.

    Parameters
    ----------
    module : torch.nn.Module
        The distributed module to unparallelize.

    Returns
    -------
    torch.nn.Module
        The module with DTensor parameters replaced by regular tensors.

    Warning
    -------
    This function is for testing purposes only. Do not use in production code.
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
    r"""Generate a random image-like tensor.

    Parameters
    ----------
    batch_size : int
        Number of samples in the batch.
    C_in : int
        Number of input channels.
    spatial_shape : tuple[int, ...]
        Spatial dimensions (H, W) for 2D or (D, H, W) for 3D, etc.
    device : torch.device, optional
        Device to create the tensor on.
    dtype : torch.dtype, optional
        Data type of the tensor.

    Returns
    -------
    torch.Tensor
        Random tensor of shape ``(batch_size, C_in, *spatial_shape)``.
    """
    return torch.randn(batch_size, C_in, *spatial_shape, device=device, dtype=dtype)


def sharded_to_local(container):
    r"""Convert a ShardTensor or DTensor to a local (full) tensor.

    Recursively processes containers (dicts, lists, tuples) to convert any
    ShardTensor or DTensor instances to their full tensor representation.

    Parameters
    ----------
    container : Any
        A ShardTensor, DTensor, dict, iterable, or other value.

    Returns
    -------
    Any
        The same structure with ShardTensor/DTensor replaced by full tensors.
        If the original tensor required gradients, the returned tensor will
        be detached and have ``requires_grad=True``.
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
    r"""Validate ShardTensor specification consistency.

    Cross-checks the tensor's dimensions and shapes to ensure the sharding
    specification is internally consistent. Verifies that:

    - Sharded mesh dimensions have corresponding sharding shapes
    - Sharding shapes have correct length for the mesh size
    - Local tensor shape matches the listed shape for this rank

    Parameters
    ----------
    shard_tensor : ShardTensor
        The ShardTensor to validate.
    group : Optional[dist.ProcessGroup], optional
        Process group for collective assertions.

    Raises
    ------
    AssertionError
        If any consistency check fails on any rank.
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
) -> bool:
    r"""Compare a local tensor output with a distributed (ShardTensor) output.

    Validates that the distributed output matches the local output within
    tolerances, and that the ShardTensor spec is internally consistent.

    Parameters
    ----------
    output : torch.Tensor
        The reference local tensor output.
    d_output : ShardTensor or torch.Tensor
        The distributed output to compare.
    atol : float
        Absolute tolerance for comparison.
    rtol : float
        Relative tolerance for comparison.
    group : Optional[dist.ProcessGroup], optional
        Process group for collective assertions.

    Returns
    -------
    bool
        ``True`` if comparison passes on all ranks.
    """
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
    r"""Default loss function for testing: compute mean of output.

    Parameters
    ----------
    output : torch.Tensor
        The tensor to compute loss on.

    Returns
    -------
    torch.Tensor
        Scalar mean of the output.
    """
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
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> None:
    r"""Numerically validate a ShardTensor operation against local computation.

    Runs the same module on both distributed (ShardTensor) inputs and local
    (full tensor) inputs, then compares the results to verify correctness.

    Parameters
    ----------
    mesh : DeviceMesh
        The device mesh for distributed computation.
    module : torch.nn.Module
        The module to test.
    input_args : Iterable
        Positional arguments to the module (may contain ShardTensors).
    input_kwargs : dict
        Keyword arguments to the module (may contain ShardTensors).
    check_grads : bool, default=False
        If ``True``, also verify gradients match between distributed and local.
    fwd_comparison_fn : callable, default=default_tensor_comparison
        Function to compare forward outputs.
    loss_fn : callable, default=default_loss_fn
        Function to compute loss for backward pass (if ``check_grads=True``).
    atol : float, default=1e-5
        Absolute tolerance for comparisons.
    rtol : float, default=1e-5
        Relative tolerance for comparisons.
    group : Optional[dist.ProcessGroup], optional
        Process group for collective assertions.
    amp : bool, default=False
        If ``True``, wrap forward and backward passes in
        ``torch.amp.autocast("cuda")`` for automatic mixed precision testing.
    amp_dtype : torch.dtype, default=torch.float16
        The dtype to use for autocast when ``amp=True``.  Common choices are
        ``torch.float16`` and ``torch.bfloat16``.

    Raises
    ------
    AssertionError
        If forward outputs don't match or (if checking grads) gradients don't match.
    """
    # Make sure the module's parameters all align on ever rank of the mesh:
    d_module = distribute_module(module, device_mesh=mesh)
    # (By default this replicates)

    # Then, get a local copy of the parameters
    module = copy.deepcopy(d_module)
    module = unparallelize_module(module)

    # Now, get the local version of the data:
    local_input_args = sharded_to_local(input_args)
    local_input_kwargs = sharded_to_local(input_kwargs)

    with torch.amp.autocast("cuda", enabled=amp, dtype=amp_dtype):
        # Run the module on the local data:
        output = module(*local_input_args, **local_input_kwargs)
        # Run the distributed module on the distributed data:
        d_output = d_module(*input_args, **input_kwargs)

    fwd_comparison_fn(output, d_output, atol, rtol, group)

    if check_grads:
        with torch.amp.autocast("cuda", enabled=amp, dtype=amp_dtype):
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
