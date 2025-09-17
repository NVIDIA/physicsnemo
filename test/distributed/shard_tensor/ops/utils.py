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


def default_tensor_comparison(output, d_output, atol, rtol):
    # We assume a single output!

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
