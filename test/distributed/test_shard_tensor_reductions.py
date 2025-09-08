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


import pytest

from physicsnemo.utils.version_check import check_module_requirements

try:
    check_module_requirements("physicsnemo.distributed.shard_tensor")
    ST_AVAILABLE = True
except ImportError:
    pytest.skip(
        "Skipping test because physicsnemo.distributed.shard_tensor is not available",
        allow_module_level=True,
    )


if ST_AVAILABLE:
    from torch.distributed.tensor.placement_types import Shard

    from physicsnemo.distributed import scatter_tensor


import torch

from physicsnemo.distributed import DistributedManager


@pytest.mark.multigpu_static
def test_consecutive_reductions(
    distributed_mesh,
):
    dm = DistributedManager()

    def two_reduction_operation(output, target):
        mask = target > 0.0

        num = torch.sum(mask * (output - target) ** 2.0, (1,))
        denom = torch.sum(mask)

        return torch.mean(num / denom)

    torch.manual_seed(dm.world_size)

    full_output = torch.randn(2, 400, 5, requires_grad=False).to(dm.device)
    full_target = torch.randn(2, 400, 5, requires_grad=False).to(dm.device)
    baseline = two_reduction_operation(full_output, full_target)

    # Scatter it:
    placements = (Shard(1),)
    shard_output = scatter_tensor(
        full_output,
        0,
        distributed_mesh,
        placements,
        global_shape=full_output.shape,
        dtype=full_output.dtype,
        requires_grad=False,
    )
    shard_target = scatter_tensor(
        full_target,
        0,
        distributed_mesh,
        placements,
        global_shape=full_target.shape,
        dtype=full_target.dtype,
        requires_grad=False,
    )

    sharded_result = two_reduction_operation(shard_output, shard_target)

    full_result = sharded_result.full_tensor()

    assert torch.allclose(baseline, full_result)


@pytest.mark.multigpu_static
@pytest.mark.parametrize("op", ["sum", "mean"])
@pytest.mark.parametrize("backward", [True])
@pytest.mark.parametrize("dim", [None, 0, (0, 1)])
@pytest.mark.parametrize("in_place", [True, False])
def test_shard_tensor_reduction(
    distributed_mesh, op, backward, dim, in_place, verbose=False
):
    dm = DistributedManager()

    # Create a random-valued tensor of at least rank 3:
    full_input = torch.randn(2, 128, 2, requires_grad=backward).to(dm.device)

    # Scatter it:
    placements = (Shard(1),)
    shard_tensor = scatter_tensor(
        full_input,
        0,
        distributed_mesh,
        placements,
        global_shape=full_input.shape,
        dtype=full_input.dtype,
        requires_grad=backward,
    )

    if verbose:
        print(
            f"Shard tensor global shape: {shard_tensor.shape} and local shape: {shard_tensor._local_tensor.shape}"
        )

    # For this test, we're testing that the reduction of the tensor works correctly

    # This means we're calling things like `shard_tensor.max()` or `shard_tensor.mean()`
    # and looking to get the right answers

    # Note that calling `full_tensor` is already checked in the initialize file but that's
    # also, technically, a reduction.

    args = ()
    kwargs = {"dim": dim}

    full_input = shard_tensor.full_tensor().detach().requires_grad_(True)

    if in_place:
        if op == "sum":
            partial_result = shard_tensor.sum(*args, **kwargs)
            full_result = full_input.sum(*args, **kwargs)
        elif op == "min":
            partial_result = shard_tensor.min(*args, **kwargs)
            full_result = full_input.min(*args, **kwargs)
        elif op == "max":
            partial_result = shard_tensor.max(*args, **kwargs)
            full_result = full_input.max(*args, **kwargs)
        elif op == "mean":
            partial_result = shard_tensor.mean(*args, **kwargs)
            full_result = full_input.mean(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported operation: {op}")
    else:
        if op == "sum":
            partial_result = torch.sum(shard_tensor, *args, **kwargs)
            full_result = torch.sum(full_input, *args, **kwargs)
        elif op == "min":
            partial_result = torch.min(shard_tensor, *args, **kwargs)
            full_result = torch.min(full_input, *args, **kwargs)
        elif op == "max":
            partial_result = torch.max(shard_tensor, *args, **kwargs)
            full_result = torch.max(full_input, *args, **kwargs)
        elif op == "mean":
            partial_result = torch.mean(shard_tensor, *args, **kwargs)
            full_result = torch.mean(full_input, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported operation: {op}")
    resolved_partial_result = partial_result.full_tensor()

    if verbose:
        print(f"Partial first: {resolved_partial_result}")
        print(f"All gather first: {full_result}")

    assert torch.allclose(resolved_partial_result, full_result, atol=1e-6)

    if backward:
        if len(full_result.shape) != 0:
            full_result.sum().backward()
        else:
            full_result.backward()

        standard_grads = full_input.grad

        if len(partial_result.shape) != 0:
            partial_result.sum().backward()
        else:
            partial_result.backward()

        sharded_grads = shard_tensor.grad.full_tensor()

        # Ensure gradient values agree:
        assert torch.allclose(standard_grads, sharded_grads)

        # Make sure that the sharded gradients have the same placement and sharding sizes as the original tensor
        assert shard_tensor.grad._spec.placements == shard_tensor._spec.placements
        assert (
            shard_tensor.grad._spec.sharding_shapes()
            == shard_tensor._spec.sharding_shapes()
        )


# @pytest.mark.multigpu
# def test_consecutive_reductions():
#     num_gpus = torch.cuda.device_count()
#     assert num_gpus >= 2, "Not enough GPUs available for test"

#     mesh_names = ["domain"]
#     mesh_sizes = [-1]

#     torch.multiprocessing.set_start_method("spawn", force=True)

#     torch.multiprocessing.spawn(
#         run_consecutive_reductions,
#         args=(
#             num_gpus,
#             mesh_names,
#             mesh_sizes,
#         ),
#         nprocs=num_gpus,
#         join=True,
#         daemon=True,
#     )
