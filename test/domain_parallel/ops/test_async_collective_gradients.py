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

r"""Regression tests for asynchronous functional-collective gradients."""

import pytest
import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed.tensor.placement_types import Replicate, Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import scatter_tensor
from physicsnemo.domain_parallel.shard_utils.grad_ops import GradReducer


def _assert_plain_gradient(grad: torch.Tensor | None) -> None:
    r"""Assert that a gradient is materialized rather than asynchronously wrapped."""
    assert grad is not None
    assert not isinstance(grad, funcol.AsyncCollectiveTensor)
    assert type(grad) is torch.Tensor


def _work_registry_size() -> int:
    r"""Return the number of functional collectives still awaiting a wait."""
    return torch._C._distributed_c10d._get_work_registry_size()


@pytest.mark.multigpu_static
@pytest.mark.parametrize(
    ("placement", "reduces"),
    [
        pytest.param(Shard(0), True, id="sharded-ref"),
        pytest.param(Replicate(), False, id="replicated-ref"),
    ],
)
def test_grad_reducer_waits_before_return(distributed_mesh, placement, reduces):
    r"""GradReducer must not return an AsyncCollectiveTensor to autograd.

    It all-reduces over the ref spec's sharded mesh dims and is the identity
    when the ref spec is replicated.
    """
    initial_registry_size = _work_registry_size()
    dm = DistributedManager()
    source = torch.ones(8, device=dm.device)
    sharded = scatter_tensor(
        source,
        global_src=0,
        mesh=distributed_mesh,
        placements=(placement,),
        requires_grad=False,
    )

    leaf = torch.ones(8, device=dm.device, requires_grad=True)
    hook_grads = []
    leaf.register_hook(hook_grads.append)

    GradReducer.apply(leaf, sharded._spec).sum().backward()

    assert len(hook_grads) == 1
    _assert_plain_gradient(hook_grads[0])
    _assert_plain_gradient(leaf.grad)
    expected = distributed_mesh.size() if reduces else 1.0
    torch.testing.assert_close(leaf.grad, torch.full_like(leaf, expected))
    assert _work_registry_size() == initial_registry_size


@pytest.mark.multigpu_static
def test_group_norm_waits_for_parameter_gradients(distributed_mesh):
    r"""GroupNorm weight and bias gradients must be materialized before return."""
    initial_registry_size = _work_registry_size()
    dm = DistributedManager()
    image = torch.randn(2, 4, 16, device=dm.device)
    sharded_image = scatter_tensor(
        image,
        global_src=0,
        mesh=distributed_mesh,
        placements=(Shard(2),),
        requires_grad=True,
    )
    module = torch.nn.GroupNorm(num_groups=2, num_channels=4).to(dm.device)

    hook_grads = {name: [] for name in ("weight", "bias")}
    module.weight.register_hook(hook_grads["weight"].append)
    module.bias.register_hook(hook_grads["bias"].append)

    module(sharded_image).mean().backward()

    for name, parameter in module.named_parameters():
        assert len(hook_grads[name]) == 1
        _assert_plain_gradient(hook_grads[name][0])
        _assert_plain_gradient(parameter.grad)
    assert _work_registry_size() == initial_registry_size
