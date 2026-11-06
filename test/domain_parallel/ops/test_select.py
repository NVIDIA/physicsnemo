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

"""
Test selection operations on ShardTensor.  This file tests
both torch.select and torch.index_select.  We use a 3D tensor to
do the tests, it has no special significance.

``index_select`` is covered both off the sharded dimension (a purely local
op) and along it (the routed gather, with a sharded or a replicated index).
"""

import pytest
import torch
from torch.distributed.tensor.placement_types import Replicate, Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import scatter_tensor

from .utils import numerical_shard_tensor_check


class SelectWrapper(torch.nn.Module):
    """
    Wrapper class for testing torch.select operation.
    """

    def __init__(self, target_dim: int, index: int):
        super(SelectWrapper, self).__init__()
        self.target_dim = target_dim
        self.index = index

    def forward(self, tensor: torch.Tensor):
        return torch.select(tensor, self.target_dim, self.index)


class IndexSelectWrapper(torch.nn.Module):
    """
    Wrapper class for testing torch.index_select operation.
    """

    def __init__(self, target_dim: int):
        super(IndexSelectWrapper, self).__init__()
        self.target_dim = target_dim

    def forward(self, tensor: torch.Tensor, index: torch.Tensor):
        return torch.index_select(tensor, self.target_dim, index.flatten())


@pytest.mark.multigpu_static
@pytest.mark.parametrize("backward", [False, True])
def test_select_operation(
    distributed_mesh,
    backward,
):
    """``torch.select`` on a ``Shard(2)`` tensor along an unsharded dim (local op)."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (128, 128, 128)
    target_dim = 1
    index = 2

    original_tensor = torch.rand(shape, device=dm.device, requires_grad=backward)

    placements = (Shard(2),)

    # Scatter the original tensor and index to all ranks
    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=placements,
        requires_grad=True,
    )

    module = SelectWrapper(target_dim=target_dim, index=index)

    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [
            sharded_tensor,
        ],
        {},
        check_grads=backward,
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("backward", [False, True])
def test_index_select_operation(
    distributed_mesh,
    backward,
):
    """``index_select`` off the sharded dim: the index is gathered, the shard is
    selected locally, and the output keeps the input placement."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (128, 128, 128)
    target_dim = 1
    N = 256

    original_tensor = torch.rand(shape, device=dm.device, requires_grad=backward)
    index = torch.randint(
        low=0, high=shape[target_dim] - 1, size=(N,), device=dm.device
    ).reshape(int(N / 2), -1)

    placements = (Shard(2),)

    # Scatter the original tensor and index to all ranks
    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=placements,
        requires_grad=True,
    )
    sharded_index = scatter_tensor(
        index,
        global_src=0,
        mesh=distributed_mesh,
        placements=(Shard(0),),
        requires_grad=False,
    )

    module = IndexSelectWrapper(target_dim=target_dim)

    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [sharded_tensor, sharded_index],
        {},
        check_grads=backward,
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("target_dim", [0, 1])
@pytest.mark.parametrize("backward", [False, True])
def test_index_select_along_sharded_dim(
    distributed_mesh,
    target_dim,
    backward,
):
    """``index_select`` along the sharded dim with a sharded index (routed gather).

    The index holds global positions along the sharded dim that reference
    every rank; the output is sharded like the index.
    """

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (61, 131, 8)  # uneven on 2, 4 and 8 ranks
    N = 97

    original_tensor = torch.rand(shape, device=dm.device, requires_grad=backward)
    index = torch.randint(low=0, high=shape[target_dim], size=(N,), device=dm.device)

    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=(Shard(target_dim),),
        requires_grad=backward,
    )
    sharded_index = scatter_tensor(
        index,
        global_src=0,
        mesh=distributed_mesh,
        placements=(Shard(0),),
        requires_grad=False,
    )

    def check_output(output):
        assert output._spec.placements == (Shard(target_dim),)

    numerical_shard_tensor_check(
        distributed_mesh,
        IndexSelectWrapper(target_dim=target_dim),
        [sharded_tensor, sharded_index],
        {},
        check_grads=backward,
        output_check_fn=check_output,
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("target_dim", [0, 1])
@pytest.mark.parametrize("backward", [False, True])
def test_index_select_along_sharded_dim_replicated_index(
    distributed_mesh,
    target_dim,
    backward,
):
    """``index_select`` along the sharded dim with a replicated index.

    Every rank asks for the same rows, so every rank holds the full result;
    in backward each rank routes only its share of the requests.
    """

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (61, 131, 8)
    N = 97

    original_tensor = torch.rand(shape, device=dm.device, requires_grad=backward)
    index = torch.randint(low=0, high=shape[target_dim], size=(N,), device=dm.device)

    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=(Shard(target_dim),),
        requires_grad=backward,
    )
    sharded_index = scatter_tensor(
        index,
        global_src=0,
        mesh=distributed_mesh,
        placements=(Replicate(),),
        requires_grad=False,
    )

    def check_output(output):
        assert output._spec.placements == (Replicate(),)

    numerical_shard_tensor_check(
        distributed_mesh,
        IndexSelectWrapper(target_dim=target_dim),
        [sharded_tensor, sharded_index],
        {},
        check_grads=backward,
        output_check_fn=check_output,
    )


class IndexSelectMethodWrapper(torch.nn.Module):
    """``tensor.index_select(dim=..., index=...)``: method spelling with keywords."""

    def __init__(self, target_dim: int):
        super().__init__()
        self.target_dim = target_dim

    def forward(self, tensor: torch.Tensor, index: torch.Tensor):
        return tensor.index_select(dim=self.target_dim, index=index)


@pytest.mark.multigpu_static
def test_index_select_method_spelling(distributed_mesh):
    """``Tensor.index_select`` with keyword arguments takes the same handler."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (61, 8)
    original_tensor = torch.rand(shape, device=dm.device, requires_grad=True)
    index = torch.randint(low=0, high=shape[0], size=(97,), device=dm.device)

    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=(Shard(0),),
        requires_grad=True,
    )
    sharded_index = scatter_tensor(
        index,
        global_src=0,
        mesh=distributed_mesh,
        placements=(Shard(0),),
        requires_grad=False,
    )

    numerical_shard_tensor_check(
        distributed_mesh,
        IndexSelectMethodWrapper(target_dim=0),
        [sharded_tensor, sharded_index],
        {},
        check_grads=True,
    )


# ---------------------------------------------------------------------------
# aten.select / select_backward with a negative dim (dispatch-level handlers)
# ---------------------------------------------------------------------------


@pytest.mark.multigpu_static
@pytest.mark.parametrize("dim", [-1, 4])
def test_select_negative_dim_keeps_shard_dim(distributed_mesh, dim):
    r"""The dispatch handlers see ``dim`` as given: a negative dim must not
    shift the shard dim (``Shard(2)`` stays ``Shard(2)`` when the last dim goes),
    and select_backward must insert the size at the same position."""
    dm = DistributedManager()
    full = torch.randn(2, 3, 8 * distributed_mesh.size(0), 5, 4, device=dm.device)
    st = scatter_tensor(full, 0, distributed_mesh, (Shard(2),))

    with torch._C.DisableTorchFunctionSubclass():
        out = torch.ops.aten.select.int(st, dim, 1)
        back = torch.ops.aten.select_backward.default(out, list(full.shape), dim, 1)

    assert out._spec.placements == (Shard(2),)
    assert tuple(out.shape) == (2, 3, full.shape[2], 5)
    torch.testing.assert_close(out.full_tensor(), full.select(dim, 1))
    assert back._spec.placements == (Shard(2),)
    torch.testing.assert_close(
        back.full_tensor(),
        torch.ops.aten.select_backward.default(
            full.select(dim, 1), list(full.shape), dim, 1
        ),
    )


@pytest.mark.multigpu_static
def test_select_negative_dim_on_shard_dim_rejected(distributed_mesh):
    r"""Selecting the sharded dim through a negative index is rejected too."""
    from physicsnemo.domain_parallel.shard_utils.patch_core import MissingShardPatch

    dm = DistributedManager()
    full = torch.randn(2, 8 * distributed_mesh.size(0), device=dm.device)
    st = scatter_tensor(full, 0, distributed_mesh, (Shard(1),))
    with torch._C.DisableTorchFunctionSubclass(), pytest.raises(MissingShardPatch):
        torch.ops.aten.select.int(st, -1, 0)


@pytest.mark.multigpu_static
def test_compiled_stack_backward_keeps_shard_dim(distributed_mesh):
    r"""``torch.stack(..., dim=-1)`` (RoPE's rotate-half) under compile: its
    backward selects with ``dim=-1``; the input grad must stay ``Shard(2)``."""
    dm = DistributedManager()
    full = torch.randn(1, 4, 8 * distributed_mesh.size(0), 6, 8, device=dm.device)
    st = scatter_tensor(full, 0, distributed_mesh, (Shard(2),), requires_grad=True)

    def rotate_half(x):
        return torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)

    torch._dynamo.reset()
    out = torch.compile(rotate_half, backend="aot_eager", dynamic=False)(st)
    out.square().sum().backward()

    ref = full.clone().requires_grad_(True)
    rotate_half(ref).square().sum().backward()
    assert st.grad._spec.placements == (Shard(2),)
    torch.testing.assert_close(st.grad.full_tensor(), ref.grad)
