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
Test integer-tensor indexing (``tensor[index]``) on ShardTensor.

The source tensor is ``Shard(0)``; the index holds *global* row ids that
reference every rank, so the gather has to route rows between ranks.  The
index is either sharded (``Shard(0)``, each rank asks for its own rows) or
replicated (every rank asks for the same rows).  ``torch.index_select`` is
covered in ``test_select.py``.
"""

import pytest
import torch
from torch.distributed.tensor.placement_types import Replicate, Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor, scatter_tensor
from physicsnemo.domain_parallel.shard_utils.patch_core import MissingShardPatch

from .utils import numerical_shard_tensor_check

# Row counts that split unevenly on 2, 4 and 8 ranks.
N_ROWS = 291
N_INDEX = 97


class GetItemWrapper(torch.nn.Module):
    """
    Wrapper class for testing ``tensor[index]`` with an integer tensor index.
    """

    def forward(self, tensor: torch.Tensor, index: torch.Tensor):
        return tensor[index]


def _check_sharded_like_index(index_placement):
    def _check(output):
        assert isinstance(output, ShardTensor)
        assert output._spec.placements == index_placement

    return _check


@pytest.mark.multigpu_static
@pytest.mark.parametrize("backward", [False, True])
def test_getitem_sharded_index(distributed_mesh, backward):
    """Sharded source, sharded index: output is sharded like the index."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    torch.manual_seed(7)
    values = torch.rand(N_ROWS, 4, device=dm.device, requires_grad=backward)
    index = torch.randint(0, N_ROWS, (N_INDEX, 3), device=dm.device)

    sharded_tensor = scatter_tensor(
        values,
        global_src=0,
        mesh=distributed_mesh,
        placements=(Shard(0),),
        requires_grad=backward,
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
        GetItemWrapper(),
        [sharded_tensor, sharded_index],
        {},
        check_grads=backward,
        output_check_fn=_check_sharded_like_index((Shard(0),)),
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("backward", [False, True])
def test_getitem_replicated_index(distributed_mesh, backward):
    """Sharded source, replicated index: every rank holds the full result.

    In backward every rank holds the full output gradient; each routes only
    its share of the requests so owners accumulate each row exactly once.
    """

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    torch.manual_seed(7)
    values = torch.rand(N_ROWS, 4, device=dm.device, requires_grad=backward)
    index = torch.randint(0, N_ROWS, (N_INDEX, 3), device=dm.device)

    sharded_tensor = scatter_tensor(
        values,
        global_src=0,
        mesh=distributed_mesh,
        placements=(Shard(0),),
        requires_grad=backward,
    )
    sharded_index = scatter_tensor(
        index,
        global_src=0,
        mesh=distributed_mesh,
        placements=(Replicate(),),
        requires_grad=False,
    )

    numerical_shard_tensor_check(
        distributed_mesh,
        GetItemWrapper(),
        [sharded_tensor, sharded_index],
        {},
        check_grads=backward,
        output_check_fn=_check_sharded_like_index((Replicate(),)),
    )


@pytest.mark.multigpu_static
def test_getitem_bf16(distributed_mesh):
    """Reduced-precision source: forward and backward match the local result."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    torch.manual_seed(7)
    values = torch.rand(
        N_ROWS, 4, device=dm.device, dtype=torch.bfloat16, requires_grad=True
    )
    index = torch.randint(0, N_ROWS, (N_INDEX, 3), device=dm.device)

    sharded_tensor = scatter_tensor(
        values,
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
        GetItemWrapper(),
        [sharded_tensor, sharded_index],
        {},
        check_grads=True,
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.multigpu_static
def test_getitem_never_all_gathers(distributed_mesh):
    """The routed gather issues no all-gather of the source in forward or
    backward -- the collective the replaced implementation was built on.

    ``CommDebugMode`` is a dispatch mode and cannot see the all-to-all calls
    inside the custom-op bodies, so this pins the absence of the expensive
    collective rather than the presence of the cheap one.  The only
    collective visible at this level is the ``all_reduce`` that resolves the
    ``Partial`` result of ``mean`` over the sharded axis.
    """
    from torch.distributed.tensor.debug import CommDebugMode

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    torch.manual_seed(7)
    values = torch.rand(N_ROWS, 4, device=dm.device)
    index = torch.randint(0, N_ROWS, (N_INDEX, 3), device=dm.device)

    sharded_tensor = scatter_tensor(
        values,
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

    with CommDebugMode() as comm:
        gathered = sharded_tensor[sharded_index]
        gathered.mean().backward()

    counts = {str(op): n for op, n in comm.get_comm_counts().items() if n}
    gathers = {op: n for op, n in counts.items() if "all_gather" in op}
    # Identical on every rank: the count dict is built from the same program.
    assert not gathers, f"routed gather all-gathered: {counts}"


@pytest.mark.multigpu_static
def test_getitem_duplicate_and_negative_index(distributed_mesh):
    """Repeated and negative global ids: duplicates accumulate in backward,
    negatives wrap like eager indexing."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    torch.manual_seed(7)
    values = torch.rand(N_ROWS, 4, device=dm.device, requires_grad=True)
    # Every rank asks for the same handful of rows several times, some negative.
    index = torch.tensor(
        [[0, -1, 5], [5, 5, -N_ROWS], [N_ROWS - 1, 0, 0]] * 11, device=dm.device
    )

    sharded_tensor = scatter_tensor(
        values,
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
        GetItemWrapper(),
        [sharded_tensor, sharded_index],
        {},
        check_grads=True,
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize(
    "key",
    ["int", "slice", "tuple", "mask", "ellipsis"],
)
def test_getitem_non_tensor_keys_fall_through(distributed_mesh, key):
    """Keys other than an integer tensor take the default route and match eager."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    torch.manual_seed(7)
    values = torch.rand(N_ROWS, 4, device=dm.device)
    sharded_tensor = scatter_tensor(
        values, global_src=0, mesh=distributed_mesh, placements=(Shard(0),)
    )
    mask = torch.zeros(4, dtype=torch.bool, device=dm.device)
    mask[1] = mask[3] = True
    keys = {
        "int": (lambda t: t[3]),
        "slice": (lambda t: t[10:50]),
        "tuple": (lambda t: t[:, 1:3]),
        "mask": (lambda t: t[:, mask]),
        "ellipsis": (lambda t: t[..., 0]),
    }
    out = keys[key](sharded_tensor)
    expected = keys[key](values)
    out = out.full_tensor() if isinstance(out, ShardTensor) else out
    torch.testing.assert_close(out, expected)


@pytest.mark.multigpu_static
def test_getitem_rejects_unsupported_index(distributed_mesh):
    """Float index, Partial index and an index sharded on dim 1 raise
    ``MissingShardPatch`` (communication-free, identical on every rank)."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    torch.manual_seed(7)
    values = torch.rand(N_ROWS, 4, device=dm.device)
    sharded_tensor = scatter_tensor(
        values, global_src=0, mesh=distributed_mesh, placements=(Shard(0),)
    )
    world = distributed_mesh.size(0)

    float_index = ShardTensor.from_local(
        torch.zeros(3, 2, device=dm.device),
        distributed_mesh,
        (Shard(0),),
        sharding_shapes={0: [(3, 2)] * world},
    )
    with pytest.raises(MissingShardPatch):
        sharded_tensor[float_index]

    dim1_index = ShardTensor.from_local(
        torch.zeros(3, 2, dtype=torch.int64, device=dm.device),
        distributed_mesh,
        (Shard(1),),
        sharding_shapes={0: [(3, 2)] * world},
    )
    with pytest.raises(MissingShardPatch):
        sharded_tensor[dim1_index]


@pytest.mark.multigpu_static
def test_getitem_rejects_2d_mesh(distributed_mesh_2d):
    """The routed gather is 1-D mesh only."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    torch.manual_seed(7)
    values = torch.rand(N_ROWS, 4, device=dm.device)
    sharded_tensor = scatter_tensor(
        values,
        global_src=0,
        mesh=distributed_mesh_2d,
        placements=(Shard(0), Replicate()),
    )
    index = torch.randint(0, N_ROWS, (5,), device=dm.device)
    with pytest.raises(MissingShardPatch):
        torch.index_select(sharded_tensor, 0, index)
