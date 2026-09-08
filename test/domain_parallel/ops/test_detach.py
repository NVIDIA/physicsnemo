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

r"""``aten.detach_`` handler on ShardTensor (``custom_ops/_tensor_ops.py``).

DTensor registers no sharding strategy for the in-place ``detach_``, so the
fallback route raises ``NotImplementedError`` -- AOT autograd's joint tracing
emits ``detach_`` on ShardTensors, which broke the compiled domain-parallel
recipe backward. The handler detaches the local tensor and rewraps with the
same spec.
"""

import pytest
import torch
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor, scatter_tensor

pytestmark = [pytest.mark.multigpu_static, pytest.mark.timeout(120)]

# Uneven on 2/4/8 ranks.
_N = 19


def test_detach_inplace_op_on_sharded(distributed_mesh):
    r"""``aten.detach_`` on a non-leaf sharded tensor: same values, same
    layout, autograd state dropped, local storage shared."""
    dm = DistributedManager()
    torch.manual_seed(7)
    a = torch.randn(_N, 3, device=dm.device)
    a_s = scatter_tensor(a, 0, distributed_mesh, (Shard(0),), requires_grad=True)

    y = a_s * 2.0
    assert y.grad_fn is not None

    result = torch.ops.aten.detach_(y)

    assert isinstance(result, ShardTensor)
    assert result._spec.placements == y._spec.placements
    assert result._spec.sharding_shapes() == y._spec.sharding_shapes()
    assert result.requires_grad is False
    assert result.grad_fn is None
    assert (
        result._local_tensor.untyped_storage().data_ptr()
        == y._local_tensor.untyped_storage().data_ptr()
    )
    torch.testing.assert_close(result.full_tensor(), a * 2.0)
    # The handler returns a new wrapper rather than mutating ``y`` in place:
    # re-entering ``aten.detach_`` on the wrapper from inside its own dispatch
    # would recurse, and the caller (AOT functionalization) consumes the
    # returned value. ``y`` therefore keeps its autograd state.
    assert y.grad_fn is not None
