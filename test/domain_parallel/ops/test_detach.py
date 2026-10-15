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
recipe backward. ``detach_`` must still be in place on the wrapper itself.
"""

import pytest
import torch
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor, scatter_tensor

pytestmark = [pytest.mark.multigpu_static, pytest.mark.timeout(120)]

# Uneven on 2/4/8 ranks.
_N = 19


@pytest.mark.parametrize(
    "detach_",
    [torch.Tensor.detach_, torch.ops.aten.detach_],
    ids=["method", "aten"],
)
def test_detach_inplace_op_on_sharded(distributed_mesh, detach_):
    r"""``detach_`` on a non-leaf sharded tensor detaches *that* tensor in
    place: same values, layout and local storage; autograd state dropped."""
    dm = DistributedManager()
    torch.manual_seed(7)
    a = torch.randn(_N, 3, device=dm.device)
    a_s = scatter_tensor(a, 0, distributed_mesh, (Shard(0),), requires_grad=True)

    y = a_s * 2.0
    assert y.grad_fn is not None
    spec, ptr = y._spec, y._local_tensor.untyped_storage().data_ptr()

    result = detach_(y)

    assert result is y
    assert isinstance(y, ShardTensor)
    assert y._spec.placements == spec.placements
    assert y._spec.sharding_shapes() == spec.sharding_shapes()
    assert y.requires_grad is False
    assert y.grad_fn is None
    assert y._local_tensor.untyped_storage().data_ptr() == ptr
    torch.testing.assert_close(y.full_tensor(), a * 2.0)
