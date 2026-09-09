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

r"""Tests for ``F.linear`` on ShardTensor inputs sharded on non-feature dims.

The critical case is a rank-4 **unevenly** sharded input, which the DTensor
fallback rejects (its linear decomposition flattens leading dims and cannot
represent the resulting uneven chunks; see ``shard_utils/linear_patches.py``).
3-D inputs are covered as the regression guard for the DoMINO / transolver
shapes that predate the handler.
"""

import pytest
import torch
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import scatter_tensor

from .utils import numerical_shard_tensor_check

# Not divisible by 2/4/8: exercises uneven sharding on every world size.
_N_UNEVEN = 2345
_N_EVEN = 2048


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
@pytest.mark.parametrize("n_points", [_N_EVEN, _N_UNEVEN], ids=["even", "uneven"])
@pytest.mark.parametrize("shape", ["3d", "4d"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("backward", [False, True])
def test_linear_sharded_points(distributed_mesh, n_points, shape, bias, backward):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()

    if shape == "3d":
        full = torch.randn(1, n_points, 8, device=dm.device)
    else:
        full = torch.randn(1, n_points, 4, 8, device=dm.device)

    sharded = scatter_tensor(
        full, 0, distributed_mesh, (Shard(1),), requires_grad=backward
    )
    module = torch.nn.Linear(8, 32, bias=bias).to(dm.device)

    numerical_shard_tensor_check(
        distributed_mesh, module, [sharded], {}, check_grads=backward
    )
