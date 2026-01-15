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

import pytest
import torch
from torch.distributed.tensor.placement_types import Replicate, Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import scatter_tensor
from physicsnemo.nn.neighbors import knn

from .utils import numerical_shard_tensor_check


class kNNModule(torch.nn.Module):
    def __init__(
        self,
        num_neighbors=4,
    ):
        super().__init__()

        self.num_neighbors = num_neighbors

    def forward(self, points, queries):
        return knn(points, queries, self.num_neighbors)


@pytest.mark.multigpu_static
@pytest.mark.parametrize("scatter_points", [True, False])
@pytest.mark.parametrize("scatter_queries", [True, False])
def test_knn_1dmesh(
    distributed_mesh,
    scatter_points: bool,
    scatter_queries: bool,
):
    dm = DistributedManager()

    # Generate random points for the points and queries
    # Use sizes divisible by common GPU counts (2, 4, 8) to avoid edge cases
    points = torch.randn(1025, 3).to(dm.device)
    queries = torch.randn(2048, 3).to(dm.device)

    # Distribute the inputs:
    points_placements = (Shard(0),) if scatter_points else (Replicate(),)
    queries_placements = (Shard(0),) if scatter_queries else (Replicate(),)

    sharded_points = scatter_tensor(points, 0, distributed_mesh, points_placements)
    sharded_queries = scatter_tensor(queries, 0, distributed_mesh, queries_placements)

    print(f"Sharded points: {sharded_points} on rank {dm.rank}")

    print(f"sharding shapes: {sharded_points._spec.sharding_shapes()}")
    print(f"sharding shapes: {sharded_queries._spec.sharding_shapes()}")

    module = kNNModule()

    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [sharded_points, sharded_queries],
        {},
        check_grads=False,
    )
