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

import numpy
import pytest
import torch
from torch.distributed.tensor import distribute_module
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import scatter_tensor
from physicsnemo.models.transolver import Transolver


@pytest.mark.multigpu_static
@pytest.mark.parametrize("n_dims", [2, 3])
def test_transolver_nd_distributed(
    distributed_mesh,
    n_dims,
):
    """Test transolver 2D and 3D distributed forward pass"""

    dm = DistributedManager()

    spatial_dims = (128,) * n_dims

    # Construct transolver model
    model = Transolver(
        structured_shape=spatial_dims,
        n_layers=8,
        n_hidden=64,
        dropout=0,
        n_head=4,
        time_input=False,
        act="gelu",
        mlp_ratio=1,
        functional_dim=3,
        embedding_dim=5,
        out_dim=2,
        slice_num=32,
        ref=1,
        unified_pos=False,
        use_te=False,
    ).to(dm.device)

    # Create data:
    image_embedding = torch.randn(1, *spatial_dims, 5).to(dm.device)
    functional_input = torch.randn(1, *spatial_dims, 3).to(dm.device)

    # Scatter the data
    placements = (Shard(1),)

    sharded_image_embedding = scatter_tensor(
        image_embedding, 0, distributed_mesh, placements, requires_grad=False
    )
    sharded_functional_input = scatter_tensor(
        functional_input, 0, distributed_mesh, placements, requires_grad=False
    )

    sharded_image_embedding = sharded_image_embedding.reshape(1, -1, 5)
    sharded_functional_input = sharded_functional_input.reshape(1, -1, 3)

    model = distribute_module(model, device_mesh=distributed_mesh)

    # Run model
    output = model(sharded_image_embedding, sharded_functional_input)

    # Check output
    assert output.shape == (1, numpy.prod(spatial_dims), 2)

    # Make sure the output is sharded, too:
    assert output._spec.placements == sharded_image_embedding._spec.placements


@pytest.mark.multigpu_static
def test_transolver_irregular_distributed(
    distributed_mesh,
):
    """Test transolver irregular distributed forward pass"""

    dm = DistributedManager()

    spatial_dims = (16384,)

    # Construct transolver model
    model = Transolver(
        structured_shape=None,
        n_layers=8,
        n_hidden=64,
        dropout=0,
        n_head=4,
        time_input=False,
        act="gelu",
        mlp_ratio=1,
        functional_dim=3,
        embedding_dim=5,
        out_dim=2,
        slice_num=32,
        ref=1,
        unified_pos=False,
        use_te=False,
    ).to(dm.device)

    # Create data:
    image_embedding = torch.randn(1, *spatial_dims, 5).to(dm.device)
    functional_input = torch.randn(1, *spatial_dims, 3).to(dm.device)

    # Scatter the data
    placements = (Shard(1),)

    sharded_image_embedding = scatter_tensor(
        image_embedding, 0, distributed_mesh, placements, requires_grad=False
    )
    sharded_functional_input = scatter_tensor(
        functional_input, 0, distributed_mesh, placements, requires_grad=False
    )

    # Distribute the model to DTensor:
    model = distribute_module(model, device_mesh=distributed_mesh)

    # Run model
    output = model(sharded_image_embedding, sharded_functional_input)

    # Check output
    assert output.shape == (1, *spatial_dims, 2)

    # Make sure the output is sharded, too:
    assert output._spec.placements == sharded_image_embedding._spec.placements
