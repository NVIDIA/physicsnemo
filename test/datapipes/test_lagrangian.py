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
import torch
from pytest_utils import import_or_fail

from . import common

dgl = pytest.importorskip("dgl")


Tensor = torch.Tensor


@pytest.fixture
def data_dir(nfs_data_dir):
    return nfs_data_dir.joinpath("datasets/water")


@import_or_fail(["tensorflow", "dgl"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_lagrangian_dataset_constructor(data_dir, device, pytestconfig):
    from physicsnemo.datapipes.gnn.lagrangian_dataset import LagrangianDataset

    # Test successful construction
    dataset = LagrangianDataset(
        data_dir=data_dir,
        split="valid",
        num_sequences=2,  # Use a small number for testing
        num_steps=10,  # Use a small number for testing
    )

    # iterate datapipe is iterable
    common.check_datapipe_iterable(dataset)

    # Test getting an item
    graph = dataset[0]
    # new DGL (2.4+) uses dgl.heterograph.DGLGraph, previous DGL is dgl.DGLGraph
    assert isinstance(graph, dgl.DGLGraph) or isinstance(
        graph, dgl.heterograph.DGLGraph
    )

    # Test graph properties
    assert "x" in graph.ndata
    assert "y" in graph.ndata
    assert graph.ndata["x"].shape[-1] > 0  # node features
    assert graph.ndata["y"].shape[-1] > 0  # node targets


@import_or_fail(["tensorflow", "dgl"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_graph_construction(device, pytestconfig):
    from physicsnemo.datapipes.gnn.lagrangian_dataset import compute_edge_index

    mesh_pos = torch.tensor([[0.0, 0.0], [0.01, 0.0], [1.0, 1.0]], device=device)
    radius = 0.015

    edge_index = compute_edge_index(mesh_pos, radius)

    # Check connectivity
    assert any((edge_index[0] == 0) & (edge_index[1] == 1))
    assert any((edge_index[0] == 1) & (edge_index[1] == 0))
    assert not any((edge_index[0] == 0) & (edge_index[1] == 2))
