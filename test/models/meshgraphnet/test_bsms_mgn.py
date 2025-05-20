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
from models.common import validate_forward_accuracy
from pytest_utils import import_or_fail

dgl = pytest.importorskip("dgl")


@pytest.fixture
def ahmed_data_dir(nfs_data_dir):
    return nfs_data_dir.joinpath("datasets/ahmed_body")


@import_or_fail(["sparse_dot_mkl", "dgl"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_bsms_mgn_forward(pytestconfig, device):
    from physicsnemo.datapipes.gnn.bsms import BistrideMultiLayerGraphDataset
    from physicsnemo.models.meshgraphnet.bsms_mgn import BiStrideMeshGraphNet

    torch.manual_seed(1)

    # Create a simple graph.
    num_nodes = 8
    edges = (
        torch.arange(num_nodes - 1),
        torch.arange(num_nodes - 1) + 1,
    )
    pos = torch.randn((num_nodes, 3))

    graph = dgl.graph(edges)
    graph = dgl.to_bidirected(graph)

    num_layers = 2
    input_dim_nodes = 10
    input_dim_edges = 4
    output_dim = 4

    graph.ndata["pos"] = pos
    graph.ndata["x"] = torch.randn(num_nodes, input_dim_nodes)
    graph.edata["x"] = torch.randn(graph.num_edges(), input_dim_edges)

    dataset = BistrideMultiLayerGraphDataset([graph], num_layers)
    assert len(dataset) == 1

    # Create a model.
    model = BiStrideMeshGraphNet(
        input_dim_nodes=input_dim_nodes,
        input_dim_edges=input_dim_edges,
        output_dim=output_dim,
        num_layers_bistride=num_layers,
        processor_size=2,
        hidden_dim_processor=32,
        hidden_dim_node_encoder=16,
        hidden_dim_edge_encoder=16,
    ).to(device)
    model.eval()

    s0 = dataset[0]
    g0 = s0["graph"].to(device)
    ms_edges0 = s0["ms_edges"]
    ms_ids0 = s0["ms_ids"]
    node_features = g0.ndata["x"]
    edge_features = g0.edata["x"]
    pred = model(node_features, edge_features, g0, ms_edges0, ms_ids0)

    # Check output shape.
    assert pred.shape == (g0.num_nodes(), output_dim)

    assert validate_forward_accuracy(
        model,
        (node_features, edge_features, g0, ms_edges0, ms_ids0),
    )


@import_or_fail(["sparse_dot_mkl", "dgl"])
def test_bsms_mgn_ahmed(pytestconfig, ahmed_data_dir):
    from physicsnemo.datapipes.gnn.ahmed_body_dataset import AhmedBodyDataset
    from physicsnemo.datapipes.gnn.bsms import BistrideMultiLayerGraphDataset
    from physicsnemo.models.meshgraphnet.bsms_mgn import BiStrideMeshGraphNet

    device = torch.device("cuda:0")

    torch.manual_seed(1)

    # Construct multi-scale dataset out of standard Ahmed Body dataset.
    ahmed_dataset = AhmedBodyDataset(
        data_dir=ahmed_data_dir,
        split="train",
        num_samples=2,
    )

    num_levels = 2
    dataset = BistrideMultiLayerGraphDataset(ahmed_dataset, num_levels)

    output_dim = 4
    # Construct model.
    model = BiStrideMeshGraphNet(
        input_dim_nodes=11,
        input_dim_edges=4,
        output_dim=output_dim,
        processor_size=2,
        hidden_dim_processor=32,
        hidden_dim_node_encoder=16,
        hidden_dim_edge_encoder=16,
    ).to(device)

    s0 = dataset[0]
    g0 = s0["graph"].to(device)
    ms_edges0 = s0["ms_edges"]
    ms_ids0 = s0["ms_ids"]
    pred = model(g0.ndata["x"], g0.edata["x"], g0, ms_edges0, ms_ids0)

    # Check output shape.
    assert pred.shape == (g0.num_nodes(), output_dim)
