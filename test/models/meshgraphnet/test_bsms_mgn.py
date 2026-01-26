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

from test.common import validate_forward_accuracy
from test.conftest import requires_module


@pytest.fixture
def ahmed_data_dir(nfs_data_dir):
    return nfs_data_dir.joinpath("datasets/ahmed_body")


@requires_module(["sparse_dot_mkl", "torch_geometric", "torch_scatter"])
def test_bsms_mgn_forward(pytestconfig, device, set_physicsnemo_force_te):
    import torch_geometric as pyg

    torch.manual_seed(1)

    from physicsnemo.datapipes.gnn.bsms import BistrideMultiLayerGraphDataset
    from physicsnemo.models.meshgraphnet.bsms_mgn import BiStrideMeshGraphNet

    # Create a simple graph.
    num_nodes = 8
    edges = (
        torch.arange(num_nodes - 1),
        torch.arange(num_nodes - 1) + 1,
    )
    edges = torch.stack(edges, dim=0).long()
    edges = pyg.utils.to_undirected(edges)
    pos = torch.randn((num_nodes, 3))

    graph = pyg.data.Data(edge_index=edges)

    num_layers = 2
    input_dim_nodes = 10
    input_dim_edges = 4
    output_dim = 4

    graph.pos = pos
    graph.x = torch.randn(num_nodes, input_dim_nodes)
    graph.edge_attr = torch.randn(graph.num_edges, input_dim_edges)

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
    node_features = g0.x
    edge_features = g0.edge_attr
    pred = model(node_features, edge_features, g0, ms_edges0, ms_ids0)

    # Check output shape.
    assert pred.shape == (g0.num_nodes, output_dim)

    assert validate_forward_accuracy(
        model,
        (node_features, edge_features, g0, ms_edges0, ms_ids0),
        file_name="models/data/bistridemeshgraphnet_output.pth",
    )


def test_bsms_mgn_constructor(device, pytestconfig, set_physicsnemo_force_te):
    """Test BiStrideMeshGraphNet constructor: public attributes are set correctly"""
    from physicsnemo.models.meshgraphnet.bsms_mgn import BiStrideMeshGraphNet

    kw = dict(
        input_dim_nodes=7,
        input_dim_edges=5,
        output_dim=3,
        processor_size=4,
        mlp_activation_fn="relu",
        num_layers_node_processor=2,
        num_layers_edge_processor=2,
        num_mesh_levels=3,
        bistride_pos_dim=3,
        num_layers_bistride=2,
        bistride_unet_levels=2,
        hidden_dim_processor=64,
        hidden_dim_node_encoder=32,
        num_layers_node_encoder=2,
        hidden_dim_edge_encoder=16,
        num_layers_edge_encoder=1,
        hidden_dim_node_decoder=32,
        num_layers_node_decoder=1,
        aggregation="sum",
        do_concat_trick=False,
        num_processor_checkpoint_segments=0,
        recompute_activation=False,
    )

    model = BiStrideMeshGraphNet(**kw).to(device)

    # Public attributes reflect constructor args
    assert model.input_dim_nodes == kw["input_dim_nodes"]
    assert model.input_dim_edges == kw["input_dim_edges"]
    assert model.output_dim == kw["output_dim"]
    assert model.bistride_unet_levels == kw["bistride_unet_levels"]

    # Key submodules exist
    assert hasattr(model, "edge_encoder")
    assert hasattr(model, "node_encoder")
    assert hasattr(model, "processor")
    assert hasattr(model, "node_decoder")
    assert hasattr(model, "bistride_processor")


@requires_module(["sparse_dot_mkl", "torch_geometric", "torch_scatter"])
def test_bsms_mgn_shape_validation(pytestconfig, device, set_physicsnemo_force_te):
    """Test shape validation errors for BiStrideMeshGraphNet.forward"""
    import torch_geometric as pyg

    from physicsnemo.models.meshgraphnet.bsms_mgn import BiStrideMeshGraphNet

    model = BiStrideMeshGraphNet(
        input_dim_nodes=4,
        input_dim_edges=3,
        output_dim=2,
        processor_size=2,
        hidden_dim_processor=16,
        hidden_dim_node_encoder=8,
        hidden_dim_edge_encoder=8,
        num_layers_bistride=1,
    ).to(device)
    model.eval()

    # Simple line graph
    num_nodes = 6
    edges = torch.stack(
        [torch.arange(num_nodes - 1), torch.arange(num_nodes - 1) + 1], dim=0
    ).long()
    edges = pyg.utils.to_undirected(edges)
    graph = pyg.data.Data(edge_index=edges).to(device)

    good_node = torch.randn(num_nodes, 4).to(device)
    good_edge = torch.randn(graph.num_edges, 3).to(device)
    ms_edges = []
    ms_ids = []

    # Wrong node feature dimension
    bad_node = torch.randn(num_nodes, 5).to(device)
    with pytest.raises(ValueError, match=r"Expected tensor of shape \(N_nodes, 4\)"):
        _ = model(bad_node, good_edge, graph, ms_edges, ms_ids)

    # Wrong edge feature dimension
    bad_edge = torch.randn(graph.num_edges, 2).to(device)
    with pytest.raises(ValueError, match=r"Expected tensor of shape \(N_edges, 3\)"):
        _ = model(good_node, bad_edge, graph, ms_edges, ms_ids)

    # Wrong node feature rank (ndim)
    bad_node_rank = torch.randn(2, num_nodes, 4).to(device)
    with pytest.raises(ValueError, match=r"Expected tensor of shape \(N_nodes, 4\)"):
        _ = model(bad_node_rank, good_edge, graph, ms_edges, ms_ids)

    # Wrong edge feature rank (ndim)
    bad_edge_rank = torch.randn(2, graph.num_edges, 3).to(device)
    with pytest.raises(ValueError, match=r"Expected tensor of shape \(N_edges, 3\)"):
        _ = model(good_node, bad_edge_rank, graph, ms_edges, ms_ids)


@requires_module(["sparse_dot_mkl", "torch_geometric", "torch_scatter"])
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
    pred = model(g0.x, g0.edge_attr, g0, ms_edges0, ms_ids0)

    # Check output shape.
    assert pred.shape == (g0.num_nodes, output_dim)
