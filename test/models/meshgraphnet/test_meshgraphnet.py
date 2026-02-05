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
# ruff: noqa: E402
import random

import numpy as np
import pytest
import torch

pytest.importorskip("torch_geometric")

from test import common
from test.conftest import requires_module
from test.models.meshgraphnet.utils import rand_graph


@requires_module("torch_geometric")
def test_meshgraphnet_forward(device, pytestconfig, set_physicsnemo_force_te):
    """Test mehsgraphnet forward pass"""

    import torch_geometric as pyg

    from physicsnemo.models.meshgraphnet import MeshGraphNet

    torch.manual_seed(0)
    np.random.seed(0)
    # Construct MGN model
    model = MeshGraphNet(
        input_dim_nodes=4,
        input_dim_edges=3,
        output_dim=2,
    ).to(device)

    bsize = 2
    num_nodes, num_edges = 20, 10
    # NOTE dgl's random graph generator does not behave consistently even after fixing dgl's random seed.
    # Instead, numpy adj matrices are created in COO format and are then converted to dgl graphs.
    graphs = []
    for _ in range(bsize):
        src = torch.tensor([np.random.randint(num_nodes) for _ in range(num_edges)])
        dst = torch.tensor([np.random.randint(num_nodes) for _ in range(num_edges)])
        graphs.append(
            pyg.data.Data(
                edge_index=torch.stack([src, dst], dim=0),
                num_nodes=num_nodes,
            ).to(device)
        )
    graph = pyg.data.Batch.from_data_list(graphs)
    node_features = torch.randn(graph.num_nodes, 4).to(device)
    edge_features = torch.randn(graph.num_edges, 3).to(device)
    assert common.validate_forward_accuracy(
        model,
        (node_features, edge_features, graph),
        file_name="models/meshgraphnet/data/meshgraphnet_output.pth",
    )


@requires_module("torch_geometric")
def test_mehsgraphnet_constructor(device, pytestconfig, set_physicsnemo_force_te):
    """Test mehsgraphnet constructor options"""
    import torch_geometric as pyg

    # Define dictionary of constructor args
    arg_list = [
        {
            "input_dim_nodes": random.randint(1, 10),
            "input_dim_edges": random.randint(1, 4),
            "output_dim": random.randint(1, 10),
            "processor_size": random.randint(1, 15),
            "num_layers_node_processor": 2,
            "num_layers_edge_processor": 2,
            "hidden_dim_node_encoder": 256,
            "num_layers_node_encoder": 2,
            "hidden_dim_edge_encoder": 256,
            "num_layers_edge_encoder": 2,
            "hidden_dim_node_decoder": 256,
            "num_layers_node_decoder": 2,
        },
        {
            "input_dim_nodes": random.randint(1, 5),
            "input_dim_edges": random.randint(1, 8),
            "output_dim": random.randint(1, 5),
            "processor_size": random.randint(1, 15),
            "num_layers_node_processor": 1,
            "num_layers_edge_processor": 1,
            "hidden_dim_node_encoder": 128,
            "num_layers_node_encoder": 1,
            "hidden_dim_edge_encoder": 128,
            "num_layers_edge_encoder": 1,
            "hidden_dim_node_decoder": 128,
            "num_layers_node_decoder": 1,
        },
    ]

    from physicsnemo.models.meshgraphnet import MeshGraphNet

    for kw_args in arg_list:
        # Construct mehsgraphnet model
        model = MeshGraphNet(**kw_args).to(device)

        bsize = random.randint(1, 16)
        num_nodes, num_edges = random.randint(10, 25), random.randint(10, 20)
        graph = pyg.data.Batch.from_data_list(
            [rand_graph(num_nodes, num_edges, device) for _ in range(bsize)]
        )
        node_features = torch.randn(bsize * num_nodes, kw_args["input_dim_nodes"]).to(
            device
        )
        edge_features = torch.randn(bsize * num_edges, kw_args["input_dim_edges"]).to(
            device
        )
        outvar = model(node_features, edge_features, graph)
        assert outvar.shape == (bsize * num_nodes, kw_args["output_dim"])

        # Check public attributes reflect constructor args
        assert model.input_dim_nodes == kw_args["input_dim_nodes"]
        assert model.input_dim_edges == kw_args["input_dim_edges"]
        assert model.output_dim == kw_args["output_dim"]

        # Check key submodules exist
        assert hasattr(model, "edge_encoder")
        assert hasattr(model, "node_encoder")
        assert hasattr(model, "processor")
        assert hasattr(model, "node_decoder")


@requires_module("torch_geometric")
def test_meshgraphnet_optims(device, pytestconfig, set_physicsnemo_force_te):
    """Test meshgraphnet optimizations"""

    import torch_geometric as pyg

    from physicsnemo.models.meshgraphnet import MeshGraphNet

    def setup_model():
        """Set up fresh model and inputs for each optim test"""
        # Construct MGN model
        model = MeshGraphNet(
            input_dim_nodes=2,
            input_dim_edges=2,
            output_dim=2,
        ).to(device)

        bsize = random.randint(1, 8)
        num_nodes, num_edges = random.randint(15, 30), random.randint(15, 25)
        graph = pyg.data.Batch.from_data_list(
            [rand_graph(num_nodes, num_edges, device) for _ in range(bsize)]
        )
        node_features = torch.randn(bsize * num_nodes, 2).to(device)
        edge_features = torch.randn(bsize * num_edges, 2).to(device)
        return model, [node_features, edge_features, graph]

    # Ideally always check graphs first
    model, invar = setup_model()
    assert common.validate_cuda_graphs(model, (*invar,))
    # Check JIT
    model, invar = setup_model()
    assert common.validate_jit(model, (*invar,))
    # Check AMP
    model, invar = setup_model()
    assert common.validate_amp(model, (*invar,))
    # Check Combo
    model, invar = setup_model()
    assert common.validate_combo_optims(model, (*invar,))


@requires_module("torch_geometric")
def test_meshgraphnet_checkpoint(device, pytestconfig, set_physicsnemo_force_te):
    """Test meshgraphnet checkpoint save/load"""

    import torch_geometric as pyg

    from physicsnemo.models.meshgraphnet import MeshGraphNet

    # Construct MGN model
    model_1 = MeshGraphNet(
        input_dim_nodes=4,
        input_dim_edges=3,
        output_dim=4,
    ).to(device)

    model_2 = MeshGraphNet(
        input_dim_nodes=4,
        input_dim_edges=3,
        output_dim=4,
    ).to(device)

    bsize = random.randint(1, 8)
    num_nodes, num_edges = random.randint(5, 15), random.randint(10, 25)
    graph = pyg.data.Batch.from_data_list(
        [rand_graph(num_nodes, num_edges, device) for _ in range(bsize)]
    )
    node_features = torch.randn(bsize * num_nodes, 4).to(device)
    edge_features = torch.randn(bsize * num_edges, 3).to(device)
    assert common.validate_checkpoint(
        model_1,
        model_2,
        (
            node_features,
            edge_features,
            graph,
        ),
    )


@requires_module("torch_geometric")
@common.check_ort_version()
def test_meshgraphnet_deploy(device, pytestconfig, set_physicsnemo_force_te):
    """Test mesh-graph net deployment support"""

    import torch_geometric as pyg

    from physicsnemo.models.meshgraphnet import MeshGraphNet

    # Construct MGN model
    model = MeshGraphNet(
        input_dim_nodes=4,
        input_dim_edges=3,
        output_dim=4,
    ).to(device)

    bsize = random.randint(1, 8)
    num_nodes, num_edges = random.randint(5, 10), random.randint(10, 15)
    graph = pyg.data.Batch.from_data_list(
        [rand_graph(num_nodes, num_edges, device) for _ in range(bsize)]
    )
    node_features = torch.randn(bsize * num_nodes, 4).to(device)
    edge_features = torch.randn(bsize * num_edges, 3).to(device)
    invar = (
        node_features,
        edge_features,
        graph,
    )
    assert common.validate_onnx_export(model, invar)
    assert common.validate_onnx_runtime(model, invar)


@requires_module("torch_geometric")
def test_meshgraphnet_shape_validation(device, pytestconfig, set_physicsnemo_force_te):
    """Test shape validation errors for MeshGraphNet.forward"""
    import torch_geometric as pyg

    from physicsnemo.models.meshgraphnet import MeshGraphNet

    model = MeshGraphNet(
        input_dim_nodes=4,
        input_dim_edges=3,
        output_dim=2,
    ).to(device)

    # Single graph
    num_nodes, num_edges = 12, 16
    graph = pyg.data.Batch.from_data_list([rand_graph(num_nodes, num_edges, device)])

    # Wrong node feature dimension (second dim)
    bad_node = torch.randn(graph.num_nodes, 5).to(device)
    good_edge = torch.randn(graph.num_edges, 3).to(device)
    with pytest.raises(ValueError, match=r"Expected tensor of shape \(N_nodes, 4\)"):
        _ = model(bad_node, good_edge, graph)

    # Wrong edge feature dimension (second dim)
    good_node = torch.randn(graph.num_nodes, 4).to(device)
    bad_edge = torch.randn(graph.num_edges, 2).to(device)
    with pytest.raises(ValueError, match=r"Expected tensor of shape \(N_edges, 3\)"):
        _ = model(good_node, bad_edge, graph)

    # Wrong node feature rank (ndim)
    bad_node_rank = torch.randn(2, graph.num_nodes, 4).to(device)
    with pytest.raises(ValueError, match=r"Expected tensor of shape \(N_nodes, 4\)"):
        _ = model(bad_node_rank, good_edge, graph)
