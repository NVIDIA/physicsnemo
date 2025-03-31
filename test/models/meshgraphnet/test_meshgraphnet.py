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
# ruff: noqa: E402
import os
import random
import sys

import numpy as np
import pytest
import torch

script_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(script_path), ".."))

import common
from pytest_utils import import_or_fail

dgl = pytest.importorskip("dgl")


@import_or_fail("dgl")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_meshgraphnet_forward(device, pytestconfig):
    """Test mehsgraphnet forward pass"""

    from physicsnemo.models.meshgraphnet import MeshGraphNet

    torch.manual_seed(0)
    dgl.seed(0)
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
        graphs.append(dgl.graph((src, dst)).to(device))
    graph = dgl.batch(graphs)
    node_features = torch.randn(graph.num_nodes(), 4).to(device)
    edge_features = torch.randn(graph.num_edges(), 3).to(device)
    assert common.validate_forward_accuracy(
        model, (node_features, edge_features, graph)
    )


@import_or_fail("dgl")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_mehsgraphnet_constructor(device, pytestconfig):
    """Test mehsgraphnet constructor options"""

    from physicsnemo.models.meshgraphnet import MeshGraphNet

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
    for kw_args in arg_list:
        # Construct mehsgraphnet model
        model = MeshGraphNet(**kw_args).to(device)

        bsize = random.randint(1, 16)
        num_nodes, num_edges = random.randint(10, 25), random.randint(10, 20)
        graph = dgl.batch(
            [dgl.rand_graph(num_nodes, num_edges).to(device) for _ in range(bsize)]
        )
        node_features = torch.randn(bsize * num_nodes, kw_args["input_dim_nodes"]).to(
            device
        )
        edge_features = torch.randn(bsize * num_edges, kw_args["input_dim_edges"]).to(
            device
        )
        outvar = model(node_features, edge_features, graph)
        assert outvar.shape == (bsize * num_nodes, kw_args["output_dim"])


@import_or_fail("dgl")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_meshgraphnet_optims(device, pytestconfig):
    """Test meshgraphnet optimizations"""

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
        graph = dgl.batch(
            [dgl.rand_graph(num_nodes, num_edges).to(device) for _ in range(bsize)]
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


@import_or_fail("dgl")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_meshgraphnet_checkpoint(device, pytestconfig):
    """Test meshgraphnet checkpoint save/load"""

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
    graph = dgl.batch(
        [dgl.rand_graph(num_nodes, num_edges).to(device) for _ in range(bsize)]
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


@import_or_fail("dgl")
@common.check_ort_version()
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_meshgraphnet_deploy(device, pytestconfig):
    """Test mesh-graph net deployment support"""

    from physicsnemo.models.meshgraphnet import MeshGraphNet

    # Construct MGN model
    model = MeshGraphNet(
        input_dim_nodes=4,
        input_dim_edges=3,
        output_dim=4,
    ).to(device)

    bsize = random.randint(1, 8)
    num_nodes, num_edges = random.randint(5, 10), random.randint(10, 15)
    graph = dgl.batch(
        [dgl.rand_graph(num_nodes, num_edges).to(device) for _ in range(bsize)]
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
