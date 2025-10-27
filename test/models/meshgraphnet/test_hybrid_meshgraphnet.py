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
# ruff: noqa: E402
import os
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
def test_hybrid_meshgraphnet_forward(device, pytestconfig, set_physicsnemo_force_te):
    """Test hybrid meshgraphnet forward pass"""

    from physicsnemo.models.meshgraphnet import HybridMeshGraphNet

    torch.manual_seed(0)
    dgl.seed(0)
    np.random.seed(0)
    # Construct MGN model
    model = HybridMeshGraphNet(
        input_dim_nodes=4,
        input_dim_edges=3,
        output_dim=2,
    ).to(device)

    num_nodes, num_mesh_edges, num_world_edges = 20, 10, 10
    # NOTE dgl's random graph generator does not behave consistently even after fixing dgl's random seed.
    # Instead, numpy adj matrices are created in COO format and are then converted to dgl graphs.
    src = torch.tensor([np.random.randint(num_nodes) for _ in range(num_mesh_edges)])
    dst = torch.tensor([np.random.randint(num_nodes) for _ in range(num_mesh_edges)])
    graph = dgl.graph((src, dst)).to(device)
    src = torch.tensor(
        [np.random.randint(num_nodes) for _ in range(num_world_edges)]
    ).to(device)
    dst = torch.tensor(
        [np.random.randint(num_nodes) for _ in range(num_world_edges)]
    ).to(device)
    graph = dgl.add_edges(graph, src, dst)

    node_features = torch.randn(num_nodes, 4).to(device)
    mesh_edge_features = torch.randn(num_mesh_edges, 3).to(device)
    world_edge_features = torch.randn(num_world_edges, 3).to(device)
    assert common.validate_forward_accuracy(
        model,
        (node_features, mesh_edge_features, world_edge_features, graph),
        rtol=1e-2,
        atol=1e-2,
    )


@import_or_fail("dgl")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_hybrid_meshgraphnet_constructor(
    device, pytestconfig, set_physicsnemo_force_te
):
    """Test hybrid meshgraphnet constructor options"""

    from physicsnemo.models.meshgraphnet import HybridMeshGraphNet

    torch.manual_seed(0)
    dgl.seed(0)
    np.random.seed(0)

    # Define dictionary of constructor args - simplified
    arg_list = [
        {
            "input_dim_nodes": 4,
            "input_dim_edges": 3,
            "output_dim": 2,
        },
        {
            "input_dim_nodes": 6,
            "input_dim_edges": 4,
            "output_dim": 3,
            "processor_size": 64,
        },
    ]

    for kw_args in arg_list:
        # Construct hybrid meshgraphnet model
        model = HybridMeshGraphNet(**kw_args).to(device)

        num_nodes, num_mesh_edges, num_world_edges = 15, 8, 8
        # Create mesh edges
        src = torch.tensor(
            [np.random.randint(num_nodes) for _ in range(num_mesh_edges)]
        )
        dst = torch.tensor(
            [np.random.randint(num_nodes) for _ in range(num_mesh_edges)]
        )
        graph = dgl.graph((src, dst)).to(device)
        # Add world edges with proper device handling
        src = torch.tensor(
            [np.random.randint(num_nodes) for _ in range(num_world_edges)]
        ).to(device)
        dst = torch.tensor(
            [np.random.randint(num_nodes) for _ in range(num_world_edges)]
        ).to(device)
        graph = dgl.add_edges(graph, src, dst)

        node_features = torch.randn(num_nodes, kw_args["input_dim_nodes"]).to(device)
        mesh_edge_features = torch.randn(num_mesh_edges, kw_args["input_dim_edges"]).to(
            device
        )
        world_edge_features = torch.randn(
            num_world_edges, kw_args["input_dim_edges"]
        ).to(device)

        outvar = model(node_features, mesh_edge_features, world_edge_features, graph)
        assert outvar.shape == (num_nodes, kw_args["output_dim"])


@import_or_fail("dgl")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_hybrid_meshgraphnet_optims(device, pytestconfig, set_physicsnemo_force_te):
    """Test hybrid meshgraphnet optimizations"""

    from physicsnemo.models.meshgraphnet import HybridMeshGraphNet

    def setup_model():
        """Set up fresh model and inputs for each optim test"""
        torch.manual_seed(0)
        dgl.seed(0)
        np.random.seed(0)

        model = HybridMeshGraphNet(
            input_dim_nodes=4,
            input_dim_edges=3,
            output_dim=2,
        ).to(device)

        num_nodes, num_mesh_edges, num_world_edges = 15, 8, 8
        src = torch.tensor(
            [np.random.randint(num_nodes) for _ in range(num_mesh_edges)]
        )
        dst = torch.tensor(
            [np.random.randint(num_nodes) for _ in range(num_mesh_edges)]
        )
        graph = dgl.graph((src, dst)).to(device)
        src = torch.tensor(
            [np.random.randint(num_nodes) for _ in range(num_world_edges)]
        ).to(device)
        dst = torch.tensor(
            [np.random.randint(num_nodes) for _ in range(num_world_edges)]
        ).to(device)
        graph = dgl.add_edges(graph, src, dst)

        node_features = torch.randn(num_nodes, 4).to(device)
        mesh_edge_features = torch.randn(num_mesh_edges, 3).to(device)
        world_edge_features = torch.randn(num_world_edges, 3).to(device)
        return model, [node_features, mesh_edge_features, world_edge_features, graph]

    # Check optimizations
    model, invar = setup_model()
    assert common.validate_cuda_graphs(model, (*invar,))
    model, invar = setup_model()
    assert common.validate_jit(model, (*invar,))
    model, invar = setup_model()
    assert common.validate_amp(model, (*invar,))
    model, invar = setup_model()
    assert common.validate_combo_optims(model, (*invar,))


@import_or_fail("dgl")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_hybrid_meshgraphnet_checkpoint(device, pytestconfig, set_physicsnemo_force_te):
    """Test hybrid meshgraphnet checkpoint save/load"""

    from physicsnemo.models.meshgraphnet import HybridMeshGraphNet

    torch.manual_seed(0)
    dgl.seed(0)
    np.random.seed(0)

    model_1 = HybridMeshGraphNet(
        input_dim_nodes=4,
        input_dim_edges=3,
        output_dim=2,
    ).to(device)

    model_2 = HybridMeshGraphNet(
        input_dim_nodes=4,
        input_dim_edges=3,
        output_dim=2,
    ).to(device)

    num_nodes, num_mesh_edges, num_world_edges = 15, 8, 8
    src = torch.tensor([np.random.randint(num_nodes) for _ in range(num_mesh_edges)])
    dst = torch.tensor([np.random.randint(num_nodes) for _ in range(num_mesh_edges)])
    graph = dgl.graph((src, dst)).to(device)
    src = torch.tensor(
        [np.random.randint(num_nodes) for _ in range(num_world_edges)]
    ).to(device)
    dst = torch.tensor(
        [np.random.randint(num_nodes) for _ in range(num_world_edges)]
    ).to(device)
    graph = dgl.add_edges(graph, src, dst)

    node_features = torch.randn(num_nodes, 4).to(device)
    mesh_edge_features = torch.randn(num_mesh_edges, 3).to(device)
    world_edge_features = torch.randn(num_world_edges, 3).to(device)

    assert common.validate_checkpoint(
        model_1,
        model_2,
        (node_features, mesh_edge_features, world_edge_features, graph),
    )


@import_or_fail("dgl")
@common.check_ort_version()
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_hybrid_meshgraphnet_deploy(device, pytestconfig, set_physicsnemo_force_te):
    """Test hybrid meshgraphnet deployment support"""

    from physicsnemo.models.meshgraphnet import HybridMeshGraphNet

    torch.manual_seed(0)
    dgl.seed(0)
    np.random.seed(0)

    model = HybridMeshGraphNet(
        input_dim_nodes=4,
        input_dim_edges=3,
        output_dim=2,
    ).to(device)

    num_nodes, num_mesh_edges, num_world_edges = 10, 6, 6
    src = torch.tensor([np.random.randint(num_nodes) for _ in range(num_mesh_edges)])
    dst = torch.tensor([np.random.randint(num_nodes) for _ in range(num_mesh_edges)])
    graph = dgl.graph((src, dst)).to(device)
    src = torch.tensor(
        [np.random.randint(num_nodes) for _ in range(num_world_edges)]
    ).to(device)
    dst = torch.tensor(
        [np.random.randint(num_nodes) for _ in range(num_world_edges)]
    ).to(device)
    graph = dgl.add_edges(graph, src, dst)

    node_features = torch.randn(num_nodes, 4).to(device)
    mesh_edge_features = torch.randn(num_mesh_edges, 3).to(device)
    world_edge_features = torch.randn(num_world_edges, 3).to(device)

    invar = (node_features, mesh_edge_features, world_edge_features, graph)
    assert common.validate_onnx_export(model, invar)
    assert common.validate_onnx_runtime(model, invar)
