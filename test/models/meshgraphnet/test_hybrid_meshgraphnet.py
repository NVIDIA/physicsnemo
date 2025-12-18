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
import numpy as np
import pytest
import torch

pytest.importorskip("torch_geometric")

from test import common
from test.conftest import requires_module


@requires_module("torch_geometric")
def test_hybrid_meshgraphnet_forward(device, pytestconfig, set_physicsnemo_force_te):
    """Test hybrid meshgraphnet forward pass"""

    import torch_geometric as pyg

    from physicsnemo.models.meshgraphnet import HybridMeshGraphNet

    torch.manual_seed(0)
    np.random.seed(0)
    # Construct MGN model
    model = HybridMeshGraphNet(
        input_dim_nodes=4,
        input_dim_edges=3,
        output_dim=2,
    ).to(device)

    num_nodes, num_mesh_edges, num_world_edges = 20, 10, 10
    # Create mesh edges.
    mesh_src = torch.tensor(
        [np.random.randint(num_nodes) for _ in range(num_mesh_edges)]
    )
    mesh_dst = torch.tensor(
        [np.random.randint(num_nodes) for _ in range(num_mesh_edges)]
    )
    mesh_edge_index = torch.stack([mesh_src, mesh_dst], dim=0)
    # Create world edges.
    world_src = torch.tensor(
        [np.random.randint(num_nodes) for _ in range(num_world_edges)]
    )
    world_dst = torch.tensor(
        [np.random.randint(num_nodes) for _ in range(num_world_edges)]
    )
    world_edge_index = torch.stack([world_src, world_dst], dim=0)
    # Combine edges and create graph.
    edge_index = torch.cat([mesh_edge_index, world_edge_index], dim=1)
    graph = pyg.data.Data(edge_index=edge_index, num_nodes=num_nodes).to(device)

    node_features = torch.randn(num_nodes, 4).to(device)
    mesh_edge_features = torch.randn(num_mesh_edges, 3).to(device)
    world_edge_features = torch.randn(num_world_edges, 3).to(device)
    assert common.validate_forward_accuracy(
        model,
        (node_features, mesh_edge_features, world_edge_features, graph),
        rtol=1e-2,
        atol=1e-2,
        file_name="models/meshgraphnet/data/hybridmeshgraphnet_output.pth",
    )


@requires_module("torch_geometric")
def test_hybrid_meshgraphnet_constructor(
    device, pytestconfig, set_physicsnemo_force_te
):
    """Test hybrid meshgraphnet constructor options"""

    import torch_geometric as pyg

    from physicsnemo.models.meshgraphnet import HybridMeshGraphNet

    torch.manual_seed(0)
    np.random.seed(0)

    # Define dictionary of constructor args - simplified.
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
        # Construct hybrid meshgraphnet model.
        model = HybridMeshGraphNet(**kw_args).to(device)

        num_nodes, num_mesh_edges, num_world_edges = 15, 8, 8
        # Create mesh edges.
        mesh_src = torch.tensor(
            [np.random.randint(num_nodes) for _ in range(num_mesh_edges)]
        )
        mesh_dst = torch.tensor(
            [np.random.randint(num_nodes) for _ in range(num_mesh_edges)]
        )
        mesh_edge_index = torch.stack([mesh_src, mesh_dst], dim=0)
        # Create world edges.
        world_src = torch.tensor(
            [np.random.randint(num_nodes) for _ in range(num_world_edges)]
        )
        world_dst = torch.tensor(
            [np.random.randint(num_nodes) for _ in range(num_world_edges)]
        )
        world_edge_index = torch.stack([world_src, world_dst], dim=0)
        # Combine edges and create graph.
        edge_index = torch.cat([mesh_edge_index, world_edge_index], dim=1)
        graph = pyg.data.Data(edge_index=edge_index, num_nodes=num_nodes).to(device)

        node_features = torch.randn(num_nodes, kw_args["input_dim_nodes"]).to(device)
        mesh_edge_features = torch.randn(num_mesh_edges, kw_args["input_dim_edges"]).to(
            device
        )
        world_edge_features = torch.randn(
            num_world_edges, kw_args["input_dim_edges"]
        ).to(device)

        outvar = model(node_features, mesh_edge_features, world_edge_features, graph)
        assert outvar.shape == (num_nodes, kw_args["output_dim"])


@requires_module("torch_geometric")
def test_hybrid_meshgraphnet_optims(device, pytestconfig, set_physicsnemo_force_te):
    """Test hybrid meshgraphnet optimizations"""

    import torch_geometric as pyg

    from physicsnemo.models.meshgraphnet import HybridMeshGraphNet

    def setup_model():
        """Set up fresh model and inputs for each optim test."""
        torch.manual_seed(0)
        np.random.seed(0)

        model = HybridMeshGraphNet(
            input_dim_nodes=4,
            input_dim_edges=3,
            output_dim=2,
        ).to(device)

        num_nodes, num_mesh_edges, num_world_edges = 15, 8, 8
        # Create mesh edges.
        mesh_src = torch.tensor(
            [np.random.randint(num_nodes) for _ in range(num_mesh_edges)]
        )
        mesh_dst = torch.tensor(
            [np.random.randint(num_nodes) for _ in range(num_mesh_edges)]
        )
        mesh_edge_index = torch.stack([mesh_src, mesh_dst], dim=0)
        # Create world edges.
        world_src = torch.tensor(
            [np.random.randint(num_nodes) for _ in range(num_world_edges)]
        )
        world_dst = torch.tensor(
            [np.random.randint(num_nodes) for _ in range(num_world_edges)]
        )
        world_edge_index = torch.stack([world_src, world_dst], dim=0)
        # Combine edges and create graph.
        edge_index = torch.cat([mesh_edge_index, world_edge_index], dim=1)
        graph = pyg.data.Data(edge_index=edge_index, num_nodes=num_nodes).to(device)

        node_features = torch.randn(num_nodes, 4).to(device)
        mesh_edge_features = torch.randn(num_mesh_edges, 3).to(device)
        world_edge_features = torch.randn(num_world_edges, 3).to(device)
        return model, [node_features, mesh_edge_features, world_edge_features, graph]

    # Check optimizations.
    model, invar = setup_model()
    assert common.validate_cuda_graphs(model, (*invar,))
    model, invar = setup_model()
    assert common.validate_jit(model, (*invar,))
    model, invar = setup_model()
    assert common.validate_amp(model, (*invar,))
    model, invar = setup_model()
    assert common.validate_combo_optims(model, (*invar,))


@requires_module("torch_geometric")
def test_hybrid_meshgraphnet_checkpoint(device, pytestconfig, set_physicsnemo_force_te):
    """Test hybrid meshgraphnet checkpoint save/load"""

    import torch_geometric as pyg

    from physicsnemo.models.meshgraphnet import HybridMeshGraphNet

    torch.manual_seed(0)
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
    # Create mesh edges.
    mesh_src = torch.tensor(
        [np.random.randint(num_nodes) for _ in range(num_mesh_edges)]
    )
    mesh_dst = torch.tensor(
        [np.random.randint(num_nodes) for _ in range(num_mesh_edges)]
    )
    mesh_edge_index = torch.stack([mesh_src, mesh_dst], dim=0)
    # Create world edges.
    world_src = torch.tensor(
        [np.random.randint(num_nodes) for _ in range(num_world_edges)]
    )
    world_dst = torch.tensor(
        [np.random.randint(num_nodes) for _ in range(num_world_edges)]
    )
    world_edge_index = torch.stack([world_src, world_dst], dim=0)
    # Combine edges and create graph.
    edge_index = torch.cat([mesh_edge_index, world_edge_index], dim=1)
    graph = pyg.data.Data(edge_index=edge_index, num_nodes=num_nodes).to(device)

    node_features = torch.randn(num_nodes, 4).to(device)
    mesh_edge_features = torch.randn(num_mesh_edges, 3).to(device)
    world_edge_features = torch.randn(num_world_edges, 3).to(device)

    assert common.validate_checkpoint(
        model_1,
        model_2,
        (node_features, mesh_edge_features, world_edge_features, graph),
    )


@requires_module("torch_geometric")
@common.check_ort_version()
def test_hybrid_meshgraphnet_deploy(device, pytestconfig, set_physicsnemo_force_te):
    """Test hybrid meshgraphnet deployment support"""

    import torch_geometric as pyg

    from physicsnemo.models.meshgraphnet import HybridMeshGraphNet

    torch.manual_seed(0)
    np.random.seed(0)

    model = HybridMeshGraphNet(
        input_dim_nodes=4,
        input_dim_edges=3,
        output_dim=2,
    ).to(device)

    num_nodes, num_mesh_edges, num_world_edges = 10, 6, 6
    # Create mesh edges.
    mesh_src = torch.tensor(
        [np.random.randint(num_nodes) for _ in range(num_mesh_edges)]
    )
    mesh_dst = torch.tensor(
        [np.random.randint(num_nodes) for _ in range(num_mesh_edges)]
    )
    mesh_edge_index = torch.stack([mesh_src, mesh_dst], dim=0)
    # Create world edges.
    world_src = torch.tensor(
        [np.random.randint(num_nodes) for _ in range(num_world_edges)]
    )
    world_dst = torch.tensor(
        [np.random.randint(num_nodes) for _ in range(num_world_edges)]
    )
    world_edge_index = torch.stack([world_src, world_dst], dim=0)
    # Combine edges and create graph.
    edge_index = torch.cat([mesh_edge_index, world_edge_index], dim=1)
    graph = pyg.data.Data(edge_index=edge_index, num_nodes=num_nodes).to(device)

    node_features = torch.randn(num_nodes, 4).to(device)
    mesh_edge_features = torch.randn(num_mesh_edges, 3).to(device)
    world_edge_features = torch.randn(num_world_edges, 3).to(device)

    invar = (node_features, mesh_edge_features, world_edge_features, graph)
    assert common.validate_onnx_export(model, invar)
    assert common.validate_onnx_runtime(model, invar)
