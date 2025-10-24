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

import numpy as np
import pytest
import torch
from pytest_utils import import_or_fail
from torch.testing import assert_close


@import_or_fail(["dgl", "torch_geometric"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_hybrid_meshgraphnet_dgl_pyg_equivalence(
    device, pytestconfig, set_physicsnemo_force_te
):
    """Test that HybridMeshGraphNet produces equivalent outputs for DGL and PyG graphs."""
    # (DGL2PYG): remove this once DGL is removed.

    import dgl
    from torch_geometric.data import Data as PyGData

    from physicsnemo.models.meshgraphnet import HybridMeshGraphNet

    # Set seeds for reproducibility.
    torch.manual_seed(42)
    dgl.seed(42)
    np.random.seed(42)

    # Test parameters.
    num_nodes = 10
    num_mesh_edges = 8
    num_world_edges = 7
    input_dim_nodes = 6
    input_dim_edges = 4
    output_dim = 3

    # Create HybridMeshGraphNet.
    model = HybridMeshGraphNet(
        input_dim_nodes=input_dim_nodes,
        input_dim_edges=input_dim_edges,
        output_dim=output_dim,
        processor_size=2,  # Small for faster testing
        hidden_dim_processor=16,
        hidden_dim_node_encoder=16,
        hidden_dim_edge_encoder=16,
        hidden_dim_node_decoder=16,
        num_layers_node_processor=1,
        num_layers_edge_processor=1,
        num_layers_node_encoder=1,
        num_layers_edge_encoder=1,
        num_layers_node_decoder=1,
    ).to(device)

    # Create random edge connectivity for mesh edges.
    mesh_src_nodes = torch.randint(0, num_nodes, (num_mesh_edges,), device=device)
    mesh_dst_nodes = torch.randint(0, num_nodes, (num_mesh_edges,), device=device)

    # Create random edge connectivity for world edges.
    world_src_nodes = torch.randint(0, num_nodes, (num_world_edges,), device=device)
    world_dst_nodes = torch.randint(0, num_nodes, (num_world_edges,), device=device)

    # Create node and edge features.
    node_features = torch.randn(num_nodes, input_dim_nodes, device=device)
    mesh_edge_features = torch.randn(num_mesh_edges, input_dim_edges, device=device)
    world_edge_features = torch.randn(num_world_edges, input_dim_edges, device=device)

    # Create DGL graph with both edge types.
    dgl_graph = dgl.graph((mesh_src_nodes, mesh_dst_nodes)).to(device)
    dgl_graph = dgl.add_edges(dgl_graph, world_src_nodes, world_dst_nodes)

    # Create PyG graph by concatenating both edge types.
    # PyG concatenates edge indices and edge attributes.
    all_src_nodes = torch.cat([mesh_src_nodes, world_src_nodes], dim=0).to(device)
    all_dst_nodes = torch.cat([mesh_dst_nodes, world_dst_nodes], dim=0).to(device)
    edge_index = torch.stack([all_src_nodes, all_dst_nodes], dim=0)
    pyg_graph = PyGData(edge_index=edge_index)

    # Forward pass with DGL graph.
    output_dgl = model(
        node_features, mesh_edge_features, world_edge_features, dgl_graph
    )

    # Forward pass with PyG graph.
    output_pyg = model(
        node_features, mesh_edge_features, world_edge_features, pyg_graph
    )

    # Verify outputs are equivalent.
    assert_close(output_dgl, output_pyg)

    # Verify output shapes.
    assert output_dgl.shape == (num_nodes, output_dim)
    assert output_pyg.shape == (num_nodes, output_dim)


@import_or_fail(["dgl", "torch_geometric"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_hybrid_meshgraphnet_gradient_equivalence(
    device, pytestconfig, set_physicsnemo_force_te
):
    """Test that HybridMeshGraphNet produces equivalent gradients for DGL and PyG graphs."""
    # (DGL2PYG): remove this once DGL is removed.

    import dgl
    from torch_geometric.data import Data as PyGData

    from physicsnemo.models.meshgraphnet import HybridMeshGraphNet

    # Set seeds for reproducibility.
    torch.manual_seed(123)
    dgl.seed(123)
    np.random.seed(123)

    # Test parameters.
    num_nodes = 8
    num_mesh_edges = 6
    num_world_edges = 5
    input_dim_nodes = 4
    input_dim_edges = 3
    output_dim = 2

    # Create identical HybridMeshGraphNets.
    model_dgl = HybridMeshGraphNet(
        input_dim_nodes=input_dim_nodes,
        input_dim_edges=input_dim_edges,
        output_dim=output_dim,
        processor_size=2,  # Small for faster testing
        hidden_dim_processor=8,
        hidden_dim_node_encoder=8,
        hidden_dim_edge_encoder=8,
        hidden_dim_node_decoder=8,
        num_layers_node_processor=1,
        num_layers_edge_processor=1,
        num_layers_node_encoder=1,
        num_layers_edge_encoder=1,
        num_layers_node_decoder=1,
    ).to(device)

    model_pyg = HybridMeshGraphNet(
        input_dim_nodes=input_dim_nodes,
        input_dim_edges=input_dim_edges,
        output_dim=output_dim,
        processor_size=2,  # Small for faster testing
        hidden_dim_processor=8,
        hidden_dim_node_encoder=8,
        hidden_dim_edge_encoder=8,
        hidden_dim_node_decoder=8,
        num_layers_node_processor=1,
        num_layers_edge_processor=1,
        num_layers_node_encoder=1,
        num_layers_edge_encoder=1,
        num_layers_node_decoder=1,
    ).to(device)

    # Copy weights to ensure identical models.
    model_pyg.load_state_dict(model_dgl.state_dict())

    # Create random edge connectivity.
    mesh_src_nodes = torch.randint(0, num_nodes, (num_mesh_edges,), device=device)
    mesh_dst_nodes = torch.randint(0, num_nodes, (num_mesh_edges,), device=device)
    world_src_nodes = torch.randint(0, num_nodes, (num_world_edges,), device=device)
    world_dst_nodes = torch.randint(0, num_nodes, (num_world_edges,), device=device)

    # Create node and edge features (requires_grad for gradient test).
    node_features_dgl = torch.randn(
        num_nodes, input_dim_nodes, device=device, requires_grad=True
    )
    mesh_edge_features_dgl = torch.randn(
        num_mesh_edges, input_dim_edges, device=device, requires_grad=True
    )
    world_edge_features_dgl = torch.randn(
        num_world_edges, input_dim_edges, device=device, requires_grad=True
    )

    node_features_pyg = node_features_dgl.clone().detach().requires_grad_(True)
    mesh_edge_features_pyg = (
        mesh_edge_features_dgl.clone().detach().requires_grad_(True)
    )
    world_edge_features_pyg = (
        world_edge_features_dgl.clone().detach().requires_grad_(True)
    )

    # Create DGL graph.
    dgl_graph = dgl.graph((mesh_src_nodes, mesh_dst_nodes)).to(device)
    dgl_graph = dgl.add_edges(dgl_graph, world_src_nodes, world_dst_nodes)

    # Create PyG graph.
    all_src_nodes = torch.cat([mesh_src_nodes, world_src_nodes], dim=0).to(device)
    all_dst_nodes = torch.cat([mesh_dst_nodes, world_dst_nodes], dim=0).to(device)
    edge_index = torch.stack([all_src_nodes, all_dst_nodes], dim=0)
    pyg_graph = PyGData(edge_index=edge_index)

    # Forward pass with DGL graph.
    output_dgl = model_dgl(
        node_features_dgl, mesh_edge_features_dgl, world_edge_features_dgl, dgl_graph
    )

    # Forward pass with PyG graph.
    output_pyg = model_pyg(
        node_features_pyg, mesh_edge_features_pyg, world_edge_features_pyg, pyg_graph
    )

    # Create identical loss functions.
    loss_dgl = output_dgl.sum()
    loss_pyg = output_pyg.sum()

    # Backward pass.
    loss_dgl.backward()
    loss_pyg.backward()

    # Compare input feature gradients.
    assert_close(
        node_features_dgl.grad,
        node_features_pyg.grad,
    )
    assert_close(
        mesh_edge_features_dgl.grad,
        mesh_edge_features_pyg.grad,
    )
    assert_close(
        world_edge_features_dgl.grad,
        world_edge_features_pyg.grad,
    )

    # Compare model parameter gradients.
    for (name_dgl, param_dgl), (name_pyg, param_pyg) in zip(
        model_dgl.named_parameters(), model_pyg.named_parameters()
    ):
        assert name_dgl == name_pyg, (
            f"Parameter names should match: {name_dgl} vs {name_pyg}"
        )
        assert_close(
            param_dgl.grad,
            param_pyg.grad,
        )


@import_or_fail(["dgl", "torch_geometric"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_hybrid_meshgraphnet_hetero_edge_processing(
    device, pytestconfig, set_physicsnemo_force_te
):
    """Test that HybridMeshGraphNet properly handles separate mesh and world edge features."""
    # This test focuses on the key difference of Hybrid MGN: heterogeneous edge processing.

    import dgl
    from torch_geometric.data import Data as PyGData

    from physicsnemo.models.meshgraphnet import HybridMeshGraphNet

    # Set seeds for reproducibility.
    torch.manual_seed(456)
    dgl.seed(456)
    np.random.seed(456)

    # Test parameters.
    num_nodes = 6
    num_mesh_edges = 4
    num_world_edges = 3
    input_dim_nodes = 5
    input_dim_edges = 4
    output_dim = 2

    # Create HybridMeshGraphNet.
    model = HybridMeshGraphNet(
        input_dim_nodes=input_dim_nodes,
        input_dim_edges=input_dim_edges,
        output_dim=output_dim,
        processor_size=1,  # Single layer for focused testing
        hidden_dim_processor=8,
        hidden_dim_node_encoder=8,
        hidden_dim_edge_encoder=8,
        hidden_dim_node_decoder=8,
        num_layers_node_processor=1,
        num_layers_edge_processor=1,
        num_layers_node_encoder=1,
        num_layers_edge_encoder=1,
        num_layers_node_decoder=1,
    ).to(device)

    # Create edge connectivity.
    mesh_src_nodes = torch.randint(0, num_nodes, (num_mesh_edges,), device=device)
    mesh_dst_nodes = torch.randint(0, num_nodes, (num_mesh_edges,), device=device)
    world_src_nodes = torch.randint(0, num_nodes, (num_world_edges,), device=device)
    world_dst_nodes = torch.randint(0, num_nodes, (num_world_edges,), device=device)

    # Create features.
    node_features = torch.randn(num_nodes, input_dim_nodes, device=device)

    # Create DIFFERENT mesh and world edge features to test separate processing.
    mesh_edge_features = torch.ones(
        num_mesh_edges, input_dim_edges, device=device
    )  # All ones
    world_edge_features = torch.zeros(
        num_world_edges, input_dim_edges, device=device
    )  # All zeros

    # Create graphs.
    dgl_graph = dgl.graph((mesh_src_nodes, mesh_dst_nodes)).to(device)
    dgl_graph = dgl.add_edges(dgl_graph, world_src_nodes, world_dst_nodes)

    all_src_nodes = torch.cat([mesh_src_nodes, world_src_nodes], dim=0).to(device)
    all_dst_nodes = torch.cat([mesh_dst_nodes, world_dst_nodes], dim=0).to(device)
    edge_index = torch.stack([all_src_nodes, all_dst_nodes], dim=0)
    pyg_graph = PyGData(edge_index=edge_index)

    # Forward pass.
    output_dgl = model(
        node_features, mesh_edge_features, world_edge_features, dgl_graph
    )
    output_pyg = model(
        node_features, mesh_edge_features, world_edge_features, pyg_graph
    )

    # Verify outputs are equivalent between DGL and PyG.
    assert_close(output_dgl, output_pyg)

    # Verify output shapes.
    assert output_dgl.shape == (num_nodes, output_dim)
    assert output_pyg.shape == (num_nodes, output_dim)

    # Verify model actually processes features (output should not be zeros).
    assert not torch.allclose(output_dgl, torch.zeros_like(output_dgl))
    assert not torch.allclose(output_pyg, torch.zeros_like(output_pyg))
