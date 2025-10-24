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

"""GraphCast DGL/PyG equivalency tests.

These tests ensure that GraphCast components produce equivalent outputs
when using DGL and PyG backends.
"""

import os
import sys

import pytest
import torch
from torch.testing import assert_close

script_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(script_path), ".."))

from graphcast.utils import compare_quantiles, create_random_input, fix_random_seeds
from pytest_utils import import_or_fail

# Disable flash attention for consistent behavior.
os.environ["NVTE_FLASH_ATTN"] = "0"


@import_or_fail(["dgl", "torch_geometric"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@torch.no_grad()
def test_graphcast_net_dgl_pyg_equivalence(
    device, pytestconfig, set_physicsnemo_force_te
):
    """Test that GraphCastNet produces equivalent outputs for DGL and PyG backends."""
    # (DGL2PYG): remove this once DGL is removed.

    from physicsnemo.models.graphcast.graph_cast_net import GraphCastNet

    # Set seeds for reproducibility.
    fix_random_seeds()

    # Test parameters - small graph and inputs for fast testing.
    model_kwargs = {
        "mesh_level": 1,
        "input_res": (8, 16),
        "input_dim_grid_nodes": 4,
        "input_dim_mesh_nodes": 3,
        "input_dim_edges": 4,
        "output_dim_grid_nodes": 4,
        "processor_layers": 3,
        "hidden_dim": 8,
        "hidden_layers": 1,
        "do_concat_trick": False,
    }

    # Create input data.
    x = create_random_input(
        model_kwargs["input_res"], model_kwargs["input_dim_grid_nodes"]
    ).to(device)

    # Create models.
    model_dgl = GraphCastNet(**model_kwargs, graph_backend="dgl").to(device)
    model_pyg = GraphCastNet(**model_kwargs, graph_backend="pyg").to(device)

    # Copy weights to ensure identical models.
    model_pyg.load_state_dict(model_dgl.state_dict())

    # Forward pass with both backends.
    output_dgl = model_dgl(x)
    output_pyg = model_pyg(x)

    # Verify outputs are equivalent.
    assert_close(output_dgl, output_pyg, rtol=1e-4, atol=1e-5)

    # Verify output shapes.
    assert output_dgl.shape == output_pyg.shape
    expected_shape = (
        1,
        model_kwargs["output_dim_grid_nodes"],
        *model_kwargs["input_res"],
    )
    assert output_dgl.shape == expected_shape


@import_or_fail(["dgl", "torch_geometric"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_graphcast_net_gradient_equivalence(
    device, pytestconfig, set_physicsnemo_force_te
):
    """Test that GraphCastNet produces equivalent gradients for DGL and PyG backends."""
    # (DGL2PYG): remove this once DGL is removed.

    from physicsnemo.models.graphcast.graph_cast_net import GraphCastNet

    # Set seeds for reproducibility.
    fix_random_seeds()

    # Test parameters - small for gradient tests.
    model_kwargs = {
        "mesh_level": 1,
        "input_res": (6, 12),
        "input_dim_grid_nodes": 3,
        "input_dim_mesh_nodes": 3,
        "input_dim_edges": 4,
        "output_dim_grid_nodes": 3,
        "processor_layers": 3,
        "hidden_dim": 6,
        "hidden_layers": 1,
        "do_concat_trick": False,
    }

    # Create input data with gradients.
    x_dgl = create_random_input(
        model_kwargs["input_res"], model_kwargs["input_dim_grid_nodes"]
    )
    x_dgl = x_dgl.to(device).requires_grad_(True)

    x_pyg = x_dgl.clone().detach().requires_grad_(True)

    # Create identical models.
    torch.manual_seed(42)
    model_dgl = GraphCastNet(**model_kwargs, graph_backend="dgl").to(device)

    torch.manual_seed(42)
    model_pyg = GraphCastNet(**model_kwargs, graph_backend="pyg").to(device)

    # Forward pass and compute loss.
    output_dgl = model_dgl(x_dgl)
    loss_dgl = output_dgl.sum()
    loss_dgl.backward()

    output_pyg = model_pyg(x_pyg)
    loss_pyg = output_pyg.sum()
    loss_pyg.backward()

    # Compare input gradients.
    assert_close(x_dgl.grad, x_pyg.grad, rtol=1e-4, atol=1e-5)

    # Compare model parameter gradients.
    for (name_dgl, param_dgl), (name_pyg, param_pyg) in zip(
        model_dgl.named_parameters(), model_pyg.named_parameters()
    ):
        assert name_dgl == name_pyg, (
            f"Parameter names should match: {name_dgl} vs {name_pyg}"
        )
        if param_dgl.grad is not None and param_pyg.grad is not None:
            assert_close(param_dgl.grad, param_pyg.grad, rtol=1e-4, atol=1e-5)


@import_or_fail(["dgl", "torch_geometric", "torch_sparse"])
@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("processor_type", ["MessagePassing", "GraphTransformer"])
@torch.no_grad()
def test_graphcast_processor_dgl_pyg_equivalence(
    device, processor_type, pytestconfig, set_physicsnemo_force_te
):
    """Test that GraphCast processors produce equivalent outputs for DGL and PyG backends."""
    # (DGL2PYG): remove this once DGL is removed.

    from physicsnemo.models.graphcast.graph_cast_net import GraphCastNet

    # Set seeds for reproducibility.
    fix_random_seeds()

    # Test parameters.
    model_kwargs = {
        "mesh_level": 1,
        "input_res": (6, 12),
        "input_dim_grid_nodes": 3,
        "input_dim_mesh_nodes": 3,
        "input_dim_edges": 4,
        "output_dim_grid_nodes": 3,
        "processor_type": processor_type,
        "processor_layers": 3,
        "hidden_dim": 8,
        "hidden_layers": 1,
        "khop_neighbors": 2 if processor_type == "GraphTransformer" else 0,
        "num_attention_heads": 2 if processor_type == "GraphTransformer" else 4,
        "do_concat_trick": False,
    }

    # Create input data.
    x = create_random_input(
        model_kwargs["input_res"], model_kwargs["input_dim_grid_nodes"]
    ).to(device)

    # Create models.
    model_dgl = GraphCastNet(**model_kwargs, graph_backend="dgl").to(device)
    model_pyg = GraphCastNet(**model_kwargs, graph_backend="pyg").to(device)

    # Copy weights to ensure identical models.
    model_pyg.load_state_dict(model_dgl.state_dict())

    # Forward pass with both backends.
    output_dgl = model_dgl(x)
    output_pyg = model_pyg(x)

    # Verify outputs are equivalent.
    # In this case, we are comparing the quantiles of the outputs
    # since the outputs are close to 0 and have higher errors in case of GraphTransformer.
    # This is a more robust way to compare the outputs than increasing
    # the tolerances in assert_close.
    compare_quantiles(
        output_dgl, output_pyg, [0.25, 0.5, 0.75, 0.95], [1e-4, 1e-4, 2e-4, 3e-4]
    )


@import_or_fail(["dgl", "torch_geometric"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("do_concat_trick", [False, True])
@torch.no_grad()
def test_graphcast_concat_trick_dgl_pyg_equivalence(
    device, do_concat_trick, pytestconfig, set_physicsnemo_force_te
):
    """Test that GraphCast concat trick produces equivalent outputs for DGL and PyG backends."""
    # (DGL2PYG): remove this once DGL is removed.

    from physicsnemo.models.graphcast.graph_cast_net import GraphCastNet

    # Set seeds for reproducibility.
    fix_random_seeds()

    # Test parameters.
    model_kwargs = {
        "mesh_level": 1,
        "input_res": (6, 12),
        "input_dim_grid_nodes": 3,
        "input_dim_mesh_nodes": 3,
        "input_dim_edges": 4,
        "output_dim_grid_nodes": 3,
        "processor_layers": 3,
        "hidden_dim": 8,
        "hidden_layers": 1,
        "do_concat_trick": do_concat_trick,
    }

    # Create input data.
    x = create_random_input(
        model_kwargs["input_res"], model_kwargs["input_dim_grid_nodes"]
    ).to(device)

    # Create models.
    model_dgl = GraphCastNet(**model_kwargs, graph_backend="dgl").to(device)
    model_pyg = GraphCastNet(**model_kwargs, graph_backend="pyg").to(device)

    # Copy weights to ensure identical models.
    model_pyg.load_state_dict(model_dgl.state_dict())

    # Forward pass with both backends.
    output_dgl = model_dgl(x)
    output_pyg = model_pyg(x)

    # Verify outputs are equivalent.
    assert_close(output_dgl, output_pyg, rtol=1e-4, atol=1e-5)


@import_or_fail(["dgl", "torch_geometric"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_graphcast_graph_creation_equivalence(
    device, pytestconfig, set_physicsnemo_force_te
):
    """Test that Graph class creates equivalent graphs for DGL and PyG backends."""
    # (DGL2PYG): remove this once DGL is removed.

    from physicsnemo.utils.graphcast.graph import Graph

    # Set seeds for reproducibility
    fix_random_seeds()

    # Test parameters
    input_res = (6, 12)
    mesh_level = 1

    # Create lat/lon grid
    latitudes = torch.linspace(-90, 90, steps=input_res[0])
    longitudes = torch.linspace(-180, 180, steps=input_res[1] + 1)[1:]
    lat_lon_grid = torch.stack(
        torch.meshgrid(latitudes, longitudes, indexing="ij"), dim=-1
    )

    # Create Graph objects with different backends
    graph_dgl = Graph(lat_lon_grid, mesh_level, multimesh=True, backend="dgl")
    graph_pyg = Graph(lat_lon_grid, mesh_level, multimesh=True, backend="pyg")

    # Create all three graph types
    mesh_graph_dgl, mask_dgl = graph_dgl.create_mesh_graph(verbose=False)
    g2m_graph_dgl = graph_dgl.create_g2m_graph(verbose=False)
    m2g_graph_dgl = graph_dgl.create_m2g_graph(verbose=False)

    mesh_graph_pyg, mask_pyg = graph_pyg.create_mesh_graph(verbose=False)
    g2m_graph_pyg = graph_pyg.create_g2m_graph(verbose=False)
    m2g_graph_pyg = graph_pyg.create_m2g_graph(verbose=False)

    # Test mesh graph equivalence
    if graph_dgl.backend.name == "dgl":
        mesh_node_features_dgl = mesh_graph_dgl.ndata["x"]
        mesh_edge_features_dgl = mesh_graph_dgl.edata["x"]
    else:
        mesh_node_features_dgl = mesh_graph_dgl.x
        mesh_edge_features_dgl = mesh_graph_dgl.edge_attr

    if graph_pyg.backend.name == "pyg":
        mesh_node_features_pyg = mesh_graph_pyg.x
        mesh_edge_features_pyg = mesh_graph_pyg.edge_attr
    else:
        mesh_node_features_pyg = mesh_graph_pyg.ndata["x"]
        mesh_edge_features_pyg = mesh_graph_pyg.edata["x"]

    # Verify that mesh graphs have same number of nodes and edges
    assert mesh_node_features_dgl.shape[0] == mesh_node_features_pyg.shape[0]
    assert mesh_edge_features_dgl.shape[0] == mesh_edge_features_pyg.shape[0]

    # Test g2m graph structure - these should have the same connectivity patterns
    if hasattr(g2m_graph_dgl, "edata"):
        g2m_edge_features_dgl = g2m_graph_dgl.edata["x"]
    else:
        g2m_edge_features_dgl = g2m_graph_dgl.edge_attr

    if hasattr(g2m_graph_pyg, "edge_attr"):
        g2m_edge_features_pyg = g2m_graph_pyg.edge_attr
    else:
        g2m_edge_features_pyg = g2m_graph_pyg.edata["x"]

    # Test m2g graph structure
    if hasattr(m2g_graph_dgl, "edata"):
        m2g_edge_features_dgl = m2g_graph_dgl.edata["x"]
    else:
        m2g_edge_features_dgl = m2g_graph_dgl.edge_attr

    if hasattr(m2g_graph_pyg, "edge_attr"):
        m2g_edge_features_pyg = m2g_graph_pyg.edge_attr
    else:
        m2g_edge_features_pyg = m2g_graph_pyg.edata["x"]

    # Verify edge feature shapes are equivalent
    assert g2m_edge_features_dgl.shape == g2m_edge_features_pyg.shape
    assert m2g_edge_features_dgl.shape == m2g_edge_features_pyg.shape


@import_or_fail(["dgl", "torch_geometric"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("aggregation", ["sum", "mean"])
@torch.no_grad()
def test_graphcast_encoder_decoder_dgl_pyg_equivalence(
    device, aggregation, pytestconfig, set_physicsnemo_force_te
):
    """Test that GraphCast encoder/decoder produce equivalent outputs for DGL and PyG backends."""
    # (DGL2PYG): remove this once DGL is removed.

    from physicsnemo.models.gnn_layers.mesh_graph_decoder import MeshGraphDecoder
    from physicsnemo.models.gnn_layers.mesh_graph_encoder import MeshGraphEncoder
    from physicsnemo.utils.graphcast.graph import Graph

    # Set seeds for reproducibility.
    fix_random_seeds()

    # Test parameters.
    input_res = (6, 12)
    mesh_level = 1
    hidden_dim = 8

    # Create lat/lon grid.
    latitudes = torch.linspace(-90, 90, steps=input_res[0])
    longitudes = torch.linspace(-180, 180, steps=input_res[1] + 1)[1:]
    lat_lon_grid = torch.stack(
        torch.meshgrid(latitudes, longitudes, indexing="ij"), dim=-1
    )

    # Create graphs.
    graph_dgl = Graph(lat_lon_grid, mesh_level, multimesh=True, backend="dgl")
    graph_pyg = Graph(lat_lon_grid, mesh_level, multimesh=True, backend="pyg")

    g2m_graph_dgl = graph_dgl.create_g2m_graph(verbose=False).to(device)
    m2g_graph_dgl = graph_dgl.create_m2g_graph(verbose=False).to(device)

    g2m_graph_pyg = graph_pyg.create_g2m_graph(verbose=False).to(device)
    m2g_graph_pyg = graph_pyg.create_m2g_graph(verbose=False).to(device)

    # Create encoder and decoder.
    encoder = MeshGraphEncoder(
        aggregation=aggregation,
        input_dim_src_nodes=hidden_dim,
        input_dim_dst_nodes=hidden_dim,
        input_dim_edges=hidden_dim,
        output_dim_src_nodes=hidden_dim,
        output_dim_dst_nodes=hidden_dim,
        output_dim_edges=hidden_dim,
        hidden_dim=hidden_dim,
        hidden_layers=1,
    ).to(device)

    decoder = MeshGraphDecoder(
        aggregation=aggregation,
        input_dim_src_nodes=hidden_dim,
        input_dim_dst_nodes=hidden_dim,
        input_dim_edges=hidden_dim,
        output_dim_dst_nodes=hidden_dim,
        output_dim_edges=hidden_dim,
        hidden_dim=hidden_dim,
        hidden_layers=1,
    ).to(device)

    # Create dummy features.
    grid_nodes = input_res[0] * input_res[1]
    mesh_nodes = len(graph_dgl.mesh_vertices)

    grid_features = torch.randn(grid_nodes, hidden_dim, device=device)
    mesh_features = torch.randn(mesh_nodes, hidden_dim, device=device)

    # Get edge features.
    g2m_edge_features_dgl = torch.randn(
        g2m_graph_dgl.num_edges(), hidden_dim, device=device
    )
    g2m_edge_features_pyg = g2m_edge_features_dgl.clone()

    # Test encoder.
    grid_out_dgl, mesh_out_dgl = encoder(
        g2m_edge_features_dgl, grid_features, mesh_features, g2m_graph_dgl
    )

    grid_out_pyg, mesh_out_pyg = encoder(
        g2m_edge_features_pyg,
        grid_features.clone(),
        mesh_features.clone(),
        g2m_graph_pyg,
    )

    # Verify encoder outputs are equivalent (allowing for small numerical differences)
    assert_close(grid_out_dgl, grid_out_pyg, rtol=1e-3, atol=1e-4)
    assert_close(mesh_out_dgl, mesh_out_pyg, rtol=1e-3, atol=1e-4)

    # Test decoder with encoder outputs
    m2g_edge_features_dgl = torch.randn(
        m2g_graph_dgl.num_edges(), hidden_dim, device=device
    )
    m2g_edge_features_pyg = m2g_edge_features_dgl.clone()

    grid_decoded_dgl = decoder(
        m2g_edge_features_dgl, grid_out_dgl, mesh_out_dgl, m2g_graph_dgl
    )

    grid_decoded_pyg = decoder(
        m2g_edge_features_pyg, grid_out_pyg, mesh_out_pyg, m2g_graph_pyg
    )

    # Verify decoder outputs are equivalent
    assert_close(grid_decoded_dgl, grid_decoded_pyg, rtol=1e-3, atol=1e-4)


@import_or_fail(["dgl", "torch_geometric"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@torch.no_grad()
def test_graphcast_multimesh_dgl_pyg_equivalence(
    device, pytestconfig, set_physicsnemo_force_te
):
    """Test that GraphCast with multimesh produces equivalent outputs for DGL and PyG backends."""
    # (DGL2PYG): remove this once DGL is removed.

    from physicsnemo.models.graphcast.graph_cast_net import GraphCastNet

    # Set seeds for reproducibility.
    fix_random_seeds()

    # Test parameters with multimesh.
    model_kwargs = {
        "mesh_level": 2,
        "multimesh": True,
        "input_res": (6, 12),
        "input_dim_grid_nodes": 3,
        "input_dim_mesh_nodes": 3,
        "input_dim_edges": 4,
        "output_dim_grid_nodes": 3,
        "processor_layers": 3,
        "hidden_dim": 8,
        "hidden_layers": 1,
        "do_concat_trick": False,
    }

    # Create input data
    x = create_random_input(
        model_kwargs["input_res"], model_kwargs["input_dim_grid_nodes"]
    ).to(device)

    # Create models.
    model_dgl = GraphCastNet(**model_kwargs, graph_backend="dgl").to(device)
    model_pyg = GraphCastNet(**model_kwargs, graph_backend="pyg").to(device)

    # Copy weights to ensure identical models.
    model_pyg.load_state_dict(model_dgl.state_dict())

    # Forward pass with both backends.
    output_dgl = model_dgl(x)
    output_pyg = model_pyg(x)

    # Verify outputs are equivalent.
    assert_close(output_dgl, output_pyg, rtol=1e-4, atol=1e-5)

    # Verify output shapes.
    assert output_dgl.shape == output_pyg.shape


@import_or_fail(["dgl", "torch_geometric"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@torch.no_grad()
def test_graphcast_different_resolutions_dgl_pyg_equivalence(
    device, pytestconfig, set_physicsnemo_force_te
):
    """Test that GraphCast with different input resolutions produces equivalent outputs for DGL and PyG backends."""
    # (DGL2PYG): remove this once DGL is removed.

    from physicsnemo.models.graphcast.graph_cast_net import GraphCastNet

    # Test different resolutions.
    resolutions = [(4, 8), (6, 12), (8, 16)]

    for input_res in resolutions:
        # Set seeds for reproducibility.
        fix_random_seeds()

        # Test parameters.
        model_kwargs = {
            "mesh_level": 1,
            "input_res": input_res,
            "input_dim_grid_nodes": 2,
            "input_dim_mesh_nodes": 3,
            "input_dim_edges": 4,
            "output_dim_grid_nodes": 2,
            "processor_layers": 3,
            "hidden_dim": 6,
            "hidden_layers": 1,
            "do_concat_trick": False,
        }

        # Create input data.
        x = create_random_input(
            model_kwargs["input_res"], model_kwargs["input_dim_grid_nodes"]
        )
        x = x.to(device)

        # Create models.
        model_dgl = GraphCastNet(**model_kwargs, graph_backend="dgl").to(device)
        model_pyg = GraphCastNet(**model_kwargs, graph_backend="pyg").to(device)

        # Copy weights to ensure identical models.
        model_pyg.load_state_dict(model_dgl.state_dict())

        # Forward pass with both backends.
        output_dgl = model_dgl(x)
        output_pyg = model_pyg(x)

        # Verify outputs are equivalent.
        assert_close(output_dgl, output_pyg, rtol=1e-4, atol=1e-5)

        # Verify output shapes.
        assert output_dgl.shape == output_pyg.shape
        expected_shape = (1, model_kwargs["output_dim_grid_nodes"], *input_res)
        assert output_dgl.shape == expected_shape
