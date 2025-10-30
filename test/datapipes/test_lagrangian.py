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

import numpy as np
import pytest
import torch
from pytest_utils import import_or_fail

from . import common

dgl = pytest.importorskip("dgl")


Tensor = torch.Tensor


@pytest.fixture
def data_dir(nfs_data_dir):
    return nfs_data_dir.joinpath("datasets/water")


@import_or_fail(["tensorflow", "torch_geometric", "torch_scatter"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_lagrangian_dataset_constructor(data_dir, device, pytestconfig):
    from torch_geometric.data import Data as PyGData

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
    assert isinstance(graph, PyGData)

    # Test graph properties
    assert graph.x.shape[-1] > 0  # node features
    assert graph.y.shape[-1] > 0  # node targets


@import_or_fail(["tensorflow", "dgl"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_lagrangian_dataset_constructor_dgl(data_dir, device, pytestconfig):
    from physicsnemo.datapipes.gnn.lagrangian_dataset_dgl import LagrangianDataset

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


@import_or_fail(["tensorflow", "torch_geometric", "torch_scatter"])
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


@import_or_fail(["tensorflow", "dgl", "torch_geometric", "torch_scatter"])
@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_lagrangian_dgl_pyg_equivalence(data_dir, split, pytestconfig):
    """Test that PyG and DGL versions of LagrangianDataset produce equivalent outputs."""
    # (DGL2PYG): remove this once DGL is removed.

    from physicsnemo.datapipes.gnn.lagrangian_dataset import (
        LagrangianDataset as LagrangianDatasetPyG,
    )
    from physicsnemo.datapipes.gnn.lagrangian_dataset_dgl import (
        LagrangianDataset as LagrangianDatasetDGL,
    )

    # Use small dataset for testing.
    num_sequences = 2
    num_steps = 10
    noise_std = 0.0

    # Create both datasets with identical parameters.
    dataset_pyg = LagrangianDatasetPyG(
        data_dir=data_dir,
        split=split,
        num_sequences=num_sequences,
        num_steps=num_steps,
        noise_std=noise_std,
    )

    dataset_dgl = LagrangianDatasetDGL(
        data_dir=data_dir,
        split=split,
        num_sequences=num_sequences,
        num_steps=num_steps,
        noise_std=noise_std,
    )

    # Check that datasets have the same length.
    assert len(dataset_pyg) == len(dataset_dgl)

    # Test multiple samples.
    for idx in [0, 1, len(dataset_pyg) - 1]:
        pyg_graph = dataset_pyg[idx]
        dgl_graph = dataset_dgl[idx]

        # Compare node features (x).
        assert (pyg_graph.x == dgl_graph.ndata["x"]).all()

        # Compare node targets (y).
        assert (pyg_graph.y == dgl_graph.ndata["y"]).all()

        # Compare positions.
        assert (pyg_graph.pos == dgl_graph.ndata["pos"]).all()

        # Compare masks.
        assert (pyg_graph.mask == dgl_graph.ndata["mask"]).all()

        # Compare edge attributes.
        assert (pyg_graph.edge_attr == dgl_graph.edata["x"]).all()

        # Compare graph structure (edge connectivity).
        # Convert DGL graph to PyG format for comparison.
        dgl_src, dgl_dst = dgl_graph.edges()
        dgl_edge_index = torch.stack([dgl_src, dgl_dst], dim=0).long()

        # Sort edges for consistent comparison (both should have same connectivity).
        pyg_sorted_idx = np.lexsort(
            (pyg_graph.edge_index[1].numpy(), pyg_graph.edge_index[0].numpy())
        )
        dgl_sorted_idx = np.lexsort((dgl_edge_index[1], dgl_edge_index[0]))

        torch.testing.assert_close(
            pyg_graph.edge_index[:, pyg_sorted_idx],
            dgl_edge_index[:, dgl_sorted_idx],
        )

        # Verify the edge attributes are also in the same order.
        torch.testing.assert_close(
            pyg_graph.edge_attr[pyg_sorted_idx],
            dgl_graph.edata["x"][dgl_sorted_idx],
        )
