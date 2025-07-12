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

import numpy as np
import pytest
import torch
from pytest_utils import import_or_fail

from . import common


@pytest.fixture
def data_dir(nfs_data_dir):
    return nfs_data_dir.joinpath("datasets/vortex_shedding/cylinder_flow")


@import_or_fail(["tensorflow"])
@pytest.mark.parametrize(
    "split, num_nodes, num_edges",
    [("train", 1876, 10788), ("valid", 1896, 10908), ("test", 1923, 11070)],
)
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_vortex_shedding_constructor(
    data_dir, split, num_nodes, num_edges, device, pytestconfig
):
    from physicsnemo.datapipes.gnn.vortex_shedding_dataset import VortexSheddingDataset

    num_samples = 2
    num_steps = 4
    dataset = VortexSheddingDataset(
        data_dir=data_dir,
        split=split,
        num_samples=num_samples,
        num_steps=num_steps,
    )

    common.check_datapipe_iterable(dataset)
    assert len(dataset) == num_samples * (num_steps - 1)
    x0 = dataset[0]
    # For validation and test splits, the dataset returns
    # a tuple of (graph, cells, rollout_mask).
    if split != "train":
        x0, *_ = x0
    assert x0.x.shape == (num_nodes, 6)
    assert x0.y.shape == (num_nodes, 3)
    assert x0.edge_index.shape == (2, num_edges)
    assert x0.edge_attr.shape == (num_edges, 3)
    if split != "train":
        assert x0.mesh_pos.shape == (num_nodes, 2)


@import_or_fail(["tensorflow", "dgl"])
@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_vortex_shedding_dgl_pyg_equivalence(data_dir, split, pytestconfig):
    """Test that PyG and DGL versions of VortexSheddingDataset produce equivalent outputs."""
    # (DGL2PYG): remove this once DGL is removed.

    from physicsnemo.datapipes.gnn.vortex_shedding_dataset import (
        VortexSheddingDataset as VortexSheddingDatasetPyG,
    )
    from physicsnemo.datapipes.gnn.vortex_shedding_dataset_dgl import (
        VortexSheddingDataset as VortexSheddingDatasetDGL,
    )

    # Use small dataset for testing.
    num_samples = 2
    num_steps = 4
    noise_std = 0.0

    # Create both datasets with identical parameters.
    dataset_pyg = VortexSheddingDatasetPyG(
        data_dir=data_dir,
        split=split,
        num_samples=num_samples,
        num_steps=num_steps,
        noise_std=noise_std,
    )

    dataset_dgl = VortexSheddingDatasetDGL(
        data_dir=data_dir,
        split=split,
        num_samples=num_samples,
        num_steps=num_steps,
        noise_std=noise_std,
    )

    # Check that datasets have the same length.
    assert len(dataset_pyg) == len(dataset_dgl)

    # Test multiple samples.
    for idx in [0, 1, len(dataset_pyg) - 1]:
        pyg_graph = dataset_pyg[idx]
        dgl_graph = dataset_dgl[idx]

        if split != "train":
            # For non-train splits, unpack the tuple.
            pyg_graph, *_ = pyg_graph
            dgl_graph, *_ = dgl_graph

        # Compare node features (x)
        assert (pyg_graph.x == dgl_graph.ndata["x"]).all()

        # Compare node targets (y)
        assert (pyg_graph.y == dgl_graph.ndata["y"]).all()

        # Compare edge attributes
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
