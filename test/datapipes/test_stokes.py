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

Tensor = torch.Tensor


@pytest.fixture
def data_dir(nfs_data_dir):
    return nfs_data_dir.joinpath("datasets/stokes")


@import_or_fail(["vtk", "pyvista", "dgl"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_stokes_constructor(data_dir, device, pytestconfig):
    from physicsnemo.datapipes.gnn.stokes_dataset import StokesDataset

    # construct dataset
    dataset = StokesDataset(
        data_dir=data_dir,
        split="train",
        num_samples=2,
    )

    # iterate datapipe is iterable
    common.check_datapipe_iterable(dataset)

    # check for failure from invalid dir
    try:
        # init dataset with empty path
        # if dataset throws an IO error then this should pass
        dataset = StokesDataset(
            data_dir="/null_path",
            split="train",
            num_samples=2,
        )
        raise IOError("Failed to raise error given null data path")
    except IOError:
        pass

    # check invalid split
    try:
        # if dataset throws an IO error then this should pass
        dataset = StokesDataset(
            data_dir=data_dir,
            invar_keys=[
                "pos",
                "markers",
            ],
            split="valid",
            num_samples=2,
        )
        raise IOError("Failed to raise error given invalid split")
    except IOError:
        pass


@import_or_fail(["vtk", "pyvista", "dgl", "torch_geometric", "torch_scatter"])
@pytest.mark.parametrize("split", ["train"])
def test_stokes_dgl_pyg_equivalence(data_dir, split, pytestconfig):
    """Test that PyG and DGL versions of StokesDataset produce equivalent outputs."""
    # (DGL2PYG): remove this once DGL is removed.

    from physicsnemo.datapipes.gnn.stokes_dataset import (
        StokesDataset as StokesDatasetPyG,
    )
    from physicsnemo.datapipes.gnn.stokes_dataset_dgl import (
        StokesDataset as StokesDatasetDGL,
    )

    # Use small dataset for testing.
    num_samples = 2

    # Create both datasets with identical parameters.
    dataset_pyg = StokesDatasetPyG(
        data_dir=data_dir,
        split=split,
        num_samples=num_samples,
        invar_keys=["pos", "marker"],
        outvar_keys=["u", "v", "p"],
        normalize_keys=["u", "v", "p"],
    )

    dataset_dgl = StokesDatasetDGL(
        data_dir=data_dir,
        split=split,
        num_samples=num_samples,
        invar_keys=["pos", "marker"],
        outvar_keys=["u", "v", "p"],
        normalize_keys=["u", "v", "p"],
        force_reload=False,
        name="dataset_dgl",
        verbose=False,
    )

    # Check that datasets have the same length.
    assert len(dataset_pyg) == len(dataset_dgl)

    # Test multiple samples.
    for idx in [0, len(dataset_pyg) - 1]:
        pyg_graph = dataset_pyg[idx]
        dgl_graph = dataset_dgl[idx]

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
