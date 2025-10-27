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

"""
Unit tests for the HydroGraphDataset datapipe.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from pytest_utils import import_or_fail
from torch.testing import assert_close

from . import common

Tensor = torch.Tensor


@pytest.fixture(scope="session")
def hydrograph_data_dir(nfs_data_dir, tmp_path_factory):
    """
    Make a **writable copy** of the tiny HydroGraph dataset so tests can
    freely create cache files without touching the pristine NFS copy.
    """
    src = nfs_data_dir.joinpath("datasets/hydrographnet_tiny")
    dst = tmp_path_factory.mktemp("hydrograph_unit_test")
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return Path(dst)


@import_or_fail(["torch_geometric", "torch_scatter", "scipy", "tqdm"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_hydrograph_constructor(hydrograph_data_dir, device, pytestconfig):
    """Constructor & basic iteration checks."""

    from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset

    # -- build a tiny train‑split dataset ------------------------------------
    dataset = HydroGraphDataset(
        data_dir=hydrograph_data_dir,
        split="train",
        num_samples=2,
        n_time_steps=2,
        k=2,
        noise_type="none",
    )

    common.check_datapipe_iterable(dataset)
    assert len(dataset) > 0

    sample = dataset[0]
    if isinstance(sample, tuple):  # physics / push‑forward mode
        g, physics = sample
        assert isinstance(physics, dict)
    else:
        g = sample
    assert g.x.shape[0] == g.num_nodes
    assert g.edge_attr.shape[0] == g.num_edges

    # -- invalid split --------------------------------------------------------
    with pytest.raises(ValueError):
        _ = HydroGraphDataset(
            data_dir=hydrograph_data_dir,
            split="validation",
            num_samples=1,
        )

    # -- test‑split rollout length -------------------------------------------
    rollout_len = 5
    test_ds = HydroGraphDataset(
        data_dir=hydrograph_data_dir,
        split="test",
        num_samples=1,
        n_time_steps=2,
        rollout_length=rollout_len,
    )
    g_test, rollout = test_ds[0]
    for key in ["inflow", "precipitation", "water_depth_gt", "volume_gt"]:
        assert rollout[key].shape[0] == rollout_len
    assert g_test.num_nodes > 0


@import_or_fail(["torch_geometric", "torch_scatter", "scipy", "tqdm", "dgl"])
@pytest.mark.parametrize("split", ["train", "test"])
def test_hydrographnet_dgl_pyg_equivalence(hydrograph_data_dir, split, pytestconfig):
    """Test that PyG and DGL versions of HydroGraphDataset produce equivalent outputs."""
    # (DGL2PYG): remove this once DGL is removed.

    from physicsnemo.datapipes.gnn.hydrographnet_dataset import (
        HydroGraphDataset as HydroGraphDatasetPyG,
    )
    from physicsnemo.datapipes.gnn.hydrographnet_dataset_dgl import (
        HydroGraphDataset as HydroGraphDatasetDGL,
    )

    # Use small dataset for testing.
    num_samples = 2
    n_time_steps = 2
    k = 2
    noise_std = 0.0
    rollout_length = 3 if split == "test" else None

    # Create both datasets with identical parameters.
    dataset_pyg = HydroGraphDatasetPyG(
        data_dir=hydrograph_data_dir,
        split=split,
        num_samples=num_samples,
        n_time_steps=n_time_steps,
        k=k,
        noise_std=noise_std,
        rollout_length=rollout_length,
    )

    dataset_dgl = HydroGraphDatasetDGL(
        data_dir=hydrograph_data_dir,
        split=split,
        num_samples=num_samples,
        n_time_steps=n_time_steps,
        k=k,
        noise_std=noise_std,
        rollout_length=rollout_length,
    )

    # Check that datasets have the same length.
    assert len(dataset_pyg) == len(dataset_dgl)

    # Test multiple samples.
    for idx in [0, len(dataset_pyg) - 1]:
        pyg_result = dataset_pyg[idx]
        dgl_result = dataset_dgl[idx]

        if split == "test":
            # For test split, unpack the tuple (graph, rollout_data).
            pyg_graph, pyg_rollout = pyg_result
            dgl_graph, dgl_rollout = dgl_result

            # Compare rollout data.
            for key in ["inflow", "precipitation", "water_depth_gt", "volume_gt"]:
                assert_close(pyg_rollout[key], dgl_rollout[key])
        else:
            # For train split, could be just graph or (graph, physics_data).
            if isinstance(pyg_result, tuple):
                pyg_graph, pyg_physics = pyg_result
                dgl_graph, dgl_physics = dgl_result
                # Compare physics data if present.
                for key in pyg_physics.keys():
                    assert abs(pyg_physics[key] - dgl_physics[key]) < 1e-5
            else:
                pyg_graph = pyg_result
                dgl_graph = dgl_result

        # Compare node features (x).
        assert_close(pyg_graph.x, dgl_graph.ndata["x"])

        # Compare node targets (y) for train split.
        if split != "test":
            assert_close(pyg_graph.y, dgl_graph.ndata["y"])

        # Compare edge attributes.
        assert_close(pyg_graph.edge_attr, dgl_graph.edata["x"])

        # Compare graph structure (edge connectivity).
        # Convert DGL graph to PyG format for comparison.
        dgl_src, dgl_dst = dgl_graph.edges()
        dgl_edge_index = torch.stack([dgl_src, dgl_dst], dim=0).long()

        # Sort edges for consistent comparison (both should have same connectivity).
        pyg_sorted_idx = np.lexsort(
            (pyg_graph.edge_index[1].numpy(), pyg_graph.edge_index[0].numpy())
        )
        dgl_sorted_idx = np.lexsort((dgl_edge_index[1], dgl_edge_index[0]))

        assert_close(
            pyg_graph.edge_index[:, pyg_sorted_idx],
            dgl_edge_index[:, dgl_sorted_idx],
        )

        # Verify the edge attributes are also in the same order.
        assert_close(
            pyg_graph.edge_attr[pyg_sorted_idx],
            dgl_graph.edata["x"][dgl_sorted_idx],
        )
