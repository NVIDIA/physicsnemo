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

import pytest
import torch
from pytest_utils import import_or_fail

dgl = pytest.importorskip("dgl")


@pytest.fixture
def ahmed_data_dir(nfs_data_dir):
    return nfs_data_dir.joinpath("datasets/ahmed_body")


@import_or_fail("sparse_dot_mkl")
def test_bsms_init(pytestconfig):
    from physicsnemo.datapipes.gnn.bsms import BistrideMultiLayerGraph

    torch.manual_seed(1)

    # Create a simple graph.
    num_nodes = 4
    edges = (
        torch.arange(num_nodes - 1),
        torch.arange(num_nodes - 1) + 1,
    )
    pos = torch.randn((num_nodes, 3))

    graph = dgl.graph(edges)
    graph = dgl.to_bidirected(graph)

    graph.ndata["pos"] = pos

    # Convert to multi-scale graph.
    num_layers = 1
    gloader = BistrideMultiLayerGraph(graph, num_layers)

    # Get multi-scale graphs.
    ms_gs, ms_edges, ms_ids = gloader.get_multi_layer_graphs()

    assert len(ms_gs) == 2, "Expected 2 graphs."
    assert len(ms_edges) == 2, "Expected 2 graphs."
    assert len(ms_ids) == 1, "Expected 1 subsampled graph."


@import_or_fail("sparse_dot_mkl")
def test_bsms_ahmed_dataset(pytestconfig, ahmed_data_dir):
    from physicsnemo.datapipes.gnn.ahmed_body_dataset import AhmedBodyDataset
    from physicsnemo.datapipes.gnn.bsms import BistrideMultiLayerGraphDataset

    split = "train"
    # Construct multi-scale dataset out of standard Ahmed Body dataset.
    ahmed_dataset = AhmedBodyDataset(
        data_dir=ahmed_data_dir,
        split=split,
        num_samples=2,
    )

    num_levels = 2
    ms_dataset = BistrideMultiLayerGraphDataset(
        ahmed_dataset,
        num_levels,
    )

    assert len(ms_dataset) == 2

    g0 = ms_dataset[0]
    assert g0["graph"].num_nodes() == 70661
    assert len(g0["ms_edges"]) == 3
    assert len(g0["ms_ids"]) == 2


@import_or_fail("sparse_dot_mkl")
def test_bsms_ahmed_dataset_caching(pytestconfig, ahmed_data_dir, tmp_path):
    from physicsnemo.datapipes.gnn.ahmed_body_dataset import AhmedBodyDataset
    from physicsnemo.datapipes.gnn.bsms import BistrideMultiLayerGraphDataset

    split = "train"
    # Construct multi-scale dataset out of standard Ahmed Body dataset.
    ahmed_dataset = AhmedBodyDataset(
        data_dir=ahmed_data_dir,
        split=split,
        num_samples=2,
    )

    num_levels = 2

    ms_dataset1 = BistrideMultiLayerGraphDataset(
        ahmed_dataset,
        num_levels,
        cache_dir=None,
    )

    ms_dataset2 = BistrideMultiLayerGraphDataset(
        ahmed_dataset,
        num_levels,
        cache_dir=tmp_path / split,
    )

    # Non-caching dataset.
    g0_1 = ms_dataset1[0]
    # First pass - cache is empty.
    g0_2 = ms_dataset2[0]
    assert (g0_1["ms_edges"][0] == g0_2["ms_edges"][0]).all()
    assert (g0_1["ms_ids"][0] == g0_2["ms_ids"][0]).all()
    # Second pass - should read from the cache.
    g0_2 = ms_dataset2[0]
    assert (g0_1["ms_edges"][0] == g0_2["ms_edges"][0]).all()
    assert (g0_1["ms_ids"][0] == g0_2["ms_ids"][0]).all()
