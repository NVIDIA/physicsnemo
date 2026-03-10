# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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

from test.conftest import requires_module

from . import common


@pytest.fixture
def data_dir(nfs_data_dir):
    return nfs_data_dir.joinpath("datasets/vortex_shedding/cylinder_flow")


@requires_module(["tfrecord"])
@pytest.mark.parametrize(
    "split, num_nodes, num_edges",
    [("train", 1876, 10788), ("valid", 1896, 10908), ("test", 1923, 11070)],
)
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
