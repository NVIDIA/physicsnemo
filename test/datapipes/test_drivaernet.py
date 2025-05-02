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

from . import common

Tensor = torch.Tensor


@pytest.fixture
def data_dir(nfs_data_dir):
    return nfs_data_dir.joinpath("datasets/drivaernet/")


@import_or_fail(["vtk", "pyvista", "dgl"])
@pytest.mark.parametrize("cache_graph", [True, False])
def test_drivaernet_init(data_dir, cache_graph, tmp_path, pytestconfig):

    from physicsnemo.datapipes.gnn.drivaernet_dataset import DrivAerNetDataset

    cache_dir = tmp_path / "cache" if cache_graph else None
    # Construct dataset
    dataset = DrivAerNetDataset(
        data_dir=data_dir,
        split="train",
        num_samples=1,
        cache_dir=cache_dir,
    )

    # Check if datapipe is iterable. This will iterate over the dataset
    # and cache graphs, if requested.
    assert common.check_datapipe_iterable(dataset)

    assert len(dataset) == 1

    # Get the first graph with the corresponding C_d.
    sample = dataset[0]
    g0 = sample["graph"]
    c_d_0 = sample["c_d"].item()

    # Some simple checks.
    assert g0.ndata["x"].shape[0] == g0.ndata["y"].shape[0]

    torch.testing.assert_close(c_d_0, 0.346311)
