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

"""
Unit tests for the HydroGraphDataset datapipe.
"""

import shutil
from pathlib import Path

import pytest
import torch
from pytest_utils import import_or_fail

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


@import_or_fail(["dgl", "scipy", "tqdm"])
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
        verbose=False,
    )

    common.check_datapipe_iterable(dataset)
    assert len(dataset) > 0

    sample = dataset[0]
    if isinstance(sample, tuple):  # physics / push‑forward mode
        g, physics = sample
        assert isinstance(physics, dict)
    else:
        g = sample
    assert g.ndata["x"].shape[0] == g.num_nodes()
    assert g.edata["x"].shape[0] == g.num_edges()

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
    assert g_test.num_nodes() > 0
