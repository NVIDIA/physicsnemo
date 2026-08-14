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

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent))

import datapipe
import zarr_reader


def _create_store(store_path: Path, num_timesteps: int = 4, num_nodes: int = 5):
    store_path.mkdir(exist_ok=True)
    mesh_pos = np.random.randn(num_timesteps, num_nodes, 3).astype(np.float32)
    thickness = np.ones(num_nodes, dtype=np.float32)
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    store = zarr.open(str(store_path), mode="w")
    store.create_array("mesh_pos", data=mesh_pos)
    store.create_array("thickness", data=thickness)
    store.create_array("edges", data=edges)
    return mesh_pos


@pytest.fixture
def lazy_zarr_dir():
    """Temporary directory with two mock Zarr crash stores."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        for i in range(2):
            _create_store(temp_path / f"Run{i:03d}.zarr")
        yield temp_dir


def test_lazy_datapipe_defers_materialization(lazy_zarr_dir):
    """Lazy datasets keep mesh tensors unset until __getitem__ is called."""
    reader = zarr_reader.Reader(lazy_load=True)
    stats_dir = Path(lazy_zarr_dir) / "stats"

    dataset = datapipe.CrashPointCloudDataset(
        reader=reader,
        data_dir=lazy_zarr_dir,
        split="train",
        num_samples=2,
        num_steps=4,
        static_features=["thickness"],
        stats_dir=str(stats_dir),
    )

    assert dataset._lazy_mode is True
    assert dataset.mesh_pos_seq == [None, None]

    sample = dataset[0]
    assert sample.node_features["coords"].shape == (5, 3)
    assert dataset.mesh_pos_seq[0] is not None
    assert dataset.mesh_pos_seq[1] is None
