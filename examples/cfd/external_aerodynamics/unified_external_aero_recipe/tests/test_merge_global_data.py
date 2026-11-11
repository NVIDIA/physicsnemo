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

"""Tests for the recipe-local global-data mesh reader."""

import torch
from merge_global_data import MeshReaderWithGlobalData
from tensordict import TensorDict

from physicsnemo.mesh import Mesh


def test_merge_global_data_preserves_geometry_caches(tmp_path):
    """Merging case metadata leaves reusable geometry caches populated."""
    mesh = Mesh(
        points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        cells=torch.tensor([[0, 1, 2]]),
        global_data={"existing": torch.tensor(1.0)},
    )
    expected_centroids = mesh.cell_centroids.clone()
    mesh.save(tmp_path / "sample.pmsh")
    TensorDict(
        {"external": torch.tensor(2.0)},
        batch_size=[],
    ).save(tmp_path / "global_data")

    reader = MeshReaderWithGlobalData(
        tmp_path,
        pattern="*.pmsh",
        merge_global_data_from="../global_data",
    )
    result = reader._load_sample(0)

    assert set(result.global_data.keys()) == {"existing", "external"}
    assert ("cell", "centroids") in result._cache.keys(
        include_nested=True,
        leaves_only=True,
    )
    torch.testing.assert_close(result.cell_centroids, expected_centroids)
