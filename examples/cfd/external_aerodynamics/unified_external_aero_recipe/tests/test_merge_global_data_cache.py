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

"""Tests for MeshReaderWithGlobalData's DatasetCache integration.

The external ``global_data`` tensordict (a handful of freestream scalars)
is otherwise re-read from storage on every sample load, every epoch --
the canonical small-read hot spot the reader cache exists to remove.
"""

from __future__ import annotations

import pytest
import torch
from merge_global_data import MeshReaderWithGlobalData
from tensordict import TensorDict

from physicsnemo.datapipes.caching import DatasetCache
from physicsnemo.mesh.primitives.basic import two_triangles_2d


@pytest.fixture
def data_root(tmp_path):
    """Two cases, each: a boundary mesh plus a sibling global_data tensordict."""
    root = tmp_path / "data"
    base = two_triangles_2d.load()
    for i in range(2):
        case = root / f"run_{i}"
        case.mkdir(parents=True)
        base.clone().save(case / "vehicle.pmsh")
        TensorDict(
            {
                "U_inf": torch.tensor(30.0 + i),
                "rho_inf": torch.tensor(1.2),
            },
            batch_size=[],
        ).memmap_(str(case / "global_data"))
    return root


def _read_all(reader):
    return [reader[i][0] for i in range(len(reader))]


class TestMergeGlobalDataCache:
    """DatasetCache integration for MeshReaderWithGlobalData."""

    def test_matches_uncached(self, data_root, tmp_path):
        """Cached reads (mesh + merged globals) match uncached, cold and warm."""
        kwargs = dict(
            pattern="run_*/vehicle.pmsh",
            merge_global_data_from="../global_data",
        )
        plain = MeshReaderWithGlobalData(data_root, **kwargs)
        cached = MeshReaderWithGlobalData(
            data_root,
            **kwargs,
            cache=DatasetCache(ram_bytes_limit=2**24, disk_dir=tmp_path / "cache"),
        )
        for _ in range(2):  # cold then warm
            for a, b in zip(_read_all(plain), _read_all(cached)):
                assert torch.equal(a.points, b.points)
                assert torch.equal(a.global_data["U_inf"], b.global_data["U_inf"])
                assert torch.equal(a.global_data["rho_inf"], b.global_data["rho_inf"])

    def test_external_read_cached_per_directory(self, data_root, monkeypatch):
        """Warm reads never re-load the external global_data tensordict."""
        cache = DatasetCache(ram_bytes_limit=2**24)
        reader = MeshReaderWithGlobalData(
            data_root,
            pattern="run_*/vehicle.pmsh",
            merge_global_data_from="../global_data",
            cache=cache,
        )
        expected = _read_all(reader)  # populate

        def boom(*args, **kwargs):
            raise AssertionError("external global_data re-read on warm sample")

        monkeypatch.setattr(TensorDict, "load_memmap", boom)
        for a, b in zip(expected, _read_all(reader)):
            assert torch.equal(a.global_data["U_inf"], b.global_data["U_inf"])

    def test_collision_still_raises(self, data_root):
        """The cache must not mask the key-collision data-layer check."""
        base = two_triangles_2d.load()
        base.global_data["U_inf"] = torch.tensor(1.0)  # collides with external
        base.save(data_root / "run_0" / "vehicle.pmsh")

        reader = MeshReaderWithGlobalData(
            data_root,
            pattern="run_*/vehicle.pmsh",
            merge_global_data_from="../global_data",
            cache=DatasetCache(ram_bytes_limit=2**24),
        )
        with pytest.raises(ValueError, match="collision"):
            reader[0]
        # And repeatedly: the collision check runs on every read, warm or not.
        with pytest.raises(ValueError, match="collision"):
            reader[0]
