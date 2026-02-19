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

"""Tests for Mesh caching behavior.

Tests validate that mesh._cache (nested TensorDict with "cell" and "point"
sub-TensorDicts) correctly stores and retrieves cached computed values.
"""

import pytest
import torch

from physicsnemo.mesh import Mesh


class TestFreshMeshEmptyCache:
    """Tests that a freshly constructed Mesh has empty caches."""

    def test_fresh_mesh_cell_cache_empty(self):
        """Test that a freshly constructed Mesh has empty cell cache."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        assert len(mesh._cache["cell"].keys()) == 0

    def test_fresh_mesh_point_cache_empty(self):
        """Test that a freshly constructed Mesh has empty point cache."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        assert len(mesh._cache["point"].keys()) == 0


class TestAccessPopulatesCache:
    """Tests that accessing computed properties populates mesh._cache."""

    def test_cell_centroids_populates_cache(self):
        """Test that accessing mesh.cell_centroids populates mesh._cache['cell', 'centroids']."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        assert mesh._cache.get(("cell", "centroids"), None) is None

        _ = mesh.cell_centroids

        assert "centroids" in mesh._cache["cell"].keys()
        assert mesh._cache["cell", "centroids"] is not None
        assert mesh._cache["cell", "centroids"].shape == (1, 2)

    def test_cell_areas_populates_cache(self):
        """Test that accessing mesh.cell_areas populates mesh._cache['cell', 'areas']."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        assert mesh._cache.get(("cell", "areas"), None) is None

        _ = mesh.cell_areas

        assert "areas" in mesh._cache["cell"].keys()
        assert mesh._cache["cell", "areas"] is not None
        assert mesh._cache["cell", "areas"].shape == (1,)


class TestCustomValueOverride:
    """Tests that writing to mesh._cache overrides property return values."""

    def test_custom_centroids_returned(self):
        """Test that writing mesh._cache['cell', 'centroids'] = custom_value makes mesh.cell_centroids return it."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        custom_value = torch.tensor([[99.0, 99.0]], dtype=torch.float32)
        mesh._cache["cell", "centroids"] = custom_value

        result = mesh.cell_centroids
        assert torch.equal(result, custom_value)


class TestCacheGet:
    """Tests for mesh._cache.get(('cell', key), None) and ('point', key)."""

    def test_get_returns_none_when_not_set(self):
        """Test that _cache.get returns None when key is not in cache."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        result = mesh._cache.get(("cell", "areas"), None)
        assert result is None

    def test_get_returns_value_when_set(self):
        """Test that _cache.get returns the cached value when present."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        custom_value = torch.tensor([42.0], dtype=torch.float32)
        mesh._cache["cell", "areas"] = custom_value

        result = mesh._cache.get(("cell", "areas"), None)
        assert result is not None
        assert torch.equal(result, custom_value)


class TestCacheStore:
    """Tests for storing values in mesh._cache."""

    def test_store_creates_entry(self):
        """Test that assigning to _cache creates the entry."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        value = torch.randn(1, 2)
        mesh._cache["cell", "centroids"] = value

        assert "centroids" in mesh._cache["cell"].keys()
        assert torch.equal(mesh._cache["cell", "centroids"], value)

    def test_store_overwrites_existing(self):
        """Test that assigning overwrites existing cached value."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        old_value = torch.randn(1, 2)
        new_value = torch.randn(1, 2)
        mesh._cache["cell", "centroids"] = old_value
        mesh._cache["cell", "centroids"] = new_value

        stored = mesh._cache["cell", "centroids"]
        assert torch.equal(stored, new_value)
        assert not torch.equal(stored, old_value)

    def test_store_multiple_keys(self):
        """Test that multiple keys can be stored in cell cache."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        centroids = torch.randn(1, 2)
        areas = torch.randn(1)
        normals = torch.randn(1, 2)
        mesh._cache["cell", "centroids"] = centroids
        mesh._cache["cell", "areas"] = areas
        mesh._cache["cell", "normals"] = normals

        assert torch.equal(mesh._cache["cell", "centroids"], centroids)
        assert torch.equal(mesh._cache["cell", "areas"], areas)
        assert torch.equal(mesh._cache["cell", "normals"], normals)


class TestCacheCellPointSeparation:
    """Tests that cell and point caches are separate."""

    def test_cell_and_point_caches_independent(self):
        """Test that cell and point caches are independent sub-TensorDicts."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        mesh._cache["cell", "centroids"] = torch.randn(1, 2)
        mesh._cache["point", "normals"] = torch.randn(3, 2)

        assert "centroids" in mesh._cache["cell"].keys()
        assert "normals" in mesh._cache["point"].keys()
        assert "centroids" not in mesh._cache["point"].keys()
        assert "normals" not in mesh._cache["cell"].keys()


class TestCacheDevices:
    """Tests for device handling in cache operations."""

    def test_cache_cpu(self):
        """Test caching on CPU mesh."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        _ = mesh.cell_centroids
        cached = mesh._cache["cell", "centroids"]

        assert cached is not None
        assert cached.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cache_cuda(self):
        """Test caching on CUDA mesh."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32, device="cuda"
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device="cuda")
        mesh = Mesh(points=points, cells=cells)

        _ = mesh.cell_centroids
        cached = mesh._cache["cell", "centroids"]

        assert cached is not None
        assert cached.device.type == "cuda"
