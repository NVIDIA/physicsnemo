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


class TaggedMesh(Mesh):
    """Custom mesh subtype used to guard functional update behavior."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "constructed_spatial_dims", self.n_spatial_dims)

    def tag(self) -> str:
        return "tagged"


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


class TestStripCaches:
    """Tests for selective cache removal."""

    def test_keep_retains_only_selected_nested_keys(self):
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)
        _ = mesh.cell_areas
        _ = mesh.cell_centroids
        mesh._cache["point", "custom"] = torch.ones(mesh.n_points)

        stripped = mesh.strip_caches(keep=[("cell", "areas"), ("point", "custom")])

        torch.testing.assert_close(stripped._cache["cell", "areas"], mesh.cell_areas)
        torch.testing.assert_close(
            stripped._cache["point", "custom"], mesh._cache["point", "custom"]
        )
        assert stripped._cache.get(("cell", "centroids"), None) is None
        assert set(stripped._cache.keys()) == {"cell", "point", "topology"}

    def test_keep_ignores_missing_keys(self):
        mesh = Mesh(points=torch.zeros(1, 2))

        stripped = mesh.strip_caches(keep=[("cell", "missing")])

        assert set(stripped._cache.keys()) == {"cell", "point", "topology"}
        assert not stripped._cache["cell"].keys()

    def test_keep_retains_complete_top_level_cache(self):
        mesh = Mesh(points=torch.zeros(1, 2))
        mesh._cache["topology", "custom"] = torch.tensor(1)

        stripped = mesh.strip_caches(keep="topology")

        torch.testing.assert_close(
            stripped._cache["topology", "custom"],
            mesh._cache["topology", "custom"],
        )

    def test_keep_accepts_one_nested_key_as_a_tuple(self):
        mesh = Mesh(
            points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            cells=torch.tensor([[0, 1, 2]]),
        )
        _ = mesh.cell_areas
        _ = mesh.cell_centroids

        stripped = mesh.strip_caches(keep=("cell", "areas"))

        torch.testing.assert_close(stripped._cache["cell", "areas"], mesh.cell_areas)
        assert stripped._cache.get(("cell", "centroids"), None) is None

    def test_retained_cache_containers_are_independent(self):
        mesh = Mesh(points=torch.zeros(1, 2))
        mesh._cache["topology", "original"] = torch.tensor(1)

        stripped = mesh.strip_caches(keep=["topology"])
        stripped._cache["topology", "derived"] = torch.tensor(2)

        assert mesh._cache.get(("topology", "derived"), None) is None
        torch.testing.assert_close(
            stripped._cache["topology", "original"],
            mesh._cache["topology", "original"],
        )

    def test_data_containers_are_independent_and_tensor_leaves_are_shared(self):
        mesh = Mesh(
            points=torch.zeros(1, 2),
            point_data={"value": torch.tensor([1.0])},
        )

        stripped = mesh.strip_caches()
        stripped.point_data["derived"] = torch.tensor([2.0])

        assert stripped.point_data is not mesh.point_data
        assert stripped.point_data["value"] is mesh.point_data["value"]
        assert "derived" not in mesh.point_data


class TestWithPoints:
    """Tests for cache-aware point-coordinate replacement."""

    @staticmethod
    def _cached_triangle() -> Mesh:
        mesh = Mesh(
            points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            cells=torch.tensor([[0, 1, 2]]),
            point_data={"value": torch.arange(3)},
            cell_data={"region": torch.ones(1)},
            global_data={"case": torch.tensor(4)},
        )
        _ = mesh.cell_areas
        mesh._cache["topology", "sentinel"] = torch.tensor(7)
        return mesh

    def test_replaces_points_and_retains_topology_by_default(self):
        mesh = self._cached_triangle()
        points = mesh.points + 2.0

        updated = mesh.with_points(points)

        torch.testing.assert_close(updated.points, points)
        torch.testing.assert_close(updated.cells, mesh.cells)
        torch.testing.assert_close(
            updated.point_data["value"], mesh.point_data["value"]
        )
        torch.testing.assert_close(
            updated.cell_data["region"], mesh.cell_data["region"]
        )
        torch.testing.assert_close(
            updated.global_data["case"], mesh.global_data["case"]
        )
        assert updated._cache.get(("cell", "areas"), None) is None
        torch.testing.assert_close(
            updated._cache["topology", "sentinel"],
            mesh._cache["topology", "sentinel"],
        )

    def test_keep_can_retain_selected_geometry_cache(self):
        mesh = self._cached_triangle()

        updated = mesh.with_points(
            mesh.points.clone(),
            keep=("cell", "areas"),
        )

        torch.testing.assert_close(updated._cache["cell", "areas"], mesh.cell_areas)
        assert updated._cache.get(("topology", "sentinel"), None) is None

    def test_accepts_new_spatial_dimension(self):
        mesh = self._cached_triangle()
        points = torch.nn.functional.pad(mesh.points, (0, 1))

        updated = mesh.with_points(points)

        assert updated.points.shape == (mesh.n_points, 3)
        torch.testing.assert_close(updated.cells, mesh.cells)

    def test_rejects_changed_point_count(self):
        mesh = self._cached_triangle()

        with pytest.raises(RuntimeError, match="must preserve point indexing"):
            mesh.with_points(mesh.points[:-1])

    def test_rejects_non_matrix_points(self):
        mesh = self._cached_triangle()

        with pytest.raises(RuntimeError, match="replacement coordinates with shape"):
            mesh.with_points(torch.zeros(mesh.n_points))

    def test_result_containers_are_independent(self):
        mesh = self._cached_triangle()

        updated = mesh.with_points(mesh.points.clone())
        updated.point_data["derived"] = torch.zeros(mesh.n_points)
        updated._cache["topology", "derived"] = torch.tensor(2)

        assert "derived" not in mesh.point_data
        assert mesh._cache.get(("topology", "derived"), None) is None


class TestWithCells:
    """Tests for cache-aware cell-connectivity replacement."""

    @staticmethod
    def _cached_triangle() -> Mesh:
        mesh = Mesh(
            points=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            cells=torch.tensor([[0, 1, 2]]),
            point_data={"value": torch.arange(3)},
            cell_data={"region": torch.ones(1)},
            global_data={"case": torch.tensor(4)},
        )
        _ = mesh.cell_areas
        _ = mesh.cell_normals
        mesh._cache["topology", "sentinel"] = torch.tensor(7)
        return mesh

    def test_replaces_cells_and_clears_caches_by_default(self):
        mesh = self._cached_triangle()
        cells = mesh.cells[:, [0, 2, 1]]

        updated = mesh.with_cells(cells)

        torch.testing.assert_close(updated.points, mesh.points)
        torch.testing.assert_close(updated.cells, cells)
        torch.testing.assert_close(
            updated.point_data["value"], mesh.point_data["value"]
        )
        torch.testing.assert_close(
            updated.cell_data["region"], mesh.cell_data["region"]
        )
        torch.testing.assert_close(
            updated.global_data["case"], mesh.global_data["case"]
        )
        assert not updated._cache["cell"].keys()
        assert not updated._cache["point"].keys()
        assert not updated._cache["topology"].keys()
        torch.testing.assert_close(updated.cell_normals, -mesh.cell_normals)

    def test_keep_can_retain_selected_cache(self):
        mesh = self._cached_triangle()

        updated = mesh.with_cells(
            mesh.cells[:, [0, 2, 1]],
            keep=("cell", "areas"),
        )

        torch.testing.assert_close(updated._cache["cell", "areas"], mesh.cell_areas)
        assert updated._cache.get(("cell", "normals"), None) is None
        assert updated._cache.get(("topology", "sentinel"), None) is None

    def test_rejects_changed_cell_count(self):
        mesh = self._cached_triangle()

        with pytest.raises(RuntimeError, match="must preserve cell indexing"):
            mesh.with_cells(mesh.cells[:0])

    def test_rejects_changed_simplex_type(self):
        mesh = self._cached_triangle()
        tetrahedra = torch.cat([mesh.cells, mesh.cells[:, :1]], dim=1)

        with pytest.raises(RuntimeError, match="simplex type"):
            mesh.with_cells(tetrahedra)

    def test_rejects_non_matrix_cells(self):
        mesh = self._cached_triangle()

        with pytest.raises(RuntimeError, match="replacement connectivity with shape"):
            mesh.with_cells(mesh.cells[0])

    def test_rejects_floating_point_cells(self):
        mesh = self._cached_triangle()

        with pytest.raises(TypeError, match="int-like dtype"):
            mesh.with_cells(mesh.cells.to(torch.float32))

    def test_result_containers_are_independent(self):
        mesh = self._cached_triangle()

        updated = mesh.with_cells(mesh.cells.clone())
        updated.cell_data["derived"] = torch.zeros(mesh.n_cells)
        updated._cache["cell", "derived"] = torch.zeros(mesh.n_cells)

        assert "derived" not in mesh.cell_data
        assert mesh._cache.get(("cell", "derived"), None) is None


@pytest.mark.parametrize(
    "operation",
    ["with_points", "with_cells", "with_data", "displace"],
)
def test_cache_aware_updates_preserve_concrete_mesh_type(operation: str):
    mesh = TaggedMesh(
        points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        cells=torch.tensor([[0, 1, 2]]),
    )

    if operation == "with_points":
        updated = mesh.with_points(torch.nn.functional.pad(mesh.points, (0, 1)))
    elif operation == "with_cells":
        updated = mesh.with_cells(mesh.cells.clone())
    elif operation == "with_data":
        updated = mesh.with_data(point_data={"value": torch.arange(mesh.n_points)})
    else:
        updated = mesh.displace(torch.zeros_like(mesh.points))

    assert type(updated) is TaggedMesh
    assert updated.tag() == "tagged"
    assert updated.constructed_spatial_dims == updated.n_spatial_dims
