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

"""Tests for adjacency caching on Mesh objects.

Validates that adjacency computations are cached correctly:
- Cache hit returns identical results without recomputation
- Topology-changing operations invalidate the cache
- Geometry-only operations (transforms) preserve the cache
"""

import pytest
import torch

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.transformations.geometric import (
    rotate,
    scale,
    transform,
    translate,
)

### Fixtures


@pytest.fixture
def triangle_mesh_2d():
    """Two triangles sharing an edge in 2D."""
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 0.5]],
        dtype=torch.float32,
    )
    cells = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64)
    return Mesh(points=points, cells=cells)


@pytest.fixture
def triangle_mesh_3d():
    """Two triangles sharing an edge in 3D (codimension-1 surface)."""
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [1.5, 0.5, 0.0]],
        dtype=torch.float32,
    )
    cells = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64)
    return Mesh(points=points, cells=cells)


@pytest.fixture
def tet_mesh():
    """Two tetrahedra sharing a face in 3D."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0],
            [0.5, 0.5, -1.0],
        ],
        dtype=torch.float32,
    )
    cells = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4]], dtype=torch.int64)
    return Mesh(points=points, cells=cells)


### Helpers


def _topology_cache_keys(mesh: Mesh) -> set[str]:
    """Return the set of keys currently stored in the topology cache."""
    topo = mesh._cache.get("topology", None)
    return set(topo.keys()) if topo is not None else set()


def _assert_adjacency_equal(adj_a, adj_b):
    """Assert two Adjacency objects have identical offsets and indices."""
    torch.testing.assert_close(adj_a.offsets, adj_b.offsets)
    torch.testing.assert_close(adj_a.indices, adj_b.indices)


### Cache hit tests


class TestCacheHit:
    """Second call returns cached result without recomputation."""

    def test_point_to_cells_cached(self, triangle_mesh_2d):
        adj1 = triangle_mesh_2d.get_point_to_cells_adjacency()
        adj2 = triangle_mesh_2d.get_point_to_cells_adjacency()
        assert adj1.offsets.data_ptr() == adj2.offsets.data_ptr()
        assert adj1.indices.data_ptr() == adj2.indices.data_ptr()

    def test_point_to_points_cached(self, triangle_mesh_2d):
        adj1 = triangle_mesh_2d.get_point_to_points_adjacency()
        adj2 = triangle_mesh_2d.get_point_to_points_adjacency()
        assert adj1.offsets.data_ptr() == adj2.offsets.data_ptr()
        assert adj1.indices.data_ptr() == adj2.indices.data_ptr()

    def test_cell_to_cells_cached(self, triangle_mesh_2d):
        adj1 = triangle_mesh_2d.get_cell_to_cells_adjacency()
        adj2 = triangle_mesh_2d.get_cell_to_cells_adjacency()
        assert adj1.offsets.data_ptr() == adj2.offsets.data_ptr()
        assert adj1.indices.data_ptr() == adj2.indices.data_ptr()

    def test_cell_to_points_cached(self, triangle_mesh_2d):
        adj1 = triangle_mesh_2d.get_cell_to_points_adjacency()
        adj2 = triangle_mesh_2d.get_cell_to_points_adjacency()
        assert adj1.offsets.data_ptr() == adj2.offsets.data_ptr()
        assert adj1.indices.data_ptr() == adj2.indices.data_ptr()

    def test_cell_to_cells_different_codimensions_cached_independently(self, tet_mesh):
        """Different codimension values get independent cache entries."""
        adj_codim1 = tet_mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
        adj_codim2 = tet_mesh.get_cell_to_cells_adjacency(adjacency_codimension=2)

        # Both should be cached now
        adj_codim1_again = tet_mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
        adj_codim2_again = tet_mesh.get_cell_to_cells_adjacency(adjacency_codimension=2)

        assert adj_codim1.offsets.data_ptr() == adj_codim1_again.offsets.data_ptr()
        assert adj_codim2.offsets.data_ptr() == adj_codim2_again.offsets.data_ptr()
        # codim-1 and codim-2 should yield different results
        assert adj_codim1.offsets.data_ptr() != adj_codim2.offsets.data_ptr()

    def test_empty_topology_cache_initially(self, triangle_mesh_2d):
        """No adjacency is cached before first access."""
        assert len(_topology_cache_keys(triangle_mesh_2d)) == 0


### Correctness tests


class TestCachedCorrectness:
    """Cached adjacency matches fresh (uncached) computation."""

    def test_point_to_cells_correctness(self, triangle_mesh_2d):
        from physicsnemo.mesh.neighbors import get_point_to_cells_adjacency

        cached = triangle_mesh_2d.get_point_to_cells_adjacency()
        fresh = get_point_to_cells_adjacency(triangle_mesh_2d)
        _assert_adjacency_equal(cached, fresh)

    def test_point_to_points_correctness(self, triangle_mesh_2d):
        from physicsnemo.mesh.neighbors import get_point_to_points_adjacency

        cached = triangle_mesh_2d.get_point_to_points_adjacency()
        fresh = get_point_to_points_adjacency(triangle_mesh_2d)
        _assert_adjacency_equal(cached, fresh)

    def test_cell_to_cells_correctness(self, triangle_mesh_2d):
        from physicsnemo.mesh.neighbors import get_cell_to_cells_adjacency

        cached = triangle_mesh_2d.get_cell_to_cells_adjacency()
        fresh = get_cell_to_cells_adjacency(triangle_mesh_2d)
        _assert_adjacency_equal(cached, fresh)

    def test_cell_to_points_correctness(self, triangle_mesh_2d):
        from physicsnemo.mesh.neighbors import get_cell_to_points_adjacency

        cached = triangle_mesh_2d.get_cell_to_points_adjacency()
        fresh = get_cell_to_points_adjacency(triangle_mesh_2d)
        _assert_adjacency_equal(cached, fresh)


### Invalidation tests


class TestInvalidation:
    """Topology-changing operations must invalidate the adjacency cache."""

    def test_slice_cells_invalidates(self, triangle_mesh_2d):
        triangle_mesh_2d.get_point_to_points_adjacency()
        assert len(_topology_cache_keys(triangle_mesh_2d)) > 0

        sliced = triangle_mesh_2d.slice_cells(torch.tensor([0]))
        assert len(_topology_cache_keys(sliced)) == 0

    def test_pad_invalidates(self, triangle_mesh_2d):
        triangle_mesh_2d.get_cell_to_cells_adjacency()
        assert len(_topology_cache_keys(triangle_mesh_2d)) > 0

        padded = triangle_mesh_2d.pad(target_n_cells=10)
        assert len(_topology_cache_keys(padded)) == 0

    def test_strip_caches_invalidates(self, triangle_mesh_2d):
        triangle_mesh_2d.get_point_to_cells_adjacency()
        assert len(_topology_cache_keys(triangle_mesh_2d)) > 0

        stripped = triangle_mesh_2d.strip_caches()
        assert len(_topology_cache_keys(stripped)) == 0

    def test_slice_cells_then_recompute_gives_correct_result(self, triangle_mesh_2d):
        """After slicing, recomputed adjacency reflects the new topology."""
        sliced = triangle_mesh_2d.slice_cells(torch.tensor([0]))
        adj = sliced.get_cell_to_cells_adjacency()
        # Single cell has no neighbors
        assert adj.to_list() == [[]]


### Propagation tests


class TestPropagation:
    """Geometry-only operations must preserve the adjacency cache."""

    def test_translate_preserves(self, triangle_mesh_2d):
        triangle_mesh_2d.get_point_to_points_adjacency()
        original_keys = _topology_cache_keys(triangle_mesh_2d)

        translated = translate(triangle_mesh_2d, [1.0, 2.0])
        assert _topology_cache_keys(translated) == original_keys

    def test_translate_preserves_correctness(self, triangle_mesh_2d):
        adj_before = triangle_mesh_2d.get_point_to_points_adjacency()

        translated = translate(triangle_mesh_2d, [5.0, -3.0])
        adj_after = translated.get_point_to_points_adjacency()

        _assert_adjacency_equal(adj_before, adj_after)
        assert adj_before.offsets.data_ptr() == adj_after.offsets.data_ptr()

    def test_rotate_preserves(self, triangle_mesh_2d):
        triangle_mesh_2d.get_cell_to_cells_adjacency()
        original_keys = _topology_cache_keys(triangle_mesh_2d)

        rotated = rotate(triangle_mesh_2d, angle=0.5)
        assert _topology_cache_keys(rotated) == original_keys

    def test_scale_preserves(self, triangle_mesh_3d):
        triangle_mesh_3d.get_point_to_cells_adjacency()
        original_keys = _topology_cache_keys(triangle_mesh_3d)

        scaled = scale(triangle_mesh_3d, factor=2.0)
        assert _topology_cache_keys(scaled) == original_keys

    def test_general_transform_preserves(self, triangle_mesh_2d):
        triangle_mesh_2d.get_cell_to_points_adjacency()
        original_keys = _topology_cache_keys(triangle_mesh_2d)

        matrix = torch.tensor([[2.0, 0.5], [0.0, 1.0]])
        transformed = transform(triangle_mesh_2d, matrix)
        assert _topology_cache_keys(transformed) == original_keys

    def test_chained_transforms_preserve(self, triangle_mesh_3d):
        """Multiple chained transforms all preserve the adjacency cache."""
        adj_original = triangle_mesh_3d.get_point_to_points_adjacency()

        result = translate(triangle_mesh_3d, [1.0, 0.0, 0.0])
        result = rotate(result, angle=0.3, axis="z")
        result = scale(result, factor=2.0)
        result = translate(result, [0.0, -1.0, 0.0])

        adj_final = result.get_point_to_points_adjacency()
        _assert_adjacency_equal(adj_original, adj_final)
        assert adj_original.offsets.data_ptr() == adj_final.offsets.data_ptr()

    def test_all_four_adjacencies_propagate_through_translate(self, triangle_mesh_2d):
        """All four adjacency types survive a translate."""
        triangle_mesh_2d.get_point_to_cells_adjacency()
        triangle_mesh_2d.get_point_to_points_adjacency()
        triangle_mesh_2d.get_cell_to_cells_adjacency()
        triangle_mesh_2d.get_cell_to_points_adjacency()
        n_keys = len(_topology_cache_keys(triangle_mesh_2d))
        assert n_keys == 4  # one sub-TensorDict per adjacency type

        translated = translate(triangle_mesh_2d, [1.0, 2.0])
        assert len(_topology_cache_keys(translated)) == n_keys


### Backward compatibility


class TestBackwardCompat:
    """Meshes created with caches missing the 'topology' key are handled."""

    def test_cache_without_topology_key(self):
        """Simulate loading a mesh saved before the topology cache existed."""
        from tensordict import TensorDict

        points = torch.randn(4, 2)
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64)
        old_cache = TensorDict(
            {
                "cell": TensorDict({}, batch_size=[2], device=points.device),
                "point": TensorDict({}, batch_size=[4], device=points.device),
            },
            device=points.device,
        )
        mesh = Mesh(points=points, cells=cells, _cache=old_cache)

        # Adjacency computation should work normally (topology key is
        # lazily created on first write, not eagerly backfilled)
        adj = mesh.get_point_to_points_adjacency()
        assert adj.n_sources == 4
