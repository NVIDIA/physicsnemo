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

"""Tests for Mesh serialization round-trips (memmap and pickle).

TensorDict records zero-element tensors in memmap metadata but does not write a
backing file for them. These tests verify that mesh serialization reconstructs
those tensors with their original shape and dtype.
"""

import os
import shutil
import subprocess
import sys

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.mesh.domain_mesh import DomainMesh
from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.primitives.basic import two_triangles_2d

### Memmap Round-Trip Tests ###


class TestMemmapRoundTrip:
    """Tests for Mesh.save() / Mesh.load() (memmap) round-trip."""

    def test_regular_mesh(self, tmp_path):
        """Regular mesh with non-empty cells survives memmap round-trip."""
        mesh = two_triangles_2d.load()
        mesh.save(tmp_path / "mesh.pt")
        loaded = Mesh.load(tmp_path / "mesh.pt")

        assert loaded.cells is not None
        assert loaded.cells.shape == mesh.cells.shape
        assert torch.equal(loaded.cells, mesh.cells)
        assert torch.allclose(loaded.points, mesh.points)

    def test_point_cloud_direct(self, tmp_path):
        """Point cloud created via constructor (cells=None) survives memmap round-trip."""
        pc = Mesh(points=torch.randn(10, 3))

        assert pc.cells is not None
        assert pc.cells.shape == (0, 1)

        pc.save(tmp_path / "pc.pt")
        loaded = Mesh.load(tmp_path / "pc.pt")

        assert loaded.cells is not None, "cells should not be None after memmap load"
        assert loaded.cells.shape == (0, 1)
        assert loaded.cells.dtype == torch.long

    def test_point_cloud_via_to_point_cloud(self, tmp_path):
        """Point cloud created via to_point_cloud() survives memmap round-trip."""
        pc = two_triangles_2d.load().to_point_cloud()

        assert pc.cells.shape == (0, 1)

        pc.save(tmp_path / "pc.pt")
        loaded = Mesh.load(tmp_path / "pc.pt")

        assert loaded.cells is not None, "cells should not be None after memmap load"
        assert loaded.cells.shape == (0, 1)
        assert loaded.cells.dtype == torch.long

    def test_n_manifold_dims_after_load(self, tmp_path):
        """n_manifold_dims works correctly after memmap round-trip of point cloud."""
        pc = two_triangles_2d.load().to_point_cloud()
        assert pc.n_manifold_dims == 0

        pc.save(tmp_path / "pc.pt")
        loaded = Mesh.load(tmp_path / "pc.pt")

        assert loaded.n_manifold_dims == 0

    def test_n_cells_after_load(self, tmp_path):
        """n_cells returns 0 for a point cloud after memmap round-trip."""
        pc = two_triangles_2d.load().to_point_cloud()
        assert pc.n_cells == 0

        pc.save(tmp_path / "pc.pt")
        loaded = Mesh.load(tmp_path / "pc.pt")

        assert loaded.n_cells == 0

    def test_repr_after_load(self, tmp_path):
        """repr does not raise for a point cloud after memmap round-trip."""
        pc = two_triangles_2d.load().to_point_cloud()
        pc.save(tmp_path / "pc.pt")
        loaded = Mesh.load(tmp_path / "pc.pt")

        result = repr(loaded)
        assert "n_manifold_dims=0, n_spatial_dims=2" in result
        assert "n_cells=0" in result

    def test_points_preserved(self, tmp_path):
        """Point coordinates survive memmap round-trip exactly."""
        pc = two_triangles_2d.load().to_point_cloud()
        pc.save(tmp_path / "pc.pt")
        loaded = Mesh.load(tmp_path / "pc.pt")

        assert torch.allclose(loaded.points, pc.points)

    def test_with_point_data(self, tmp_path):
        """point_data survives memmap round-trip alongside empty cells."""
        mesh = Mesh(
            points=torch.randn(5, 3),
            point_data={"velocity": torch.randn(5, 3), "pressure": torch.randn(5)},
        )
        mesh.save(tmp_path / "mesh.pt")
        loaded = Mesh.load(tmp_path / "mesh.pt")

        assert loaded.cells is not None
        assert loaded.cells.shape == (0, 1)
        assert "velocity" in loaded.point_data.keys()
        assert "pressure" in loaded.point_data.keys()
        assert torch.allclose(
            loaded.point_data["velocity"], mesh.point_data["velocity"]
        )

    def test_with_cell_and_global_data(self, tmp_path):
        """cell_data and global_data survive memmap round-trip on a real mesh."""
        # Use a 2-triangle mesh so n_cells > 0 (cell_data is non-trivial).
        mesh = two_triangles_2d.load()
        mesh.point_data["p_scalar"] = torch.randn(mesh.n_points)
        mesh.point_data["p_vector"] = torch.randn(mesh.n_points, 3)
        mesh.cell_data["c_scalar"] = torch.randn(mesh.n_cells)
        mesh.cell_data["c_vector"] = torch.randn(mesh.n_cells, 3)
        mesh.global_data["g_scalar"] = torch.tensor(1.5)
        mesh.global_data["g_vector"] = torch.tensor([1.0, 2.0, 3.0])

        mesh.save(tmp_path / "mesh.pt")
        loaded = Mesh.load(tmp_path / "mesh.pt")

        assert set(loaded.point_data.keys()) == {"p_scalar", "p_vector"}
        assert set(loaded.cell_data.keys()) == {"c_scalar", "c_vector"}
        assert set(loaded.global_data.keys()) == {"g_scalar", "g_vector"}

        for field in ("point_data", "cell_data", "global_data"):
            for key in getattr(mesh, field).keys():
                assert torch.allclose(
                    getattr(loaded, field)[key], getattr(mesh, field)[key]
                ), f"{field}[{key!r}] mismatch after round-trip"

    def test_all_dimension_configs(self, tmp_path, dims_all):
        """All (n_spatial_dims, n_manifold_dims) configurations survive memmap round-trip."""
        from test.mesh.conftest import create_simple_mesh

        n_spatial_dims, n_manifold_dims = dims_all
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims)
        mesh.save(tmp_path / "mesh.pt")
        loaded = Mesh.load(tmp_path / "mesh.pt")

        assert loaded.cells is not None
        assert loaded.n_manifold_dims == mesh.n_manifold_dims
        assert loaded.n_spatial_dims == mesh.n_spatial_dims
        assert loaded.n_points == mesh.n_points
        assert loaded.n_cells == mesh.n_cells

    @pytest.mark.parametrize("vertices_per_cell", [1, 2, 3, 4])
    def test_empty_cells_preserve_shape_and_dtype(self, tmp_path, vertices_per_cell):
        """Empty connectivity retains its manifold dimension and dtype."""
        mesh = Mesh(
            points=torch.randn(5, 4),
            cells=torch.empty((0, vertices_per_cell), dtype=torch.int32),
        )

        mesh.save(tmp_path / "mesh.pt")
        loaded = Mesh.load(tmp_path / "mesh.pt")

        assert loaded.cells.shape == (0, vertices_per_cell)
        assert loaded.cells.dtype == torch.int32
        assert loaded.n_manifold_dims == vertices_per_cell - 1

    def test_fully_empty_geometry_preserves_shape_and_dtype(self, tmp_path):
        """A mesh with no points or cells can be loaded without losing metadata."""
        mesh = Mesh(
            points=torch.empty((0, 4), dtype=torch.float64),
            cells=torch.empty((0, 3), dtype=torch.int32),
        )

        mesh.save(tmp_path / "mesh.pt")
        loaded = Mesh.load(tmp_path / "mesh.pt")

        assert loaded.points.shape == (0, 4)
        assert loaded.points.dtype == torch.float64
        assert loaded.cells.shape == (0, 3)
        assert loaded.cells.dtype == torch.int32
        assert loaded.n_spatial_dims == 4
        assert loaded.n_manifold_dims == 2

    def test_zero_element_data_preserves_nested_shapes_and_dtypes(self, tmp_path):
        """Zero-width user fields survive at every association and nesting level."""
        point_data = TensorDict(
            {
                "embedding": torch.empty((3, 0), dtype=torch.float16),
                "nested": TensorDict(
                    {"features": torch.empty((3, 2, 0), dtype=torch.float64)},
                    batch_size=[3],
                ),
            },
            batch_size=[3],
        )
        mesh = Mesh(
            points=torch.randn(3, 3),
            cells=torch.tensor([[0, 1, 2]], dtype=torch.int64),
            point_data=point_data,
            cell_data={"embedding": torch.empty((1, 0), dtype=torch.int16)},
            global_data={"embedding": torch.empty((2, 0), dtype=torch.float64)},
        )

        mesh.save(tmp_path / "mesh.pt")
        loaded = Mesh.load(tmp_path / "mesh.pt")

        assert loaded.point_data["embedding"].shape == (3, 0)
        assert loaded.point_data["embedding"].dtype == torch.float16
        assert loaded.point_data["nested", "features"].shape == (3, 2, 0)
        assert loaded.point_data["nested", "features"].dtype == torch.float64
        assert loaded.cell_data["embedding"].shape == (1, 0)
        assert loaded.cell_data["embedding"].dtype == torch.int16
        assert loaded.global_data["embedding"].shape == (2, 0)
        assert loaded.global_data["embedding"].dtype == torch.float64

    def test_empty_cached_adjacency_round_trip(self, tmp_path):
        """A cached adjacency with no indices remains loadable and empty."""
        mesh = Mesh(points=torch.randn(4, 3))
        expected = mesh.get_point_to_points_adjacency().to_list()

        mesh.save(tmp_path / "mesh.pt")
        loaded = Mesh.load(tmp_path / "mesh.pt")
        adjacency = loaded.get_point_to_points_adjacency()

        assert adjacency.to_list() == expected
        assert adjacency.indices.shape == (0,)
        assert adjacency.indices.dtype == torch.int64

    def test_domain_mesh_preserves_nested_empty_tensors(self, tmp_path):
        """Domain, interior, and boundary empties all survive one round-trip."""
        interior = Mesh(
            points=torch.randn(4, 3),
            cells=torch.empty((0, 4), dtype=torch.int32),
        )
        boundary = Mesh(
            points=torch.empty((0, 3), dtype=torch.float64),
            cells=torch.empty((0, 3), dtype=torch.int32),
            point_data={"embedding": torch.empty((0, 2), dtype=torch.float16)},
        )
        domain = DomainMesh(
            interior=interior,
            boundaries={"empty": boundary},
            global_data={"embedding": torch.empty((2, 0), dtype=torch.float64)},
        )

        domain.save(tmp_path / "domain.pt")
        loaded = DomainMesh.load(tmp_path / "domain.pt")

        assert loaded.interior.cells.shape == (0, 4)
        assert loaded.interior.cells.dtype == torch.int32
        assert loaded.boundaries["empty"].points.shape == (0, 3)
        assert loaded.boundaries["empty"].points.dtype == torch.float64
        assert loaded.boundaries["empty"].cells.shape == (0, 3)
        assert loaded.boundaries["empty"].cells.dtype == torch.int32
        assert loaded.boundaries["empty"].point_data["embedding"].shape == (0, 2)
        assert loaded.boundaries["empty"].point_data["embedding"].dtype == torch.float16
        assert loaded.global_data["embedding"].shape == (2, 0)
        assert loaded.global_data["embedding"].dtype == torch.float64

    def test_restored_empties_share_the_device_of_loaded_tensors(
        self, tmp_path, device
    ):
        """Rebuilt empties land on the same device as the tensors read from disk.

        ``Mesh.__post_init__`` rejects a mesh whose ``points`` and ``cells``
        disagree on device, so a rebuilt tensor placed on the wrong device
        breaks the load outright rather than degrading quietly.
        """
        mesh = Mesh(
            points=torch.randn(4, 3, device=device),
            cells=torch.empty((0, 3), dtype=torch.int32, device=device),
            point_data={"embedding": torch.empty((4, 0), device=device)},
        )

        mesh.save(tmp_path / "mesh.pt")
        loaded = Mesh.load(tmp_path / "mesh.pt")

        assert loaded.cells.device == loaded.points.device
        assert loaded.point_data["embedding"].device == loaded.points.device

    def test_missing_tensordict_directory_raises_clear_error(self, tmp_path):
        """An incomplete save directory reports what is missing, not a raw OSError."""
        mesh = two_triangles_2d.load()
        mesh.save(tmp_path / "mesh.pt")
        shutil.rmtree(tmp_path / "mesh.pt" / "_tensordict")

        with pytest.raises(
            ValueError, match="_tensordict directory seems to be missing"
        ):
            Mesh.load(tmp_path / "mesh.pt")

    @pytest.mark.xfail(
        strict=True,
        reason="A jagged tensor's recorded shape is the shape of its shape "
        "tensor, which lives in a separate file, so an empty one cannot be "
        "rebuilt from meta.json alone. Documented in _serialization.py; if this "
        "starts passing, update those Notes.",
    )
    def test_empty_jagged_tensor_round_trip(self, tmp_path):
        """Known limitation: empty nested (jagged) tensors are still dropped."""
        mesh = Mesh(
            points=torch.randn(2, 3),
            cells=torch.empty((0, 3), dtype=torch.int64),
        )
        mesh.global_data["jagged"] = torch.nested.nested_tensor(
            [torch.empty((0, 2)), torch.empty((0, 2))]
        )

        mesh.save(tmp_path / "mesh.pt")
        loaded = Mesh.load(tmp_path / "mesh.pt")

        assert "jagged" in loaded.global_data


### Fresh-Process Load Tests ###

# Deliberately imports nothing but ``Mesh``: reaching for an adjacency helper
# here would register ``Adjacency`` by hand and mask the bug under test.
_LOAD_MESH_WITH_CACHED_ADJACENCY = """
import sys

from physicsnemo.mesh.mesh import Mesh

mesh = Mesh.load(sys.argv[1])
adjacency = mesh._cache["topology", "point_to_points"]
print(f"{type(adjacency).__name__} {tuple(adjacency.indices.shape)}")
"""


class TestFreshProcessLoad:
    """Loading must work in a process that has only ever imported ``Mesh``.

    This is the offline-preprocessing / dataloader-worker pattern: one process
    writes meshes to disk, an unrelated one reads them back.
    """

    def test_cached_adjacency_loads_in_a_fresh_process(self, tmp_path):
        """A saved topology cache reconstructs without touching the adjacency API.

        ``TensorDict.load_memmap`` resolves the ``Adjacency`` held in
        ``_cache["topology"]`` by looking its class up in tensordict's registry,
        which is only populated as an import side effect. If nothing imports
        ``physicsnemo.mesh.neighbors``, the load dies with
        ``RuntimeError: Could not find name ...Adjacency`` -- and note this
        happens for a perfectly ordinary non-empty adjacency, independent of the
        zero-element handling the rest of this module covers.

        Runs out-of-process because any in-process test would already have the
        class registered by an earlier import.
        """
        mesh = two_triangles_2d.load()
        adjacency = mesh.get_point_to_points_adjacency()
        assert adjacency.indices.numel() > 0, "want a populated cache here"
        mesh.save(tmp_path / "mesh.pt")

        # Hand the child our own sys.path so it imports the same physicsnemo we
        # are testing (matters in a git worktree, where the installed
        # distribution resolves elsewhere).
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
        result = subprocess.run(  # noqa: S603 - interpreter and snippet are test constants
            [
                sys.executable,
                "-c",
                _LOAD_MESH_WITH_CACHED_ADJACENCY,
                str(tmp_path / "mesh.pt"),
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
            check=False,
        )

        assert result.returncode == 0, (
            f"fresh-process load failed:\n{result.stdout}\n{result.stderr}"
        )
        assert result.stdout.strip() == f"Adjacency {tuple(adjacency.indices.shape)}"


### Pickle (torch.save / torch.load) Round-Trip Tests ###


class TestTorchSaveLoadRoundTrip:
    """Tests for torch.save() / torch.load() (pickle) round-trip.

    These should already work without the Mesh.load() fix, since pickle
    preserves empty tensors. Included as regression tests.
    """

    def test_regular_mesh(self, tmp_path):
        """Regular mesh survives torch.save/load round-trip."""
        mesh = two_triangles_2d.load()
        path = tmp_path / "mesh.pt"
        torch.save(mesh, path)
        loaded = torch.load(path, weights_only=False)

        assert loaded.cells is not None
        assert torch.equal(loaded.cells, mesh.cells)

    def test_point_cloud(self, tmp_path):
        """Point cloud survives torch.save/load round-trip."""
        pc = two_triangles_2d.load().to_point_cloud()
        path = tmp_path / "pc.pt"
        torch.save(pc, path)
        loaded = torch.load(path, weights_only=False)

        assert loaded.cells is not None
        assert loaded.cells.shape == (0, 1)
        assert loaded.n_manifold_dims == 0
        assert repr(loaded)  # should not raise
