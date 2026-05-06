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

"""Tests for MeshReader, DomainMeshReader, and DomainMesh transform integration."""

import pytest
import torch

from physicsnemo.datapipes.mesh_dataset import MeshDataset
from physicsnemo.datapipes.readers.mesh import (
    DomainMeshReader,
    MeshReader,
    _contiguous_block_slice,
)
from physicsnemo.datapipes.transforms.mesh import (
    CenterMesh,
    RandomScaleMesh,
    ScaleMesh,
    apply_to_tensordict_mesh,
)
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.primitives.basic import (
    single_triangle_3d,
    two_triangles_2d,
)


class TestMeshReader:
    """Tests for MeshReader (single-mesh)."""

    def test_len_and_getitem(self, tmp_path):
        mesh = two_triangles_2d.load()
        mesh.save(tmp_path / "a.pmsh")
        mesh.save(tmp_path / "b.pmsh")
        reader = MeshReader(tmp_path, pattern="*.pmsh")
        assert len(reader) == 2
        m, meta = reader[0]
        assert isinstance(m, Mesh)
        assert m.n_points == mesh.n_points
        assert "source_path" in meta
        assert "index" in meta
        assert meta["index"] == 0

    def test_negative_index(self, tmp_path):
        mesh = two_triangles_2d.load()
        mesh.save(tmp_path / "single.pmsh")
        reader = MeshReader(tmp_path, pattern="*.pmsh")
        m1, _ = reader[0]
        m2, _ = reader[-1]
        assert m1.n_points == m2.n_points

    def test_iter(self, tmp_path):
        mesh = two_triangles_2d.load()
        for i in range(3):
            mesh.save(tmp_path / f"m{i}.pmsh")
        reader = MeshReader(tmp_path, pattern="*.pmsh")
        samples = list(reader)
        assert len(samples) == 3
        for m, meta in samples:
            assert isinstance(m, Mesh)
            assert isinstance(meta, dict)

    def test_subsample_n_points(self, tmp_path):
        mesh = Mesh(points=torch.randn(50, 3))
        mesh.save(tmp_path / "m.pt")
        reader = MeshReader(tmp_path, pattern="*.pt", subsample_n_points=10)
        reader.set_generator(torch.Generator().manual_seed(0))
        loaded, _ = reader[0]
        assert loaded.n_points == 10


class TestContiguousBlockSlice:
    """Tests for the ``_contiguous_block_slice`` helper."""

    def test_guard_returns_full_range(self):
        # When total <= k, the helper returns the full [0, total) range.
        assert _contiguous_block_slice(5, 5) == slice(0, 5)
        assert _contiguous_block_slice(3, 10) == slice(0, 3)

    def test_last_start_reachable_regression(self):
        # Regression: with total == k + 1 the only non-zero valid start is
        # total - k == 1.  Prior to the off-by-one fix this branch sampled
        # from torch.randint(0, 1, ...), which is deterministic at 0 and
        # therefore never produced start == 1.
        total, k = 11, 10
        gen = torch.Generator().manual_seed(0)
        starts = {
            _contiguous_block_slice(total, k, generator=gen).start for _ in range(200)
        }
        assert starts == {0, 1}

    def test_bounds_and_max_start_reached(self):
        total, k = 100, 10
        gen = torch.Generator().manual_seed(123)
        starts = []
        for _ in range(2000):
            sl = _contiguous_block_slice(total, k, generator=gen)
            assert 0 <= sl.start
            assert sl.stop - sl.start == k
            assert sl.stop <= total
            starts.append(sl.start)
        assert min(starts) == 0
        assert max(starts) == total - k

    def test_determinism(self):
        total, k = 64, 8
        gen_a = torch.Generator().manual_seed(42)
        gen_b = torch.Generator().manual_seed(42)
        for _ in range(50):
            sl_a = _contiguous_block_slice(total, k, generator=gen_a)
            sl_b = _contiguous_block_slice(total, k, generator=gen_b)
            assert sl_a == sl_b


class TestDomainMeshReader:
    """Tests for DomainMeshReader (DomainMesh per sample)."""

    def _make_domain_mesh(self):
        """Create a simple DomainMesh for testing."""
        interior = Mesh(points=torch.randn(10, 3))
        wall = single_triangle_3d.load()
        inlet = single_triangle_3d.load()
        return DomainMesh(
            interior=interior,
            boundaries={"wall": wall, "inlet": inlet},
            global_data={"Re": torch.tensor(1e6)},
        )

    def test_len_and_getitem(self, tmp_path):
        dm = self._make_domain_mesh()
        dm.save(tmp_path / "sample_a.pdmsh")
        dm.save(tmp_path / "sample_b.pdmsh")
        reader = DomainMeshReader(tmp_path, pattern="*.pdmsh")
        assert len(reader) == 2
        loaded, meta = reader[0]
        assert isinstance(loaded, DomainMesh)
        assert loaded.interior.n_points == dm.interior.n_points
        assert "source_path" in meta
        assert "index" in meta
        assert meta["index"] == 0

    def test_boundary_names_in_metadata(self, tmp_path):
        dm = self._make_domain_mesh()
        dm.save(tmp_path / "dm.pdmsh")
        reader = DomainMeshReader(tmp_path, pattern="*.pdmsh")
        _, meta = reader[0]
        assert sorted(meta["boundary_names"]) == ["inlet", "wall"]

    def test_no_boundaries(self, tmp_path):
        dm = DomainMesh(interior=Mesh(points=torch.randn(5, 3)))
        dm.save(tmp_path / "bare.pdmsh")
        reader = DomainMeshReader(tmp_path, pattern="*.pdmsh")
        loaded, meta = reader[0]
        assert loaded.n_boundaries == 0
        assert meta["boundary_names"] == []

    def test_iter(self, tmp_path):
        dm = self._make_domain_mesh()
        for i in range(3):
            dm.save(tmp_path / f"dm{i}.pdmsh")
        reader = DomainMeshReader(tmp_path, pattern="*.pdmsh")
        samples = list(reader)
        assert len(samples) == 3
        for loaded, meta in samples:
            assert isinstance(loaded, DomainMesh)
            assert isinstance(meta, dict)

    def test_global_data_preserved(self, tmp_path):
        dm = self._make_domain_mesh()
        dm.save(tmp_path / "dm.pdmsh")
        reader = DomainMeshReader(tmp_path, pattern="*.pdmsh")
        loaded, _ = reader[0]
        assert "Re" in loaded.global_data.keys()


class TestMeshDataset:
    """Tests for MeshDataset with mesh transforms."""

    def test_single_mesh_with_transform(self, tmp_path):
        mesh = two_triangles_2d.load()
        mesh.save(tmp_path / "m.pmsh")
        reader = MeshReader(tmp_path, pattern="*.pmsh")
        ds = MeshDataset(reader, transforms=[ScaleMesh(2.0)])
        m, meta = ds[0]
        assert isinstance(m, Mesh)
        assert m.n_points == mesh.n_points

    def test_domain_mesh_with_transform(self, tmp_path):
        interior = Mesh(points=torch.randn(10, 3))
        wall = single_triangle_3d.load()
        dm = DomainMesh(
            interior=interior,
            boundaries={"wall": wall},
        )
        dm.save(tmp_path / "dm.pdmsh")
        reader = DomainMeshReader(tmp_path, pattern="*.pdmsh")
        ds = MeshDataset(reader, transforms=[ScaleMesh(0.5)])
        loaded, meta = ds[0]
        assert isinstance(loaded, DomainMesh)
        assert loaded.interior.n_points == interior.n_points
        assert loaded.n_boundaries == 1

    def test_domain_mesh_transform_applies_to_all(self, tmp_path):
        interior = Mesh(
            points=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        )
        wall = Mesh(
            points=torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        )
        dm = DomainMesh(interior=interior, boundaries={"wall": wall})
        dm.save(tmp_path / "dm.pdmsh")
        reader = DomainMeshReader(tmp_path, pattern="*.pdmsh")
        ds = MeshDataset(reader, transforms=[ScaleMesh(2.0)])
        loaded, _ = ds[0]
        assert torch.allclose(
            loaded.interior.points,
            torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        )
        assert torch.allclose(
            loaded.boundaries["wall"].points,
            torch.tensor([[2.0, 0.0, 0.0], [6.0, 0.0, 0.0]]),
        )


class TestDomainMeshTransforms:
    """Tests for DomainMesh-aware transform behavior via apply_to_domain."""

    def test_scale_transforms_domain_global_data(self, tmp_path):
        """ScaleMesh with transform_global_data=True should scale domain global_data."""
        dm = DomainMesh(
            interior=Mesh(
                points=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            ),
            global_data={"velocity": torch.tensor([1.0, 0.0, 0.0])},
        )
        dm.save(tmp_path / "dm.pdmsh")
        reader = DomainMeshReader(tmp_path, pattern="*.pdmsh")
        ds = MeshDataset(
            reader,
            transforms=[ScaleMesh(2.0, transform_global_data=True)],
        )
        loaded, _ = ds[0]
        assert torch.allclose(
            loaded.global_data["velocity"],
            torch.tensor([2.0, 0.0, 0.0]),
        )

    def test_scale_preserves_domain_global_data_by_default(self, tmp_path):
        """ScaleMesh without transform_global_data leaves domain global_data unchanged."""
        dm = DomainMesh(
            interior=Mesh(
                points=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            ),
            global_data={"velocity": torch.tensor([1.0, 0.0, 0.0])},
        )
        dm.save(tmp_path / "dm.pdmsh")
        reader = DomainMeshReader(tmp_path, pattern="*.pdmsh")
        ds = MeshDataset(reader, transforms=[ScaleMesh(2.0)])
        loaded, _ = ds[0]
        assert torch.allclose(
            loaded.global_data["velocity"],
            torch.tensor([1.0, 0.0, 0.0]),
        )

    def test_random_scale_consistent_across_meshes(self, tmp_path):
        """RandomScaleMesh should apply the same factor to interior and boundaries."""
        interior = Mesh(
            points=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        wall = Mesh(
            points=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        dm = DomainMesh(interior=interior, boundaries={"wall": wall})
        dm.save(tmp_path / "dm.pdmsh")

        aug = RandomScaleMesh(
            distribution=torch.distributions.Uniform(0.5, 2.0),
        )
        aug.set_generator(torch.Generator().manual_seed(42))
        reader = DomainMeshReader(tmp_path, pattern="*.pdmsh")
        ds = MeshDataset(
            reader,
            transforms=[aug],
        )
        loaded, _ = ds[0]

        interior_factor = loaded.interior.points[1, 0].item()
        wall_factor = loaded.boundaries["wall"].points[1, 0].item()
        assert interior_factor == pytest.approx(wall_factor)

    def test_center_mesh_uses_interior_com(self, tmp_path):
        """CenterMesh should center by interior COM, not per-mesh COM."""
        interior = Mesh(
            points=torch.tensor(
                [
                    [2.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                ]
            ),
        )
        wall = Mesh(
            points=torch.tensor(
                [
                    [10.0, 0.0, 0.0],
                    [12.0, 0.0, 0.0],
                ]
            ),
        )
        dm = DomainMesh(interior=interior, boundaries={"wall": wall})
        dm.save(tmp_path / "dm.pdmsh")
        reader = DomainMeshReader(tmp_path, pattern="*.pdmsh")
        ds = MeshDataset(
            reader,
            transforms=[CenterMesh(use_area_weighting=False)],
        )
        loaded, _ = ds[0]

        interior_com = loaded.interior.points.mean(dim=0)
        assert torch.allclose(interior_com, torch.zeros(3), atol=1e-6)

        expected_wall = torch.tensor(
            [
                [10.0 - 3.0, 0.0, 0.0],
                [12.0 - 3.0, 0.0, 0.0],
            ]
        )
        assert torch.allclose(loaded.boundaries["wall"].points, expected_wall)


class TestMeshDatasetStreams:
    """Tests for MeshDataset prefetching with real CUDA streams."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prefetch_with_stream(self, tmp_path):
        """Prefetch with a CUDA stream transfers data to GPU."""
        mesh = two_triangles_2d.load()
        mesh.save(tmp_path / "m.pmsh")
        reader = MeshReader(tmp_path, pattern="*.pmsh", pin_memory=True)
        ds = MeshDataset(reader, device="cuda:0")

        stream = torch.cuda.Stream()
        ds.prefetch(0, stream=stream)

        data, _ = ds[0]
        assert data.points.device.type == "cuda"
        torch.cuda.synchronize()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prefetch_with_stream_and_transforms(self, tmp_path):
        """Prefetch with CUDA stream applies transforms on GPU."""
        mesh = Mesh(
            points=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        mesh.save(tmp_path / "m.pmsh")
        reader = MeshReader(tmp_path, pattern="*.pmsh", pin_memory=True)
        ds = MeshDataset(reader, transforms=[ScaleMesh(2.0)], device="cuda:0")

        stream = torch.cuda.Stream()
        ds.prefetch(0, stream=stream)

        data, _ = ds[0]
        assert data.points.device.type == "cuda"
        expected = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], device="cuda:0")
        assert torch.allclose(data.points, expected)
        torch.cuda.synchronize()


class TestMeshReaderSubsamplingRNG:
    """Tests for MeshReader subsampling RNG reproducibility."""

    def test_subsample_reproducible(self, tmp_path):
        """Same generator seed yields identical subsampled points."""
        mesh = Mesh(points=torch.randn(100, 3))
        mesh.save(tmp_path / "m.pmsh")
        reader = MeshReader(tmp_path, pattern="*.pmsh", subsample_n_points=10)

        reader.set_generator(torch.Generator().manual_seed(42))
        data1, _ = reader[0]

        reader.set_generator(torch.Generator().manual_seed(42))
        data2, _ = reader[0]

        assert torch.equal(data1.points, data2.points)

    def test_subsample_epoch_changes_output(self, tmp_path):
        """Different epochs produce different subsampled slices."""
        mesh = Mesh(points=torch.randn(100, 3))
        mesh.save(tmp_path / "m.pmsh")
        reader = MeshReader(tmp_path, pattern="*.pmsh", subsample_n_points=10)

        reader.set_generator(torch.Generator().manual_seed(42))
        reader.set_epoch(0)
        data_e0, _ = reader[0]

        reader.set_generator(torch.Generator().manual_seed(42))
        reader.set_epoch(1)
        data_e1, _ = reader[0]

        assert not torch.equal(data_e0.points, data_e1.points)


class TestDomainMeshReaderExtraBoundaries:
    """Tests for DomainMeshReader extra_boundaries feature."""

    def test_extra_boundaries_loaded(self, tmp_path):
        """Extra boundary mesh is loaded alongside the DomainMesh."""
        interior = Mesh(points=torch.randn(10, 3))
        dm = DomainMesh(interior=interior)
        case_dir = tmp_path / "case"
        case_dir.mkdir()
        dm.save(case_dir / "domain.pdmsh")

        farfield = Mesh(points=torch.randn(7, 3))
        farfield.save(case_dir / "farfield_001.pmsh")

        reader = DomainMeshReader(
            tmp_path,
            pattern="**/*.pdmsh",
            extra_boundaries={"farfield": {"pattern": "farfield*.pmsh"}},
        )
        assert len(reader) == 1

        loaded, meta = reader[0]
        assert "farfield" in loaded.boundary_names
        assert loaded.boundaries["farfield"].n_points == 7

    def test_extra_boundaries_missing_raises(self, tmp_path):
        """Missing extra boundary file raises FileNotFoundError."""
        interior = Mesh(points=torch.randn(10, 3))
        dm = DomainMesh(interior=interior)
        dm.save(tmp_path / "domain.pdmsh")

        reader = DomainMeshReader(
            tmp_path,
            pattern="*.pdmsh",
            extra_boundaries={"missing_bnd": {"pattern": "nonexistent*.pmsh"}},
        )

        with pytest.raises(FileNotFoundError, match="nonexistent"):
            _ = reader[0]


class TestApplyToTensorDictMesh:
    """Tests for apply_to_tensordict_mesh helper (standalone utility)."""

    def test_scale_each(self):
        from tensordict import TensorDict

        mesh = two_triangles_2d.load()
        original_points = mesh.points.clone()
        td = TensorDict({"x": mesh, "y": mesh.clone()}, batch_size=[])
        out = apply_to_tensordict_mesh(td, ScaleMesh(3.0))
        assert out["x"].n_points == mesh.n_points
        assert "x" in out
        assert "y" in out
        assert torch.allclose(out["x"].points, original_points * 3.0)
        assert torch.allclose(out["y"].points, original_points * 3.0)
