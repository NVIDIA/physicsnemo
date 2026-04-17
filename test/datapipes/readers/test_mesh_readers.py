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
from physicsnemo.datapipes.readers.mesh import DomainMeshReader, MeshReader
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
        mesh.save(tmp_path / "a.pt")
        mesh.save(tmp_path / "b.pt")
        reader = MeshReader(tmp_path, pattern="*.pt")
        assert len(reader) == 2
        m, meta = reader[0]
        assert isinstance(m, Mesh)
        assert m.n_points == mesh.n_points
        assert "source_path" in meta
        assert "index" in meta
        assert meta["index"] == 0

    def test_negative_index(self, tmp_path):
        mesh = two_triangles_2d.load()
        mesh.save(tmp_path / "single.pt")
        reader = MeshReader(tmp_path, pattern="*.pt")
        m1, _ = reader[0]
        m2, _ = reader[-1]
        assert m1.n_points == m2.n_points

    def test_iter(self, tmp_path):
        mesh = two_triangles_2d.load()
        for i in range(3):
            mesh.save(tmp_path / f"m{i}.pt")
        reader = MeshReader(tmp_path, pattern="*.pt")
        samples = list(reader)
        assert len(samples) == 3
        for m, meta in samples:
            assert isinstance(m, Mesh)
            assert isinstance(meta, dict)


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
        dm.save(tmp_path / "sample_a.pt")
        dm.save(tmp_path / "sample_b.pt")
        reader = DomainMeshReader(tmp_path, pattern="*.pt")
        assert len(reader) == 2
        loaded, meta = reader[0]
        assert isinstance(loaded, DomainMesh)
        assert loaded.interior.n_points == dm.interior.n_points
        assert "source_path" in meta
        assert "index" in meta
        assert meta["index"] == 0

    def test_boundary_names_in_metadata(self, tmp_path):
        dm = self._make_domain_mesh()
        dm.save(tmp_path / "dm.pt")
        reader = DomainMeshReader(tmp_path, pattern="*.pt")
        _, meta = reader[0]
        assert sorted(meta["boundary_names"]) == ["inlet", "wall"]

    def test_no_boundaries(self, tmp_path):
        dm = DomainMesh(interior=Mesh(points=torch.randn(5, 3)))
        dm.save(tmp_path / "bare.pt")
        reader = DomainMeshReader(tmp_path, pattern="*.pt")
        loaded, meta = reader[0]
        assert loaded.n_boundaries == 0
        assert meta["boundary_names"] == []

    def test_iter(self, tmp_path):
        dm = self._make_domain_mesh()
        for i in range(3):
            dm.save(tmp_path / f"dm{i}.pt")
        reader = DomainMeshReader(tmp_path, pattern="*.pt")
        samples = list(reader)
        assert len(samples) == 3
        for loaded, meta in samples:
            assert isinstance(loaded, DomainMesh)
            assert isinstance(meta, dict)

    def test_global_data_preserved(self, tmp_path):
        dm = self._make_domain_mesh()
        dm.save(tmp_path / "dm.pt")
        reader = DomainMeshReader(tmp_path, pattern="*.pt")
        loaded, _ = reader[0]
        assert "Re" in loaded.global_data.keys()


class TestMeshDataset:
    """Tests for MeshDataset with mesh transforms."""

    def test_single_mesh_with_transform(self, tmp_path):
        mesh = two_triangles_2d.load()
        mesh.save(tmp_path / "m.pt")
        reader = MeshReader(tmp_path, pattern="*.pt")
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
        dm.save(tmp_path / "dm.pt")
        reader = DomainMeshReader(tmp_path, pattern="*.pt")
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
        dm.save(tmp_path / "dm.pt")
        reader = DomainMeshReader(tmp_path, pattern="*.pt")
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
        dm.save(tmp_path / "dm.pt")
        reader = DomainMeshReader(tmp_path, pattern="*.pt")
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
        dm.save(tmp_path / "dm.pt")
        reader = DomainMeshReader(tmp_path, pattern="*.pt")
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
        dm.save(tmp_path / "dm.pt")

        aug = RandomScaleMesh(
            distribution=torch.distributions.Uniform(0.5, 2.0),
        )
        aug.set_generator(torch.Generator().manual_seed(42))
        reader = DomainMeshReader(tmp_path, pattern="*.pt")
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
        dm.save(tmp_path / "dm.pt")
        reader = DomainMeshReader(tmp_path, pattern="*.pt")
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


class TestApplyToTensorDictMesh:
    """Tests for apply_to_tensordict_mesh helper (standalone utility)."""

    def test_scale_each(self):
        from tensordict import TensorDict

        mesh = two_triangles_2d.load()
        td = TensorDict({"x": mesh, "y": mesh.clone()}, batch_size=[])
        out = apply_to_tensordict_mesh(td, ScaleMesh(3.0))
        assert out["x"].n_points == mesh.n_points
        assert "x" in out
        assert "y" in out
