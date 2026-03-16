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

"""Tests for MeshReader and MultiMeshReader."""

import pytest
from pathlib import Path
from tensordict import TensorDict

from physicsnemo.datapipes.readers.mesh import MeshReader, MultiMeshReader
from physicsnemo.datapipes.mesh_dataset import MeshDataset
from physicsnemo.datapipes.transforms.mesh import (
    ScaleMesh,
    TranslateMesh,
    RandomScaleMesh,
    apply_to_tensordict_mesh,
)
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.basic import two_triangles_2d


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
        for (m, meta) in samples:
            assert isinstance(m, Mesh)
            assert isinstance(meta, dict)


class TestMultiMeshReader:
    """Tests for MultiMeshReader (multi-mesh per sample)."""

    def test_len_and_getitem(self, tmp_path):
        mesh = two_triangles_2d.load()
        run1 = tmp_path / "run_1"
        run1.mkdir()
        mesh.save(run1 / "boundary.pt")
        mesh.save(run1 / "volume.pt")
        run2 = tmp_path / "run_2"
        run2.mkdir()
        mesh.save(run2 / "boundary.pt")
        reader = MultiMeshReader(tmp_path, mesh_pattern="*.pt")
        assert len(reader) == 2
        td, meta = reader[0]
        assert isinstance(td, TensorDict)
        assert "boundary" in td and "volume" in td
        assert isinstance(td["boundary"], Mesh)
        assert meta["mesh_names"] == ["boundary", "volume"]

    def test_single_mesh_in_subdir(self, tmp_path):
        mesh = two_triangles_2d.load()
        run = tmp_path / "run_0"
        run.mkdir()
        mesh.save(run / "only.pt")
        reader = MultiMeshReader(tmp_path, mesh_pattern="*.pt")
        td, _ = reader[0]
        assert "only" in td
        assert isinstance(td["only"], Mesh)


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

    def test_multi_mesh_with_transform(self, tmp_path):
        mesh = two_triangles_2d.load()
        run = tmp_path / "run_1"
        run.mkdir()
        mesh.save(run / "a.pt")
        mesh.save(run / "b.pt")
        reader = MultiMeshReader(tmp_path, mesh_pattern="*.pt")
        ds = MeshDataset(reader, transforms=[ScaleMesh(0.5)])
        td, meta = ds[0]
        assert isinstance(td, TensorDict)
        assert td["a"].n_points == mesh.n_points


class TestApplyToTensorDictMesh:
    """Tests for apply_to_tensordict_mesh helper."""

    def test_scale_each(self, tmp_path):
        mesh = two_triangles_2d.load()
        run = tmp_path / "run_1"
        run.mkdir()
        mesh.save(run / "x.pt")
        reader = MultiMeshReader(tmp_path, mesh_pattern="*.pt")
        td, _ = reader[0]
        out = apply_to_tensordict_mesh(td, ScaleMesh(3.0))
        assert out["x"].n_points == mesh.n_points
        assert "x" in out
