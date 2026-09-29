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

"""The mesh readers map saved meshes read-only, whatever the file permissions.

``tensordict`` maps owner-writable memmap files read-write and shared, which on a
networked file system makes concurrent readers of one case wait on write locks
(a training hang observed on a Lustre-hosted dataset). The readers therefore
load under ``read_only_memory_maps``; these tests check that the override is in
force exactly during the load, is restored afterwards, nests, and that loading a
writable directory through the reader produces the same mesh as a direct load.
"""

import threading

import pytest
import torch

pytest.importorskip("tensordict")
import importlib  # noqa: E402

tdm = importlib.import_module("tensordict.memmap")

from physicsnemo.datapipes.readers.mesh import (  # noqa: E402
    DomainMeshReader,
    MeshReader,
    read_only_memory_maps,
)
from physicsnemo.mesh import DomainMesh, Mesh  # noqa: E402


def _triangle_mesh(seed: int = 0) -> Mesh:
    g = torch.Generator().manual_seed(seed)
    points = torch.randn(6, 3, generator=g)
    cells = torch.tensor([[0, 1, 2], [2, 3, 4], [4, 5, 0]])
    return Mesh(points=points, cells=cells)


def test_context_forces_read_only_and_restores(tmp_path):
    original = tdm._is_writable
    path = tmp_path / "f.memmap"
    path.write_bytes(b"\0" * 8)
    assert original(path) is True  # the file is writable by us
    with read_only_memory_maps():
        assert tdm._is_writable(path) is False
        with read_only_memory_maps():  # nested use keeps the override
            assert tdm._is_writable(path) is False
        assert tdm._is_writable(path) is False
    assert tdm._is_writable is original
    assert tdm._is_writable(path) is True


def test_context_restores_after_exception(tmp_path):
    original = tdm._is_writable
    with pytest.raises(RuntimeError):
        with read_only_memory_maps():
            raise RuntimeError("boom")
    assert tdm._is_writable is original


def test_context_is_thread_safe():
    original = tdm._is_writable
    seen = []

    def worker():
        with read_only_memory_maps():
            seen.append(tdm._is_writable("anything") is False)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(seen) and len(seen) == 8
    assert tdm._is_writable is original


def test_mesh_reader_loads_writable_directory_read_only(tmp_path):
    mesh = _triangle_mesh()
    mesh.save(tmp_path / "case_0.pmsh")
    original = tdm._is_writable
    reader = MeshReader(str(tmp_path), pattern="*.pmsh")
    loaded, meta = reader[0]
    assert isinstance(loaded, Mesh)
    torch.testing.assert_close(loaded.points, mesh.points)
    assert tdm._is_writable is original  # restored after the load
    assert "source_path" in meta


def test_domain_mesh_reader_round_trip(tmp_path):
    interior = _triangle_mesh(1)
    boundary = _triangle_mesh(2)
    dm = DomainMesh(interior=interior, boundaries={"wall": boundary})
    dm.save(tmp_path / "case_0.pdmsh")
    original = tdm._is_writable
    reader = DomainMeshReader(str(tmp_path), pattern="*.pdmsh")
    loaded, _ = reader[0]
    assert isinstance(loaded, DomainMesh)
    torch.testing.assert_close(loaded.interior.points, interior.points)
    torch.testing.assert_close(loaded.boundaries["wall"].points, boundary.points)
    assert tdm._is_writable is original
