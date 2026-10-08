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

"""MeshReader / DomainMeshReader integration with zarr stores, including the
subsample push-down (window reads must be bitwise-identical to an eager load
followed by in-memory subsampling)."""

from pathlib import Path

import pytest
import torch
from conftest import (
    assert_meshes_equal,
    make_domain_mesh,
    make_mesh,
    make_point_cloud,
)

from physicsnemo.datapipes.readers.mesh import (
    DomainMeshReader,
    MeshReader,
    _subsample_mesh,
)
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.calculus.measure import point_measures, set_point_measures
from physicsnemo.mesh.io import from_zarr, to_zarr


def test_mesh_reader_reads_zarr(tmp_path):
    mesh = make_mesh()
    to_zarr(mesh, tmp_path / "sample.mesh.zarr")
    reader = MeshReader(tmp_path, pattern="*.zarr")
    back, metadata = reader[0]
    assert_meshes_equal(mesh, back)
    assert metadata["index"] == 0


def test_domain_mesh_reader_reads_zarr(tmp_path):
    dm = make_domain_mesh()
    to_zarr(dm, tmp_path / "case.zarr")
    reader = DomainMeshReader(tmp_path, pattern="*.zarr")
    back, _ = reader[0]
    assert_meshes_equal(dm.interior, back.interior)
    for name in dm.boundary_names:
        assert_meshes_equal(dm.boundaries[name], back.boundaries[name])


def test_pushdown_matches_eager_subsample(tmp_path):
    """Window reads == eager full load + in-memory subsample, same seed."""
    mesh = make_mesh(n_points=500, n_cells=400, seed=21)
    to_zarr(mesh, tmp_path / "sample.mesh.zarr", chunk_rows=64)

    reader = MeshReader(tmp_path, pattern="*.zarr", subsample_n_cells=100)
    seed_gen = torch.Generator().manual_seed(1234)
    reader.set_generator(seed_gen)
    reader.set_epoch(0)
    lazy, _ = reader[0]

    from physicsnemo.datapipes._rng import spawn_generator

    eager_full = from_zarr(tmp_path / "sample.mesh.zarr")
    gen = spawn_generator(seed_gen.initial_seed(), 0, 0)
    eager = _subsample_mesh(eager_full, n_cells=100, generator=gen)

    assert lazy.n_cells == 100
    assert_meshes_equal(eager, lazy)


def test_pushdown_point_cloud_matches_eager(tmp_path):
    dm = make_domain_mesh()
    to_zarr(dm, tmp_path / "case.zarr", chunk_rows=16)

    reader = DomainMeshReader(
        tmp_path, pattern="*.zarr", subsample_n_points=20, subsample_n_cells=10
    )
    seed_gen = torch.Generator().manual_seed(99)
    reader.set_generator(seed_gen)
    reader.set_epoch(0)
    lazy, _ = reader[0]

    from physicsnemo.datapipes._rng import spawn_generator

    eager_full = from_zarr(tmp_path / "case.zarr")
    gen = spawn_generator(seed_gen.initial_seed(), 0, 0)
    interior = _subsample_mesh(eager_full.interior, 10, 20, generator=gen)
    boundaries = {
        n: _subsample_mesh(eager_full.boundaries[n], 10, 20, generator=gen)
        for n in eager_full.boundary_names
    }

    assert lazy.interior.n_points == 20
    assert_meshes_equal(interior, lazy.interior)
    for n in eager_full.boundary_names:
        assert_meshes_equal(boundaries[n], lazy.boundaries[n])


@pytest.mark.parametrize("interior_has_cells", [False, True])
@pytest.mark.parametrize(
    "options",
    [
        {"subsample_n_points": 30},
        {"subsample_n_cells": 20},
        {"subsample_n_cells": 20, "subsample_n_points": 30},
        {
            "subsample_n_cells": 20,
            "subsample_n_points": 30,
            "boundary_subsample": "cells",
        },
        {
            "subsample_n_cells": 20,
            "subsample_n_points": 30,
            "boundary_subsample": "points",
        },
        {
            "subsample_n_cells": 20,
            "subsample_n_points": 30,
            "drop_interior_cells": True,
        },
    ],
)
def test_seeded_subsamples_match_between_formats(tmp_path, interior_has_cells, options):
    """One seed selects the same rows from a zarr store as from a memmap file."""
    interior = (
        make_mesh(n_points=80, n_cells=60, seed=5)
        if interior_has_cells
        else make_point_cloud(seed=5)
    )
    domain = DomainMesh(
        interior=interior,
        boundaries={
            "wall": make_mesh(n_points=60, n_cells=50, seed=6),
            "inlet": make_mesh(n_points=50, n_cells=40, seed=7),
        },
    )
    domain.save(tmp_path / "case.pdmsh")
    to_zarr(domain, tmp_path / "case.zarr", chunk_rows=8)
    loaded = []
    for pattern in ("*.pdmsh", "*.zarr"):
        reader = DomainMeshReader(
            tmp_path, pattern=pattern, pin_memory=False, **options
        )
        reader.set_generator(torch.Generator().manual_seed(17))
        loaded.append(reader[0][0])
    from_memmap, from_store = loaded
    assert_meshes_equal(from_memmap.interior, from_store.interior)
    assert sorted(from_memmap.boundary_names) == sorted(from_store.boundary_names)
    for name in from_memmap.boundary_names:
        assert_meshes_equal(from_memmap.boundaries[name], from_store.boundaries[name])


def test_mixed_directory_discovery(tmp_path):
    """zarr and memmap samples coexist; each routes to the right loader."""
    m1 = make_mesh(seed=1)
    m2 = make_mesh(seed=2)
    to_zarr(m1, tmp_path / "a.mesh.zarr")
    m2.save(tmp_path / "b.pmsh")
    reader = MeshReader(tmp_path, pattern="*.*")
    metas = {Path(reader[i][1]["source_path"]).name: reader[i][0] for i in (0, 1)}
    assert_meshes_equal(m1, metas["a.mesh.zarr"])
    assert_meshes_equal(m2, metas["b.pmsh"])


@pytest.mark.parametrize("domain", [False, True])
@pytest.mark.parametrize("n_keep", [4, 20])
def test_point_quadrature_subsampling_matches_between_formats(tmp_path, domain, n_keep):
    """Partial reads apply the same correction as eager reads, exactly once."""
    cloud = Mesh(
        points=torch.arange(30, dtype=torch.float32).reshape(10, 3),
        point_data={"index": torch.arange(10)},
    )
    set_point_measures(cloud, torch.arange(10, dtype=torch.float32) + 1, dimension=3)
    sample = DomainMesh(interior=cloud) if domain else cloud
    sample.save(tmp_path / "sample.pmsh")
    to_zarr(sample, tmp_path / "sample.zarr", chunk_rows=2)
    reader_type = DomainMeshReader if domain else MeshReader
    loaded = []
    for pattern in ("*.pmsh", "*.zarr"):
        reader = reader_type(
            tmp_path, pattern=pattern, subsample_n_points=n_keep, pin_memory=False
        )
        reader.set_generator(torch.Generator().manual_seed(5))
        sampled, _ = reader[0]
        points = sampled.interior if domain else sampled
        expected = (points.point_data["index"] + 1).float() * (10 / min(n_keep, 10))
        torch.testing.assert_close(point_measures(points), expected)
        torch.testing.assert_close(point_measures(points.scale(2.0)), expected * 8)
        loaded.append(points)
    assert_meshes_equal(*loaded)
