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

r"""Single-process tests of the ``"local"`` domain-parallel mesh layout.

Every rank's share is read and transformed here in one process, with no
process group at all: a rank reads and transforms its share alone; the one
all-reduce of the centering statistics' partial sums is done here by summing
every rank's partials. Concatenated over the ranks, the shares must
reproduce the unsharded read and transform chain (bit for bit, except the
center and what depends on it, which match up to float rounding). The
ShardTensor wrap and the multi-rank pipeline are covered in
``test/domain_parallel/datapipes/test_local_mesh_shard_dataset.py``.
"""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.datapipes._indexing import _cyclic_block_indices
from physicsnemo.datapipes._local_shard import (
    AREA_MOMENT,
    AREA_TOTAL,
    AREA_WEIGHTED_CENTER,
    POINT_COUNT,
    POINT_MEAN,
    POINT_SUM,
    LocalMeshShard,
)
from physicsnemo.datapipes._rng import spawn_generator
from physicsnemo.datapipes.readers.mesh import (
    MeshReader,
    _subsample_mesh_cells,
    read_local_cell_shard,
)
from physicsnemo.datapipes.transforms.mesh import (
    CenterMesh,
    ComputeSurfaceNormals,
    SubsampleMesh,
)
from physicsnemo.distributed.utils import compute_split_shapes
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.io import io_zarr
from test.datapipes._local_shard_utils import (
    WINDOW as _WINDOW,
)
from test.datapipes._local_shard_utils import (
    grid_mesh,
    recipe_chain,
    write_sample,
)

_SEED = 1234
_TOL = {"atol": 1e-5, "rtol": 1e-6}


@pytest.fixture(scope="module", params=["pmsh", "zarr"])
def stored(request, tmp_path_factory):
    r"""``(open_source, full)``: a lazy opener of the stored sample, and the sample."""
    full = grid_mesh()
    root = tmp_path_factory.mktemp(f"local_shard_{request.param}")
    write_sample(full, root, "s", request.param)
    if request.param == "pmsh":
        return (lambda: Mesh.load(root / "s.pmsh", mode="r")), full
    pytest.importorskip("zarr")
    return (lambda: io_zarr._open_group(root / "s.zarr")), full


@pytest.fixture(scope="module")
def pmsh_stored(tmp_path_factory):
    r"""The ``pmsh`` store alone, for tests that are format-independent."""
    full = grid_mesh()
    root = tmp_path_factory.mktemp("local_shard_pmsh_only")
    write_sample(full, root, "s", "pmsh")
    return (lambda: Mesh.load(root / "s.pmsh", mode="r")), full


def _fresh(stored_fixture):
    # A fresh in-memory sample per test: transforms cache geometry on it.
    return stored_fixture[0], grid_mesh()


@pytest.fixture
def source(stored):
    return _fresh(stored)


@pytest.fixture
def pmsh_source(pmsh_stored):
    return _fresh(pmsh_stored)


def _window_and_reference(full: Mesh, windowed: bool):
    if not windowed:
        return None, full
    window = _cyclic_block_indices(
        full.n_cells, _WINDOW, generator=spawn_generator(_SEED, 0, 0)
    )
    reference = _subsample_mesh_cells(
        full, _WINDOW, generator=spawn_generator(_SEED, 0, 0)
    )
    return window, reference


def _shares(open_source, world, window, stats=frozenset()):
    return [
        read_local_cell_shard(
            open_source(),
            device_mesh=None,  # never touched before the wrap
            rank=r,
            world_size=world,
            cell_window=window,
            stats=stats,
        )
        for r in range(world)
    ]


def _reduced_stats(shares) -> TensorDict:
    r"""In-process stand-in for ``reduce_shard_stats``: sum the partials, divide."""
    out = {}
    for key, num, den in (
        (POINT_MEAN, POINT_SUM, POINT_COUNT),
        (AREA_WEIGHTED_CENTER, AREA_MOMENT, AREA_TOTAL),
    ):
        if num in shares[0].stats.keys():
            out[key] = sum(s.stats[num] for s in shares) / sum(
                s.stats[den] for s in shares
            )
    return TensorDict(out, batch_size=[])


def _with_stats(share, stats) -> LocalMeshShard:
    return LocalMeshShard(mesh=share.mesh, info=share.info, stats=stats)


@pytest.mark.parametrize("windowed", [False, True])
@pytest.mark.parametrize("world", [1, 3, 4])
def test_shares_reproduce_unsharded_read(source, world, windowed):
    r"""Concatenated shares and reduced partial sums equal the unsharded read."""
    open_source, full = source
    window, reference = _window_and_reference(full, windowed)
    shares = _shares(open_source, world, window, {POINT_MEAN, AREA_WEIGHTED_CENTER})

    n_cells = reference.n_cells
    assert all(
        s.info.cell_sizes == tuple(compute_split_shapes(n_cells, world)) for s in shares
    )
    point_sizes = shares[0].info.point_sizes
    for rank, share in enumerate(shares):
        mesh = share.mesh
        assert share.info.point_sizes == point_sizes
        assert mesh.n_cells == share.info.cell_sizes[rank]
        assert mesh.n_points == point_sizes[rank]
        # Cells index exactly this rank's points.
        assert int(mesh.cells.max()) == mesh.n_points - 1
        assert torch.unique(mesh.cells).numel() == mesh.n_points
        assert share.info.n_points == reference.n_points
    if world > 1:  # seam vertices are held by both neighbours
        assert sum(point_sizes) > reference.n_points

    # Per-cell geometry, point data seen through the cells, cell data and
    # the window's measure correction: exactly the unsharded rows.
    cat = torch.cat
    assert torch.equal(
        cat([s.mesh.points[s.mesh.cells] for s in shares]),
        reference.points[reference.cells],
    )
    assert torch.equal(
        cat([s.mesh.point_data["velocity"][s.mesh.cells] for s in shares]),
        reference.point_data["velocity"][reference.cells],
    )
    assert set(shares[0].mesh.cell_data.keys()) == set(reference.cell_data.keys())
    for key in reference.cell_data.keys():
        assert torch.equal(
            cat([s.mesh.cell_data[key] for s in shares]), reference.cell_data[key]
        )
    assert torch.equal(shares[0].mesh.global_data["Re"], reference.global_data["Re"])

    # Only the partial sums, in float64.
    for share in shares:
        assert set(share.stats.keys()) == {
            POINT_SUM,
            POINT_COUNT,
            AREA_MOMENT,
            AREA_TOTAL,
        }
        assert all(v.dtype == torch.float64 for v in share.stats.values())

    # Summed over the ranks, the partials give the unsharded centers
    # (seam vertices counted once).
    assert int(sum(s.stats[POINT_COUNT] for s in shares).item()) == reference.n_points
    reduced = _reduced_stats(shares)
    for key, area in ((POINT_MEAN, False), (AREA_WEIGHTED_CENTER, True)):
        torch.testing.assert_close(
            reduced[key].to(reference.points.dtype),
            CenterMesh(use_area_weighting=area)._compute_com(reference),
            **_TOL,
        )


def test_statistics_only_on_request(pmsh_source):
    r"""No statistics unless asked for; unknown names raise."""
    open_source, full = pmsh_source
    window, _ = _window_and_reference(full, True)
    (share,) = _shares(open_source, 1, window)
    assert len(share.stats.keys()) == 0
    with pytest.raises(ValueError, match="unknown shard statistics"):
        # Validation lives in the reader's request; it needs no instance state.
        MeshReader.request_shard_stats(object(), {"median"})


def test_cached_point_sizes_read_only_own_cells(pmsh_source):
    r"""Passing cached ``point_sizes`` reproduces the same shares."""
    open_source, _ = pmsh_source
    first = _shares(open_source, 3, None)
    again = [
        read_local_cell_shard(
            open_source(),
            device_mesh=None,
            rank=r,
            world_size=3,
            point_sizes=first[r].info.point_sizes,
        )
        for r in range(3)
    ]
    for a, b in zip(first, again):
        assert a.info.point_sizes == b.info.point_sizes
        assert torch.equal(a.mesh.points, b.mesh.points)
        assert torch.equal(a.mesh.cells, b.mesh.cells)


def test_too_few_cells_for_the_ranks_raises(pmsh_source):
    open_source, _ = pmsh_source
    with pytest.raises(ValueError, match="cannot split 3 cells over 4 ranks"):
        read_local_cell_shard(
            open_source(),
            device_mesh=None,
            rank=0,
            world_size=4,
            cell_window=torch.arange(3),
        )


def _apply_local(chain, payload):
    data = payload.mesh
    for t in chain:
        data = t.apply_local_shard(data, payload)
    return data


@pytest.mark.parametrize("area_weighted", [False, True])
@pytest.mark.parametrize("windowed", [False, True])
@pytest.mark.parametrize("world", [3])
def test_local_transform_chain_matches_unsharded(
    pmsh_source, world, windowed, area_weighted
):
    r"""The recipe-style chain (centering, normalization, cell normals, a
    no-op subsample, centroid interior) on each share alone gives exactly
    the unsharded chain's rows."""
    open_source, full = pmsh_source
    window, reference = _window_and_reference(full, windowed)
    chain = recipe_chain(
        area_weighted=area_weighted,
        window=_WINDOW if windowed else None,
        store_center_as="center",
    )
    stats = set()
    for t in chain:
        stats |= t.required_shard_stats()
    shares = _shares(open_source, world, window, stats)
    reduced = _reduced_stats(shares)
    outputs = [_apply_local(chain, _with_stats(s, reduced)) for s in shares]
    for t in chain:
        reference = t(reference)

    cat = torch.cat
    # Geometry is centered by the reduced center: equal up to rounding.
    torch.testing.assert_close(
        cat([o.interior.points for o in outputs]), reference.interior.points, **_TOL
    )
    for key in reference.interior.point_data.keys():
        assert torch.equal(
            cat([o.interior.point_data[key] for o in outputs]),
            reference.interior.point_data[key],
        )
    bnd = [o.boundaries["vehicle"] for o in outputs]
    ref_bnd = reference.boundaries["vehicle"]
    torch.testing.assert_close(
        cat([b.cell_data["normals"] for b in bnd]),
        ref_bnd.cell_data["normals"],
        **_TOL,
    )
    torch.testing.assert_close(
        cat([b.points[b.cells] for b in bnd]), ref_bnd.points[ref_bnd.cells], **_TOL
    )
    for o in outputs:
        torch.testing.assert_close(
            o.global_data["center"], reference.global_data["center"], **_TOL
        )
        assert "Re" not in o.global_data.keys()


def test_unsupported_transforms_raise(pmsh_source):
    open_source, full = pmsh_source
    window, _ = _window_and_reference(full, True)
    payload = _shares(open_source, 2, window)[0]
    with pytest.raises(NotImplementedError, match="subsample"):
        SubsampleMesh(n_cells=_WINDOW - 1).apply_local_shard(payload.mesh, payload)
    with pytest.raises(NotImplementedError, match="subsample"):
        SubsampleMesh(n_points=3).apply_local_shard(payload.mesh, payload)
    with pytest.raises(NotImplementedError, match="seam"):
        ComputeSurfaceNormals(store_as="point_data").apply_local_shard(
            payload.mesh, payload
        )
