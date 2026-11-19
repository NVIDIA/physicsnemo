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

r"""The ``"local"`` domain-parallel mesh layout through ``MeshDataset``.

Every rank reads its share of a triangulated grid (cells share vertices
across the shard seams) and runs a recipe-style chain -- centering,
augmentations, normalization, cell normals, a no-op subsample, a centroid
interior -- on its plain local mesh; the wrapped ShardTensor sample must
equal the unsharded pipeline: bit for bit when the transforms run on the
host, and to float32 rounding on the GPU. The whole-sample center is a
partial-sum all-reduce, so center-dependent geometry matches to rounding. A
chain without CenterMesh issues no collectives; CenterMesh adds exactly one
all_reduce per sample.

Run with ``torchrun --nproc-per-node {2,4} -m pytest --multigpu-static``.
"""

import re
import threading
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.datapipes import MeshDataset
from physicsnemo.datapipes.readers.mesh import MeshReader
from physicsnemo.datapipes.transforms.mesh import (
    CenterMesh,
    ComputeSurfaceNormals,
    MeshToTensorDict,
    RandomTranslateMesh,
    RestructureTensorDict,
    SubsampleMesh,
)
from physicsnemo.distributed import DistributedManager
from physicsnemo.distributed.utils import compute_split_shapes
from physicsnemo.domain_parallel import ShardTensor
from physicsnemo.mesh import DomainMesh
from test.datapipes._local_shard_utils import (
    WINDOW,
    grid_mesh,
    recipe_chain,
    write_sample,
)

pytestmark = [pytest.mark.multigpu_static, pytest.mark.timeout(300)]

_N_SAMPLES = 3
_SEED = 2024


def _chain(augment: bool = False, area_weighted: bool = False, center: bool = True):
    return recipe_chain(center=center, area_weighted=area_weighted, augment=augment)


def _make_root(tmp_path_factory, distributed_mesh, fmt):
    dm = DistributedManager()
    if dm.rank == 0:
        root = tmp_path_factory.mktemp(f"local_{fmt}")
        for i in range(_N_SAMPLES):
            write_sample(grid_mesh(31 + i, 50.0), root, f"sample_{i}", fmt)
        holder = [str(root)]
    else:
        holder = [None]
    dist.broadcast_object_list(holder, src=0)
    return holder[0], fmt


@pytest.fixture(scope="module", params=["pmsh", "zarr"])
def root_and_fmt_all(request, tmp_path_factory, distributed_mesh):
    return _make_root(tmp_path_factory, distributed_mesh, request.param)


@pytest.fixture(scope="module")
def root_and_fmt(tmp_path_factory, distributed_mesh):
    r"""The ``pmsh`` store, for tests that are format-independent."""
    return _make_root(tmp_path_factory, distributed_mesh, "pmsh")


def _dataset(root_and_fmt, device, *, chain, device_mesh=None, layout="local"):
    root, fmt = root_and_fmt
    kwargs = {}
    if device_mesh is not None:
        kwargs = dict(
            domain_parallel={"auto_shard_size": 2, "mesh_layout": layout},
            device_mesh=device_mesh,
        )
    reader = MeshReader(root, pattern=f"**/*.{fmt}", subsample_n_cells=WINDOW, **kwargs)
    dataset = MeshDataset(reader, transforms=chain, device=device)
    generator = torch.Generator()
    generator.manual_seed(_SEED)
    dataset.set_generator(generator)
    dataset.set_epoch(3)
    return dataset


def _assert_matches(
    sample: DomainMesh,
    reference: DomainMesh,
    world: int,
    exact: bool,
    centered: bool = False,
):
    device = sample.interior.points._local_tensor.device
    reference = reference.to(device)

    def check(actual, expected, geometry=False):
        # The center is a partial-sum all-reduce: equal to the unsharded one
        # up to float rounding, so only center-dependent geometry is relaxed.
        if exact and not (centered and geometry):
            assert torch.equal(actual, expected)
        else:
            # The reader computes the center on the host; the unsharded
            # pipeline on the GPU: float32 reduction-order rounding.
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-5)

    # Cell axis: chunk-sharded, the unsharded rows.
    interior = sample.interior
    assert isinstance(interior.points, ShardTensor)
    assert interior.points._spec.placements == (Shard(0),)
    sizes = [s[0] for s in interior.points._spec.sharding_shapes(0)]
    assert sizes == compute_split_shapes(WINDOW, world)
    check(interior.points.full_tensor(), reference.interior.points, True)
    for key in reference.interior.point_data.keys():
        # Cell areas come from the centered (then augmented) points.
        check(
            interior.point_data[key].full_tensor(),
            reference.interior.point_data[key],
            key == "_effective_measure",
        )

    # The surface: its cells index this rank's copy of their vertices on the
    # concatenated point axis, so the routed gather resolves the unsharded
    # per-cell geometry.
    bnd, ref_bnd = sample.boundaries["vehicle"], reference.boundaries["vehicle"]
    assert isinstance(bnd.cells, ShardTensor)
    assert bnd.n_cells == ref_bnd.n_cells
    check(bnd.cell_data["normals"].full_tensor(), ref_bnd.cell_data["normals"], True)
    check(
        bnd.points.full_tensor()[bnd.cells.full_tensor()],
        ref_bnd.points[ref_bnd.cells],
        True,
    )
    # Recomputed on the device on both sides (the reference's cached
    # centroids were computed on the host).
    check(
        bnd.cell_centroids.full_tensor(),
        ref_bnd.points[ref_bnd.cells].mean(dim=1),
        True,
    )
    assert "Re" not in sample.global_data.keys()
    check(sample.global_data["U_inf"], reference.global_data["U_inf"])


@pytest.mark.parametrize(
    "on_device, augment, area_weighted",
    [
        (False, False, False),
        (False, False, True),
        (True, True, False),
        (False, True, False),
        (False, True, True),
    ],
)
def test_local_layout_matches_unsharded(
    root_and_fmt_all, distributed_mesh, augment, on_device, area_weighted
):
    r"""Sharded local-layout samples equal the unsharded pipeline's."""
    dm = DistributedManager()
    device = dm.device if on_device else None
    chain = lambda: _chain(augment, area_weighted)  # noqa: E731
    sharded = _dataset(
        root_and_fmt_all, device, chain=chain(), device_mesh=distributed_mesh
    )
    reference = _dataset(root_and_fmt_all, device, chain=chain())
    try:
        for i in range(_N_SAMPLES):
            payload = sharded._load_host(i)
            assert payload.error is None
            sample, _ = sharded._consume(payload)
            expected, _ = reference[i]
            _assert_matches(
                sample,
                expected,
                distributed_mesh.size(0),
                exact=not on_device,
                centered=True,
            )
    finally:
        sharded.close()
        reference.close()


# Communication as the profiler records it: c10d / NCCL ops, functional
# collectives and ShardTensor's routed gather.
_COLLECTIVE = re.compile(
    r"nccl|gloo|c10d::|_c10d_functional|record_param_comms|routed_gather|"
    r"exchange_requests|all_?reduce|all_?gather|all_?to_?all|reduce_scatter",
    re.IGNORECASE,
)


def _collective_events(dataset, indices) -> list[str]:
    r"""Profiler event names of the loader that look like communication."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        for i in indices:
            dataset._consume(dataset._load_host(i))
    return sorted({e.name for e in prof.events() if _COLLECTIVE.search(e.name)})


@pytest.mark.parametrize("center", [None, "point_mean", "area_weighted"])
def test_collective_count(root_and_fmt, distributed_mesh, center):
    r"""No CenterMesh: no communication; centering: exactly one all_reduce per
    sample. The global layout is the positive control of the detector."""
    dm = DistributedManager()
    chain = lambda: _chain(  # noqa: E731
        True, area_weighted=center == "area_weighted", center=center is not None
    )
    dataset = _dataset(
        root_and_fmt, dm.device, chain=chain(), device_mesh=distributed_mesh
    )
    try:
        if center is None:
            assert _collective_events(dataset, range(_N_SAMPLES)) == []
            global_ = _dataset(
                root_and_fmt,
                dm.device,
                chain=chain(),
                device_mesh=distributed_mesh,
                layout="global",
            )
            try:
                assert _collective_events(global_, [0]) != []
            finally:
                global_.close()
            return
        for i in range(_N_SAMPLES):
            events = _collective_events(dataset, [i])
            assert len(events) >= 1, events
            # record_param_comms is the profiler's bookkeeping of that call.
            assert all(
                re.search(r"all_?reduce|record_param_comms", e, re.I) for e in events
            ), events
            with torch.profiler.profile() as prof:
                dataset._consume(dataset._load_host(i))
            n = sum(
                1
                for e in prof.events()
                if re.search(r"c10d::allreduce_|c10d::all_reduce", e.name, re.I)
            )
            assert n == 1, n
    finally:
        dataset.close()


def test_rank_zero_iterates_alone(root_and_fmt, distributed_mesh):
    r"""Rank 0 loads every sample while the other ranks wait at a barrier: a
    collective inside the loader would never complete. Holds only for chains
    without CenterMesh (centering all-reduces its partial sums)."""
    dm = DistributedManager()
    dist.barrier()
    if dm.rank == 0:
        dataset = _dataset(
            root_and_fmt,
            dm.device,
            chain=_chain(True, center=False),
            device_mesh=distributed_mesh,
        )
        result: list = []

        def work():
            try:
                for i in range(_N_SAMPLES):
                    sample, _ = dataset._consume(dataset._load_host(i))
                    sample.interior.points._local_tensor.sum().item()
                result.append("done")
            except Exception as e:  # noqa: BLE001
                result.append(e)

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        thread.join(timeout=120)
        dataset.close()
        assert result == ["done"], result or "rank 0 blocked alone (a collective?)"
    dist.barrier()


def test_local_layout_rejects_unsupported_chains(root_and_fmt, distributed_mesh):
    dm = DistributedManager()
    for chain, error, match in [
        ([SubsampleMesh(n_cells=WINDOW - 7)], NotImplementedError, "subsample"),
        ([ComputeSurfaceNormals(store_as="point_data")], NotImplementedError, "seam"),
        (
            [RandomTranslateMesh(), CenterMesh(use_area_weighting=False)],
            ValueError,
            "RandomTranslateMesh ran before it",
        ),
        (
            [
                MeshToTensorDict(),
                RestructureTensorDict(
                    {"input": {"vertices": "points", "connectivity": "cells"}}
                ),
            ],
            NotImplementedError,
            "cannot move 'cells'",
        ),
        (
            [
                MeshToTensorDict(),
                RestructureTensorDict({"input": {"conn": "boundaries.vehicle.cells"}}),
            ],
            NotImplementedError,
            "cannot move 'cells'",
        ),
    ]:
        dataset = _dataset(
            root_and_fmt, dm.device, chain=chain, device_mesh=distributed_mesh
        )
        try:
            with pytest.raises(error, match=match):
                dataset[0]
        finally:
            dataset.close()


def test_local_layout_restructure_without_cells(root_and_fmt, distributed_mesh):
    dm = DistributedManager()
    chain = [
        MeshToTensorDict(),
        RestructureTensorDict({"input": {"vertices": "points"}}),
    ]
    dataset = _dataset(
        root_and_fmt, dm.device, chain=chain, device_mesh=distributed_mesh
    )
    try:
        assert "input" in dataset[0][0].keys()
    finally:
        dataset.close()


def test_prepare_background_consume_dedicated_group(root_and_fmt, distributed_mesh):
    r"""After prepare_background_consume the centering all_reduce runs on a
    group other than the device mesh's, with unchanged results."""
    dm = DistributedManager()
    sharded = _dataset(
        root_and_fmt, dm.device, chain=_chain(True, False), device_mesh=distributed_mesh
    )
    reference = _dataset(root_and_fmt, dm.device, chain=_chain(True, False))
    groups = []
    orig = dist.all_reduce

    def spy(tensor, *args, **kwargs):
        groups.append(kwargs.get("group"))
        return orig(tensor, *args, **kwargs)

    try:
        sharded.prepare_background_consume()
        sharded.prepare_background_consume()  # idempotent
        mesh_group = distributed_mesh.get_group()
        assert sharded._stats_group is not None
        assert sharded._stats_group != mesh_group
        assert dist.get_process_group_ranks(
            sharded._stats_group
        ) == dist.get_process_group_ranks(mesh_group)
        with patch.object(dist, "all_reduce", spy):
            for i in range(_N_SAMPLES):
                sample, _ = sharded._consume(sharded._load_host(i))
                expected, _ = reference[i]
                _assert_matches(
                    sample,
                    expected,
                    distributed_mesh.size(0),
                    exact=False,
                    centered=True,
                )
        assert len(groups) == _N_SAMPLES
        assert all(g is sharded._stats_group for g in groups)
    finally:
        sharded.close()
        reference.close()


def test_prepare_background_consume_global_layout_raises(
    root_and_fmt, distributed_mesh
):
    dm = DistributedManager()
    dataset = _dataset(
        root_and_fmt,
        dm.device,
        chain=_chain(True),
        device_mesh=distributed_mesh,
        layout="global",
    )
    try:
        with pytest.raises(ValueError, match="global"):
            dataset.prepare_background_consume()
    finally:
        dataset.close()


def test_prepare_background_consume_no_stats(root_and_fmt, distributed_mesh):
    r"""Without centering no group is needed and still no collectives."""
    dm = DistributedManager()
    dataset = _dataset(
        root_and_fmt,
        dm.device,
        chain=_chain(True, center=False),
        device_mesh=distributed_mesh,
    )
    try:
        dataset.prepare_background_consume()
        assert _collective_events(dataset, range(_N_SAMPLES)) == []
    finally:
        dataset.close()
