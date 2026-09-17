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

r"""Tests for ``ShardTensorSpec`` hashing and the redistribute planner cache key.

``ShardTensorSpec`` hashes on its (lazily populated) ``_sharding_shapes``,
so the cached hash must be invalidated when they are assigned. Torch's
redistribute planner (``_gen_transform_infos``) is an ``lru_cache`` keyed on
the specs it is given; feeding it ``ShardTensorSpec`` objects whose shapes
populate *after* first use makes every call a miss. ``_plan_key`` strips the
spec down to a plain ``DTensorSpec`` so the cache actually hits.

Also covers ``record_consumer_stream`` with a real ``ShardTensor`` (its
``to_local`` unwrap), which needs a device mesh.
"""

import dataclasses

import pytest
import torch
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._redistribute import _gen_transform_infos
from torch.distributed.tensor.placement_types import Replicate

from physicsnemo.datapipes.protocols import record_consumer_stream
from physicsnemo.domain_parallel import ShardTensor
from physicsnemo.domain_parallel._shard_redistribute import _plan_key
from test.domain_parallel.test_redistribute import shard_tensor_factory


def _as_plain(shapes):
    r"""Normalize a sharding-shapes dict to ``{dim: ((int, ...), ...)}``."""
    return {k: tuple(tuple(int(d) for d in s) for s in v) for k, v in shapes.items()}


# ---------------------------------------------------------------------------
# _plan_key
# ---------------------------------------------------------------------------


def run_plan_key_strips_sharding_shapes(mesh):
    spec = shard_tensor_factory(mesh, uneven=True)._spec
    assert spec._sharding_shapes, "factory should infer sharding shapes"

    key = _plan_key(spec)
    assert type(key) is DTensorSpec
    assert key.mesh == spec.mesh
    assert key.placements == tuple(spec.placements)
    assert tuple(key.tensor_meta.shape) == tuple(spec.tensor_meta.shape)
    assert tuple(key.tensor_meta.stride) == tuple(spec.tensor_meta.stride)
    assert key.tensor_meta.dtype == spec.tensor_meta.dtype

    # Same layout, no sharding shapes: a different ShardTensorSpec hash, but
    # an identical plan key -- that's the whole point.
    bare = dataclasses.replace(spec, _sharding_shapes=None)
    assert hash(bare) != hash(spec)
    assert _plan_key(bare) == key
    assert hash(_plan_key(bare)) == hash(key)


@pytest.mark.multigpu_static
@pytest.mark.timeout(60)
def test_plan_key_strips_sharding_shapes_1d(distributed_mesh):
    run_plan_key_strips_sharding_shapes(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(60)
def test_plan_key_strips_sharding_shapes_2d(distributed_mesh_2d):
    run_plan_key_strips_sharding_shapes(distributed_mesh_2d)


def run_planner_cache_does_not_grow(mesh):
    if not hasattr(_gen_transform_infos, "cache_info"):
        pytest.skip("torch's redistribute planner is not lru_cache'd here")
    replicate = [Replicate()] * mesh.ndim

    def redistribute_both_variants():
        st = shard_tensor_factory(mesh, uneven=True)
        st.redistribute(placements=replicate)
        # Same layout with the shapes still unknown -> negotiation fallback.
        st_bare = ShardTensor.__new__(
            ShardTensor,
            local_tensor=st._local_tensor,
            spec=dataclasses.replace(st._spec, _sharding_shapes=None),
            requires_grad=False,
        )
        st_bare.redistribute(placements=replicate)

    # Warm every key this workload can produce.
    redistribute_both_variants()
    before = _gen_transform_infos.cache_info()

    for _ in range(3):
        redistribute_both_variants()

    after = _gen_transform_infos.cache_info()
    assert after.currsize == before.currsize, (
        f"planner cache grew {before.currsize} -> {after.currsize}"
    )
    assert after.hits > before.hits


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_planner_cache_does_not_grow_1d(distributed_mesh):
    run_planner_cache_does_not_grow(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_planner_cache_does_not_grow_2d(distributed_mesh_2d):
    run_planner_cache_does_not_grow(distributed_mesh_2d)


# ---------------------------------------------------------------------------
# ShardTensorSpec.__setattr__ hash invalidation
# ---------------------------------------------------------------------------


def run_setattr_sharding_shapes_invalidates_hash(mesh):
    spec = shard_tensor_factory(mesh, uneven=True)._spec
    bare = dataclasses.replace(spec, _sharding_shapes=None)

    h_bare = hash(bare)
    assert bare._hash == h_bare

    bare._sharding_shapes = dict(spec._sharding_shapes)
    assert bare._hash is None, "assigning _sharding_shapes must drop the cached hash"
    assert hash(bare) != h_bare
    assert hash(bare) == hash(spec)

    # An attribute that is not part of the hash leaves the cache alone.
    h_now = hash(bare)
    bare._local_shape = bare._local_shape
    assert bare._hash == h_now


@pytest.mark.multigpu_static
@pytest.mark.timeout(60)
def test_setattr_sharding_shapes_invalidates_hash_1d(distributed_mesh):
    run_setattr_sharding_shapes_invalidates_hash(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(60)
def test_setattr_sharding_shapes_invalidates_hash_2d(distributed_mesh_2d):
    run_setattr_sharding_shapes_invalidates_hash(distributed_mesh_2d)


def run_lazy_population_refreshes_hash(mesh):
    st = shard_tensor_factory(mesh, uneven=True)
    bare = dataclasses.replace(st._spec, _sharding_shapes=None)
    h_before = hash(bare)

    # Collective: every rank participates in the gather.
    populated = bare.sharding_shapes()

    assert bare._sharding_shapes is not None
    assert _as_plain(populated) == _as_plain(st._spec.sharding_shapes())
    assert hash(bare) != h_before
    assert hash(bare) == hash(st._spec)


@pytest.mark.multigpu_static
@pytest.mark.timeout(60)
def test_lazy_population_refreshes_hash_1d(distributed_mesh):
    run_lazy_population_refreshes_hash(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(60)
def test_lazy_population_refreshes_hash_2d(distributed_mesh_2d):
    run_lazy_population_refreshes_hash(distributed_mesh_2d)


# ---------------------------------------------------------------------------
# record_consumer_stream with a real ShardTensor
# ---------------------------------------------------------------------------


@pytest.mark.multigpu_static
@pytest.mark.timeout(60)
def test_record_consumer_stream_records_shard_tensor_local(
    distributed_mesh, monkeypatch
):
    recorded: list = []
    monkeypatch.setattr(
        torch.Tensor,
        "record_stream",
        lambda self, stream: recorded.append((self, stream)),
    )

    st = shard_tensor_factory(distributed_mesh, uneven=True)
    stream = torch.cuda.Stream()
    record_consumer_stream({"x": st, "cpu": torch.ones(2)}, stream)

    assert len(recorded) == 1
    tensor, seen_stream = recorded[0]
    assert type(tensor) is torch.Tensor, "wrapper must be unwrapped via to_local()"
    assert tensor is st._local_tensor
    assert seen_stream is stream
