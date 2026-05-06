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

"""Unit tests for `src/collate.py`.

Covers the two-mode batching contract:

- ``input_type='tensors'``: tensor values get padded up to ``ndim >= 2``
  and prepended with a batch dim of 1 (so 1-D token features become
  ``(1, 1, D)`` and per-element features become ``(1, N, C)``); the
  targets TensorDict gets a single ``unsqueeze(0)`` so its batch_size
  goes from ``[N]`` to ``[1, N]`` (per-element scalars become ``(1, N)``,
  per-element vectors become ``(1, N, D)``).
- ``input_type='mesh'``: forward kwargs and targets pass through
  unchanged; Mesh objects, scalar tensors, and nested dicts all stay
  in their natural shape; the targets TensorDict keeps batch_size ``[N]``.

Plus the failure paths: ``batch_size > 1`` raises ``NotImplementedError``;
unknown ``input_type`` raises ``ValueError``.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.mesh import DomainMesh, Mesh

from collate import _add_batch_dim_token, build_collate_fn


### ---------------------------------------------------------------------------
### Fixtures
### ---------------------------------------------------------------------------


@pytest.fixture
def domain(surface_domain_mesh) -> DomainMesh:
    """Small surface DomainMesh used by the collate tests.

    Wraps the shared ``surface_domain_mesh`` fixture (interior at cell
    centroids, vehicle boundary with normals) under the local name
    ``domain`` that the collate suite has historically used.
    """
    return surface_domain_mesh


### ---------------------------------------------------------------------------
### Batch-dim helpers
### ---------------------------------------------------------------------------


class TestBatchDimHelpers:
    """Unit tests for the two batch-dim padding helpers."""

    def test_token_padding_zero_dim(self):
        ### Scalar literal -> (1, 1) padded then (1, 1, 1) batch-wrapped.
        """Token padding zero dim."""
        t = torch.tensor(1.0)
        assert tuple(_add_batch_dim_token(t).shape) == (1, 1, 1)

    def test_token_padding_one_dim(self):
        ### Token feature like U_inf (3,) -> (1, 3) padded then (1, 1, 3) batched.
        """Token padding one dim."""
        t = torch.randn(3)
        assert tuple(_add_batch_dim_token(t).shape) == (1, 1, 3)

    def test_token_padding_two_dim(self):
        ### Per-element feature (N, C) needs no padding -> (1, N, C).
        """Token padding two dim."""
        t = torch.randn(10, 5)
        assert tuple(_add_batch_dim_token(t).shape) == (1, 10, 5)


### ---------------------------------------------------------------------------
### Tensor-input collate (transformer-style models)
### ---------------------------------------------------------------------------


class TestTensorInputCollate:
    """Tests for tensor input collate."""

    def test_returns_dict_with_two_keys(self, domain):
        """Returns dict with two keys."""
        spec = {"geometry": "interior.points"}
        collate = build_collate_fn(
            "tensors", spec, {"pressure": "scalar", "wss": "vector"}
        )
        batch = collate([(domain, {})])
        assert set(batch) == {"forward_kwargs", "targets"}

    def test_per_element_feature_gets_batch_dim(self, domain):
        """Per element feature gets batch dim."""
        spec = {"geometry": "interior.points"}
        collate = build_collate_fn("tensors", spec, {"pressure": "scalar"})
        batch = collate([(domain, {})])
        ### interior.points is (6, 3) -> (1, 6, 3).
        assert tuple(batch["forward_kwargs"]["geometry"].shape) == (1, 6, 3)

    def test_global_feature_gets_token_padding(self, domain):
        """Global feature gets token padding."""
        spec = {"global_embedding": "global_data.U_inf"}
        collate = build_collate_fn("tensors", spec, {"pressure": "scalar"})
        batch = collate([(domain, {})])
        ### U_inf is (3,) -> (1, 3) padded -> (1, 1, 3) batched.
        assert tuple(batch["forward_kwargs"]["global_embedding"].shape) == (1, 1, 3)

    def test_scalar_literal_in_mesh_kwargs_becomes_111(self, domain):
        ### A literal float in the spec gets resolved to a 0-d tensor;
        ### with input_type='tensors', it pads up to (1, 1) then batches
        ### to (1, 1, 1). Confirms the pad-up rule still applies to
        ### resolved literals.
        """Scalar literal in mesh kwargs becomes 111."""
        spec = {"weight": 0.42}
        collate = build_collate_fn("tensors", spec, {"pressure": "scalar"})
        batch = collate([(domain, {})])
        w = batch["forward_kwargs"]["weight"]
        assert tuple(w.shape) == (1, 1, 1)
        assert float(w) == pytest.approx(0.42)

    def test_targets_become_tensordict_with_batch_dim(self, domain):
        ### The asymmetry with forward_kwargs: scalar target leaves stay
        ### (1, N) (not padded to (1, 1, N)) so they match the model's
        ### (1, N) split-from-(B, N, C) prediction. TensorDict.unsqueeze(0)
        ### grows batch_size [6] -> [1, 6] and every leaf in lock-step.
        """Targets become tensordict with batch dim."""
        collate = build_collate_fn(
            "tensors",
            {"geometry": "interior.points"},
            {"pressure": "scalar", "wss": "vector"},
        )
        batch = collate([(domain, {})])
        assert isinstance(batch["targets"], TensorDict)
        assert batch["targets"].batch_size == torch.Size([1, 6])
        assert tuple(batch["targets"]["pressure"].shape) == (1, 6)
        assert tuple(batch["targets"]["wss"].shape) == (1, 6, 3)

    def test_list_spec_concatenates(self, domain):
        """List spec concatenates."""
        spec = {
            "embedding": [
                "interior.points",
                "interior.point_data.pressure",
            ],
        }
        collate = build_collate_fn("tensors", spec, {"pressure": "scalar"})
        batch = collate([(domain, {})])
        ### (6, 3) cat with (6, 1) on last dim -> (6, 4); batched -> (1, 6, 4).
        assert tuple(batch["forward_kwargs"]["embedding"].shape) == (1, 6, 4)

    def test_nested_dict_kwargs_recurse(self, domain):
        ### DoMINO-style: the model takes a single `data_dict` argument.
        """Nested dict kwargs recurse."""
        spec = {
            "data_dict": {
                "geometry": "interior.points",
                "global": "global_data.U_inf",
            },
        }
        collate = build_collate_fn("tensors", spec, {"pressure": "scalar"})
        batch = collate([(domain, {})])
        dd = batch["forward_kwargs"]["data_dict"]
        assert isinstance(dd, dict)
        assert tuple(dd["geometry"].shape) == (1, 6, 3)
        assert tuple(dd["global"].shape) == (1, 1, 3)


### ---------------------------------------------------------------------------
### Mesh-input collate (GLOBE-style models)
### ---------------------------------------------------------------------------


class TestMeshInputCollate:
    """Tests for mesh input collate."""

    def test_tensor_passes_through_unbatched(self, domain):
        """Tensor passes through unbatched."""
        spec = {"prediction_points": "interior.points"}
        collate = build_collate_fn(
            "mesh", spec, {"pressure": "scalar", "wss": "vector"}
        )
        batch = collate([(domain, {})])
        ### (6, 3) stays (6, 3) -- no batch dim added.
        assert tuple(batch["forward_kwargs"]["prediction_points"].shape) == (6, 3)

    def test_mesh_passes_through(self, domain):
        """Mesh passes through."""
        spec = {"boundary_meshes": {"vehicle": "boundaries.vehicle"}}
        collate = build_collate_fn("mesh", spec, {"pressure": "scalar"})
        batch = collate([(domain, {})])
        bm = batch["forward_kwargs"]["boundary_meshes"]
        assert isinstance(bm, dict)
        assert isinstance(bm["vehicle"], Mesh)
        ### Same Mesh object referenced from the source DomainMesh.
        assert bm["vehicle"] is domain.boundaries["vehicle"]

    def test_scalar_literals_stay_zero_dim(self, domain):
        """Scalar literals stay zero dim."""
        spec = {
            "reference_lengths": {"L_ref": 1.0, "delta_turb": 0.015},
        }
        collate = build_collate_fn("mesh", spec, {"pressure": "scalar"})
        batch = collate([(domain, {})])
        rl = batch["forward_kwargs"]["reference_lengths"]
        assert isinstance(rl, dict)
        for name, expected in [("L_ref", 1.0), ("delta_turb", 0.015)]:
            t = rl[name]
            assert isinstance(t, torch.Tensor)
            assert t.ndim == 0
            assert float(t) == pytest.approx(expected)

    def test_targets_are_tensordict_without_batch_dim(self, domain):
        """Targets are tensordict without batch dim."""
        collate = build_collate_fn(
            "mesh",
            {"prediction_points": "interior.points"},
            {"pressure": "scalar", "wss": "vector"},
        )
        batch = collate([(domain, {})])
        assert isinstance(batch["targets"], TensorDict)
        ### Mesh-input mode: batch_size matches the source point_data ([N]).
        assert batch["targets"].batch_size == torch.Size([6])
        assert tuple(batch["targets"]["pressure"].shape) == (6,)
        assert tuple(batch["targets"]["wss"].shape) == (6, 3)


### ---------------------------------------------------------------------------
### Failure paths
### ---------------------------------------------------------------------------


class TestFailures:
    """Tests for failures."""

    def test_invalid_input_type_raises(self):
        """Invalid input type raises."""
        with pytest.raises(ValueError, match="input_type must be"):
            build_collate_fn("invalid_mode", {}, {"pressure": "scalar"})

    def test_batch_size_greater_than_one_raises(self, domain):
        """Batch size greater than one raises."""
        collate = build_collate_fn(
            "tensors", {"geometry": "interior.points"}, {"pressure": "scalar"}
        )
        with pytest.raises(NotImplementedError, match=r"len\(samples\)=2"):
            collate([(domain, {}), (domain, {})])

    def test_empty_batch_raises(self, domain):
        """Empty batch raises."""
        collate = build_collate_fn(
            "tensors", {"geometry": "interior.points"}, {"pressure": "scalar"}
        )
        with pytest.raises(NotImplementedError, match=r"len\(samples\)=0"):
            collate([])

    def test_missing_target_in_mesh_raises(self, domain):
        """Missing target in mesh raises."""
        collate = build_collate_fn(
            "tensors",
            {"geometry": "interior.points"},
            {"missing_field": "scalar"},
        )
        with pytest.raises(KeyError, match="not found in interior.point_data"):
            collate([(domain, {})])
