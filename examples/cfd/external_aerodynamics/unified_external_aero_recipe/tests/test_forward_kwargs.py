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

"""Unit tests for `src/forward_kwargs.py`.

Covers the path resolver (`walk_path`), the per-value resolver
(`resolve_spec`), the top-level resolver with two-pass `expand_like`
handling (`resolve_forward_kwargs`), and target extraction
(`extract_targets`).
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.mesh import DomainMesh, Mesh

from forward_kwargs import (
    extract_targets,
    resolve_forward_kwargs,
    resolve_spec,
    walk_path,
)


### ---------------------------------------------------------------------------
### Fixtures
### ---------------------------------------------------------------------------


@pytest.fixture
def simple_mesh() -> Mesh:
    """A small triangle mesh in 3D with point_data and global_data."""
    return Mesh(
        points=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        cells=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        point_data={"phi": torch.tensor([1.0, 2.0, 3.0])},
        cell_data={"normals": torch.tensor([[0.0, 0.0, 1.0]])},
        global_data={"U_inf": torch.tensor([30.0, 0.0, 0.0])},
    )


@pytest.fixture
def simple_domain() -> DomainMesh:
    """A DomainMesh with a 10-point interior and one ``vehicle`` boundary.

    Wraps the shared :func:`conftest.make_surface_domain_mesh` factory
    with ``n_cells=10`` so the ``forward_kwargs`` tests can assert on
    ``(10, 3)`` / ``(10, 6)`` / ``(10, 4)`` shapes.
    """
    from conftest import make_surface_domain_mesh

    return make_surface_domain_mesh(n_cells=10)


### ---------------------------------------------------------------------------
### walk_path
### ---------------------------------------------------------------------------


class TestWalkPath:
    """Tests for walk path."""

    def test_empty_string_returns_source(self, simple_domain):
        """Empty string returns source."""
        assert walk_path(simple_domain, "") is simple_domain

    def test_single_dot_returns_source(self, simple_domain):
        """Single dot returns source."""
        assert walk_path(simple_domain, ".") is simple_domain

    def test_single_attr_via_getattr(self, simple_domain):
        ### `interior` is a Mesh attribute on DomainMesh.
        """Single attr via getattr."""
        result = walk_path(simple_domain, "interior")
        assert isinstance(result, Mesh)
        assert result is simple_domain.interior

    def test_chained_attr_then_item(self, simple_domain):
        ### `boundaries` (attr) -> `vehicle` (TensorDict item).
        """Chained attr then item."""
        result = walk_path(simple_domain, "boundaries.vehicle")
        assert isinstance(result, Mesh)

    def test_deep_path_to_tensor(self, simple_domain):
        ### `boundaries.vehicle.cell_data.normals` walks attr/item/attr/item.
        """Deep path to tensor."""
        result = walk_path(simple_domain, "boundaries.vehicle.cell_data.normals")
        assert isinstance(result, torch.Tensor)
        assert torch.equal(
            result, simple_domain.boundaries["vehicle"].cell_data["normals"]
        )

    def test_global_data_lookup(self, simple_domain):
        """Global data lookup."""
        result = walk_path(simple_domain, "global_data.U_inf")
        assert torch.equal(result, simple_domain.global_data["U_inf"])

    def test_interior_point_data_field(self, simple_domain):
        """Interior point data field."""
        result = walk_path(simple_domain, "interior.point_data.pressure")
        assert torch.equal(result, simple_domain.interior.point_data["pressure"])

    def test_invalid_path_raises_keyerror(self, simple_domain):
        """Invalid path raises keyerror."""
        with pytest.raises(KeyError, match="Cannot resolve segment"):
            walk_path(simple_domain, "interior.point_data.nonexistent_field")

    def test_invalid_first_segment_raises_keyerror(self, simple_domain):
        """Invalid first segment raises keyerror."""
        with pytest.raises(KeyError, match="Cannot resolve segment"):
            walk_path(simple_domain, "no_such_attribute")


### ---------------------------------------------------------------------------
### resolve_spec
### ---------------------------------------------------------------------------


class TestResolveSpec:
    """Tests for resolve spec."""

    def test_string_path(self, simple_domain):
        """String path."""
        result = resolve_spec("interior.points", simple_domain)
        assert torch.equal(result, simple_domain.interior.points)

    def test_int_literal_becomes_scalar_tensor(self, simple_domain):
        """Int literal becomes scalar tensor."""
        result = resolve_spec(42, simple_domain)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0
        assert result.dtype == torch.float32
        assert float(result) == 42.0

    def test_float_literal_becomes_scalar_tensor(self, simple_domain):
        """Float literal becomes scalar tensor."""
        result = resolve_spec(0.015, simple_domain)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0
        assert result.dtype == torch.float32
        assert float(result) == pytest.approx(0.015)

    def test_bool_literal_raises_typeerror(self, simple_domain):
        ### bool is a subclass of int, so a missing guard would silently
        ### coerce True / False to tensor(1.0) / tensor(0.0). resolve_spec
        ### raises explicitly so the YAML author sees an actionable error
        ### (almost always a "did you mean a numeric flag?" config bug).
        """Bool literal raises TypeError."""
        with pytest.raises(TypeError, match="Boolean spec values are not supported"):
            resolve_spec(True, simple_domain)

    def test_list_concatenates_on_last_dim(self, simple_domain):
        ### Both interior.points (10, 3) and a fake (10, 3) tensor.
        ### Use boundaries.vehicle.cell_data.normals broadcast manually.
        ### Build a domain where two paths have matching shapes.
        ### Here, just use interior.points twice.
        """List concatenates on last dim."""
        result = resolve_spec(["interior.points", "interior.points"], simple_domain)
        assert tuple(result.shape) == (10, 6)

    def test_list_with_ndim_alignment(self, simple_domain):
        ### A scalar field (10,) and a vector field (10, 3) -- the scalar
        ### should be unsqueezed on dim -1 so they cat on the last dim.
        """List with ndim alignment."""
        result = resolve_spec(
            ["interior.point_data.pressure", "interior.point_data.wss"],
            simple_domain,
        )
        assert tuple(result.shape) == (10, 4)

    def test_list_of_non_tensors_raises(self, simple_domain):
        ### A path that resolves to a Mesh (not a tensor) must trigger
        ### the type check. Numeric literals would auto-resolve to 0-d
        ### tensors and instead trip torch.cat shape errors -- those are
        ### a separate (but valid) failure mode.
        """List of non tensors raises."""
        with pytest.raises(TypeError, match="must resolve to tensors"):
            resolve_spec(["interior.points", "boundaries.vehicle"], simple_domain)

    def test_nested_dict_recurses(self, simple_domain):
        ### Models with dict-valued kwargs (GLOBE's reference_lengths,
        ### boundary_meshes) need this.
        """Nested dict recurses."""
        result = resolve_spec(
            {
                "L_ref": 1.0,
                "delta_turb": 0.015,
                "vehicle": "boundaries.vehicle",
            },
            simple_domain,
        )
        assert isinstance(result, dict)
        assert set(result) == {"L_ref", "delta_turb", "vehicle"}
        assert isinstance(result["L_ref"], torch.Tensor) and result["L_ref"].ndim == 0
        assert float(result["L_ref"]) == 1.0
        assert isinstance(result["vehicle"], Mesh)

    def test_modifier_dict_raises_on_direct_call(self, simple_domain):
        ### Modifier specs must be resolved by `resolve_forward_kwargs`
        ### (which has access to other already-resolved kwargs); they
        ### cannot be resolved by `resolve_spec` directly.
        """Modifier dict raises on direct call."""
        with pytest.raises(ValueError, match="must be resolved via"):
            resolve_spec(
                {"source": "global_data.U_inf", "expand_like": "embedding"},
                simple_domain,
            )

    def test_none_passes_through(self, simple_domain):
        """None passes through."""
        assert resolve_spec(None, simple_domain) is None

    def test_already_tensor_passes_through(self, simple_domain):
        """Already tensor passes through."""
        t = torch.randn(3, 4)
        assert resolve_spec(t, simple_domain) is t


### ---------------------------------------------------------------------------
### resolve_forward_kwargs (including expand_like)
### ---------------------------------------------------------------------------


class TestResolveForwardKwargs:
    """Tests for resolve forward kwargs."""

    def test_minimal_spec(self, simple_domain):
        """Minimal spec."""
        spec = {
            "geometry": "interior.points",
            "global_embedding": "global_data.U_inf",
        }
        result = resolve_forward_kwargs(spec, simple_domain)
        assert set(result) == {"geometry", "global_embedding"}
        assert tuple(result["geometry"].shape) == (10, 3)
        assert tuple(result["global_embedding"].shape) == (3,)

    def test_globe_style_dict_kwargs(self, simple_domain):
        """Globe style dict kwargs."""
        spec = {
            "prediction_points": "interior.points",
            "boundary_meshes": {"vehicle": "boundaries.vehicle"},
            "reference_lengths": {"L_ref": 1.0, "delta_turb": 0.015},
        }
        result = resolve_forward_kwargs(spec, simple_domain)
        assert tuple(result["prediction_points"].shape) == (10, 3)
        assert isinstance(result["boundary_meshes"], dict)
        assert isinstance(result["boundary_meshes"]["vehicle"], Mesh)
        assert isinstance(result["reference_lengths"], dict)
        assert float(result["reference_lengths"]["L_ref"]) == 1.0

    def test_expand_like_two_pass(self, simple_domain):
        ### `embedding` is (10, 3); `fx` source is (3,) which gets padded
        ### to (1, 3) then expanded along axis -2 to match embedding's
        ### second-to-last dim (= 10).
        """Expand like two pass."""
        spec = {
            "embedding": "interior.points",
            "fx": {
                "source": "global_data.U_inf",
                "expand_like": "embedding",
            },
        }
        result = resolve_forward_kwargs(spec, simple_domain)
        assert tuple(result["embedding"].shape) == (10, 3)
        assert tuple(result["fx"].shape) == (10, 3)
        ### Each row of fx should equal U_inf.
        u_inf = simple_domain.global_data["U_inf"]
        for i in range(10):
            assert torch.equal(result["fx"][i], u_inf)

    def test_expand_like_with_concat_embedding(self, simple_domain):
        ### Embedding here is the cat of points and pressure; fx still
        ### expands to match its axis -2 (= 10).
        """Expand like with concat embedding."""
        spec = {
            "embedding": [
                "interior.points",
                "interior.point_data.pressure",
            ],
            "fx": {"source": "global_data.U_inf", "expand_like": "embedding"},
        }
        result = resolve_forward_kwargs(spec, simple_domain)
        assert tuple(result["embedding"].shape) == (10, 4)
        assert tuple(result["fx"].shape) == (10, 3)

    def test_expand_like_missing_reference_raises(self, simple_domain):
        """Expand like missing reference raises."""
        spec = {
            "fx": {
                "source": "global_data.U_inf",
                "expand_like": "no_such_kwarg",
            },
        }
        with pytest.raises(KeyError, match="expand_like references"):
            resolve_forward_kwargs(spec, simple_domain)

    def test_expand_like_non_tensor_reference_raises(self, simple_domain):
        ### `boundary_meshes` resolves to a dict, not a tensor; expand_like
        ### should reject that.
        """Expand like non tensor reference raises."""
        spec = {
            "boundary_meshes": {"vehicle": "boundaries.vehicle"},
            "fx": {
                "source": "global_data.U_inf",
                "expand_like": "boundary_meshes",
            },
        }
        with pytest.raises(TypeError, match="must resolve to a tensor"):
            resolve_forward_kwargs(spec, simple_domain)

    def test_expand_like_non_tensor_source_raises(self, simple_domain):
        """Expand like non tensor source raises."""
        spec = {
            "ref": "interior.points",
            "broken": {"source": "boundaries.vehicle", "expand_like": "ref"},
        }
        with pytest.raises(TypeError, match="source must resolve to a tensor"):
            resolve_forward_kwargs(spec, simple_domain)

    def test_expand_like_one_d_reference_raises_clearly(self, simple_domain):
        ### A 1-D reference (e.g. a per-element scalar field) has no axis -2;
        ### the resolver must catch that explicitly instead of leaking the
        ### bare `IndexError: tuple index out of range` from `ref.shape[-2]`.
        """Expand like one d reference raises clearly."""
        spec = {
            "embedding": "interior.point_data.pressure",  # 1-D, shape (N,)
            "fx": {"source": "global_data.U_inf", "expand_like": "embedding"},
        }
        with pytest.raises(ValueError, match="must be at least 2-D"):
            resolve_forward_kwargs(spec, simple_domain)

    def test_expand_like_resolution_order_independent(self, simple_domain):
        ### Whether `fx` is declared before or after `embedding`, both
        ### orderings should resolve correctly because pass 1 handles
        ### non-modifier specs first.
        """Expand like resolution order independent."""
        spec_a = {
            "embedding": "interior.points",
            "fx": {"source": "global_data.U_inf", "expand_like": "embedding"},
        }
        spec_b = {
            "fx": {"source": "global_data.U_inf", "expand_like": "embedding"},
            "embedding": "interior.points",
        }
        result_a = resolve_forward_kwargs(spec_a, simple_domain)
        result_b = resolve_forward_kwargs(spec_b, simple_domain)
        assert torch.equal(result_a["fx"], result_b["fx"])
        assert torch.equal(result_a["embedding"], result_b["embedding"])


### ---------------------------------------------------------------------------
### extract_targets
### ---------------------------------------------------------------------------


class TestExtractTargets:
    """Tests for extract targets."""

    def test_domain_mesh_input(self, simple_domain):
        """Domain mesh input."""
        result = extract_targets(simple_domain, {"pressure": "scalar", "wss": "vector"})
        ### Result is a TensorDict whose batch_size matches the source
        ### point_data ([N] for the interior point cloud).
        assert isinstance(result, TensorDict)
        assert result.batch_size == torch.Size([10])
        assert set(result.keys()) == {"pressure", "wss"}
        assert torch.equal(
            result["pressure"], simple_domain.interior.point_data["pressure"]
        )
        assert torch.equal(result["wss"], simple_domain.interior.point_data["wss"])

    def test_bare_mesh_input(self):
        ### Backward-compatible: extract_targets accepts a bare Mesh too,
        ### using its `point_data` directly. Useful for ad-hoc calls outside
        ### the DomainMesh-native pipeline.
        """Bare mesh input."""
        mesh = Mesh(
            points=torch.randn(5, 3),
            point_data={"foo": torch.randn(5)},
        )
        result = extract_targets(mesh, {"foo": "scalar"})
        assert isinstance(result, TensorDict)
        assert result.batch_size == torch.Size([5])
        assert torch.equal(result["foo"], mesh.point_data["foo"])

    def test_missing_target_raises_keyerror(self, simple_domain):
        """Missing target raises keyerror."""
        with pytest.raises(KeyError, match="not found in interior.point_data"):
            extract_targets(simple_domain, {"missing_field": "scalar"})

    def test_unsupported_input_type_raises(self):
        """Unsupported input type raises."""
        with pytest.raises(TypeError, match="Expected DomainMesh or Mesh"):
            extract_targets(torch.randn(5, 3), {"pressure": "scalar"})  # type: ignore[arg-type]

    def test_target_config_subset_of_available(self, simple_domain):
        ### Only requested fields are returned, even if the mesh has more.
        """Target config subset of available."""
        result = extract_targets(simple_domain, {"pressure": "scalar"})
        assert isinstance(result, TensorDict)
        assert set(result.keys()) == {"pressure"}
