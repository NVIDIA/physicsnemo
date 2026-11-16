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

"""Validation tests for :class:`GLOBE`.

The forward-time tests pin the contract that a model instantiated with a
given ``boundary_source_schemas`` / ``global_schema`` accepts any
``boundary_meshes`` / ``global_data`` that *contains* every declared
leaf with matching rank.  Extra leaves are silently dropped by the
``select`` filter inside :class:`GLOBE`; missing leaves and rank
mismatches raise ``ValueError`` at forward entry.  The construction-time
tests pin the attributes GLOBE stores, which arguments it refuses, and that
its arguments survive a checkpoint.
"""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.core import Module
from physicsnemo.mesh import FieldSchema, Mesh, RankSpec
from physicsnemo.mesh.primitives.procedural import lumpy_sphere
from physicsnemo.models.globe import GLOBE
from test.models.globe._helpers import assert_globe_attributes

### Test fixtures and helpers
# `lumpy_sphere(subdivisions=1)` -> 80 triangle faces. Validation runs in
# eager mode at forward entry, and the silently-accepts cases run a real
# (small) forward to confirm the select-based filter actually drops the
# extras instead of just hiding them.
N_PREDICTION_POINTS = 5


def _make_model(
    *,
    boundary_source_schemas: dict | None = None,
    global_schema: dict | None = None,
) -> GLOBE:
    """Build a minimal GLOBE that only varies the bits the test cares about."""
    return GLOBE(
        n_spatial_dims=3,
        output_schema={"pressure": {"rank": 0}},
        boundary_source_schemas=boundary_source_schemas
        if boundary_source_schemas is not None
        else {"no_slip": {}},
        global_schema=global_schema,
        reference_length_names=["test_length"],
        reference_area=1.0,
        hidden_layer_sizes=[8],
    ).eval()


def _mesh_with_cell_data(cell_data: dict[str, torch.Tensor] | None) -> Mesh:
    """Reconstruct a lumpy_sphere mesh with a custom ``cell_data`` dict."""
    base = lumpy_sphere.load(subdivisions=1)
    return Mesh(points=base.points, cells=base.cells, cell_data=cell_data)


@pytest.fixture
def prediction_points() -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(0)
    return torch.randn(N_PREDICTION_POINTS, 3, generator=g)


@pytest.fixture
def reference_lengths() -> dict[str, torch.Tensor]:
    return {"test_length": torch.tensor(1.0, dtype=torch.float32)}


### Test cases
def test_globe_silently_accepts_extra_cell_data_keys(
    prediction_points: torch.Tensor,
    reference_lengths: dict[str, torch.Tensor],
) -> None:
    """Extra ``cell_data`` leaves not in the schema are dropped by ``select``.

    Forward must succeed and produce finite outputs that match the
    no-extras baseline (extras genuinely don't influence the kernel).
    """
    model = _make_model(boundary_source_schemas={"no_slip": {}})
    n_cells = lumpy_sphere.load(subdivisions=1).n_cells
    mesh_with_extra = _mesh_with_cell_data({"unexpected_field": torch.zeros(n_cells)})
    mesh_clean = _mesh_with_cell_data(None)

    with torch.no_grad():
        out_extra = model(
            prediction_points=prediction_points,
            boundary_meshes={"no_slip": mesh_with_extra},
            reference_lengths=reference_lengths,
        )
        out_clean = model(
            prediction_points=prediction_points,
            boundary_meshes={"no_slip": mesh_clean},
            reference_lengths=reference_lengths,
        )

    pressure_extra = out_extra.point_data["pressure"]
    pressure_clean = out_clean.point_data["pressure"]
    assert torch.all(torch.isfinite(pressure_extra))
    torch.testing.assert_close(pressure_extra, pressure_clean)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"output_schema": {"stress": {"rank": 2, "symmetric": True}}},
            r"output_schema\['stress'\]: rank-2 fields are not supported",
        ),
        (
            {"output_schema": {"vorticity": {"rank": 1, "parity": "odd"}}},
            r"output_schema\['vorticity'\]: pseudotensors",
        ),
        (
            {"boundary_source_schemas": {"no_slip": {"stress": {"rank": 2}}}},
            r"boundary_source_schemas\['no_slip'\]\['stress'\]: rank-2",
        ),
        (
            {
                "boundary_source_schemas": {
                    "no_slip": {"omega": RankSpec(1, parity="odd")}
                }
            },
            r"boundary_source_schemas\['no_slip'\]\['omega'\]: pseudotensors",
        ),
        (
            {"global_schema": {"spin": {"rank": 1, "parity": "odd"}}},
            r"global_schema\['spin'\]: pseudotensors",
        ),
    ],
)
def test_globe_refuses_declarations_it_does_not_implement(kwargs, match) -> None:
    """GLOBE's kernels handle true scalars and vectors only; anything else
    would be silently dropped (rank >= 2) or transform with the wrong sign
    under reflections (pseudotensors), so construction refuses it."""
    base = {
        "output_schema": {"pressure": {"rank": 0}},
        "boundary_source_schemas": {"no_slip": {}},
    }
    with pytest.raises(NotImplementedError, match=match):
        GLOBE(
            n_spatial_dims=3,
            reference_length_names=["test_length"],
            reference_area=1.0,
            hidden_layer_sizes=[8],
            **(base | kwargs),
        )


def test_globe_checkpoints_schema_objects(tmp_path) -> None:
    """FieldSchema / RankSpec arguments serialize into the ``.mdlus`` metadata
    as they stand and load back to the same model."""
    torch.manual_seed(0)
    model = GLOBE(
        n_spatial_dims=3,
        output_schema=FieldSchema.parse(
            {"surface": {"C_p": {"rank": 0}}, "C_f": RankSpec(1)}
        ),
        boundary_source_schemas={"no_slip": {"alpha": RankSpec(0)}},
        global_schema={"U_inf": RankSpec(1)},
        reference_length_names=["test_length"],
        reference_area=1.0,
        n_communication_hyperlayers=1,
        hidden_layer_sizes=[8],
    )
    path = tmp_path / "globe.mdlus"
    model.save(path)
    loaded = GLOBE.from_checkpoint(path)
    assert loaded.output_schema == model.output_schema
    assert loaded.boundary_source_schemas == model.boundary_source_schemas
    assert loaded.global_schema == model.global_schema
    for (name, a), b in zip(model.state_dict().items(), loaded.state_dict().values()):
        assert torch.equal(a, b), name


def test_globe_rejects_missing_cell_data_keys(
    prediction_points: torch.Tensor,
    reference_lengths: dict[str, torch.Tensor],
) -> None:
    """A declared cell_data key absent from the input mesh must raise."""
    model = _make_model(boundary_source_schemas={"no_slip": {"alpha": {"rank": 0}}})
    empty_mesh = _mesh_with_cell_data(None)

    with pytest.raises(ValueError, match=r"missing field 'alpha'"):
        with torch.no_grad():
            model(
                prediction_points=prediction_points,
                boundary_meshes={"no_slip": empty_mesh},
                reference_lengths=reference_lengths,
            )


def test_globe_rejects_cell_data_rank_mismatch(
    prediction_points: torch.Tensor,
    reference_lengths: dict[str, torch.Tensor],
) -> None:
    """Declaring a scalar but passing a vector (or vice versa) must raise."""
    model = _make_model(boundary_source_schemas={"no_slip": {"alpha": {"rank": 0}}})
    n_cells = lumpy_sphere.load(subdivisions=1).n_cells
    bad_mesh = _mesh_with_cell_data({"alpha": torch.zeros(n_cells, 3)})

    with pytest.raises(
        ValueError, match=r"rank mismatch for 'alpha': declared 0, got 1"
    ):
        with torch.no_grad():
            model(
                prediction_points=prediction_points,
                boundary_meshes={"no_slip": bad_mesh},
                reference_lengths=reference_lengths,
            )


def test_globe_silently_accepts_extra_global_data_keys(
    prediction_points: torch.Tensor,
    reference_lengths: dict[str, torch.Tensor],
) -> None:
    """Extra ``global_data`` leaves not in ``global_schema`` are dropped.

    Forward must succeed and match the no-extras baseline.
    """
    model = _make_model(boundary_source_schemas={"no_slip": {}})
    empty_mesh = _mesh_with_cell_data(None)
    extra_global = TensorDict({"unexpected_global": torch.tensor(0.0)}, batch_size=[])

    with torch.no_grad():
        out_extra = model(
            prediction_points=prediction_points,
            boundary_meshes={"no_slip": empty_mesh},
            reference_lengths=reference_lengths,
            global_data=extra_global,
        )
        out_clean = model(
            prediction_points=prediction_points,
            boundary_meshes={"no_slip": empty_mesh},
            reference_lengths=reference_lengths,
        )

    pressure_extra = out_extra.point_data["pressure"]
    pressure_clean = out_clean.point_data["pressure"]
    assert torch.all(torch.isfinite(pressure_extra))
    torch.testing.assert_close(pressure_extra, pressure_clean)


### Constructor arguments: only the required ones, then every optional one
### set to a value other than its default.
REQUIRED_KWARGS = {
    "n_spatial_dims": 3,
    "output_schema": {"pressure": {"rank": 0}, "velocity": {"rank": 1}},
    "boundary_source_schemas": {"no_slip": {"alpha": {"rank": 0}}, "freestream": {}},
    "reference_length_names": ["delta_FS", "chord"],
    "reference_area": 2.0,
}
CUSTOM_KWARGS = {
    "global_schema": {"U_inf": {"rank": 1}},
    "n_communication_hyperlayers": 1,
    "n_latent_scalars": 3,
    "n_latent_vectors": 2,
    "smoothing_radius": 1e-6,
    "hidden_layer_sizes": [8, 4],
    "n_spherical_harmonics": 2,
    "theta": 0.5,
    "leaf_size": 4,
    "network_type": "mlp",
    "self_regularization_beta": 0.5,
    "latent_compression_scale": 10.0,
    "expand_far_targets": False,
    "use_gradient_checkpointing": False,
    "tree_build_device": "cpu",
}


@pytest.mark.parametrize(
    "optional_kwargs", [{}, CUSTOM_KWARGS], ids=["defaults", "custom"]
)
def test_globe_constructor_attributes(optional_kwargs: dict) -> None:
    """GLOBE stores its constructor arguments as public attributes (see
    :func:`assert_globe_attributes`)."""
    model = GLOBE(**REQUIRED_KWARGS, **optional_kwargs)
    assert_globe_attributes(model, REQUIRED_KWARGS | optional_kwargs)


@pytest.mark.parametrize("n_spatial_dims", [1, 4])
def test_globe_rejects_unsupported_n_spatial_dims(n_spatial_dims: int) -> None:
    """GLOBE is defined in 2 and 3 dimensions only."""
    with pytest.raises(ValueError, match=r"n_spatial_dims must be 2 or 3"):
        GLOBE(**(REQUIRED_KWARGS | {"n_spatial_dims": n_spatial_dims}))


@pytest.mark.parametrize("prediction_chunk_size", [0, -1, 2.5, "big"])
def test_globe_rejects_invalid_prediction_chunk_size(
    monkeypatch,
    prediction_points: torch.Tensor,
    reference_lengths: dict[str, torch.Tensor],
    prediction_chunk_size,
) -> None:
    """An invalid ``prediction_chunk_size`` raises before any computation."""
    model = _make_model()

    def fail(*args, **kwargs):
        raise AssertionError("trees were built before validation")

    monkeypatch.setattr(model, "_build_trees_and_plans", fail)
    with pytest.raises(ValueError, match=r"prediction_chunk_size"):
        model(
            prediction_points=prediction_points,
            boundary_meshes={"no_slip": _mesh_with_cell_data(None)},
            reference_lengths=reference_lengths,
            prediction_chunk_size=prediction_chunk_size,
        )


def test_globe_checkpoints_tree_build_device(tmp_path) -> None:
    """A ``torch.device`` given as ``tree_build_device`` survives a checkpoint."""
    model = GLOBE(
        n_spatial_dims=3,
        output_schema={"pressure": {"rank": 0}},
        boundary_source_schemas={"no_slip": {}},
        reference_length_names=["test_length"],
        reference_area=1.0,
        n_communication_hyperlayers=1,
        hidden_layer_sizes=[8],
        tree_build_device=torch.device("cpu"),
    )
    path = tmp_path / "globe.mdlus"
    model.save(path)
    loaded = Module.from_checkpoint(path)
    assert isinstance(loaded, GLOBE)
    assert loaded.tree_build_device == torch.device("cpu")
