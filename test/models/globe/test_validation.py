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

"""Subset-validation tests for :class:`GLOBE.forward`.

These tests pin the contract that a model instantiated with a given
``boundary_source_data_ranks`` / ``global_data_ranks`` accepts any
``boundary_meshes`` / ``global_data`` that *contains* every declared
leaf with matching rank.  Extra leaves are silently dropped by the
``select`` filter inside :class:`GLOBE`; missing leaves and rank
mismatches raise ``ValueError`` at forward entry.
"""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.experimental.models.globe.model import GLOBE
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.procedural import lumpy_sphere

### Test fixtures and helpers
# `lumpy_sphere(subdivisions=1)` -> 80 triangle faces. Validation runs in
# eager mode at forward entry, and the silently-accepts cases run a real
# (small) forward to confirm the select-based filter actually drops the
# extras instead of just hiding them.
N_PREDICTION_POINTS = 5


def _make_model(
    *,
    boundary_source_data_ranks: dict | None = None,
    global_data_ranks: dict | None = None,
) -> GLOBE:
    """Build a minimal GLOBE that only varies the bits the test cares about."""
    return GLOBE(
        n_spatial_dims=3,
        output_field_ranks={"pressure": 0},
        boundary_source_data_ranks=boundary_source_data_ranks
        if boundary_source_data_ranks is not None
        else {"no_slip": {}},
        global_data_ranks=global_data_ranks,
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
    """Extra ``cell_data`` leaves not in the rank spec are dropped by ``select``.

    Forward must succeed and produce finite outputs that match the
    no-extras baseline (extras genuinely don't influence the kernel).
    """
    model = _make_model(boundary_source_data_ranks={"no_slip": {}})
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


def test_globe_rejects_missing_cell_data_keys(
    prediction_points: torch.Tensor,
    reference_lengths: dict[str, torch.Tensor],
) -> None:
    """A declared cell_data key absent from the input mesh must raise."""
    model = _make_model(boundary_source_data_ranks={"no_slip": {"alpha": 0}})
    empty_mesh = _mesh_with_cell_data(None)

    with pytest.raises(ValueError, match=r"missing leaf 'alpha'"):
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
    model = _make_model(boundary_source_data_ranks={"no_slip": {"alpha": 0}})
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
    """Extra ``global_data`` leaves not in ``global_data_ranks`` are dropped.

    Forward must succeed and match the no-extras baseline.
    """
    model = _make_model(boundary_source_data_ranks={"no_slip": {}})
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
