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

"""Unit tests for `src/forces.py` (integrated force / moment coefficients).

The physics is pinned on a closed tetrahedron, where the answer is
analytic:

- A uniform pressure coefficient over a *closed* surface yields zero net
  pressure force (the closed-surface identity :math:`\\oint \\mathbf{n}\\,dA = 0`),
  so CD/CL/CS are all ~0 -- this also confirms the normals are
  consistently outward and the pressure-force sign is right.
- A uniform skin-friction coefficient :math:`C_f = [c, 0, 0]` integrates
  to a force :math:`[c \\cdot A_\\text{total}, 0, 0]`, so with the flow
  along x the drag is :math:`c \\cdot A_\\text{total}` and lift/side are 0.

The rest pins the config/aggregation glue (`build_axis_frame`,
`surface_force_fields`, `ForceContext`, `ForceAccumulator`).
"""

from __future__ import annotations

import forces
import pytest
import torch
from conftest import make_surface_domain_mesh, make_volume_domain_mesh
from omegaconf import OmegaConf

from physicsnemo.mesh import Mesh


def _closed_tetrahedron() -> Mesh:
    """A regular tetrahedron centered at the origin, all faces wound outward."""
    verts = torch.tensor(
        [[1.0, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=torch.float32
    )
    faces = torch.tensor([[1, 3, 2], [0, 2, 3], [0, 3, 1], [0, 1, 2]])
    mesh = Mesh(points=verts, cells=faces)
    ### Guard the winding convention: every normal must point away from the
    ### origin-centered body, or the pressure-force sign tests are invalid.
    assert ((mesh.cell_normals * mesh.cell_centroids).sum(-1) > 0).all()
    return mesh


_COMMON = dict(
    flow_direction=torch.tensor([1.0, 0.0, 0.0]),
    up_direction=torch.tensor([0.0, 0.0, 1.0]),
    moment_center=torch.zeros(3),
    reference_area=1.0,
    reference_length=1.0,
    length_scale=1.0,
)


### ---------------------------------------------------------------------------
### Physics
### ---------------------------------------------------------------------------


def test_uniform_pressure_on_closed_surface_is_zero_force():
    """Uniform Cp over a closed surface -> ~zero net pressure force and moment."""
    vehicle = _closed_tetrahedron()
    n = vehicle.n_cells
    res = forces.force_moment_coefficients(
        vehicle, torch.ones(n), torch.zeros(n, 3), **_COMMON
    )
    ### The closed-surface identity (integral of n dA = 0) covers the
    ### moments too, so all six coefficients vanish.
    for key in forces.COEFFICIENT_NAMES:
        assert abs(res[key]) < 1e-4, f"{key}={res[key]} should be ~0 for a closed body"


def test_uniform_shear_gives_drag_equal_to_c_times_area():
    """Uniform Cf=[c,0,0] -> CD = c * total_area, zero lift/side."""
    vehicle = _closed_tetrahedron()
    n = vehicle.n_cells
    area_total = float(vehicle.cell_areas.sum())
    c = 2.0
    cf = torch.tensor([[c, 0.0, 0.0]]).repeat(n, 1)
    res = forces.force_moment_coefficients(vehicle, torch.zeros(n), cf, **_COMMON)
    assert res["CD"] == pytest.approx(c * area_total, abs=1e-3)
    assert abs(res["CL"]) < 1e-4 and abs(res["CS"]) < 1e-4


def test_uniform_shear_moment_about_offset_center():
    """Uniform traction + offset moment center -> analytic yaw moment.

    For uniform traction ``t`` over a closed surface whose area-weighted
    centroid is the origin, ``M = -x_ref x (t * A_total)``. With
    ``t = [c, 0, 0]`` and ``x_ref = [0, 1, 0]`` that is ``[0, 0, c * A]``:
    the entire moment lands on the lift axis (yaw, CMY), pinning the
    ``cross(arm, traction)`` order, the moment-center subtraction, and the
    CMR/CMP/CMY projections at once.
    """
    vehicle = _closed_tetrahedron()
    n = vehicle.n_cells
    area_total = float(vehicle.cell_areas.sum())
    c = 2.0
    cf = torch.tensor([[c, 0.0, 0.0]]).repeat(n, 1)
    res = forces.force_moment_coefficients(
        vehicle,
        torch.zeros(n),
        cf,
        **{**_COMMON, "moment_center": torch.tensor([0.0, 1.0, 0.0])},
    )
    assert res["CMY"] == pytest.approx(c * area_total, abs=1e-3)
    assert abs(res["CMR"]) < 1e-4
    assert abs(res["CMP"]) < 1e-4


def test_reference_area_scales_coefficients():
    """Coefficients scale as 1 / reference_area."""
    vehicle = _closed_tetrahedron()
    n = vehicle.n_cells
    cf = torch.tensor([[1.0, 0.0, 0.0]]).repeat(n, 1)
    base = forces.force_moment_coefficients(vehicle, torch.zeros(n), cf, **_COMMON)
    scaled = forces.force_moment_coefficients(
        vehicle, torch.zeros(n), cf, **{**_COMMON, "reference_area": 4.0}
    )
    assert scaled["CD"] == pytest.approx(base["CD"] / 4.0, rel=1e-5)


### ---------------------------------------------------------------------------
### Axis frame
### ---------------------------------------------------------------------------


def test_build_axis_frame_orthonormal():
    """drag/lift/side are unit-length, mutually orthogonal, with drag along flow."""
    drag, lift, side = forces.build_axis_frame(
        torch.tensor([2.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 3.0])
    )
    for v in (drag, lift, side):
        assert v.norm().item() == pytest.approx(1.0, abs=1e-5)
    assert (drag @ lift).item() == pytest.approx(0.0, abs=1e-5)
    assert (drag @ side).item() == pytest.approx(0.0, abs=1e-5)
    assert (lift @ side).item() == pytest.approx(0.0, abs=1e-5)
    ### drag is along the (normalized) flow direction.
    assert torch.allclose(drag, torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)


def test_build_axis_frame_degenerate_up_parallel_to_flow():
    """up parallel to flow still yields a valid orthonormal frame."""
    drag, lift, side = forces.build_axis_frame(
        torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0, 0.0, 1.0])
    )
    for v in (drag, lift, side):
        assert v.norm().item() == pytest.approx(1.0, abs=1e-5)
    assert (drag @ lift).item() == pytest.approx(0.0, abs=1e-5)
    assert (lift @ side).item() == pytest.approx(0.0, abs=1e-5)


### ---------------------------------------------------------------------------
### Field identification / config
### ---------------------------------------------------------------------------


def test_surface_force_fields_identifies_cp_cf():
    """A surface field map (pressure + stress) resolves to its (pressure, shear) keys."""
    assert forces.surface_force_fields({"pressure": "pressure", "wss": "stress"}) == (
        "pressure",
        "wss",
    )


def test_surface_force_fields_missing_shear_returns_none():
    """No stress field (the volume contract) returns None: forces need a shear field."""
    # volume contract: pressure present, no stress field
    assert (
        forces.surface_force_fields(
            {"velocity": "velocity", "pressure": "pressure", "nut": "identity"}
        )
        is None
    )


def _force_cfg(**overrides):
    base = {
        "enabled": True,
        "reference_area": 1.0,
        "reference_length": None,
        "moment_center": [0.0, 0.0, 0.0],
        "up_direction": [0.0, 0.0, 1.0],
    }
    base.update(overrides)
    return OmegaConf.create(base)


def test_force_context_from_config_variants():
    """from_config builds for surface fields; disabled/None/non-surface all give None."""
    ft = {"pressure": "pressure", "wss": "stress"}
    ctx = forces.ForceContext.from_config(_force_cfg(), ft, "cpu")
    assert ctx is not None
    assert ctx.pressure_field == "pressure" and ctx.shear_field == "wss"
    # disabled / absent / no-surface-fields all yield None
    assert forces.ForceContext.from_config(_force_cfg(enabled=False), ft, "cpu") is None
    assert forces.ForceContext.from_config(None, ft, "cpu") is None
    assert (
        forces.ForceContext.from_config(_force_cfg(), {"velocity": "velocity"}, "cpu")
        is None
    )


def test_force_context_coefficients_surface_and_volume():
    """coefficients() integrates on a surface mesh and skips (None) a volume mesh."""
    ft = {"pressure": "pressure", "wss": "stress"}
    ctx = forces.ForceContext.from_config(_force_cfg(), ft, "cpu")

    surf = make_surface_domain_mesh({"pressure": "scalar", "wss": "vector"}, n_cells=32)
    coeff = surf.interior.point_data.select("pressure", "wss")
    out = ctx.coefficients(surf, coeff, coeff, normalizer=None)
    assert out is not None
    pred, true = out
    assert set(pred) == set(forces.COEFFICIENT_NAMES)
    # pred == true here (same input), so MAE-relevant diffs are zero
    for name in forces.COEFFICIENT_NAMES:
        assert pred[name] == true[name]

    # A volume DomainMesh (interior points != vehicle cells) is skipped.
    vol = make_volume_domain_mesh(n_pts=120)
    vcoeff = vol.interior.point_data.select("velocity", "pressure", "nut")
    assert ctx.coefficients(vol, vcoeff, vcoeff, normalizer=None) is None


### ---------------------------------------------------------------------------
### Accumulator
### ---------------------------------------------------------------------------


def test_force_accumulator_means_and_mae():
    """Accumulator reports per-coefficient pred/true means and MAE over samples."""
    acc = forces.ForceAccumulator()
    pred_a = dict.fromkeys(forces.COEFFICIENT_NAMES, 1.0)
    true_a = dict.fromkeys(forces.COEFFICIENT_NAMES, 0.0)
    pred_b = dict.fromkeys(forces.COEFFICIENT_NAMES, 3.0)
    true_b = dict.fromkeys(forces.COEFFICIENT_NAMES, 4.0)
    acc.update(pred_a, true_a)
    acc.update(pred_b, true_b)

    assert acc.count == 2
    rows, summary = acc.summary()
    assert len(rows) == len(forces.COEFFICIENT_NAMES)
    cd = summary["CD"]
    assert cd["pred_mean"] == pytest.approx(2.0)  # (1 + 3) / 2
    assert cd["true_mean"] == pytest.approx(2.0)  # (0 + 4) / 2
    assert cd["mae"] == pytest.approx(1.0)  # (|1-0| + |3-4|) / 2


def test_force_accumulator_keys_are_rank_invariant():
    """A never-updated accumulator carries the same `totals` keys as an updated one.

    ``infer._allreduce_sums`` folds ``totals`` into one fixed-length tensor
    for the cross-rank all-reduce, so the key set must not depend on whether
    a given rank's shard happened to contain a surface sample. A rank that
    never calls ``update`` (e.g. an empty or all-volume shard) must still
    expose every coefficient key, else the collective would mismatch and
    hang/abort.
    """
    fresh = forces.ForceAccumulator()
    updated = forces.ForceAccumulator()
    updated.update(
        dict.fromkeys(forces.COEFFICIENT_NAMES, 1.0),
        dict.fromkeys(forces.COEFFICIENT_NAMES, 0.5),
    )
    assert fresh.totals.keys() == updated.totals.keys()
    assert len(fresh.totals) == 3 * len(forces.COEFFICIENT_NAMES)
    ### A fresh accumulator contributes only zeros (and count 0), so it is a
    ### no-op in the all-reduced sums.
    assert set(fresh.totals.values()) == {0.0}
    assert fresh.count == 0
