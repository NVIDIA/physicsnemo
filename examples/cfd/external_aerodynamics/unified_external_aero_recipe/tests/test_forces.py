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
  consistently oriented. It does not pin the sign of the pressure force:
  a non-uniform :math:`C_p = [1, 2, 3, 4]` does, with the force
  :math:`-\\sum C_p\\,\\mathbf{n}\\,A = (-8, -4, 0)`.
- A uniform skin-friction coefficient :math:`C_f = [c, 0, 0]` integrates
  to a force :math:`[c \\cdot A_\\text{total}, 0, 0]`, so with the flow
  along x the drag is :math:`c \\cdot A_\\text{total}` and lift/side are 0.
  That holds when the wall shear stress acts on the body; when it acts on
  the fluid, the force on the body has the opposite sign.
- A surface wound inward, declared as such, gives the same coefficients as
  the same surface wound outward. Undeclared, its pressure force has the
  opposite sign.

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
from physicsnemo.mesh.calculus.measure import scale_measures


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


def test_wall_shear_on_the_fluid_gives_the_opposite_friction_force():
    """A stress on the fluid along -x (OpenFOAM's sign) is a drag along +x."""
    vehicle = _closed_tetrahedron()
    n = vehicle.n_cells
    area_total = float(vehicle.cell_areas.sum())
    c = 2.0
    ### The wall pulls the fluid upstream, so the fluid pulls the body downstream.
    cf_on_fluid = torch.tensor([[-c, 0.0, 0.0]]).repeat(n, 1)
    res = forces.force_moment_coefficients(
        vehicle, torch.zeros(n), cf_on_fluid, **_COMMON, wall_shear_acts_on="fluid"
    )
    assert res["CD"] == pytest.approx(c * area_total, abs=1e-3)
    assert abs(res["CL"]) < 1e-4 and abs(res["CS"]) < 1e-4

    ### The pressure force does not depend on the wall-shear side.
    cp = torch.arange(n, dtype=torch.float32)
    on_body = forces.force_moment_coefficients(
        vehicle, cp, torch.zeros(n, 3), **_COMMON
    )
    on_fluid = forces.force_moment_coefficients(
        vehicle, cp, torch.zeros(n, 3), **_COMMON, wall_shear_acts_on="fluid"
    )
    assert on_body == on_fluid


def test_inward_normals_declared_give_the_outward_result():
    """A surface wound inward integrates like its outward twin once declared."""
    outward = _closed_tetrahedron()
    ### Swapping two vertices of every face reverses the winding.
    inward = Mesh(points=outward.points, cells=outward.cells[:, [0, 2, 1]])
    assert ((inward.cell_normals * inward.cell_centroids).sum(-1) < 0).all()

    n = outward.n_cells
    cp = torch.arange(1, n + 1, dtype=torch.float32)
    cf = torch.tensor([[2.0, -1.0, 0.5]]).repeat(n, 1)
    reference = forces.force_moment_coefficients(outward, cp, cf, **_COMMON)
    declared = forces.force_moment_coefficients(
        inward, cp, cf, **_COMMON, surface_normals_point="inward"
    )
    undeclared = forces.force_moment_coefficients(inward, cp, cf, **_COMMON)
    ### The friction part does not depend on the orientation; without the
    ### declaration only the pressure part changes sign.
    friction = forces.force_moment_coefficients(outward, torch.zeros(n), cf, **_COMMON)
    ### This pins the sign of the pressure force itself: for Cp = [1, 2, 3, 4]
    ### on this tetrahedron, -sum(Cp n A) = (-8, -4, 0).
    assert reference["CD"] - friction["CD"] == pytest.approx(-8.0, abs=1e-4)
    assert reference["CS"] - friction["CS"] == pytest.approx(-4.0, abs=1e-4)
    assert reference["CL"] - friction["CL"] == pytest.approx(0.0, abs=1e-4)
    for name in forces.COEFFICIENT_NAMES:
        assert declared[name] == pytest.approx(reference[name], abs=1e-4)
        pressure = reference[name] - friction[name]
        assert undeclared[name] == pytest.approx(friction[name] - pressure, abs=1e-4)


def test_conventions_must_be_spelled_right():
    """A misspelled value fails loudly, in the function and at context setup."""
    vehicle = _closed_tetrahedron()
    n = vehicle.n_cells
    ft = {"pressure": "pressure", "wss": "stress"}
    for key, bad in (("wall_shear_acts_on", "wall"), ("surface_normals_point", "in")):
        with pytest.raises(ValueError, match=key):
            forces.force_moment_coefficients(
                vehicle, torch.zeros(n), torch.zeros(n, 3), **_COMMON, **{key: bad}
            )
        with pytest.raises(ValueError, match=key):
            forces.ForceContext.from_config(
                _force_cfg(), ft, "cpu", dataset_cfg=OmegaConf.create({key: bad})
            )


@pytest.mark.parametrize("length_scale", [1.0, 3.0])
def test_subsampled_surface_with_effective_measures_recovers_full_surface_drag(
    length_scale,
):
    """Effective measures make the integral over kept cells estimate the full one."""
    full = _closed_tetrahedron()
    kept = torch.tensor([0, 1])
    unweighted = full.slice_cells(kept)
    weighted = full.slice_cells(kept)
    scale_measures(weighted, full.n_cells / weighted.n_cells)

    c = 2.0
    cp = torch.zeros(len(kept))
    cf = torch.tensor([[c, 0.0, 0.0]]).repeat(len(kept), 1)
    common = {**_COMMON, "length_scale": length_scale}
    res = forces.force_moment_coefficients(weighted, cp, cf, **common)
    res_unweighted = forces.force_moment_coefficients(unweighted, cp, cf, **common)

    ### A regular tetrahedron's faces have equal area, so two of four cells
    ### weighted by 4/2 reproduce the full-surface value exactly; without
    ### the weights the integral covers only the kept half.
    expected = c * float(full.cell_areas.sum()) * length_scale**2
    assert res["CD"] == pytest.approx(expected, abs=1e-3)
    assert res_unweighted["CD"] == pytest.approx(expected / 2, abs=1e-3)


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
    # outward normals and wall shear on the body unless the dataset says otherwise
    assert ctx.surface_normals_point == "outward"
    assert ctx.wall_shear_acts_on == "body"
    declared = forces.ForceContext.from_config(
        _force_cfg(),
        ft,
        "cpu",
        dataset_cfg=OmegaConf.create(
            {"surface_normals_point": "inward", "wall_shear_acts_on": "fluid"}
        ),
    )
    assert declared.surface_normals_point == "inward"
    assert declared.wall_shear_acts_on == "fluid"
    # a dataset config without the keys keeps the defaults
    plain = forces.ForceContext.from_config(
        _force_cfg(), ft, "cpu", dataset_cfg=OmegaConf.create({"targets": {}})
    )
    assert (plain.surface_normals_point, plain.wall_shear_acts_on) == (
        "outward",
        "body",
    )
    # disabled / absent / no-surface-fields all yield None
    assert forces.ForceContext.from_config(_force_cfg(enabled=False), ft, "cpu") is None
    assert forces.ForceContext.from_config(None, ft, "cpu") is None
    assert (
        forces.ForceContext.from_config(_force_cfg(), {"velocity": "velocity"}, "cpu")
        is None
    )


def test_force_context_coefficients_surface_and_volume():
    """coefficients() integrates on a surface mesh and skips (None) a volume mesh."""
    torch.manual_seed(0)
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

    # The context passes the dataset's conventions on: wall shear on the fluid
    # changes the sign of the friction part, inward normals that of the
    # pressure part, and both together that of every coefficient. A sign
    # change is exact in floating point, so these compare without a tolerance.
    only_pressure = coeff.clone()
    only_pressure["wss"] = torch.zeros_like(coeff["wss"])
    only_friction = coeff.clone()
    only_friction["pressure"] = torch.zeros_like(coeff["pressure"])

    def integrated(fields, **conventions):
        declared = forces.ForceContext.from_config(
            _force_cfg(), ft, "cpu", dataset_cfg=OmegaConf.create(conventions)
        )
        return declared.coefficients(surf, fields, fields, normalizer=None)[0]

    def negated(coefficients):
        return {name: -value for name, value in coefficients.items()}

    pressure_part = integrated(only_pressure)
    friction_part = integrated(only_friction)
    assert integrated(only_pressure, wall_shear_acts_on="fluid") == pressure_part
    assert integrated(only_friction, wall_shear_acts_on="fluid") == negated(
        friction_part
    )
    assert integrated(only_pressure, surface_normals_point="inward") == negated(
        pressure_part
    )
    assert integrated(only_friction, surface_normals_point="inward") == friction_part
    assert integrated(
        coeff, surface_normals_point="inward", wall_shear_acts_on="fluid"
    ) == negated(pred)

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


@pytest.mark.parametrize(
    ("dataset", "normals", "wall_shear"),
    [
        ("drivaer_ml_surface", "inward", "fluid"),  # OpenFOAM
        ("highlift_surface", "inward", "body"),
        ("shift_suv_estate_surface", "outward", "body"),
        ("shift_suv_fastback_surface", "outward", "body"),
    ],
)
def test_shipped_dataset_yamls_declare_their_conventions(dataset, normals, wall_shear):
    """The surface datasets declare the conventions measured on their data."""
    from pathlib import Path

    datasets_dir = Path(__file__).resolve().parent.parent / "datasets"
    ds_yaml = OmegaConf.load(datasets_dir / f"{dataset}.yaml")
    ### Build the context from the YAML the way infer.py does, so a misspelled
    ### key or value in a shipped YAML fails here.
    ctx = forces.ForceContext.from_config(
        _force_cfg(), {"pressure": "pressure", "wss": "stress"}, "cpu", ds_yaml
    )
    assert ctx.surface_normals_point == normals
    assert ctx.wall_shear_acts_on == wall_shear
