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

"""ISLA global inputs: zero or more rank-1 global inputs (directions) and zero or
more rank-0 global inputs (scalars, one name per scalar), read from ``global_data``
by name. The contracts tested here are exact (float64, tolerance 1e-10): SE(3)
covariance and measure-scale invariance for K = 0 and K = 2 directions, magnitude
invariance of every direction, scalar conditioning that changes the output while
preserving every covariance contract, the K = 1 parameter layout, and the fused
kernel refusing K outside one to four."""

import pytest
import torch

from physicsnemo.experimental.models.isla import ISLA
from test.experimental.models.isla._meshes import (
    cloud,
    live,
    orthogonal,
    outputs,
    rescale_measures,
    same,
    soup,
    transform,
)

D = torch.float64
PV = {"p": {"rank": 0}, "v": {"rank": 1}}


def _boundary(n=200, seed=0):
    pts, nrm, w, _ = cloud(n, seed=seed, scale=(1.0, 1.0, 1.0))
    return soup(pts, nrm, w)


def _model(global_schema=None, **kw):
    torch.manual_seed(0)
    if global_schema is None:
        global_schema = {"g": {"rank": 1}}
    kw.setdefault("boundary_schemas", {"wall": {}})
    return live(
        ISLA(
            output_schema=PV,
            global_schema=global_schema,
            hidden=32,
            n_layers=2,
            n_slices=8,
            **kw,
        )
        .to(D)
        .eval()
    )


def _rotate_globals(m, gd, R):
    """Rotate the rank-1 entries of a global_data dict, leave the rank-0 ones."""
    return {k: (v @ R.T if m.global_schema[k].rank == 1 else v) for k, v in gd.items()}


def _run(m, mesh, gd, q=None):
    with torch.no_grad():
        return m(
            boundary_meshes={"wall": mesh},
            prediction_points=q,
            prediction_boundaries=None if q is not None else ["wall"],
            global_data=gd,
        )


def _assert_se3(m, mesh, gd, q=None, atol=1e-10):
    R = orthogonal(3, 5)
    t = torch.tensor([2.0, -5.0, 1.0], dtype=D)
    base = _run(m, mesh, gd, q)
    moved = _run(
        m,
        transform(mesh, R, t),
        _rotate_globals(m, gd, R),
        None if q is None else q @ R.T + t,
    )
    assert torch.allclose(
        moved.point_data["p"], base.point_data["p"], atol=atol, rtol=0.0
    )
    assert torch.allclose(
        moved.point_data["v"], base.point_data["v"] @ R.T, atol=atol, rtol=0.0
    )
    ### Measure-scale invariance: the RMS-distance unit uses normalized weights and
    ### the routing bias is a log-weight, so a uniform rescale changes nothing.
    rescaled = _run(m, rescale_measures(mesh, 137.0), gd, q)
    assert same(rescaled, base, atol=1e-9)
    return base


def test_zero_global_vectors():
    """K = 0: no direction anywhere in the problem (steady conduction with pinned
    boundary values). The seed is the constant alone and the model is still exactly
    SE(3)-covariant and measure-scale invariant; global_data may be None."""
    m = _model(global_schema={})
    mesh = _boundary()
    assert m.boundary_conditions["wall"].seed.in_features == 1  # the constant alone
    base = _assert_se3(m, mesh, {})
    assert all(torch.isfinite(o).all() for o in outputs(base).values())
    none = _run(m, mesh, None)
    assert same(none, base, atol=0.0, rtol=0.0)
    ### extra leaves in global_data are ignored, and the output carries none of them
    extra = _run(m, mesh, {"g": torch.randn(3, dtype=D)})
    assert (
        same(extra, base, atol=0.0, rtol=0.0) and list(extra.global_data.keys()) == []
    )


def test_two_global_vectors():
    """K = 2 (a freestream and a gravity direction): both directions rotate with the
    geometry, the output rotates with them, and rescaling either's magnitude, by
    any factor, changes nothing."""
    m = _model(global_schema={"g1": {"rank": 1}, "g2": {"rank": 1}})
    mesh = _boundary()
    gv = torch.randn(2, 3, dtype=D, generator=torch.Generator().manual_seed(1))
    gd = {"g1": gv[0], "g2": gv[1]}
    assert (
        m.boundary_conditions["wall"].seed.in_features == 1 + 2
    )  # the constant, n.g_1, n.g_2
    base = _assert_se3(m, mesh, gd)
    scaled = _run(m, mesh, {"g1": 3.0 * gv[0], "g2": 0.2 * gv[1]})
    swapped = _run(m, mesh, {"g1": gv[1], "g2": gv[0]})
    assert same(scaled, base, atol=1e-10)
    tiny = _run(m, mesh, {"g1": 1e-14 * gv[0], "g2": 1e14 * gv[1]})
    assert same(tiny, base, atol=1e-10)
    assert not same(swapped, base, atol=1e-3)  # the two inputs are distinct channels
    with pytest.raises(ValueError, match="3 values per case"):
        _run(m, mesh, {"g1": torch.randn(5, dtype=D), "g2": gv[1]})
    ### the selected global data ride onto the output Mesh
    assert set(base.global_data.keys()) == {"g1", "g2"}
    assert torch.equal(base.global_data["g1"], gv[0])


@pytest.mark.parametrize("declared", [0, 2])
def test_a_derived_direction_counts_among_the_global_vectors(declared):
    """A boundary velocity source derives the direction g, the last of the K global
    directions: K = 1 with none declared, K = 3 after two declared ones; the seed,
    the head basis and every covariance contract count it, and it rotates with the
    source."""
    gs = {f"g{k}": {"rank": 1} for k in range(declared)}
    m = _model(
        gs,
        boundary_schemas={
            "wall": {"u": {"rank": 1, "units": {"U": 1}, "defines": "U"}}
        },
    )
    K = declared + 1
    assert m._n_global_vectors == K and m._n_true_basis == 3 + 2 * K
    assert (
        m.boundary_conditions["wall"].seed.in_features == 1 + K + (K + 1) + 1
    )  # 1, n.g_k, u.(n, g_k), |u|
    pts, nrm, w, _ = cloud(200, seed=0, scale=(1.0, 1.0, 1.0))
    u = torch.randn(
        200, 3, dtype=D, generator=torch.Generator().manual_seed(3)
    ) + torch.tensor([4.0, 0, 0], dtype=D)
    mesh = soup(pts, nrm, w, cell_data={"u": u})
    gd = {
        f"g{k}": torch.randn(3, dtype=D, generator=torch.Generator().manual_seed(k))
        for k in range(declared)
    }
    R, t = orthogonal(3, 5), torch.tensor([2.0, -5.0, 1.0], dtype=D)
    base = _run(m, mesh, gd)
    moved = _run(
        m,
        transform(mesh, R, t, specs=m.boundary_schemas["wall"]),
        _rotate_globals(m, gd, R),
    )
    assert torch.allclose(
        moved.point_data["p"], base.point_data["p"], atol=1e-10, rtol=0.0
    )
    assert torch.allclose(
        moved.point_data["v"], base.point_data["v"] @ R.T, atol=1e-10, rtol=0.0
    )


def test_two_vectors_interior_path():
    """The interior task carries the K directions to the prediction-point side as well."""
    mesh = _boundary()
    q = torch.randn(40, 3, dtype=D, generator=torch.Generator().manual_seed(2))
    gv = torch.randn(2, 3, dtype=D, generator=torch.Generator().manual_seed(1))
    m = _model({"g1": {"rank": 1}, "g2": {"rank": 1}}, interior=True)
    _assert_se3(m, mesh, {"g1": gv[0], "g2": gv[1]}, q=q)


def test_global_scalars_condition_every_token():
    """S = 2 scalar inputs (a diffusivity and a modulus, say) change the output,
    reach the prediction-point side too, and leave every covariance contract intact."""
    mesh = _boundary()
    m = _model(
        {"g": {"rank": 1}, "cond.kappa": {"rank": 0}, "cond.E": {"rank": 0}},
        interior=True,
    )
    assert (
        m.boundary_conditions["wall"].seed.in_features == 1 + 1 + 2
    )  # the constant, n.g and the two scalars
    assert m._n_global_scalars == 2
    gv = torch.randn(3, dtype=D, generator=torch.Generator().manual_seed(1))
    gd = {
        "g": gv,
        "cond.kappa": torch.tensor(0.3, dtype=D),
        "cond.E": torch.tensor(-1.2, dtype=D),
    }
    q = torch.randn(30, 3, dtype=D, generator=torch.Generator().manual_seed(2))
    base = _assert_se3(m, mesh, gd, q=q)
    other = _run(
        m,
        mesh,
        {**gd, "cond.kappa": gd["cond.kappa"] + 1.0, "cond.E": gd["cond.E"] + 1.0},
        q,
    )
    assert not same(other, base, atol=1e-3)
    with pytest.raises(
        ValueError, match="missing the declared fields \\['cond.E', 'cond.kappa'\\]"
    ):
        _run(m, mesh, {"g": gv}, q)
    ### a rank-0 global input is one scalar; a parameter vector is several names
    with pytest.raises(ValueError, match="unknown keys"):
        _model({"g": {"rank": 1}, "params": {"rank": 0, "channels": 2}})
    with pytest.raises(ValueError, match="gives 2 rows for a batch of 1"):
        _run(m, mesh, {**gd, "cond.kappa": torch.zeros(2, dtype=D)}, q)


def test_no_global_inputs_at_all_with_scalars_only():
    """K = 0 with S = 1: a parameterized PDE family with no preferred direction."""
    mesh = _boundary()
    m = _model({"kappa": {"rank": 0}})
    assert (
        m.boundary_conditions["wall"].seed.in_features == 1 + 1
    )  # the constant and the scalar
    base = _assert_se3(m, mesh, {"kappa": torch.tensor([0.5], dtype=D)})
    other = _run(m, mesh, {"kappa": torch.tensor([1.0], dtype=D)})
    assert not same(other, base, atol=1e-3)


def test_one_vector_layout():
    """K = 1, S = 0: seed width 2 (the constant and n.g), geometry width 6; the head has 5 true basis
    vectors and 3 pseudovectors."""
    m = _model()
    assert (m._n_global_vectors, m._n_global_scalars) == (1, 0)
    assert m.boundary_conditions["wall"].seed.in_features == 1 + 1
    assert m.blocks[0].geo_logit.in_features == 6
    assert m._n_true_basis == 5 and m._n_pseudo_basis == 3


def test_bad_global_specs_rejected():
    """A negative rank and a rank-2 global input are refused at construction."""
    with pytest.raises(ValueError, match="non-negative"):
        ISLA(
            output_schema=PV,
            boundary_schemas={"wall": {}},
            global_schema={"g": {"rank": -1}},
            hidden=32,
            n_layers=1,
            n_slices=8,
        )
    with pytest.raises(NotImplementedError, match="rank 2 is not supported"):
        ISLA(
            output_schema=PV,
            boundary_schemas={"wall": {}},
            global_schema={"T": {"rank": 2}},
            hidden=32,
            n_layers=1,
            n_slices=8,
        )


def test_boundary_data_reaches_the_interior_prediction():
    """A boundary-value problem needs the per-cell boundary data (Dirichlet trace)
    on the boundary tokens. The covariance
    contracts hold, and the interior prediction depends on the boundary data."""
    pts, nrm, w, _ = cloud(200, scale=(1.0, 1.0, 1.0))
    bsc = torch.randn(200, dtype=D, generator=torch.Generator().manual_seed(3))
    q = torch.randn(30, 3, dtype=D, generator=torch.Generator().manual_seed(2)) * 0.5
    gd = {"kappa": torch.tensor(0.7, dtype=D)}
    m = _model(
        global_schema={"kappa": {"rank": 0}},
        boundary_schemas={"wall": {"trace": {"rank": 0}}},
        interior=True,
    )
    assert (
        m.boundary_conditions["wall"].seed.in_features == 1 + 2
    )  # the constant, the boundary scalar and the global scalar (no directions)
    base = _assert_se3(m, soup(pts, nrm, w, cell_data={"trace": bsc}), gd, q=q)
    other = _run(m, soup(pts, nrm, w, cell_data={"trace": bsc + 1.0}), gd, q)
    assert base.point_data["p"].shape == (30,) and not same(other, base, atol=1e-3)
