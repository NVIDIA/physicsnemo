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

"""ISLA in two and four spatial dimensions: the contracts of the three-dimensional
tests on a native two-dimensional problem (a polyline boundary, ``Mesh[1, 2]``) and
on a four-dimensional even-parity smoke test (``Mesh[3, 4]``), plus the invariants
of the dimension handling (Levi-Civita symbol, pseudovector counts, refusals)."""

import pytest
import torch

from physicsnemo.experimental.models.isla import ISLA
from physicsnemo.experimental.models.isla._geometry import exact_boundary_geometry
from physicsnemo.experimental.models.isla._heads import _levi_civita, _pseudovectors
from test.experimental.models.isla._meshes import (
    cloud,
    law,
    live,
    orthogonal,
    outputs,
    soup,
    transform,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _reflection(d: int) -> torch.Tensor:
    r = torch.eye(d)
    r[0, 0] = -1.0
    return r


def _problem(d: int, n: int = 160, seed: int = 0, dtype=torch.float32):
    pts, nrm, w, drv = cloud(n, d, seed=seed, dtype=dtype)
    return soup(pts, nrm, w), drv


def _model(d: int, outputs, **kw):
    torch.manual_seed(1)
    kw.setdefault("boundary_schemas", {"wall": {}})
    return live(
        ISLA(
            output_schema=outputs,
            global_schema={"g": {"rank": 1}},
            hidden=32,
            n_layers=2,
            n_slices=8,
            n_spatial_dims=d,
            **kw,
        )
    )


def _run(m, mesh, drv, q=None):
    with torch.no_grad():
        return outputs(
            m(
                boundary_meshes={"wall": mesh},
                prediction_points=q,
                prediction_boundaries=None if q is not None else ["wall"],
                global_data={"g": drv},
            )
        )


OUT2 = {
    "p": {"rank": 0},
    "v": {"rank": 1},
    "w": {"rank": 1, "parity": "odd"},
    "t": {"rank": 2, "symmetric": True},
    "a": {"rank": 2, "parity": "odd"},
}


def test_levi_civita_symbol_and_pseudovector_counts():
    """The Levi-Civita symbol in two and three dimensions has its signs, the
    pseudovectors of an orthonormal basis are its perpendiculars (two in 2-D, three
    in 3-D, none in 4-D), and a 2-D model's output layouts are two-dimensional."""
    e2 = _levi_civita(2)
    assert torch.equal(e2, torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64))
    e3 = _levi_civita(3)
    assert (
        e3[0, 1, 2] == 1 and e3[0, 2, 1] == -1 and e3[1, 2, 0] == 1 and e3[0, 0, 1] == 0
    )
    v = [torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])]
    ps = _pseudovectors(v, 2)
    assert (
        len(ps) == 2
        and torch.equal(ps[0], torch.tensor([[0.0, 1.0]]))
        and torch.equal(ps[1], torch.tensor([[-1.0, 0.0]]))
    )
    v3 = [
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0]]),
    ]
    assert len(_pseudovectors(v3, 3)) == 3 and _pseudovectors(v3, 4) == []
    assert _model(2, {"p": {"rank": 0}})._output_layouts["p"].dim == 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["fp32", "fp64"])
def test_two_dimensional_contracts_rotation_translation_magnitude_and_reflection(dtype):
    """On a polyline boundary: rotation equivariance of every rank and parity under a
    random SO(2) map, translation invariance, magnitude invariance of the global
    direction, and the reflection behaviour: even fields transform as tensors, odd
    fields pick up the extra sign. Float32 at 1e-4 relative; float64 at 1e-11
    absolute (roundoff of outputs of order one)."""
    d = 2
    m = _model(d, OUT2).to(dtype)
    mesh, drv = _problem(d, dtype=dtype)
    assert mesh.n_manifold_dims == 1
    q = orthogonal(d, 3).to(dtype)
    r = _reflection(d).to(dtype)
    base = _run(m, mesh, drv)
    rot = _run(m, transform(mesh, q), drv @ q.T)
    shifted = _run(m, transform(mesh, shift=torch.full((d,), 7.0, dtype=dtype)), drv)
    mag = _run(m, mesh, 3.5 * drv)
    ref = _run(m, transform(mesh, r), drv @ r.T)
    f64 = dtype == torch.float64
    tol = dict(rtol=0.0, atol=1e-11) if f64 else dict(rtol=1e-4, atol=1e-5)
    n = mesh.n_cells
    assert (
        base["p"].shape == (n,)
        and base["v"].shape == (n, 2)
        and base["t"].shape == (n, 2, 2)
    )
    torch.testing.assert_close(rot["p"], base["p"], **tol)
    torch.testing.assert_close(rot["v"], base["v"] @ q.T, **tol)
    torch.testing.assert_close(rot["w"], base["w"] @ q.T, **tol)
    torch.testing.assert_close(rot["t"], law(base["t"], q, 2), **tol)
    torch.testing.assert_close(rot["a"], law(base["a"], q, 2), **tol)
    for k in base:
        ### a shift of 7 length units moves the float32 positions' roundoff by ~1e-5
        shift_tol = tol if f64 else dict(rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(shifted[k], base[k], **shift_tol)
        torch.testing.assert_close(mag[k], base[k], **tol)
    ### det(r) = -1: even fields transform as tensors, odd fields flip in addition.
    torch.testing.assert_close(ref["p"], base["p"], **tol)
    torch.testing.assert_close(ref["v"], base["v"] @ r.T, **tol)
    torch.testing.assert_close(ref["w"], -(base["w"] @ r.T), **tol)
    torch.testing.assert_close(ref["t"], law(base["t"], r, 2), **tol)
    torch.testing.assert_close(ref["a"], -law(base["a"], r, 2), **tol)
    ### the symmetric spec is symmetric, the odd rank-2 field is nonzero (its basis has
    ### the perpendiculars and the 2-D Levi-Civita symbol as content)
    torch.testing.assert_close(base["t"], base["t"].transpose(-1, -2))
    assert base["a"].abs().max() > 0 and base["w"].abs().max() > 0


def test_two_dimensional_gradients_and_interior_task():
    """In two dimensions the interior task gives every parameter a gradient, and a
    rotation of the boundary, the global direction and the prediction points keeps p
    and rotates v, to the float32 bound of the distance channel."""
    d = 2
    mesh, drv = _problem(d)
    ### prediction-point tokens with the internal nearest-boundary geometry in 2-D
    mq = _model(d, {"p": {"rank": 0}, "v": {"rank": 1}}, interior=True)
    q = torch.randn(50, d) * 0.5
    out = outputs(
        mq(boundary_meshes={"wall": mesh}, prediction_points=q, global_data={"g": drv})
    )
    sum(v.square().mean() for v in out.values()).backward()
    missing = [k for k, p in mq.named_parameters() if p.grad is None]
    assert not missing, missing
    mq.eval()
    rq = orthogonal(d, 5).float()
    o1 = _run(mq, mesh, drv, q)
    o2 = _run(mq, transform(mesh, rq), drv @ rq.T, q @ rq.T)
    assert o1["p"].shape == (50,) and o1["v"].shape == (50, 2)
    ### float32: the rotation's rounding moves a query's distance by at most
    ### E = 2 u (|x| + max |v|) (u = eps / 2), so the soft-log distance channel by at
    ### most E / |d|: up to 3e-3 at a query 3e-4 from the wall
    dist, _ = exact_boundary_geometry(q, mesh.points, mesh.cells)
    bound = torch.finfo(q.dtype).eps * (q.norm(dim=-1) + mesh.points.norm(dim=-1).max())
    rel = bound / dist.abs()
    for got, want in ((o2["p"], o1["p"]), (o2["v"], o1["v"] @ rq.T)):
        r = rel.reshape(-1, *[1] * (want.ndim - 1))
        tol = 1e-5 + 1e-4 * want.abs() + r * want.abs().max()
        assert ((got - want).abs() <= tol).all(), (got - want).abs().max()


def test_two_dimensional_rank_inputs_keep_the_contracts():
    """Rank-1 and rank-2 boundary inputs in two dimensions: the invariant seeds and
    the block contractions are built for d components."""
    d = 2
    pts, nrm, w, drv = cloud(160, d, dtype=torch.float32)
    g = torch.Generator().manual_seed(9)
    trac = torch.randn(160, d, generator=g)
    sym = torch.randn(160, d, d, generator=g)
    sym = sym + sym.transpose(-1, -2)
    m = _model(
        d,
        {"p": {"rank": 0}, "v": {"rank": 1}},
        boundary_schemas={
            "wall": {"trac": {"rank": 1}, "sig": {"rank": 2, "symmetric": True}}
        },
    )
    mesh = soup(pts, nrm, w, cell_data={"trac": trac, "sig": sym})
    q = orthogonal(d, 9).float()
    base = _run(m, mesh, drv)
    rot = _run(m, transform(mesh, q, specs=m.boundary_schemas["wall"]), drv @ q.T)
    torch.testing.assert_close(rot["p"], base["p"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(rot["v"], base["v"] @ q.T, rtol=1e-4, atol=1e-5)


def test_four_dimensional_even_parity_smoke_and_odd_refused():
    """In four dimensions a model of even outputs runs and is rotation covariant
    (float32), with no pseudovector basis. An odd output is refused there, and so is
    one spatial dimension."""
    d = 4
    mesh, drv = _problem(d, n=120)
    assert mesh.n_manifold_dims == 3
    m = _model(d, {"p": {"rank": 0}, "v": {"rank": 1}, "t": {"rank": 2}})
    assert m._n_pseudo_basis == 0 and m._output_layouts["t"].iso[0].shape == (4, 4)
    q = orthogonal(d, 11).float()
    base = _run(m, mesh, drv)
    rot = _run(m, transform(mesh, q), drv @ q.T)
    assert base["v"].shape == (120, 4) and base["t"].shape == (120, 4, 4)
    torch.testing.assert_close(rot["p"], base["p"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(rot["v"], base["v"] @ q.T, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(rot["t"], law(base["t"], q, 2), rtol=1e-4, atol=1e-5)
    with pytest.raises(NotImplementedError):
        _model(d, {"w": {"rank": 1, "parity": "odd"}})
    with pytest.raises(ValueError):
        ISLA(
            output_schema={"p": {"rank": 0}},
            boundary_schemas={"wall": {}},
            n_spatial_dims=1,
        )


def test_dimension_mismatch_is_refused_and_two_d_is_not_embedded_three_d():
    """A 3-D mesh into a 2-D model is refused, and the surface and interior models
    have the parameter counts derived below, the same in 2-D and 3-D for one global
    direction."""
    mesh3, drv3 = _problem(3)
    m2 = _model(2, {"p": {"rank": 0}})
    with pytest.raises(
        ValueError, match="has 3 spatial dimensions; this model has n_spatial_dims=2"
    ):
        m2(
            prediction_boundaries=["wall"],
            boundary_meshes={"wall": mesh3},
            global_data={"g": drv3},
        )
    ### The 3-D model with a K = 1 direction keeps its parameter count (the head has
    ### m = 5 true and p = 3 pseudo vectors in both dimensions for K = 1, so the
    ### 2-D model of the same spec has the same count too). A surface model has the
    ### 41362 backbone-and-head parameters (the wall's input layer 2 x 32 over the
    ### constant and n.g, the shared embedding layer 32 x 32 + 32, ...) plus 2 x 32^2
    ### + 1 = 2049 for each of its three prototype synthesizers (two blocks and the
    ### head frame: the key and value maps and the gain); the interior model
    ### adds 128 for the prediction-point side (qt.seed 3 x 32 + 32 over n.g and the
    ### built-in distance pair) and 18688 for the boundary read: the kernel basis
    ### 8 x 64 + 64, 64 x 64 + 64, 64 x 16 = 5760 over the 6 + 2K = 8 pair
    ### invariants, and per block 6464 (two LayerNorms 2 x 64, the value map 32 x 64
    ### + 64, the key and query maps 2 x 32 x 32, the kernel 16 x 8, the output map
    ### 64 x 32).
    n_surface = 41362 + 3 * (2 * 32**2 + 1)
    m3 = _model(3, {"p": {"rank": 0}, "v": {"rank": 1}})
    assert sum(p.numel() for p in m3.parameters()) == n_surface
    assert (
        sum(
            p.numel()
            for p in _model(2, {"p": {"rank": 0}, "v": {"rank": 1}}).parameters()
        )
        == n_surface
    )
    assert (
        sum(
            p.numel()
            for p in _model(
                3,
                {"p": {"rank": 0}, "v": {"rank": 1}},
                interior=True,
            ).parameters()
        )
        == n_surface + 128 + 5760 + 2 * 6464
    )
