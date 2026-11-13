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

"""Contract tests for ISLA's rank-spec output heads.

Every output is declared with a rank, a symmetry and a parity and is assembled from
invariant coefficients times tensor products of the token's basis vectors (true
vectors for even parity, exactly one pseudovector factor for odd parity) plus the
isotropic tensors of that rank. The checks are exact (float64): a rank-r output
transforms as ``R^{(x)r}`` under every orthogonal map, with the extra sign ``det R``
for odd parity, and is invariant under translations.
"""

from math import comb

import pytest
import torch

from physicsnemo.experimental.models.isla import ISLA
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.fields import RankSpec
from test.experimental.models.isla._marks import requires_cuda_triton
from test.experimental.models.isla._meshes import (
    cloud,
    law,
    live,
    orthogonal,
    outputs,
    points_mesh,
    soup,
    transform,
)
from test.experimental.models.isla._model_cases import _sphere_cells

D = torch.float64

OUTPUTS = {
    "s": {"rank": 0},
    "s2": {"rank": 0},
    "v": {"rank": 1},
    "pv": {"rank": 1, "parity": "odd"},
    "t": {"rank": 2},
    "ts": {"rank": 2, "symmetric": True},
    "pt": {"rank": 2, "parity": "odd"},
    "t3": {"rank": 3},
    "t3o": {"rank": 3, "parity": "odd"},
    "t3s": {"rank": 3, "symmetric": True},
    "t4s": {"rank": 4, "symmetric": True},
}


def _boundary(n=300, seed=0):
    pts, nrm, w, _ = cloud(n, seed=seed)
    g = torch.Generator().manual_seed(seed + 100)
    drv = torch.nn.functional.normalize(torch.randn(2, 3, dtype=D, generator=g), dim=-1)
    return soup(pts, nrm, w), drv


def _globals(K):
    return {f"g{k}": {"rank": 1} for k in range(K)}


def _model(K=1, output_schema=OUTPUTS, **kw):
    torch.manual_seed(0)
    base = dict(
        hidden=32,
        n_layers=2,
        n_slices=8,
        output_schema=output_schema,
        boundary_schemas={"wall": {}},
        global_schema=_globals(K),
    )
    base.update(kw)
    return live(ISLA(**base).double().eval())


def _forward(m, mesh, drv, K, R=None, shift=None, q=None):
    """Forward pass with the boundary (and every vector input) transformed by ``R``
    and translated by ``shift``; ``q`` prediction points are moved along."""
    R = torch.eye(3, dtype=D, device=drv.device) if R is None else R.to(drv.device)
    gd = {f"g{k}": drv[k] @ R.T for k in range(K)}
    if q is not None:
        q = points_mesh(q.points @ R.T + (0 if shift is None else shift), q.point_data)
    with torch.no_grad():
        return outputs(
            m(
                boundary_meshes={"wall": transform(mesh, R, shift)},
                prediction_points=q,
                prediction_boundaries=None if q is not None else ["wall"],
                global_data=gd,
            )
        )


def _transform(x, R, spec: RankSpec):
    """``R^{(x)r} x`` on the trailing ``rank`` axes with the pseudotensor sign for odd parity."""
    return law(x, R, spec.rank) * (torch.det(R) if spec.parity == "odd" else 1.0)


@pytest.mark.parametrize("K", [0, 1, 2])
def test_coefficient_counts_match_the_docstring(K):
    """m = 2K + 3 true vectors, p = C(K+2, 2) pseudovectors, plus the isotropic terms."""
    m = _model(K)
    mm, p = 2 * K + 3, comb(K + 2, 2)
    assert m._n_true_basis == mm and m._n_pseudo_basis == p
    expected = {
        "s": 1,
        "s2": 1,
        "v": mm,
        "pv": p,
        "t": mm**2 + 1,
        "ts": mm * (mm + 1) // 2 + 1,
        "pt": 2 * p * mm,
        "t3": mm**3,
        "t3o": 3 * p * mm**2 + 1,
        "t3s": comb(mm + 2, 3),
        "t4s": comb(mm + 3, 4) + 1,
    }
    for name, n in expected.items():
        assert m._output_layouts[name].n_coeffs == n, name
    assert m.head.out_features == sum(expected.values())
    assert m._n_head_coeffs == m.head.out_features


def test_the_true_basis_has_rank_two_without_a_global_direction(monkeypatch):
    """A known limit, characterized: each polar
    complement e_theta(u, a) is -a projected off u, so it lies in span{u, a} and adds
    no rank. With K = 0 in three dimensions the true vectors n, u and e_theta(u, n)
    span only the plane of n and u at every token, so every even-parity vector output
    lies in that plane, orthogonal to the token's one pseudovector n x u. One global
    direction (K = 1) gives rank 3."""
    from physicsnemo.experimental.models.isla import model as model_mod

    seen = []
    assemble = model_mod._assemble_output

    def spy(coeffs, layout, true_vecs, pseudo_vecs, index):
        seen.append((true_vecs, pseudo_vecs))
        return assemble(coeffs, layout, true_vecs, pseudo_vecs, index)

    monkeypatch.setattr(model_mod, "_assemble_output", spy)
    mesh, drv = _boundary()
    for K, rank in ((0, 2), (1, 3)):
        seen.clear()
        out = _forward(_model(K, output_schema={"v": {"rank": 1}}), mesh, drv, K)
        true_vecs, pseudo = seen[0][0][0], seen[0][1][0]  # (N, m, 3), (N, p, 3)
        sv = torch.linalg.svdvals(true_vecs)  # (N, 3), descending
        assert (sv[:, rank - 1] > 1e-6 * sv[:, 0]).float().mean() > 0.99, K
        if rank == 2:
            assert (sv[:, 2] <= 1e-12 * sv[:, 0]).all()
            w = pseudo[:, 0]  # n x u
            dots = (out["v"] * w).sum(-1).abs()
            assert (dots <= 1e-12 * out["v"].norm(dim=-1) * w.norm(dim=-1)).all()


@pytest.mark.parametrize("K", [0, 1, 2])
def test_output_shapes(K):
    """Every declared output has its rank's shape and is finite, for K = 0, 1, 2."""
    m = _model(K)
    mesh, drv = _boundary()
    out = _forward(m, mesh, drv, K)
    assert set(out) == set(OUTPUTS)
    assert out["s"].shape == (300,) and out["s2"].shape == (300,)
    assert out["v"].shape == out["pv"].shape == (300, 3)
    assert out["t"].shape == out["ts"].shape == out["pt"].shape == (300, 3, 3)
    assert out["t3"].shape == out["t3o"].shape == out["t3s"].shape == (300, 3, 3, 3)
    assert out["t4s"].shape == (300, 3, 3, 3, 3)
    assert all(torch.isfinite(o).all() for o in out.values())


@pytest.mark.parametrize("K", [0, 1, 2])
@pytest.mark.parametrize("improper", [False, True], ids=["rotation", "reflection"])
def test_covariance_of_every_rank_and_parity(K, improper):
    """Under a random orthogonal map R (proper or improper) together with a
    translation, a rank-r even output transforms as R^{(x)r}, an odd one picks up
    det R, and a rank-0 output is unchanged (float64)."""
    m = _model(K)
    mesh, drv = _boundary()
    base = _forward(m, mesh, drv, K)
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=D)
    for seed in range(2):
        R = orthogonal(3, seed, improper)
        assert (torch.det(R) < 0) == improper
        out = _forward(m, mesh, drv, K, R=R, shift=shift)
        for name, spec in m.output_schema.items():
            assert torch.allclose(
                out[name], _transform(base[name], R, spec), atol=1e-11, rtol=0.0
            ), (name, K, improper)


@pytest.mark.parametrize("dtype, tol", [(D, 1e-10), (torch.float32, 1e-3)])
def test_a_closed_boundary_without_fields_or_globals_is_covariant(dtype, tol):
    """With K = 0 and no field on a closed boundary every first-block token is alike,
    so each slice's mean normal is the surface's vector area, zero up to roundoff;
    the floor on its norm keeps that roundoff from becoming a direction that does not
    rotate with the case."""
    p, c = _sphere_cells(2)
    m = _model(0, output_schema={"s": {"rank": 0}, "v": {"rank": 1}}, n_layers=3)
    m = m.to(dtype)

    def forward(R, shift=None):
        mesh = transform(Mesh(points=p, cells=c), R, shift)
        mesh = Mesh(points=mesh.points.to(dtype), cells=mesh.cells)
        with torch.no_grad():
            out = m(boundary_meshes={"wall": mesh}, prediction_boundaries=["wall"])
        return {k: v.double() for k, v in outputs(out).items()}

    base = forward(torch.eye(3, dtype=D))
    for seed, improper in ((0, False), (1, True)):
        R = orthogonal(3, seed, improper)
        out = forward(R, torch.tensor([1.0, 2.0, -3.0], dtype=D))
        for name, expect in (("s", base["s"]), ("v", base["v"] @ R.T)):
            err = (out[name] - expect).abs().max() / base[name].abs().max()
            assert err < tol, (name, seed, improper, err.item())


def test_symmetric_outputs_are_symmetric():
    """Symmetric outputs of rank 2, 3 and 4 are invariant under every index
    permutation tried; the general rank-2 output is not symmetric."""
    m = _model(1)
    mesh, drv = _boundary()
    out = _forward(m, mesh, drv, 1)
    ts, t3s, t4s = out["ts"], out["t3s"], out["t4s"]
    assert torch.allclose(ts, ts.transpose(-1, -2), atol=1e-12, rtol=0.0)
    for perm in ((0, 2, 3, 1), (0, 1, 3, 2), (0, 3, 2, 1)):
        assert torch.allclose(t3s, t3s.permute(*perm), atol=1e-12, rtol=0.0)
    for perm in ((0, 2, 1, 3, 4), (0, 1, 2, 4, 3), (0, 4, 3, 2, 1)):
        assert torch.allclose(t4s, t4s.permute(*perm), atol=1e-12, rtol=0.0)
    ### the general rank-2 output is generically NOT symmetric
    assert not torch.allclose(out["t"], out["t"].transpose(-1, -2), atol=1e-6)


def test_odd_outputs_are_pseudotensors_and_even_ones_are_not():
    """A pure inversion x -> -x flips every odd output's sign relative to the even
    law and leaves the even outputs on the even law; the contract is not vacuous
    because the odd basis is non-trivial (the outputs are not zero)."""
    m = _model(1)
    mesh, drv = _boundary()
    base = _forward(m, mesh, drv, 1)
    out = _forward(m, mesh, drv, 1, R=-torch.eye(3, dtype=D))
    assert torch.allclose(
        out["pv"], base["pv"], atol=1e-11, rtol=0.0
    )  # (-1)^1 * det = +1
    assert torch.allclose(out["v"], -base["v"], atol=1e-11, rtol=0.0)
    assert torch.allclose(
        out["pt"], -base["pt"], atol=1e-11, rtol=0.0
    )  # (-1)^2 * det = -1
    assert torch.allclose(out["t"], base["t"], atol=1e-11, rtol=0.0)
    assert base["pv"].norm() > 1e-3 and base["pt"].norm() > 1e-3
    assert base["t3o"].norm() > 1e-3


def test_rank_spec_grammar_is_validated():
    """The shared grammar (FieldSchema.parse) validates the fields; ISLA adds its
    rank ceilings and the refusal of pseudotensor inputs. A bare integer is not a
    field declaration."""
    base = dict(
        hidden=32,
        n_layers=1,
        n_slices=8,
        boundary_schemas={"wall": {}},
        global_schema={"g": {"rank": 1}},
    )
    with pytest.raises(NotImplementedError, match="rank 5 is not supported"):
        ISLA(output_schema={"x": {"rank": 5}}, **base)
    with pytest.raises(ValueError, match="unknown keys"):
        ISLA(output_schema={"x": {"rank": 1, "parity_odd": True}}, **base)
    with pytest.raises(TypeError, match="bool"):
        ISLA(output_schema={"x": True}, **base)
    with pytest.raises(
        TypeError, match="output_schema\\['x'\\] must be a RankSpec or a mapping"
    ):
        ISLA(output_schema={"x": 0}, **base)
    with pytest.raises(
        TypeError, match="global_schema\\['g'\\] must be a RankSpec or a mapping"
    ):
        ISLA(output_schema={"x": {"rank": 0}}, **{**base, "global_schema": {"g": 1}})
    with pytest.raises(ValueError, match="parity must be"):
        ISLA(output_schema={"x": {"rank": 1, "parity": "mixed"}}, **base)
    with pytest.raises(
        NotImplementedError, match="\\['so'\\]: pseudoscalar outputs are not"
    ):
        ISLA(
            output_schema={"s": {"rank": 0}, "so": {"rank": 0, "parity": "odd"}}, **base
        )
    with pytest.raises(ValueError, match="unknown keys \\['channels'\\]"):
        ISLA(output_schema={"x": {"rank": 0, "channels": 2}}, **base)
    with pytest.raises(
        ValueError, match="symmetric=True is only meaningful for rank >= 2"
    ):
        ISLA(output_schema={"x": {"rank": 1, "symmetric": True}}, **base)
    with pytest.raises(ValueError, match="at least one field"):
        ISLA(output_schema={}, **base)
    with pytest.raises(NotImplementedError, match="pseudotensor"):
        ISLA(
            output_schema={"p": {"rank": 0}},
            **{
                **base,
                "boundary_schemas": {"wall": {"w": {"rank": 1, "parity": "odd"}}},
            },
        )
    with pytest.raises(NotImplementedError, match="rank 3 is not supported"):
        ISLA(
            output_schema={"p": {"rank": 0}},
            **{**base, "boundary_schemas": {"wall": {"T": {"rank": 3}}}},
        )
    with pytest.raises(NotImplementedError, match="rank 2 is not supported"):
        ISLA(
            output_schema={"p": {"rank": 0}},
            **{**base, "global_schema": {"T": {"rank": 2}}},
        )
    with pytest.raises(TypeError, match="mapping"):
        ISLA(output_schema=[0, 1], **base)
    ### a RankSpec instance is a leaf too
    m = ISLA(output_schema={"sig": RankSpec(2, symmetric=True)}, **base)
    assert m.output_schema["sig"] == RankSpec(rank=2, symmetric=True, parity="even")


def test_all_parameters_receive_gradients():
    """In the interior task with outputs of every rank, every parameter receives a
    finite gradient (DDP)."""
    m = _model(1, interior=True).train()
    mesh, drv = _boundary(100)
    out = outputs(
        m(
            boundary_meshes={"wall": mesh},
            prediction_points=mesh.cell_centroids[:20] * 1.5,
            global_data={"g0": drv[0]},
        )
    )
    sum(o.square().sum() for o in out.values()).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


@pytest.mark.cuda
@requires_cuda_triton
def test_fused_kernel_does_not_touch_the_heads():
    """The fused geometry region is per layer; the heads are downstream of it, so the
    rank-2 outputs of the fused and eager models agree to roundoff in float64."""
    mesh, drv = _boundary(400)
    outputs_ = {
        "p": {"rank": 0},
        "v": {"rank": 1},
        "sig": {"rank": 2, "symmetric": True},
    }
    outs = []
    for kernel in ("eager", "fused"):
        m = _model(1, output_schema=outputs_).cuda()
        m._geo_kernel = kernel
        outs.append(
            {k: v.cpu() for k, v in _forward(m, mesh.to("cuda"), drv.cuda(), 1).items()}
        )
    for k in outs[0]:
        assert torch.allclose(outs[0][k], outs[1][k], atol=1e-10), k
