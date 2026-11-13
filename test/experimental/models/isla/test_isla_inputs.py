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

"""Contract tests for ISLA's rank-spec per-token inputs.

Rank-1 and rank-2 boundary fields (in the meshes' ``cell_data``) and
prediction-point fields (in the prediction Mesh's ``point_data``) enter only
through invariants formed with true vectors (the normal, the global directions, the
token-to-anchor direction), so every rank-0 output stays invariant and every rank
>= 1 output stays covariant when the geometry AND the fields are rotated, reflected
or translated together; the checks are exact (float64).
"""

import pytest
import torch

from physicsnemo.experimental.models.isla import ISLA
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

D = torch.float64
OUT = {"p": {"rank": 0}, "v": {"rank": 1}, "sig": {"rank": 2, "symmetric": True}}
BIN = {
    "trac": {"rank": 1},
    "T": {"rank": 2, "symmetric": True},
    "A": {"rank": 2},
    "tag": {"rank": 0},
}
DRV = torch.tensor([0.3, 1.0, 0.2], dtype=D)


def _fields(n, seed, symmetric_names, vector_names, scalar_names):
    g = torch.Generator().manual_seed(seed)
    f = {}
    for k in vector_names:
        f[k] = torch.randn(n, 3, dtype=D, generator=g)
    for k in symmetric_names:
        t = torch.randn(n, 3, 3, dtype=D, generator=g)
        f[k] = t + t.transpose(-1, -2)
    for k in scalar_names:
        f[k] = torch.rand(n, dtype=D, generator=g)
    return f


def _boundary_fields(n, seed=1):
    f = _fields(n, seed, ["T"], ["trac"], ["tag"])
    f["A"] = torch.randn(
        n, 3, 3, dtype=D, generator=torch.Generator().manual_seed(seed + 10)
    )
    return f


def _move(fields, specs, R):
    """Transform every field by its rank: rank 0 unchanged, rank 1 ``v R^T``, rank 2
    ``R T R^T``."""
    out = {}
    for k, v in fields.items():
        r = specs[k].rank
        out[k] = v if r == 0 else (v @ R.T if r == 1 else R @ v @ R.T)
    return out


def _model(**kw):
    torch.manual_seed(0)
    base = dict(
        output_schema=OUT,
        boundary_schemas={"wall": {}},
        global_schema={"g": {"rank": 1}},
        hidden=32,
        n_layers=2,
        n_slices=8,
    )
    base.update(kw)
    return ISLA(**base).double().eval()


def _assert_covariant(m, call, improper):
    """``call(R, shift)`` runs the model on the transformed problem."""
    base = call(torch.eye(3, dtype=D), torch.zeros(3, dtype=D))
    R = orthogonal(3, 4, improper)
    out = call(R, torch.tensor([2.0, -5.0, 1.0], dtype=D))
    for name, spec in m.output_schema.items():
        assert torch.allclose(
            out[name], law(base[name], R, spec.rank), atol=1e-11, rtol=0.0
        ), (
            name,
            improper,
        )
    return base


@pytest.mark.parametrize("improper", [False, True], ids=["rotation", "reflection"])
def test_boundary_inputs_covariance(improper):
    """Rank-1 and rank-2 boundary fields moved with the boundary keep every output
    covariant; the fields are live, and rotating the fields alone is not a
    symmetry."""
    m = _model(boundary_schemas={"wall": BIN})
    pts, nrm, w, _ = cloud(200)
    bf = _boundary_fields(200)
    mesh = soup(pts, nrm, w, cell_data=bf)

    def call(R, shift):
        with torch.no_grad():
            return outputs(
                m(
                    prediction_boundaries=["wall"],
                    boundary_meshes={
                        "wall": transform(
                            mesh, R, shift, specs=m.boundary_schemas["wall"]
                        )
                    },
                    global_data={"g": DRV @ R.T},
                )
            )

    base = _assert_covariant(m, call, improper)
    ### the fields are live: scaling the traction changes every output
    with torch.no_grad():
        other = outputs(
            m(
                prediction_boundaries=["wall"],
                boundary_meshes={
                    "wall": soup(
                        pts, nrm, w, cell_data={**bf, "trac": 2.0 * bf["trac"]}
                    )
                },
                global_data={"g": DRV},
            )
        )
    assert all(not torch.allclose(other[k], base[k], atol=1e-6) for k in base)
    ### a rotation of the fields ALONE (geometry fixed) is not a symmetry
    R = orthogonal(3, 6, False)
    with torch.no_grad():
        skew = outputs(
            m(
                prediction_boundaries=["wall"],
                boundary_meshes={
                    "wall": soup(
                        pts, nrm, w, cell_data=_move(bf, m.boundary_schemas["wall"], R)
                    )
                },
                global_data={"g": DRV},
            )
        )
    assert not torch.allclose(skew["p"], base["p"], atol=1e-6)


@pytest.mark.parametrize("improper", [False, True], ids=["rotation", "reflection"])
@pytest.mark.parametrize("interior", [False, True], ids=["surface", "interior"])
def test_two_vector_fields_of_one_condition_are_covariant_with_their_gradients(
    interior, improper
):
    """One condition with two rank-1 fields, whose dot product enters the seed beside
    each one's frame contractions and norm: under a rotation or a reflection with a
    translation every output maps by its rank (1e-11 absolute, float64 roundoff of
    outputs of order one), and the gradient of an invariant loss with respect to
    each field maps as that field does (``g -> g R^T``), to 1e-11 of the largest
    gradient entry; every parameter receives a gradient. Both tasks, the interior
    one with its boundary read live."""
    m = live(
        _model(
            interior=interior,
            boundary_schemas={"wall": {"a": {"rank": 1}, "b": {"rank": 1}}},
        )
    )
    F, P = 2, 2  # the frame [n_hat, g] and the two fields
    assert (
        m.boundary_conditions["wall"].inputs.n_features == P * F + P + P * (P - 1) // 2
    )
    pts, nrm, w, _ = cloud(200)
    g = torch.Generator().manual_seed(5)
    fields = {k: torch.randn(200, 3, dtype=D, generator=g) for k in "ab"}
    mesh = soup(pts, nrm, w)
    q = cloud(40, seed=1, scale=(4.0, 4.0, 4.0))[0] if interior else None

    def call(R, shift):
        moved = transform(mesh, R, shift)
        leaves = {k: (v @ R.T).requires_grad_() for k, v in fields.items()}
        m.zero_grad(set_to_none=True)
        out = outputs(
            m(
                boundary_meshes={
                    "wall": moved.with_data(
                        cell_data={**moved.cell_data.to_dict(), **leaves}
                    )
                },
                prediction_points=None if q is None else q @ R.T + shift,
                prediction_boundaries=None if interior else ["wall"],
                global_data={"g": DRV @ R.T},
            )
        )
        sum(o.square().sum() for o in out.values()).backward()
        return {k: v.detach() for k, v in out.items()}, {
            k: v.grad for k, v in leaves.items()
        }

    base, grad = call(torch.eye(3, dtype=D), torch.zeros(3, dtype=D))
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, missing
    R = orthogonal(3, 4, improper)
    out, moved_grad = call(R, torch.tensor([2.0, -5.0, 1.0], dtype=D))
    for name, spec in m.output_schema.items():
        want = law(base[name], R, spec.rank)
        assert torch.allclose(out[name], want, atol=1e-11, rtol=0.0), name
    scale = max(float(v.abs().max()) for v in grad.values())
    for k in fields:
        assert grad[k].abs().max() > 0, k
        assert torch.allclose(
            moved_grad[k], grad[k] @ R.T, atol=1e-11 * scale, rtol=0.0
        ), k


def test_geometry_width_and_seed_width():
    """One rank-1 input and one symmetric rank-2 input over K = 1: the geometry region
    gains P' = 1 + (1 + K) contracted vectors and Q = 1 quadratic form; the seed gains
    the frame cosines, the norm, the frame contractions, the trace and the norm."""
    for K in (0, 1, 2):
        m = _model(
            global_schema={f"g{k}": {"rank": 1} for k in range(K)},
            boundary_schemas={
                "wall": {"v": {"rank": 1}, "T": {"rank": 2, "symmetric": True}}
            },
        )
        F = 1 + K
        assert m.blocks[0].n_geo == 5 + K + (1 + F) + 1
        assert m._n_block_vecs == 1 + F and m._n_block_tens == 1
        wall = m.boundary_conditions["wall"]
        assert wall.inputs.n_features == (F + 1) + (F * (F + 1) // 2 + 2)
        assert wall.seed.in_features == 1 + K + wall.inputs.n_features
    ### a non-symmetric rank-2 input contracts both T a and T^T a and all F^2 pairs
    m = _model(boundary_schemas={"wall": {"A": {"rank": 2}}})
    assert (
        m._n_block_vecs == 2 * 2
        and m.boundary_conditions["wall"].inputs.n_features == 4 + 2
    )


def test_field_validation_errors():
    """Declared fields are read from cell_data / point_data by name and checked for
    shape; extras there are ignored (like GLOBE)."""
    m = _model(boundary_schemas={"wall": {"trac": {"rank": 1}, "tag": {"rank": 0}}})
    pts, nrm, w, _ = cloud(20)
    gd = {"g": torch.ones(3, dtype=D)}
    trac, tag = torch.randn(20, 3, dtype=D), torch.rand(20, dtype=D)
    with pytest.raises(
        ValueError, match="cell_data is missing the declared fields \\['tag'\\]"
    ):
        m(
            prediction_boundaries=["wall"],
            boundary_meshes={"wall": soup(pts, nrm, w, cell_data={"trac": trac})},
            global_data=gd,
        )
    with pytest.raises(
        ValueError, match="cell_data is missing the declared fields \\['tag', 'trac'\\]"
    ):
        m(
            prediction_boundaries=["wall"],
            boundary_meshes={"wall": soup(pts, nrm, w)},
            global_data=gd,
        )
    with pytest.raises(
        ValueError,
        match="\\['trac'\\]: expected shape \\(20, 3\\) \\(rank 1\\), got \\(20, 2\\)",
    ):
        m(
            prediction_boundaries=["wall"],
            boundary_meshes={
                "wall": soup(pts, nrm, w, cell_data={"trac": trac[:, :2], "tag": tag})
            },
            global_data=gd,
        )
    ### extras in cell_data are ignored
    out = m(
        prediction_boundaries=["wall"],
        boundary_meshes={
            "wall": soup(
                pts, nrm, w, cell_data={"trac": trac, "tag": tag, "extra": tag}
            )
        },
        global_data=gd,
    )
    assert out.point_data["p"].shape == (20,)
    ### the prediction points must match the model's dimension
    mq = _model(interior=True)
    q = pts[:7] * 2
    with pytest.raises(ValueError, match="prediction_points has 2 spatial dimensions"):
        mq(
            boundary_meshes={"wall": soup(pts, nrm, w)},
            prediction_points=points_mesh(q[:, :2]),
            global_data=gd,
        )


def test_fused_kernel_auto_falls_back_with_tensor_inputs():
    """The fused kernel does not take rank-2 inputs: auto falls back to eager (off CUDA
    it does anyway). The surface task takes them; the interior task (whose
    boundary read has no pair contractions for them) refuses them."""
    base = dict(
        output_schema=OUT,
        global_schema={"g": {"rank": 1}},
        hidden=32,
        n_layers=1,
        n_slices=8,
    )
    m = ISLA(boundary_schemas={"wall": {"S": {"rank": 2}}}, **base)
    assert m._geo_kernel == "auto"
    assert m._resolve_geo_kernel(torch.zeros(1, 4, 3)) == "eager"
    with pytest.raises(NotImplementedError, match="rank-2"):
        ISLA(interior=True, boundary_schemas={"wall": {"S": {"rank": 2}}}, **base)


def test_fields_keep_their_declaration_order_as_columns():
    """The field inputs fill the seed, and the head's coefficients fill the
    outputs, in declaration order, not in key order: with the condition's input
    layer reading only its constant and its first field column and the head
    writing only its first coefficient,
    the first declared input alone moves the first declared output."""
    m = (
        ISLA(
            output_schema={"z": {"rank": 0}, "a": {"rank": 0}},
            boundary_schemas={"wall": {"y": {"rank": 0}, "b.c": {"rank": 0}}},
            hidden=16,
            n_layers=1,
            n_slices=4,
        )
        .double()
        .eval()
    )
    with torch.no_grad():
        m.boundary_conditions["wall"].seed.weight[:, 2:] = (
            0  # [1, y, b.c]: no global inputs
        )
        m.head.weight[1:] = 0
        m.head.bias[1:] = 0
    pts, nrm, w, _ = cloud(30, seed=0)
    y, c = torch.rand(30, dtype=D), torch.rand(30, dtype=D)

    def run(y, c):
        with torch.no_grad():
            mesh = soup(pts, nrm, w, cell_data={"y": y, "b": {"c": c}})
            return outputs(
                m(prediction_boundaries=["wall"], boundary_meshes={"wall": mesh})
            )

    base = run(y, c)
    assert list(base) == ["z", "a"]
    assert (
        torch.equal(base["a"], torch.zeros_like(base["a"]))
        and base["z"].abs().max() > 0
    )
    assert torch.equal(run(y, c + 1)["z"], base["z"])
    assert not torch.allclose(run(y + 1, c)["z"], base["z"])
