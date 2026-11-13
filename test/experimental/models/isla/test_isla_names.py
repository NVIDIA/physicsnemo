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

"""Nested field names.

A field in ``output_schema``, a schema of ``boundary_schemas`` or ``global_schema``
is declared in the shared rank-spec grammar (:mod:`physicsnemo.mesh.fields`): a
dotted flat name (``"surface.pressure"``) and a nested group (``{"surface":
{"pressure": {"rank": 0}}}``) declare the same field. The output Mesh's ``point_data`` nests
the outputs the same way, and the inputs are read from ``cell_data`` and
``global_data`` by the nested key.
"""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.experimental.models.isla import ISLA
from test.experimental.models.isla._meshes import (
    cloud,
    live,
    orthogonal,
    points_mesh,
    soup,
    transform,
)

D = torch.float64


def _boundary(n=80, seed=0):
    pts, nrm, w, drv = cloud(n, seed=seed, scale=(1.0, 1.0, 1.0))
    return (
        pts,
        nrm,
        w,
        torch.randn(3, dtype=D, generator=torch.Generator().manual_seed(seed + 7)),
    )


OUTPUTS = {
    "surface.pressure": {"rank": 0},
    "surface.wall_shear": {"rank": 1},
    "solid.stress": {"rank": 2, "symmetric": True},
    "nut": {"rank": 0},
}


def _model(**kw):
    torch.manual_seed(0)
    base = dict(
        output_schema=OUTPUTS,
        global_schema={
            "conditions.freestream": {"rank": 1},
            "conditions.re": {"rank": 0},
        },
        boundary_schemas={
            "wall": {"bc.trace": {"rank": 0}, "bc.traction": {"rank": 1}}
        },
        hidden=32,
        n_layers=1,
        n_slices=8,
    )
    base.update(kw)
    m = live(ISLA(**base).double().eval())
    m._geo_kernel = "eager"
    return m


def test_outputs_are_nested_point_data():
    """Dotted output names come back as a nested TensorDict of point data, whose flat
    view has the declared names."""
    m = _model()
    pts, nrm, w, drv = _boundary()
    n = pts.shape[0]
    bi = {
        "bc": {"trace": torch.rand(n, dtype=D), "traction": torch.randn(n, 3, dtype=D)}
    }
    with torch.no_grad():
        out = m(
            prediction_boundaries=["wall"],
            boundary_meshes={"wall": soup(pts, nrm, w, cell_data=bi)},
            global_data={
                "conditions": {"freestream": drv, "re": torch.tensor(1.0, dtype=D)}
            },
        )
    pd = out.point_data
    assert isinstance(pd, TensorDict) and pd.batch_size == torch.Size([n])
    assert set(pd.keys()) == {"surface", "solid", "nut"}
    assert pd["surface"]["pressure"].shape == (n,)
    assert pd["surface", "wall_shear"].shape == (n, 3)
    assert pd["solid", "stress"].shape == (n, 3, 3)
    assert pd["nut"].shape == (n,)
    assert sorted(
        ".".join(k) if isinstance(k, tuple) else k
        for k in pd.keys(include_nested=True, leaves_only=True)
    ) == ["nut", "solid.stress", "surface.pressure", "surface.wall_shear"]
    ### the flat view carries the declared names back
    assert set(pd.flatten_keys(".").keys()) == set(m.output_schema)
    ### the selected global data ride onto the output, nested
    assert out.global_data["conditions", "re"].shape == ()


def test_nested_and_dotted_declarations_are_the_same_model():
    """A nested group in the constructor and the dotted flat names declare the same
    fields, in the same order, with the same parameters."""
    nested = _model(
        output_schema={
            "surface": {"pressure": {"rank": 0}, "wall_shear": {"rank": 1}},
            "solid": {"stress": {"rank": 2, "symmetric": True}},
            "nut": {"rank": 0},
        },
        global_schema={"conditions": {"freestream": {"rank": 1}, "re": {"rank": 0}}},
        boundary_schemas={
            "wall": {"bc": {"trace": {"rank": 0}, "traction": {"rank": 1}}}
        },
    )
    flat = _model()
    assert nested.output_schema == flat.output_schema
    assert nested.boundary_schemas == flat.boundary_schemas
    assert nested.global_schema == flat.global_schema
    assert nested.head.out_features == flat.head.out_features


def test_dotted_and_nested_data_are_the_same_call():
    """Cell data and global data may be given with dotted keys (plain mapping) or
    nested; the model reads the nested key either way."""
    m = _model()
    pts, nrm, w, drv = _boundary()
    n = pts.shape[0]
    re = torch.tensor(0.3, dtype=D)
    trace, trac = torch.rand(n, dtype=D), torch.randn(n, 3, dtype=D)
    forms = [
        (
            {"conditions.freestream": drv, "conditions.re": re},
            {"bc": {"trace": trace, "traction": trac}},
        ),
        (
            {"conditions": {"freestream": drv, "re": re}},
            {"bc": {"trace": trace, "traction": trac}},
        ),
        (
            TensorDict({"conditions": {"freestream": drv, "re": re}}, batch_size=[]),
            {("bc", "trace"): trace, ("bc", "traction"): trac},
        ),
    ]
    outs = []
    for gd, cd in forms:
        with torch.no_grad():
            outs.append(
                m(
                    prediction_boundaries=["wall"],
                    boundary_meshes={"wall": soup(pts, nrm, w, cell_data=cd)},
                    global_data=gd,
                )
            )
    for o in outs[1:]:
        for k in outs[0].point_data.keys(include_nested=True, leaves_only=True):
            assert torch.equal(o.point_data[k], outs[0].point_data[k]), k


def test_mixed_flat_and_nested_names_and_the_interior_task():
    """Flat and dotted output names mix, in the interior task too."""
    pts, nrm, w, drv = _boundary()
    q = torch.randn(12, 3, dtype=D) * 2.0
    m = _model(
        output_schema={"pressure": {"rank": 0}, "flow.velocity": {"rank": 1}},
        global_schema={"freestream": {"rank": 1}},
        boundary_schemas={"wall": {}},
        interior=True,
    )
    with torch.no_grad():
        out = m(
            boundary_meshes={"wall": soup(pts, nrm, w)},
            prediction_points=points_mesh(q),
            global_data={"freestream": drv},
        )
    assert out.point_data["pressure"].shape == (12,) and out.point_data[
        "flow", "velocity"
    ].shape == (12, 3)


def test_covariance_holds_on_the_nested_values():
    """With nested inputs and outputs, a rotation with a translation maps every nested
    output by its rank."""
    m = _model()
    pts, nrm, w, drv = _boundary()
    n = pts.shape[0]
    bi = {
        "bc": {"trace": torch.rand(n, dtype=D), "traction": torch.randn(n, 3, dtype=D)}
    }
    mesh = soup(pts, nrm, w, cell_data=bi)
    R = orthogonal(3, 1)
    t = torch.tensor([1.0, 2.0, -3.0], dtype=D)

    def call(R, t):
        with torch.no_grad():
            return m(
                prediction_boundaries=["wall"],
                boundary_meshes={
                    "wall": transform(mesh, R, t, specs=m.boundary_schemas["wall"])
                },
                global_data={
                    "conditions": {
                        "freestream": drv @ R.T,
                        "re": torch.tensor(1.0, dtype=D),
                    }
                },
            ).point_data

    base, out = call(torch.eye(3, dtype=D), torch.zeros(3, dtype=D)), call(R, t)
    assert torch.allclose(
        out["surface", "pressure"], base["surface", "pressure"], atol=1e-11
    )
    assert torch.allclose(out["nut"], base["nut"], atol=1e-11)
    assert torch.allclose(
        out["surface", "wall_shear"], base["surface", "wall_shear"] @ R.T, atol=1e-11
    )
    assert torch.allclose(
        out["solid", "stress"], R @ base["solid", "stress"] @ R.T, atol=1e-11
    )


def test_name_grammar_is_validated():
    """A name that is also a group, an empty path component and a name declared twice
    are refused at construction; at forward time a field must come by its declared
    dotted name."""
    base = dict(hidden=32, n_layers=1, n_slices=8, boundary_schemas={"wall": {}})
    with pytest.raises(ValueError, match="is a group of"):
        ISLA(
            output_schema={"surface": {"rank": 0}, "surface.pressure": {"rank": 0}},
            **base,
        )
    with pytest.raises(ValueError, match="empty path component"):
        ISLA(output_schema={"surface..pressure": {"rank": 0}}, **base)
    with pytest.raises(ValueError, match="empty path component"):
        ISLA(output_schema={"surface.": {"rank": 0}}, **base)
    with pytest.raises(ValueError, match="more than once"):
        ISLA(
            output_schema={
                "surface.pressure": {"rank": 0},
                "surface": {"pressure": {"rank": 0}},
            },
            **base,
        )
    ### forward-time names must match the declared dotted names
    m = ISLA(
        output_schema={"p": {"rank": 0}},
        **{**base, "boundary_schemas": {"wall": {"bc.trace": {"rank": 0}}}},
    ).double()
    pts, nrm, w, _ = _boundary()
    with pytest.raises(
        ValueError, match="missing the declared fields \\['bc.trace'\\]"
    ):
        m(
            prediction_boundaries=["wall"],
            boundary_meshes={"wall": soup(pts, nrm, w, cell_data={"trace": w})},
        )
    with pytest.raises(
        ValueError, match="missing the declared fields \\['bc.trace'\\]"
    ):
        m(
            prediction_boundaries=["wall"],
            boundary_meshes={"wall": soup(pts, nrm, w, cell_data={"bc": w})},
        )  # "bc" is a tensor, not a group
