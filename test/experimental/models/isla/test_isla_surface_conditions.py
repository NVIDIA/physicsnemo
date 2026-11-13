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

"""A surface task reads the whole boundary and predicts at its prediction boundaries.

Every given boundary's cells are tokens; the outputs are returned at the cells of the
boundaries named in ``prediction_boundaries`` only (a vehicle, with a tunnel's floor
and inlet read as conditions). The outputs are the rows of those cells, so naming more
boundaries returns more rows of the same prediction; a condition-only boundary is read
(moving it, or changing its field, changes the prediction); and every condition's
parameters receive a gradient. The symmetry contracts of several conditions are
test_isla_conditions.py's.
"""

import pytest
import torch

from physicsnemo.experimental.models.isla import ISLA
from physicsnemo.mesh import Mesh
from test.experimental.models.isla._meshes import (
    cloud,
    outputs,
    soup,
    transform,
)

D = torch.float64
OUT = {"p": {"rank": 0}, "v": {"rank": 1}, "s": {"rank": 2, "symmetric": True}}
SCHEMAS = {"vehicle": {}, "floor": {}, "inlet": {"u": {"rank": 1}}}
SIZES = {"vehicle": 60, "floor": 20, "inlet": 10}
OFFSETS = {
    "vehicle": (0.0, 0.0, 0.0),
    "floor": (0.0, 0.0, -4.0),
    "inlet": (-7.0, 0.0, 0.0),
}
GD = {"g": torch.tensor([0.6, 0.8, 0.0], dtype=D)}


def _model(**kw):
    torch.manual_seed(0)
    return (
        ISLA(
            output_schema=OUT,
            boundary_schemas=SCHEMAS,
            global_schema={"g": {"rank": 1}},
            hidden=32,
            n_layers=2,
            n_slices=8,
            **kw,
        )
        .double()
        .eval()
    )


def _meshes(seed: int = 0) -> dict[str, Mesh]:
    out = {}
    for k, (name, n) in enumerate(SIZES.items()):
        pts, nrm, w, _ = cloud(n, seed=seed + k)
        data = {}
        if name == "inlet":
            data["u"] = torch.randn(
                n, 3, dtype=D, generator=torch.Generator().manual_seed(seed + 9)
            )
        out[name] = soup(
            pts + torch.tensor(OFFSETS[name], dtype=D), nrm, w, cell_data=data
        )
    return out


def _run(m, meshes, names=("vehicle",), gd=GD):
    with torch.no_grad():
        return m(
            boundary_meshes=meshes, prediction_boundaries=list(names), global_data=gd
        )


def test_naming_more_boundaries_returns_more_rows_of_the_same_prediction():
    """The outputs are at the prediction boundaries' cell centroids, one row per cell,
    and the head reads the same tokens whichever boundaries carry the outputs: the
    vehicle's rows are the same when the floor's rows are also returned, the
    boundaries come in sorted name order (floor, inlet, vehicle) whatever order they
    are named in, and naming every boundary returns every token."""
    m, meshes = _model(), _meshes()
    every = outputs(_run(m, meshes, ("vehicle", "inlet", "floor")))
    nf, ni = SIZES["floor"], SIZES["inlet"]
    for names in [("vehicle", "floor"), ("floor", "vehicle")]:
        out = _run(m, meshes, names)
        torch.testing.assert_close(
            out.points,
            torch.cat(
                [meshes["floor"].cell_centroids, meshes["vehicle"].cell_centroids]
            ),
        )
        y = outputs(out)
        for k in y:
            torch.testing.assert_close(
                y[k], torch.cat([every[k][:nf], every[k][nf + ni :]])
            )
    out = _run(m, meshes)
    torch.testing.assert_close(out.points, meshes["vehicle"].cell_centroids)
    vehicle, n = outputs(out), SIZES["vehicle"]
    assert vehicle["v"].shape == (n, 3) and vehicle["s"].shape == (n, 3, 3)
    for k in vehicle:
        torch.testing.assert_close(vehicle[k], every[k][nf + ni :])


def test_a_condition_only_boundary_is_read():
    """The floor and the inlet carry no outputs but are read: dropping the floor, moving
    it alone, or changing the inlet's field changes the vehicle's prediction."""
    m, meshes = _model(), _meshes()
    base = outputs(_run(m, meshes))
    variants = {
        "no floor": {k: v for k, v in meshes.items() if k != "floor"},
        "floor moved": {
            **meshes,
            "floor": transform(
                meshes["floor"], shift=torch.tensor([0.0, 0.0, 1.5], dtype=D)
            ),
        },
        "inlet field": {
            **meshes,
            "inlet": meshes["inlet"].with_data(
                cell_data={
                    **meshes["inlet"].cell_data.to_dict(),
                    "u": 2.0 * meshes["inlet"].cell_data["u"],
                }
            ),
        },
    }
    for label, ms in variants.items():
        y = outputs(_run(m, ms))
        assert y["p"].shape == base["p"].shape
        assert not torch.allclose(y["p"], base["p"], atol=1e-8), label


def test_every_condition_parameter_gets_a_gradient():
    """A loss on the vehicle's outputs alone reaches every condition's input layer
    (the floor's and the inlet's tokens route into the slices the vehicle reads), so no
    parameter is left without a gradient (DDP)."""
    m, meshes = _model(), _meshes()
    m.train()
    out = m(boundary_meshes=meshes, prediction_boundaries=["vehicle"], global_data=GD)
    sum(t.square().sum() for t in outputs(out).values()).backward()
    missing = [
        n
        for n, p in m.named_parameters()
        if p.grad is None or not p.grad.abs().sum() > 0
    ]
    assert not missing, missing


def test_prediction_boundaries_is_checked():
    """Required for a surface task, a non-empty list or tuple of distinct given names;
    refused for an interior task."""
    m, meshes = _model(), _meshes()
    kw = dict(boundary_meshes=meshes, global_data=GD)
    with pytest.raises(ValueError, match="pass prediction_boundaries"):
        m(**kw)
    for bad, match in [
        ("vehicle", "list or tuple"),
        ([], "list or tuple"),
        (["vehicle", "vehicle"], "twice"),
        (["wing"], "not in boundary_meshes"),
    ]:
        with pytest.raises(ValueError, match=match):
            m(prediction_boundaries=bad, **kw)
    without_floor = {k: v for k, v in meshes.items() if k != "floor"}
    with pytest.raises(ValueError, match="not in boundary_meshes"):
        m(
            boundary_meshes=without_floor,
            prediction_boundaries=["floor"],
            global_data=GD,
        )
    assert m(prediction_boundaries=("vehicle",), **kw).n_points == SIZES["vehicle"]
    interior = _model(interior=True)
    q = torch.zeros(4, 3, dtype=D)
    with pytest.raises(ValueError, match="prediction_boundaries must be None"):
        interior(prediction_points=q, prediction_boundaries=["vehicle"], **kw)
