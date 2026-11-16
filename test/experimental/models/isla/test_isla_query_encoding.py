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

"""Continuity of the prediction points' geometry encoding at the boundary.

A prediction point's built-in geometry (its signed distance to the boundary
surface and its direction) enters the model as features of its token. Points on
the boundary, as the boundary nodes of a solid's mesh are, have a distance of the
order of floating-point rounding, whose sign rounding decides; points just off it
differ from each other by tiny distances. These tests measure what the model
ingests (the query tokens' feature vector and their direction, read by forward
hooks) for on-surface and near-surface points, across the boundary, under random
rigid motions and under uniform scalings, all in float32 as the recipe runs the
geometry, and bound the change.
"""

import pytest
import torch

from physicsnemo.experimental.models.isla import ISLA
from physicsnemo.experimental.models.isla._geometry import (
    LENGTH_SOFTENING,
    soft_log_length,
)
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.curves import circle_2d
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from test.experimental.models.isla._meshes import orthogonal, points_mesh

F = torch.float32
D = torch.float64


def _surface(dim: int, subdivisions: int = 3) -> Mesh:
    """A closed, curved surface off the origin, in float32: a 96-gon in two
    dimensions, an icosphere in three (642 vertices at 3 subdivisions, 42 at 1)."""
    if dim == 2:
        m = circle_2d.load(radius=1.3, n_points=96)
        shift = torch.tensor([0.4, -0.25])
    else:
        m = sphere_icosahedral.load(radius=1.3, subdivisions=subdivisions)
        shift = torch.tensor([0.4, -0.25, 0.15])
    return Mesh(points=(m.points.double() + shift.double()).to(F), cells=m.cells)


def _queries(mesh: Mesh):
    """Query points in float32 and their kind: the vertices and the cell centroids
    (on the surface), and points off the centroids along the cell normal at signed
    offsets ``+-delta * L`` (``L`` the circumradius 1.3)."""
    pts = mesh.points.double()
    cen = pts[mesh.cells].mean(1)
    nrm = mesh.cell_normals.double()
    groups = {"vertex": pts, "centroid": cen}
    for delta in (1e-6, 1e-4, 1e-2):
        groups[f"+{delta:g}"] = cen + 1.3 * delta * nrm
        groups[f"-{delta:g}"] = cen - 1.3 * delta * nrm
    return groups


def _model(dim: int) -> ISLA:
    torch.manual_seed(0)
    return ISLA(
        output_schema={"p": {"rank": 0}},
        boundary_schemas={"wall": {}},
        n_spatial_dims=dim,
        hidden=16,
        n_layers=1,
        n_slices=4,
        interior=True,
    ).eval()


def _ingested(model: ISLA, mesh: Mesh, q: torch.Tensor):
    """The query tokens' feature vector (the built-in geometry channels) and their
    direction, as the model ingests them, read by forward pre-hooks."""
    seen = {}
    ### The prediction points' input layer reads [n.g_k, global scalars, built-in
    ### geometry, query inputs]; without global inputs or query inputs that is the
    ### built-in geometry alone.
    h1 = model.qt.seed.register_forward_pre_hook(
        lambda mod, args: seen.__setitem__("feat", args[0].detach().clone())
    )
    ### The direction the query tokens carry into the blocks' reads.
    query_tokens = model._query_tokens

    def spy(*args, **kwargs):
        out = query_tokens(*args, **kwargs)
        seen["dir"] = out[0]["n_hat"].detach().clone()
        return out

    model._query_tokens = spy
    try:
        with torch.no_grad():
            model(boundary_meshes={"wall": mesh}, prediction_points=points_mesh(q))
    finally:
        h1.remove()
        del model._query_tokens
    return seen["feat"][0].double(), seen["dir"][0].double()


def _move(x64: torch.Tensor, R, t, k: float) -> torch.Tensor:
    """``k R x + t`` in float32, from the float32 inputs (rounded as a data pipeline
    that moves the stored float32 geometry would round)."""
    return (k * x64.to(F) @ R.to(F).T + t.to(F)).to(F)


def feature_jumps(dim: int, n_motions: int = 4, subdivisions: int = 3):
    """The largest change of the ingested features and of the direction (after
    undoing the motion) per query group: under ``n_motions`` random rigid motions
    (a rotation, proper or improper, and a translation of twice the size), under the
    uniform scalings 1e-3 and 1e3, and across the boundary (``+delta`` against
    ``-delta`` in the native frame)."""
    model = _model(dim)
    mesh = _surface(dim, subdivisions)
    groups = _queries(mesh)
    names = list(groups)
    q64 = torch.cat([groups[n] for n in names])
    sizes = [groups[n].shape[0] for n in names]

    def split(x):
        return dict(zip(names, torch.split(x, sizes)))

    feat0, dir0 = _ingested(model, mesh, q64.to(F))
    out = {
        "pose": {n: [0.0, 0.0] for n in names},
        "scale": {n: [0.0, 0.0] for n in names},
    }
    g = torch.Generator().manual_seed(7)
    moves = []
    for i in range(n_motions):
        R = orthogonal(dim, 11 + i, improper=bool(i % 2))
        t = 2.6 * torch.randn(dim, generator=g, dtype=D)
        moves.append(("pose", R, t, 1.0))
    moves += [
        ("scale", torch.eye(dim, dtype=D), torch.zeros(dim, dtype=D), k)
        for k in (1e-3, 1e3)
    ]
    for kind, R, t, k in moves:
        pts = _move(mesh.points.double(), R, t, k)
        cells = mesh.cells.clone()
        if torch.det(R) < 0:
            cells = cells[:, list(range(dim - 2)) + [dim - 1, dim - 2]]
        moved = Mesh(points=pts, cells=cells)
        feat, dirn = _ingested(model, moved, _move(q64, R, t, k))
        dfeat = split((feat - feat0).abs().amax(-1))
        ddir = split((dirn @ R - dir0).norm(dim=-1))
        for n in names:
            out[kind][n][0] = max(out[kind][n][0], float(dfeat[n].max()))
            out[kind][n][1] = max(out[kind][n][1], float(ddir[n].max()))
    f, dn = split(feat0), split(dir0)
    out["across"] = {
        f"{delta:g}": [
            float((f[f"+{delta:g}"] - f[f"-{delta:g}"]).abs().amax(-1).max()),
            float((dn[f"+{delta:g}"] - dn[f"-{delta:g}"]).norm(dim=-1).max()),
        ]
        for delta in (1e-6, 1e-4, 1e-2)
    }
    out["on_surface_feature_range"] = [
        float(torch.cat([f["vertex"], f["centroid"]]).min()),
        float(torch.cat([f["vertex"], f["centroid"]]).max()),
    ]
    return out


#: Bounds on the measured changes. The current encoding (asinh(s / 1e-6), sign(d) u)
#: gives at most about 0.45 for the on-surface and 1e-6 L points under a motion or a
#: scaling (rounding of d, times the slope 1 / 1e-6) and 1.5 across the surface at
#: +-1e-6 L (2 s / 1e-6); the signed logarithm sign(s) log(|s| + 1e-12) it replaced
#: gave 37-46 and 28.5, and the direction from the surface flipped (change 2).
FEATURE_JUMP_ON = 1.0
FEATURE_JUMP_ACROSS = 2.0
DIRECTION_JUMP = 1e-4


def _report(dim, out):
    print(f"\n[dim={dim}] feature / direction jump (max abs)")
    for kind in ("pose", "scale"):
        for n, (a, b) in out[kind].items():
            print(f"  {kind:5s} {n:9s} feature {a:10.4g}  direction {b:8.4g}")
    for delta, (a, b) in out["across"].items():
        print(f"  across +-{delta:7s} feature {a:10.4g}  direction {b:8.4g}")
    print(f"  on-surface feature range {out['on_surface_feature_range']}")


@pytest.mark.parametrize(
    "dim, subdivisions",
    [
        pytest.param(2, 3, id="2"),
        ### an icosphere of 80 cells: the three-dimensional case in the fast tier
        pytest.param(3, 1, id="3-small"),
        ### 3-8 s on CPU (an icosphere of 1280 cells under motions and scalings)
        pytest.param(3, 3, id="3", marks=pytest.mark.slow),
    ],
)
def test_query_encoding_is_continuous_at_the_surface(dim, subdivisions):
    """On-surface points (vertices, and cell centroids whose distance is a rounding
    error of either sign) and points 1e-6 L off the surface keep their features and
    their direction under rigid motions (proper and improper) and uniform scalings;
    across the surface the features change by at most their slope times the step
    and the direction not at all. The measured numbers print with ``pytest -s``."""
    out = feature_jumps(dim, subdivisions=subdivisions)
    _report(dim, out)
    for kind in ("pose", "scale"):
        for n in ("vertex", "centroid", "+1e-06", "-1e-06"):
            feature, direction = out[kind][n]
            assert feature <= FEATURE_JUMP_ON, (kind, n, feature)
            assert direction <= DIRECTION_JUMP, (kind, n, direction)
        for n, (_, direction) in out[kind].items():
            assert direction <= DIRECTION_JUMP, (kind, n, direction)
    assert out["across"]["1e-06"][0] <= FEATURE_JUMP_ACROSS, out["across"]
    for delta, (_, direction) in out["across"].items():
        assert direction <= DIRECTION_JUMP, (delta, direction)
    lo, hi = out["on_surface_feature_range"]
    assert -FEATURE_JUMP_ON <= lo <= hi <= FEATURE_JUMP_ON


def test_soft_log_length():
    """asinh(s / 1e-6): odd, zero at zero, slope at most 1 / 1e-6, and the signed
    logarithm log(2 |s| / 1e-6) away from zero."""
    s = torch.cat([torch.logspace(-12, 2, 57, dtype=D), torch.zeros(1, dtype=D)])
    phi = soft_log_length(s)
    assert torch.equal(soft_log_length(-s), -phi)
    assert soft_log_length(torch.zeros(1, dtype=D)).item() == 0.0
    assert (phi <= s / LENGTH_SOFTENING + 1e-12).all()
    far = s > 1e-3
    torch.testing.assert_close(
        phi[far], torch.log(2 * s[far] / LENGTH_SOFTENING), rtol=1e-6, atol=1e-6
    )
