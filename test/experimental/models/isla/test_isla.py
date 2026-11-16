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

"""Contract tests for ISLA (Invariant Slice Attention): every guarantee is exact by construction.

The model under test declares one scalar output ``p`` and one vector output ``v``
and one global direction ``g``. The boundary is a ``Mesh[2, 3]`` built by
:func:`soup` around a prescribed cloud (its cell centroids, normals and effective
measures are the cloud's points, normals and weights); ``forward`` returns a
point-cloud Mesh with ``point_data["p"]`` of shape ``(N,)`` and ``point_data["v"]``
of shape ``(N, 3)``.
"""

import pytest
import torch

from physicsnemo.experimental.models.isla import ISLA
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import EFFECTIVE_MEASURE_KEY
from test.experimental.models.isla._marks import DEVICES
from test.experimental.models.isla._meshes import (
    cloud,
    live,
    orthogonal,
    outputs,
    points_mesh,
    same,
    soup,
    transform,
)

# The frame is origin-free (positions enter only as point-to-anchor differences) with
# one data-derived length unit, the measure-weighted RMS pairwise distance.

PV = {"p": {"rank": 0}, "v": {"rank": 1}}
G = {"g": {"rank": 1}}
D = torch.float64


#: One boundary condition without fields, the cloud's.
WALL = {"wall": {}}


def _isla(**kw):
    kw.setdefault("boundary_schemas", WALL)
    return ISLA(output_schema=PV, global_schema=G, **kw)


def _rows(out, sl):
    """Slice the point axis of every output."""
    return {k: v[sl] for k, v in outputs(out).items()}


def _assert_rotated(rot, base, q, atol=1e-10):
    rot, base = outputs(rot), outputs(base)
    assert torch.allclose(rot["p"], base["p"], atol=atol, rtol=0.0)
    assert torch.allclose(rot["v"], base["v"] @ q.T, atol=atol, rtol=0.0)


def _rotation(seed=0):
    return orthogonal(3, seed)


def _run(m, mesh, drv, q=None, **kw):
    with torch.no_grad():
        return m(
            boundary_meshes={"wall": mesh},
            prediction_points=q,
            prediction_boundaries=None if q is not None else ["wall"],
            global_data={"g": drv},
            **kw,
        )


@pytest.fixture
def setup():
    torch.manual_seed(0)
    m = _isla(hidden=64, n_layers=3, n_slices=32).double().eval()
    ### Anisotropic cloud: generic for vehicle geometry; an isotropic cloud is the
    ### degenerate corner.
    pts, nrm, w, drv = cloud(500)
    mesh = soup(pts, nrm, w)
    return m, mesh, drv, _run(m, mesh, drv)


#: The rigid motions of :func:`test_rigid_motions_move_the_outputs`: the model's
#: arguments (beside :func:`_isla`'s), the cloud's size, whether its cells carry two
#: rank-0 fields, the global direction's magnitude, whether the motion rotates, its
#: shift, and the bound.
RIGID = {
    "rotation": (
        dict(hidden=64, n_layers=3, n_slices=32),
        500,
        False,
        1.0,
        True,
        None,
        1e-10,
    ),
    "translation": (
        dict(hidden=64, n_layers=3, n_slices=32),
        500,
        False,
        1.0,
        False,
        (3.0, -7.0, 11.0),
        1e-10,
    ),
    ### ISLA's own scales: a global velocity source, outputs in its units.
    "rotation_with_scales": (
        dict(
            hidden=32,
            n_layers=2,
            n_slices=8,
            output_schema={
                "p": {"rank": 0, "units": {"U": 2}},
                "v": {"rank": 1, "units": {"U": 1}},
            },
            global_schema={"g": {"rank": 1, "units": {"U": 1}, "defines": "U"}},
        ),
        200,
        False,
        7.0,
        True,
        None,
        1e-9,
    ),
    ### Two rank-0 boundary fields (one leaf per scalar) in cell_data ride along.
    "rotation_with_boundary_scalars": (
        dict(
            hidden=64,
            n_layers=2,
            n_slices=16,
            boundary_schemas={"wall": {"bc_a": {"rank": 0}, "bc_b": {"rank": 0}}},
        ),
        300,
        True,
        1.0,
        True,
        None,
        1e-10,
    ),
}


@pytest.mark.parametrize("case", list(RIGID))
def test_rigid_motions_move_the_outputs(case):
    """A rotation of the boundary (its fields moved with it) and the global direction
    keeps p and rotates v; a translation changes nothing."""
    kw, n, fields, scale, rotated, shift, atol = RIGID[case]
    torch.manual_seed(0)
    m = (
        ISLA(
            **{"output_schema": PV, "global_schema": G, "boundary_schemas": WALL, **kw}
        )
        .double()
        .eval()
    )
    pts, nrm, w, drv = cloud(n)
    data = (
        {"bc_a": torch.randn(n, dtype=D), "bc_b": torch.randn(n, dtype=D)}
        if fields
        else None
    )
    mesh = soup(pts, nrm, w, cell_data=data)
    base = _run(m, mesh, scale * drv)
    if fields:
        ### the constant, n.g and the two fields
        assert m.boundary_conditions["wall"].seed.in_features == 1 + 1 + 2
    q = _rotation() if rotated else None
    moved = transform(
        mesh,
        q,
        shift=None if shift is None else torch.tensor(shift, dtype=D),
        specs=m.boundary_schemas["wall"],
    )
    out = _run(m, moved, scale * drv @ q.T if rotated else scale * drv)
    if rotated:
        _assert_rotated(out, base, q, atol=atol)
    else:
        assert same(out, base, atol=atol)


@pytest.mark.parametrize("k", [0.5, 2.0, 4.0])
def test_global_vector_magnitude_invariance(setup, k):
    """Rank-1 global inputs enter as unit directions: rescaling one leaves every
    output unchanged (a physically meaningful magnitude is a rank-0 global input)."""
    m, mesh, drv, base = setup
    assert same(_run(m, mesh, drv * k), base, atol=1e-10)


def test_no_effective_measures_is_the_geometric_quadrature(setup):
    """Without ``_effective_measure`` the cells' geometric measures are the weights;
    a soup without prescribed measures has congruent cells, so this is the uniform
    quadrature: the same output as any constant measure vector."""
    m, mesh, drv, base = setup
    mesh = soup(mesh.cell_centroids, mesh.cell_normals)
    none = _run(m, mesh, drv)
    const = _run(
        m,
        mesh.with_data(
            cell_data={
                EFFECTIVE_MEASURE_KEY: torch.full((mesh.n_cells,), 0.37, dtype=D)
            }
        ),
        drv,
    )
    assert same(none, const, atol=1e-10)
    assert not same(none, base, atol=1e-3)


def test_head_runs_under_bf16_autocast():
    """Mixed-precision smoke: the head must survive bf16 autocast (the geometry is
    kept in the input precision by construction)."""
    torch.manual_seed(0)
    m = _isla(hidden=32, n_layers=1, n_slices=8)
    pts, nrm, w, drv = cloud(128, dtype=torch.float32)
    mesh = soup(pts, nrm, w)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = outputs(
            m(
                prediction_boundaries=["wall"],
                boundary_meshes={"wall": mesh},
                global_data={"g": drv},
            )
        )
    sum(o.float().sum() for o in out.values()).backward()
    assert all(torch.isfinite(o.float()).all() for o in out.values())


@pytest.mark.parametrize("interior", [False, True], ids=["surface", "interior"])
def test_every_parameter_receives_a_gradient(interior):
    """DDP requires every parameter to take part in the loss. A surface model owns
    no prediction-point parameters and an interior model uses all of its own, so
    in both tasks every parameter receives a gradient."""
    torch.manual_seed(0)
    m = live(_isla(hidden=32, n_layers=2, n_slices=8, interior=interior))
    pts, nrm, w, drv = cloud(128, dtype=torch.float32)
    q = torch.randn(20, 3) if interior else None
    out = outputs(
        m(
            boundary_meshes={"wall": soup(pts, nrm, w)},
            prediction_points=q,
            prediction_boundaries=None if q is not None else ["wall"],
            global_data={"g": drv},
        )
    )
    sum(o.sum() for o in out.values()).backward()
    unused = [n for n, p in m.named_parameters() if p.grad is None]
    assert not unused, unused
    assert (m.qt is not None) == interior
    assert not any(n.startswith("qt") for n, _ in m.named_parameters()) or interior


def test_live_makes_the_boundary_read_contribute():
    """The interior contract tests run the interior model through ``live``, which
    seeds the boundary read's zero-initialized output projection. Under it the read
    changes the prediction, and zeroing the projection again restores the dead read's
    outputs, so those tests cannot pass with a read that adds nothing."""
    torch.manual_seed(0)
    pts, nrm, w, drv = cloud(200)
    mesh = soup(pts, nrm, w)
    q = pts[:20] * 1.5
    m = _isla(hidden=32, n_layers=2, n_slices=8, interior=True).double().eval()
    assert all(not r.out.weight.any() for r in m.kernel_reads)
    dead = _run(m, mesh, drv, q)
    live(m)
    assert all(r.out.weight.any() for r in m.kernel_reads)
    out = _run(m, mesh, drv, q)
    assert (
        max(float((outputs(out)[k] - outputs(dead)[k]).abs().max()) for k in PV) > 1e-3
    )
    with torch.no_grad():
        for r in m.kernel_reads:
            r.out.weight.zero_()
    assert same(_run(m, mesh, drv, q), dead, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("interior", [False, True], ids=["surface", "interior"])
def test_geometric_scale_equivariance(interior):
    """For a model without scales: a similarity of the geometry (points x k + shift,
    areas x k^2) leaves the output exactly unchanged, for the boundary alone and with
    prediction points that are moved along. (The similarity with declared scales is
    test_isla_scales.py's.)"""
    torch.manual_seed(0)
    pts, nrm, w, drv = cloud(400)
    mesh = soup(pts, nrm, w)
    m = (
        _isla(
            hidden=64,
            n_layers=2,
            n_slices=16,
            interior=interior,
        )
        .double()
        .eval()
    )
    live(m)
    assert not m._scaled
    k = 2.7
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=D)
    q = pts[:50] * 1.5 if interior else None
    a = _run(m, mesh, drv, q)
    b = _run(
        m,
        transform(mesh, shift=shift, scale=k),
        drv,
        k * q + shift if interior else None,
    )
    assert same(b, a, atol=1e-10, rtol=0.0)


def _biased_poisson_subsample(pts, w, n_expected, bias, generator):
    """10:1 Poisson subsample with exact Horvitz-Thompson weights, biased toward x
    below the median (a data pipeline's biased subsample of a surface)."""
    x = pts[:, 0]
    b = torch.where(x < x.median(), torch.full_like(x, bias), torch.ones_like(x))
    pi = (n_expected / b.sum() * b).clamp(max=1.0)
    keep = torch.rand(x.shape[0], dtype=pi.dtype, generator=generator) < pi
    idx = keep.nonzero(as_tuple=True)[0]
    return idx, w[idx] / pi[idx]


def _frame_cloud(n=300, seed=0):
    pts, nrm, w, drv = cloud(n, seed=seed)
    return soup(pts + 5.0, nrm, w), drv


def _float32_case(device, interior):
    """A float32 model (its boundary read live for the interior task) on ``device``
    and a case of 500 cells, with 150 prediction points about four units out."""
    torch.manual_seed(0)
    m = live(_isla(hidden=64, n_layers=3, n_slices=32, interior=interior))
    pts, nrm, w, drv = cloud(500, dtype=torch.float32)
    q = cloud(150, seed=1, scale=(4.0, 4.0, 4.0), dtype=torch.float32)[0]
    return m.to(device).eval(), soup(pts, nrm, w), drv, q if interior else None


def _run_on(device, m, mesh, drv, q):
    return outputs(
        _run(m, mesh.to(device), drv.to(device), None if q is None else q.to(device))
    )


#: The float32 contracts' bound, relative to the largest output: float32 roundoff
#: (eps 1.2e-7) of the moved inputs, amplified through the network and its kernels
#: (measured at most 3e-6 on x86, 9e-6 on Grace and 2.4e-6 on a GB300 with the fused
#: kernels; the translation by about 13 units is the larger).
F32_BOUND = 1e-4


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("interior", [False, True], ids=["surface", "interior"])
def test_float32_rigid_motions_on_every_device(device, interior):
    """In float32, on CPU and on CUDA (every kernel at ``auto``): a rotation of the
    boundary, the global direction and the prediction points keeps p and rotates v,
    and a translation changes no output, to F32_BOUND of the largest output."""
    m, mesh, drv, q = _float32_case(device, interior)
    base = _run_on(device, m, mesh, drv, q)
    tol = F32_BOUND * max(float(v.abs().max()) for v in base.values())
    R = _rotation().float()
    rot = _run_on(
        device, m, transform(mesh, R), drv @ R.T, None if q is None else q @ R.T
    )
    assert torch.allclose(rot["p"], base["p"], atol=tol, rtol=0.0)
    assert torch.allclose(rot["v"], base["v"] @ R.T.to(device), atol=tol, rtol=0.0)
    t = torch.tensor([3.0, -7.0, 11.0])
    moved = _run_on(
        device, m, transform(mesh, shift=t), drv, None if q is None else q + t
    )
    assert same(moved, base, atol=tol)


@pytest.mark.slow
@pytest.mark.parametrize("interior", [False, True], ids=["surface", "interior"])
def test_the_default_configuration_keeps_the_contracts(interior):
    """At the default width, slices and layers (the release configuration), on 2000
    cells in float64, in each task (the interior task with its read live): a
    reflection with a translation keeps p and maps v, to 1e-10 absolute (float64
    roundoff through every layer). Slow: the default network on CPU."""
    torch.manual_seed(0)
    m = live(_isla(interior=interior)).double().eval()
    pts, nrm, w, drv = cloud(2000)
    mesh = soup(pts, nrm, w)
    q = cloud(200, seed=1, scale=(4.0, 4.0, 4.0))[0] if interior else None
    base = _run(m, mesh, drv, q)
    R = orthogonal(3, 0, improper=True)
    t = torch.tensor([3.0, -7.0, 11.0], dtype=D)
    moved = _run(
        m, transform(mesh, R, t), drv @ R.T, None if q is None else q @ R.T + t
    )
    _assert_rotated(moved, base, R)


def test_frame_is_sampling_consistent():
    """The origin-free frame uses no sample statistic. Under a 10:1 biased Poisson
    subsample with exact Horvitz-Thompson weights, the predictions at fixed points
    move between a uniform and a biased draw by about the measure noise, the change
    between two uniform draws (a frame centred on the sample's centroid would move
    them by much more)."""
    pts, nrm, w, drv = cloud(100000)
    q = pts[:40] * 1.2
    torch.manual_seed(1)
    m = live(_isla(hidden=32, n_layers=2, n_slices=8, interior=True).double().eval())
    n_sub = 10000
    d_bias, d_noise = [], []

    def _flat(o):
        return torch.cat([o.point_data["p"][..., None], o.point_data["v"]], dim=-1)

    def _draw(bias, g):
        idx, w_ht = _biased_poisson_subsample(pts, w, n_sub, bias, g)
        return _run(m, soup(pts[idx], nrm[idx], w_ht), drv, q)

    for seed in range(4):
        g = torch.Generator().manual_seed(100 + seed)
        o_u, o_u2, o_b = _draw(1.0, g), _draw(1.0, g), _draw(10.0, g)
        d_bias.append(float((_flat(o_b) - _flat(o_u)).norm()))
        d_noise.append(float((_flat(o_u2) - _flat(o_u)).norm()))
    ### measure-noise level: the biased draw's HT estimates have a larger variance
    ### than a uniform draw's (10x fewer points in the undersampled half), so the
    ### bias-vs-uniform difference may exceed the uniform-vs-uniform one by a
    ### bounded factor, far below the order of magnitude a centroid shift produces.
    ratio = sum(d_bias) / sum(d_noise)
    assert ratio < 4.0, (ratio, d_bias, d_noise)


def test_the_class_defaults():
    """The class defaults: a surface task (no prediction points, no boundary read) and
    the automatic geometry kernel. An interior model adds the prediction-point
    embedding over the built-in distance pair and the boundary read, with no routing
    mass."""
    m = _isla(hidden=32, n_layers=2, n_slices=8)
    assert m.interior is False and m.qt is None
    assert m.kernel_basis is None and m.kernel_reads is None
    assert m._geo_kernel == "auto"
    mi = _isla(hidden=32, n_layers=2, n_slices=8, interior=True)
    assert mi.qt.seed.in_features == 1 + 2  # n.g and the built-in distance pair
    assert not hasattr(mi.qt, "logw")  # no routing mass
    assert mi.kernel_basis is not None and len(mi.kernel_reads) == 2
    assert not any(n.startswith("qt.logw") for n, _ in mi.named_parameters())
    assert any(n.startswith("kernel_reads.") for n, _ in mi.named_parameters())


def test_the_docs_page_quick_start_runs():
    """The quick start of ISLA's docs page, a surface and an interior task, runs as a
    doctest (skipped in a tree without the docs)."""
    import doctest
    from pathlib import Path

    page = Path(__file__).parents[4] / "docs" / "api" / "models" / "isla.rst"
    if not page.exists():
        pytest.skip(f"no {page}")
    result = doctest.testfile(str(page), module_relative=False)
    assert result.attempted and not result.failed, result


def test_constructor_and_forward_are_keyword_only():
    """Every constructor argument and every forward input is keyword-only
    (inputs read as the recipe's forward_kwargs mapping and models can be
    hot-swapped without positional bookkeeping)."""
    pts, nrm, w, drv = cloud(16, dtype=torch.float32)
    mesh = soup(pts, nrm, w)
    with pytest.raises(TypeError):
        ISLA(PV, WALL, G)
    with pytest.raises(TypeError):
        ISLA(
            boundary_schemas=WALL, hidden=32, n_layers=1, n_slices=8
        )  # outputs is required
    with pytest.raises(TypeError):
        ISLA(
            output_schema=PV, hidden=32, n_layers=1, n_slices=8
        )  # so are the conditions
    m = _isla(hidden=32, n_layers=1, n_slices=8)
    with pytest.raises(TypeError):
        m({"wall": mesh}, None, {"g": drv})
    out = m(
        prediction_boundaries=["wall"],
        boundary_meshes={"wall": mesh},
        global_data={"g": drv},
    )
    assert out.point_data["p"].shape == (16,) and out.point_data["v"].shape == (16, 3)
    assert torch.allclose(out.points, mesh.cell_centroids)
    assert all(torch.isfinite(o).all() for o in outputs(out).values())


def test_interior_flag_structures_the_forward():
    """A surface model refuses prediction points; an interior model requires
    them, at least one, as a point-cloud Mesh or a bare tensor."""
    pts, nrm, w, drv = cloud(20, dtype=torch.float32)
    mesh = soup(pts, nrm, w)
    surface = _isla(hidden=32, n_layers=1, n_slices=8)
    with pytest.raises(ValueError, match="interior=False"):
        surface(
            boundary_meshes={"wall": mesh},
            prediction_points=pts[:3],
            global_data={"g": drv},
        )
    interior = _isla(hidden=32, n_layers=1, n_slices=8, interior=True)
    with pytest.raises(ValueError, match="pass prediction_points"):
        interior(
            prediction_boundaries=["wall"],
            boundary_meshes={"wall": mesh},
            global_data={"g": drv},
        )
    out = interior(
        boundary_meshes={"wall": mesh},
        prediction_points=points_mesh(pts[:3] * 2),
        global_data={"g": drv},
    )
    assert out.point_data["p"].shape == (3,)
    assert out.n_cells == 0 and torch.equal(out.points, pts[:3] * 2)
    bare = interior(
        boundary_meshes={"wall": mesh},
        prediction_points=pts[:3] * 2,
        global_data={"g": drv},
    )
    assert torch.equal(bare.point_data["p"], out.point_data["p"])
    with pytest.raises(ValueError, match="tensor of shape \\(N, 3\\)"):
        interior(
            boundary_meshes={"wall": mesh},
            prediction_points=pts[:3, :2],
            global_data={"g": drv},
        )
    for empty in (pts[:0], points_mesh(pts[:0])):
        for call in (interior, interior.query_chunks):
            with pytest.raises(ValueError, match="prediction_points has no points"):
                call(
                    boundary_meshes={"wall": mesh},
                    prediction_points=empty,
                    global_data={"g": drv},
                )


def test_mesh_validation_errors():
    """Dimension and type mismatches, and a boundary with cells but no measure,
    are refused at the API boundary with the offending mesh named."""
    pts, nrm, w, drv = cloud(20, dtype=torch.float32)
    m = _isla(hidden=32, n_layers=1, n_slices=8)
    with pytest.raises(TypeError, match="must be a physicsnemo.mesh.Mesh"):
        m(
            prediction_boundaries=["wall"],
            boundary_meshes={"wall": pts},
            global_data={"g": drv},
        )
    with pytest.raises(
        ValueError,
        match="boundary_meshes\\['wall'\\]: the boundary of a 3-dimensional domain is a Mesh\\[2, 3\\]",
    ):
        m(
            prediction_boundaries=["wall"],
            boundary_meshes={"wall": Mesh(points=pts)},
            global_data={"g": drv},
        )
    pts2, nrm2, w2, _ = cloud(20, d=2, dtype=torch.float32)
    with pytest.raises(
        ValueError, match="has 2 spatial dimensions; this model has n_spatial_dims=3"
    ):
        m(
            prediction_boundaries=["wall"],
            boundary_meshes={"wall": soup(pts2, nrm2, w2)},
            global_data={"g": drv},
        )
    with pytest.raises(
        ValueError, match="global_data is missing the declared fields \\['g'\\]"
    ):
        m(prediction_boundaries=["wall"], boundary_meshes={"wall": soup(pts, nrm, w)})
    with pytest.raises(TypeError, match="global_data must be"):
        m(
            prediction_boundaries=["wall"],
            boundary_meshes={"wall": soup(pts, nrm, w)},
            global_data=drv,
        )
    flat = soup(pts, nrm, torch.zeros_like(w))
    interior = _isla(hidden=32, n_layers=1, n_slices=8, interior=True)
    for model, where in (
        (m, {"prediction_boundaries": ["wall"]}),
        (interior, {"prediction_points": pts[:3] * 2}),
    ):
        with pytest.raises(
            ValueError, match="boundary_meshes\\['wall'\\] has cells but no measure"
        ):
            model(boundary_meshes={"wall": flat}, global_data={"g": drv}, **where)


def test_non_finite_negative_and_empty_inputs_are_refused():
    """A boundary point that is not finite, a cell measure that is negative or not
    finite, a prediction point that is not finite or not of a floating-point dtype,
    and a surface task whose prediction boundaries have no cells are refused in
    eager mode with the input named, in both tasks where they apply."""
    pts, nrm, w, drv = cloud(20, dtype=torch.float32)
    good = soup(pts, nrm, w)
    surface = _isla(hidden=32, n_layers=1, n_slices=8)
    interior = _isla(hidden=32, n_layers=1, n_slices=8, interior=True)
    tasks = (
        (surface, {"prediction_boundaries": ["wall"]}),
        (interior, {"prediction_points": pts[:3] * 2}),
    )
    nan_point = good.points.clone()
    nan_point[0, 0] = float("nan")
    bad_points = Mesh(points=nan_point, cells=good.cells, cell_data=good.cell_data)
    refused = [(bad_points, "boundary_meshes\\['wall'\\] has a non-finite point")]
    for value in (-0.3, float("nan"), float("inf")):
        measure = w.clone()
        measure[0] = value
        bad = good.with_data(cell_data={EFFECTIVE_MEASURE_KEY: measure})
        refused.append(
            (
                bad,
                "boundary_meshes\\['wall'\\] has a negative or non-finite cell measure",
            )
        )
    for model, where in tasks:
        for mesh, message in refused:
            with pytest.raises(ValueError, match=message):
                model(boundary_meshes={"wall": mesh}, global_data={"g": drv}, **where)
    for value in (float("nan"), float("inf")):
        q = pts[:3] * 2
        q[1, 2] = value
        with pytest.raises(ValueError, match="prediction_points has a non-finite"):
            interior(
                boundary_meshes={"wall": good},
                global_data={"g": drv},
                prediction_points=q,
            )
    with pytest.raises(TypeError, match="prediction_points must have a floating-point"):
        interior(
            boundary_meshes={"wall": good},
            global_data={"g": drv},
            prediction_points=torch.ones(3, 3, dtype=torch.int64),
        )
    two = _isla(hidden=32, n_layers=1, n_slices=8, boundary_schemas={"a": {}, "b": {}})
    empty = soup(pts[:0], nrm[:0], w[:0])
    with pytest.raises(
        ValueError, match="prediction_boundaries \\['b'\\] have no cells"
    ):
        two(
            boundary_meshes={"a": good, "b": empty},
            global_data={"g": drv},
            prediction_boundaries=["b"],
        )


def test_non_finite_fields_and_globals_are_refused():
    """A declared cell field or a global with a value that is not finite is refused
    in eager mode with the field named, in both tasks, instead of giving outputs
    that are NaN everywhere."""
    pts, nrm, w, drv = cloud(20, dtype=torch.float32)
    good = soup(pts, nrm, w, cell_data={"T": torch.ones(20)})
    globals_ = {"g": {"rank": 1}, "re": {"rank": 0}}
    tasks = [
        (
            ISLA(output_schema=PV, boundary_schemas={"wall": {"T": {"rank": 0}}}, global_schema=globals_,
                 hidden=32, n_layers=1, n_slices=8, interior=interior),
            {"prediction_points": pts[:3] * 2} if interior else {"prediction_boundaries": ["wall"]},
        )
        for interior in (False, True)
    ]  # fmt: skip
    gd = {"g": drv, "re": torch.tensor(0.5)}
    for model, where in tasks:
        model(boundary_meshes={"wall": good}, global_data=gd, **where)
        for value in (float("nan"), float("inf")):
            data = good.cell_data.clone()
            data["T"][3] = value
            bad = good.with_data(cell_data=data)
            with pytest.raises(
                ValueError,
                match="boundary_meshes\\['wall'\\].cell_data\\['T'\\] has a non-finite",
            ):
                model(boundary_meshes={"wall": bad}, global_data=gd, **where)
            with pytest.raises(
                ValueError, match="global_data\\['re'\\] has a non-finite"
            ):
                model(
                    boundary_meshes={"wall": good},
                    global_data={**gd, "re": torch.tensor(value)},
                    **where,
                )


def test_inputs_of_another_dtype_than_the_parameters_are_refused():
    """A boundary mesh whose floating dtype is not the parameters', in either
    direction, and prediction points wider than the parameters are refused in eager
    mode with the input named and the casts that fix it, in both tasks (not a matmul
    error deep inside). Narrower prediction points are promoted to the parameters'
    dtype: float32 points into a float64 model, and float16 or bfloat16 points into
    a float32 model, give exactly the outputs and points of the same values passed
    in the parameters' dtype. Float32 inputs to a float32 model under bfloat16
    autocast run. Casting an interior model to float64 is named as the slower path."""
    pts, nrm, w, drv = cloud(20, dtype=torch.float32)
    good = soup(pts, nrm, w)
    surface = _isla(hidden=32, n_layers=1, n_slices=8)
    interior = _isla(hidden=32, n_layers=1, n_slices=8, interior=True)
    for model, where in (
        (surface, {"prediction_boundaries": ["wall"]}),
        (interior, {"prediction_points": pts[:3] * 2}),
    ):
        with pytest.raises(
            ValueError,
            match="boundary_meshes\\['wall'\\] has torch.float64 points and this "
            "model's parameters are torch.float32: cast the input",
        ) as refused:
            model(
                boundary_meshes={"wall": good.to(torch.float64)},
                global_data={"g": drv},
                **where,
            )
        ### (the hint: a float64 interior model reads the boundary on the reference path)
        assert (
            "slower reference boundary read" in str(refused.value)
        ) == model.interior
        with torch.autocast("cpu", dtype=torch.bfloat16):
            model(boundary_meshes={"wall": good}, global_data={"g": drv}, **where)
    for q in (pts[:3].double(), points_mesh(pts[:3].double())):
        with pytest.raises(
            ValueError, match="prediction_points has torch.float64 points"
        ):
            interior(
                boundary_meshes={"wall": good},
                global_data={"g": drv},
                prediction_points=q,
            )
    interior = interior.double().eval()
    with pytest.raises(
        ValueError,
        match="boundary_meshes\\['wall'\\] has torch.float32 points and this "
        "model's parameters are torch.float64",
    ):
        interior(
            boundary_meshes={"wall": good},
            global_data={"g": drv.double()},
            prediction_points=pts[:3].double(),
        )
    kw = {
        "boundary_meshes": {"wall": good.to(torch.float64)},
        "global_data": {"g": drv.double()},
    }
    with torch.no_grad():
        promoted = interior(**kw, prediction_points=pts[:3] * 2)
        cast = interior(**kw, prediction_points=(pts[:3] * 2).double())
    assert promoted.points.dtype == torch.float64
    assert same(promoted, cast, atol=0.0)
    interior = interior.float()
    kw = {"boundary_meshes": {"wall": good}, "global_data": {"g": drv}}
    for low in (torch.float16, torch.bfloat16):
        q = (pts[:3] * 2).to(low)
        with torch.no_grad():
            promoted = interior(**kw, prediction_points=q)
            cast = interior(**kw, prediction_points=q.float())
        assert promoted.points.dtype == torch.float32, low
        assert same(promoted, cast, atol=0.0), low


def test_the_execution_attributes_are_validated():
    """An execution choice that is not one of its values (``_geo_kernel`` ``"auto"``,
    ``"eager"`` or ``"fused"``; ``_geometry_kernel`` ``"auto"``, ``"reference"`` or
    ``"fast"``; ``_tree_read_kernel`` ``"auto"`` or ``"reference"``) is a ValueError
    naming it at the first forward pass that reads it, and so is a
    ``block_activation_budget`` outside ``(0, 1]`` where the compiled blocks read it."""
    pts, nrm, w, drv = cloud(20, dtype=torch.float32)
    boundary = {
        "boundary_meshes": {"wall": soup(pts, nrm, w)},
        "global_data": {"g": drv},
    }
    surface = _isla(hidden=32, n_layers=1, n_slices=8)
    interior = _isla(hidden=32, n_layers=1, n_slices=8, interior=True)
    at_wall = {"prediction_boundaries": ["wall"]}
    at_points = {"prediction_points": pts[:3] * 2}
    for model, where, name, value in (
        (surface, at_wall, "_geo_kernel", "Fused"),
        (interior, at_points, "_geometry_kernel", "fused"),
        (interior, at_points, "_tree_read_kernel", "fast"),
    ):
        setattr(model, name, value)
        with pytest.raises(ValueError, match=f"^{name} must be one of"):
            model(**boundary, **where)
        setattr(model, name, "auto")
        model(**boundary, **where)
    for budget in (0.0, -0.5, 1.5, float("nan")):
        compiled = _isla(hidden=32, n_layers=1, n_slices=8, compile_blocks=True)
        compiled.block_activation_budget = budget
        with pytest.raises(
            ValueError, match="^block_activation_budget must be above 0"
        ):
            compiled(**boundary, **at_wall)


def test_reference_kernels_sets_every_execution_choice():
    """``reference_kernels`` is ``False`` on a new model, every region at ``"auto"``;
    ``True`` puts every region on its reference path and ``False`` puts them back at
    ``"auto"``; a region forced off its reference path reads as ``False``; a non-bool
    is refused."""
    m = _isla(hidden=32, n_layers=1, n_slices=8, interior=True)

    def kernels():
        return m._geo_kernel, m._geometry_kernel, m._tree_read_kernel

    assert m.reference_kernels is False and kernels() == ("auto",) * 3
    m.reference_kernels = True
    assert m.reference_kernels is True
    assert kernels() == ("eager", "reference", "reference")
    m.reference_kernels = False
    assert m.reference_kernels is False and kernels() == ("auto",) * 3
    m.reference_kernels = True
    m._geo_kernel = "fused"
    assert m.reference_kernels is False
    for bad in ("no", 1, None):
        with pytest.raises(TypeError, match="reference_kernels must be True or False"):
            m.reference_kernels = bad


def test_rms_distance_scale_contracts():
    """The RMS-pairwise-distance length unit keeps exact translation invariance and
    rotation covariance, is invariant to a geometric rescale of the boundary (with
    or without the matching measure rescale) AND to a uniform rescale of the
    measures alone, and needs no centre."""
    mesh, drv = _frame_cloud()
    torch.manual_seed(1)
    m = _isla(hidden=64, n_layers=2, n_slices=16).double().eval()
    shift = torch.tensor([300.0, -70.0, 1100.0], dtype=D)
    w = mesh.cell_data[EFFECTIVE_MEASURE_KEY]
    base = _run(m, mesh, drv)
    assert all(torch.isfinite(o).all() for o in outputs(base).values())
    assert same(_run(m, transform(mesh, shift=shift), drv), base, atol=1e-10)
    q = _rotation()
    _assert_rotated(_run(m, transform(mesh, q), drv @ q.T), base, q)
    assert same(_run(m, transform(mesh, scale=2.5), drv), base, atol=1e-10)
    # measure rescale alone: invariant (the unit is normalized)
    assert same(
        _run(m, mesh.with_data(cell_data={EFFECTIVE_MEASURE_KEY: w * 137.0}), drv),
        base,
        atol=1e-10,
    )
    # a rescale of the points alone is invariant too: the unit is a length of the
    # sample and the routing bias log w is uniform-rescale invariant, so measures
    # and positions decouple
    scaled_points_only = transform(mesh, scale=2.5).with_data(
        cell_data={EFFECTIVE_MEASURE_KEY: w}
    )
    assert same(_run(m, scaled_points_only, drv), base, atol=1e-10)


def _koch_curve(level: int):
    """Koch curve (plane z = 0) as a polyline of 4**level segments on the unit base."""
    import math

    pts = [torch.tensor([0.0, 0.0]), torch.tensor([1.0, 0.0])]
    rot = torch.tensor(
        [
            [math.cos(math.pi / 3), -math.sin(math.pi / 3)],
            [math.sin(math.pi / 3), math.cos(math.pi / 3)],
        ]
    )
    for _ in range(level):
        new = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            d = b - a
            p1 = a + d / 3
            p3 = a + 2 * d / 3
            p2 = p1 + rot @ (d / 3)
            new += [p1, p2, p3, b]
        pts = new
    xy = torch.stack(pts).double()
    mids = (xy[:-1] + xy[1:]) / 2
    seg = (xy[1:] - xy[:-1]).norm(dim=-1)
    return torch.cat([mids, torch.zeros(len(mids), 1, dtype=D)], dim=-1)[None], seg[
        None
    ]


def test_rms_distance_converges_on_koch_curve_where_total_measure_diverges():
    """On a fractal boundary refined dyadically, the total measure grows without
    bound ((4/3)^level for the Koch curve) while the RMS pairwise distance
    converges: the relative differences between successive levels fall below 1e-3
    from level 4 on, and the last two levels agree to 1e-4."""
    from physicsnemo.experimental.models.isla._geometry import rms_pairwise_distance

    totals, rms = [], []
    for level in range(0, 7):
        pts, w = _koch_curve(level)
        totals.append(w.sum().item())
        rms.append(rms_pairwise_distance(pts, w).item())
    for a, b in zip(totals[:-1], totals[1:]):
        assert abs(b / a - 4.0 / 3.0) < 1e-6  # the measure diverges geometrically
    rel = [abs(b - a) / a for a, b in zip(rms[:-1], rms[1:])]
    assert all(r < 1e-3 for r in rel[4:]), rel
    assert rel[-1] < 1e-4, rel
    assert rms[-1] > 0.3  # a finite length of the order of the base


def test_geo_kernel_auto_resolves_to_eager_off_cuda_and_honors_explicit_choices():
    """_geo_kernel "auto" picks the fused Triton region only where it can run: on CUDA
    with one to four global directions and at most 1024 slices. On CPU it must
    resolve to eager at the first forward pass, cache the decision, and leave the
    outputs identical to a model forced to eager with the same weights."""
    torch.manual_seed(0)
    pts, nrm, w, _ = cloud(64)
    mesh = soup(pts, nrm, w)
    drv = torch.tensor([1.0, 0.0, 0.0], dtype=D)
    auto = _isla(hidden=32, n_layers=2, n_slices=8).double().eval()
    assert auto._geo_kernel == "auto" and auto._resolve_geo_kernel(pts) == "eager"
    eager = _isla(hidden=32, n_layers=2, n_slices=8).double().eval()
    eager._geo_kernel = "eager"
    eager.load_state_dict(auto.state_dict())
    ya, ye = outputs(_run(auto, mesh, drv)), outputs(_run(eager, mesh, drv))
    assert all(torch.equal(ya[k], ye[k]) for k in ya)
    assert auto._geo_kernel_cache[1] == "eager"


def test_scalar_only_model_has_no_frame_routing_and_every_parameter_gets_a_gradient():
    """A model whose outputs are all rank 0 never reads the token basis, so the head's
    frame routing is not built (a parameter without a gradient stops DDP at the second
    step); with a vector output the frame routing exists. In both cases every
    parameter receives a gradient in the interior task, and the scalar-only outputs
    equal the head coefficients directly."""
    torch.manual_seed(0)
    pts, nrm, w, drv = cloud(200, dtype=torch.float32)
    mesh = soup(pts, nrm, w)
    q = torch.randn(30, 3) * 2.0
    for outs in (
        {"u": {"rank": 0}},
        {"u": {"rank": 0}, "q1": {"rank": 0}, "q2": {"rank": 0}},
        {"u": {"rank": 0}, "v": {"rank": 1}},
    ):
        m = ISLA(
            output_schema=outs,
            boundary_schemas=WALL,
            global_schema={"g": {"rank": 1}},
            hidden=32,
            n_layers=2,
            n_slices=8,
            interior=True,
        )
        live(m)
        has_vec = any(v["rank"] == 1 for v in outs.values())
        assert m._needs_frame == has_vec and (m.frame_assign is not None) == has_vec
        out = outputs(
            m(
                boundary_meshes={"wall": mesh},
                prediction_points=q,
                global_data={"g": drv},
            )
        )
        sum(v.square().mean() for v in out.values()).backward()
        missing = [k for k, p in m.named_parameters() if p.grad is None]
        assert not missing, missing
        assert out["u"].shape == (30,)
        if "q1" in outs:
            assert out["q2"].shape == (30,) and m.head.out_features == 3
