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

r"""ISLA brings its own scales (module docstring, *Scales*), in float64.

The scales are the length L (the full-resolution boundary's RMS pairwise distance),
the scale U and direction g that rank-1 sources define, and the scale rho that a
rank-0 global defines (``defines``). The tests check that:

- a similarity of the problem, with its dimensionless global inputs unchanged,
  leaves every internal state unchanged and scales the outputs by their units, in
  either task;
- the condition that carries the source, and the scales, can have any name;
- U, g and L are the same for every draw and under refinement;
- a model that declares nothing is the release model;
- the scales take their hand values, and an absent or zero source and bad
  declarations are refused;
- ``output_stats`` is the recipe's NormalizeMeshFields;
- a global source (an airfoil in a freestream) adds no direction;
- the public :func:`case_scales` is the model's scales bitwise, invariant under
  orthogonal maps and translations with L covariant with the coordinates, and
  refuses an undefined scale.
"""

import math

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.datapipes.transforms.mesh import NormalizeMeshFields
from physicsnemo.experimental.models.isla import ISLA, case_scales
from physicsnemo.experimental.models.isla._geometry import (
    cell_second_moments,
    rms_pairwise_distance,
)
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import cell_measures
from test.experimental.models.isla._meshes import (
    cloud,
    law,
    live,
    orthogonal,
    outputs,
    soup,
    transform,
)

D = torch.float64

#: A declaration of an external flow: the inlet's velocity is the source.
SCHEMAS = {
    "inlet": {
        "bc_velocity": {"rank": 1, "units": {"U": 1}, "defines": "U"},
        "bc_tke": {"rank": 0, "units": {"U": 2}},
        "bc_omega": {"rank": 0, "units": {"U": 1, "L": -1}},
        "bc_gamma": {"rank": 0},
    },
    "outlet": {"bc_pressure": {"rank": 0, "units": {"U": 2, "rho": 1}}},
    "wall": {},
}
GLOBALS = {
    "physics.density": {"rank": 0, "defines": "rho"},
    "re_z": {"rank": 0},  # a dimensionless global, the recipe's standardized log10 Re
}
OUTPUTS = {
    "velocity": {"rank": 1, "units": {"U": 1}},
    "pressure": {"rank": 0, "units": {"U": 2, "rho": 1}},
    "nut": {"rank": 0, "units": {"U": 1, "L": 1}},
    "omega": {"rank": 0, "units": {"U": 1, "L": -1}},
}
SIZES = {"inlet": 40, "outlet": 30, "wall": 120}
OFFSETS = {
    "inlet": (-6.0, 0.0, 0.0),
    "outlet": (6.0, 0.0, 0.0),
    "wall": (0.0, 0.0, 0.0),
}
RHO, RE_Z = 1.2, 0.3


def _units(declared: dict) -> dict:
    return {name: f.get("units", {}) for name, f in declared.items()}


def _factor(units: dict, s: float, c: float, r: float) -> float:
    """The similarity's factor on a field of these units: L -> s L, U -> c U, rho -> r rho."""
    base = {"L": s, "U": c, "rho": r}
    return math.prod(base[k] ** e for k, e in units.items())


def _model(
    interior=True,
    schemas=SCHEMAS,
    globals_=GLOBALS,
    outputs_=OUTPUTS,
    **kw,
):
    torch.manual_seed(0)
    m = ISLA(
        output_schema=outputs_,
        boundary_schemas=schemas,
        global_schema=globals_,
        interior=interior,
        hidden=32,
        n_layers=2,
        n_slices=8,
        **kw,
    )
    m._geo_kernel = "eager"
    if interior:
        m.tree_read_theta = 0.0  # the exact boundary read (a surface task has none)
    return live(m.double().eval())


def _case(schemas=SCHEMAS, sizes=SIZES, seed=0):
    """The boundary meshes (the inlet's velocity about 10 m/s along a direction, its k
    and omega positive, the outlet's pressure in Pa) and the globals."""
    g = torch.Generator().manual_seed(seed + 50)
    meshes = {}
    for k, (name, n) in enumerate(sizes.items()):
        pts, nrm, w, _ = cloud(n, seed=seed + k)
        data = {}
        for f, spec in schemas[name].items():
            shape = (n, 3) if spec["rank"] == 1 else (n,)
            data[f] = torch.rand(shape, dtype=D, generator=g) + 0.5
        if "bc_velocity" in data:
            data["bc_velocity"] = (
                10.0 * torch.tensor([0.6, 0.8, 0.0], dtype=D) + data["bc_velocity"]
            )
        meshes[name] = soup(
            pts + torch.tensor(OFFSETS.get(name, (0.0, 0.0, 0.0)), dtype=D),
            nrm,
            w,
            cell_data=data,
        )
    gd = {
        "physics.density": torch.tensor(RHO, dtype=D),
        "re_z": torch.tensor(RE_Z, dtype=D),
    }
    q = torch.randn(25, 3, dtype=D, generator=g) * 3
    return meshes, gd, q


def _similar(m, meshes, gd, q, schemas, s, c, r, R=None):
    """The case under the similarity x -> s R x, every field times its units' factor
    (a vector rotated with the geometry), rho -> r rho and the dimensionless re_z
    unchanged."""
    out = {}
    for name, mesh in meshes.items():
        moved = transform(mesh, R=R, scale=s, specs=m.boundary_schemas[name])
        data = moved.cell_data.clone()
        for f, u in _units(schemas[name]).items():
            data[f] = data[f] * _factor(u, s, c, r)
        out[name] = moved.with_data(cell_data=data)
    gd = {
        "physics.density": gd["physics.density"] * r,
        "re_z": gd["re_z"],
    }
    R = torch.eye(3, dtype=D) if R is None else R
    return out, gd, None if q is None else s * q @ R.T


def _spy(m):
    """Record the case's scales, its derived direction and the head's input of
    every forward pass."""
    seen = {"scales": [], "g": [], "h": [], "gauge": []}
    case_scales, case_inputs = m._case_scales, m._case_inputs

    def scales(*a, **k):
        out = case_scales(*a, **k)
        seen["scales"].append(out[0])
        seen["g"].append(out[1])
        return out

    def inputs(*a, **k):
        c = case_inputs(*a, **k)
        seen["gauge"].append(c.gauge)
        return c

    m._case_scales, m._case_inputs = scales, inputs
    m.norm_out.register_forward_hook(
        lambda mod, i, o: seen["h"].append(o.detach().clone())
    )
    return seen


def _run(m, meshes, gd, q=None):
    with torch.no_grad():
        return outputs(
            m(
                boundary_meshes=meshes,
                prediction_points=q,
                prediction_boundaries=None if q is not None else list(meshes),
                global_data=gd,
            )
        )


@pytest.mark.parametrize(
    "s, c, r",
    [
        (3.7, 0.29, 1.0),
        (0.05, 12.0, 830.0),
        (1e-6, 12.0, 830.0),
        (1e-9, 12.0, 830.0),
        (1e6, 0.29, 1.0),
        (1e9, 0.29, 1.0),
        (1e12, 0.29, 1.0),
    ],
)
@pytest.mark.parametrize("interior", [False, True], ids=["surface", "interior"])
def test_similarity_leaves_the_internals_and_scales_the_outputs(s, c, r, interior):
    """(x, v, rho, k, omega, p) -> (s x, c v, r rho, c^2 k, (c/s) omega, r c^2 p), with
    a rotation and re_z unchanged: L -> s L, U -> c U, g rotates, every internal state is
    unchanged, and every output is scaled by its units (and rotated by its rank), in
    either task (the surface task predicting on every condition's cells). At
    s = 1e-6 the cells' measures are about 1e-12: the floor on the measure is
    relative to the case's total, so it binds at no scale. At s = 1e9 and 1e12 the
    winding number that signs the prediction points' distance takes its float32
    moments in a power-of-two unit of the surface, so they do not overflow."""
    m = _model(interior=interior)
    seen = _spy(m)
    meshes, gd, q = _case()
    q = q if interior else None
    R = orthogonal(3, 4)
    base = _run(m, meshes, gd, q)
    moved = _run(m, *_similar(m, meshes, gd, q, SCHEMAS, s, c, r, R))
    a, b = seen["scales"]
    torch.testing.assert_close(b["L"], s * a["L"], atol=0, rtol=1e-12)
    torch.testing.assert_close(b["U"], c * a["U"], atol=0, rtol=1e-12)
    torch.testing.assert_close(b["rho"], r * a["rho"], atol=0, rtol=1e-12)
    torch.testing.assert_close(seen["g"][1], seen["g"][0] @ R.T, atol=1e-12, rtol=0)
    torch.testing.assert_close(seen["h"][1], seen["h"][0], atol=1e-9, rtol=1e-9)
    for name, spec in m.output_schema.items():
        expected = law(base[name], R, spec.rank) * _factor(
            OUTPUTS[name].get("units", {}), s, c, r
        )
        torch.testing.assert_close(
            moved[name], expected, atol=1e-10 * expected.abs().max(), rtol=1e-9
        )


def test_the_source_s_condition_can_have_any_name():
    """The code names no condition: the same physics with the source on a condition
    named ``inlet`` or ``zz`` (another canonical position) gives the same outputs for
    the same parameters."""
    renamed = {("zz" if k == "inlet" else k): v for k, v in SCHEMAS.items()}
    a, b = _model(), _model(schemas=renamed)
    state = {
        k.replace("boundary_conditions.inlet.", "boundary_conditions.zz."): v
        for k, v in a.state_dict().items()
    }
    b.load_state_dict(state)
    meshes, gd, q = _case()
    out_a = _run(a, meshes, gd, q)
    out_b = _run(
        b, {("zz" if k == "inlet" else k): v for k, v in meshes.items()}, gd, q
    )
    for name in out_a:
        torch.testing.assert_close(out_b[name], out_a[name], atol=1e-10, rtol=1e-10)


def _full_length(meshes) -> torch.Tensor:
    """The exact RMS pairwise distance of the meshes' piecewise-flat cells."""
    names = sorted(meshes)
    return rms_pairwise_distance(
        torch.cat([meshes[k].cell_centroids for k in names])[None],
        torch.cat([cell_measures(meshes[k]) for k in names])[None],
        torch.cat([cell_second_moments(meshes[k]) for k in names])[None],
    )


def test_scales_are_independent_of_the_draw_and_of_refinement():
    """U, g and L are computed on the full-resolution boundary before the token
    draw: two training draws (different starts) of 60 of the 190 cells, and a
    subdivision of the source condition's cells (each triangle into four, its data
    carried), give the same scales; L is the full mesh's exact one."""
    m = _model(n_boundary_tokens=60).train()
    seen = _spy(m)
    meshes, gd, q = _case()
    for seed in (0, 1):
        torch.manual_seed(seed)
        _run(m, meshes, gd, q)
    _run(
        m,
        {**meshes, "inlet": meshes["inlet"].subdivide(levels=1, filter="linear")},
        gd,
        q,
    )
    for later, g in zip(seen["scales"][1:], seen["g"][1:]):
        for key in ("L", "U"):
            torch.testing.assert_close(
                later[key], seen["scales"][0][key], atol=1e-13, rtol=1e-13
            )
        torch.testing.assert_close(g, seen["g"][0], atol=1e-13, rtol=1e-13)
    assert torch.equal(seen["gauge"][0], seen["gauge"][1])  # bitwise across draws
    for gauge in seen["gauge"]:
        torch.testing.assert_close(gauge, _full_length(meshes), atol=0, rtol=1e-13)
    assert seen["h"][0].shape[1] == 25 and not torch.equal(
        seen["h"][0], seen["h"][1]
    )  # the draws differ


@pytest.mark.parametrize("dim", [2, 3])
def test_the_length_unit_is_exact_under_subdivision(dim):
    """L is the RMS distance between two points of the piecewise-flat boundary, each
    cell with its own second moment: subdividing every cell (segments in two in two
    dimensions, triangles in four in three) at three levels leaves it unchanged to
    roundoff, where the centroids alone converge only as h^2 (a 4 % change here)."""
    from test.experimental.models.isla._geometry_cases import (
        _polygon,
        _sphere,
    )

    if dim == 2:
        t = torch.linspace(0, 2 * math.pi, 13, dtype=D)[:-1]
        mesh = _polygon(torch.stack([2.0 * t.cos(), 0.7 * t.sin()], -1))
    else:
        mesh = _sphere(1)
        mesh = Mesh(
            points=mesh.points * torch.tensor([2.0, 1.0, 0.5], dtype=D),
            cells=mesh.cells,
        )
    torch.manual_seed(0)
    m = (
        ISLA(
            output_schema={"p": {"rank": 0}},
            boundary_schemas={"wall": {}},
            interior=True,
            n_spatial_dims=dim,
            hidden=16,
            n_layers=1,
            n_slices=4,
        )
        .double()
        .eval()
    )
    seen = []
    inner = m._case_inputs

    def spy(*a, **k):
        c = inner(*a, **k)
        seen.append(c.gauge)
        return c

    m._case_inputs = spy
    q = torch.zeros(3, dim, dtype=D)
    lengths, centroid_only = [], []
    for _ in range(4):
        with torch.no_grad():
            m(boundary_meshes={"wall": mesh}, prediction_points=q)
        lengths.append(seen[-1].reshape(()))
        centroid_only.append(
            rms_pairwise_distance(
                mesh.cell_centroids[None], cell_measures(mesh)[None]
            ).reshape(())
        )
        mesh = mesh.subdivide(levels=1, filter="linear")
    for L in lengths[1:]:
        torch.testing.assert_close(L, lengths[0], atol=0, rtol=1e-12)
    assert abs(float(centroid_only[0] / lengths[0]) - 1) > 1e-3


@pytest.mark.parametrize("config", ["default", "dtt"])
def test_declaring_nothing_is_the_release_model(config):
    """Keys that declare nothing (``units: {}``) leave the
    model the unscaled one: bitwise equal to the release configuration with the same
    parameters, and equal to the release checkpoint's reference outputs at their
    tolerance (those outputs are bitwise on the recording architecture and differ at
    the last bits across CPU architectures, BLAS reduction order)."""
    from test.experimental.models.isla._release_cases import (
        CONFIGS,
        _instantiate_model,
        _make_inputs,
    )

    cfg = dict(CONFIGS[config])
    cfg["output_schema"] = {"p": {"rank": 0, "units": {}}, "v": {"rank": 1}}
    model, release = _instantiate_model(**cfg), _instantiate_model(**CONFIGS[config])
    assert not model._scaled and not model._fp32_head
    inputs = _make_inputs("cpu", query=(config != "default"))
    with torch.no_grad():
        out, same = outputs(model(**inputs)), outputs(release(**inputs))
    assert out.keys() == same.keys()
    for k in out:
        assert torch.equal(out[k], same[k]), k


def test_the_scales_take_their_hand_values():
    """U = sqrt(sum a |v|^2 / sum a) pooled over two sources, g = sum a v / |sum a v|,
    rho the density's; g is the last global direction."""
    schemas = {
        "a": {"v": {"rank": 1, "units": {"U": 1}, "defines": "U"}},
        "b": {"w": {"rank": 1, "units": {"U": 1}, "defines": "U"}},
    }
    m = _model(
        schemas=schemas, outputs_={"p": {"rank": 0, "units": {"U": 2, "rho": 1}}}
    )
    assert (
        m._n_global_vectors == 1 and m._n_global_scalars == 1
    )  # g, and re_z (the density role is consumed)
    seen = _spy(m)
    meshes, gd, q = _case(schemas=schemas, sizes={"a": 7, "b": 5})
    _run(m, meshes, gd, q)
    sc = seen["scales"][0]
    a = [cell_measures(meshes[k]) for k in "ab"]
    v = [meshes["a"].cell_data["v"], meshes["b"].cell_data["w"]]
    sum_a = a[0].sum() + a[1].sum()
    U = (
        ((a[0] * v[0].square().sum(-1)).sum() + (a[1] * v[1].square().sum(-1)).sum())
        / sum_a
    ).sqrt()
    m1 = (a[0][:, None] * v[0]).sum(0) + (a[1][:, None] * v[1]).sum(0)
    pts = torch.cat([meshes[k].cell_centroids for k in "ab"])
    w = torch.cat(a) / sum_a
    centre = (w[:, None] * pts).sum(0)
    J = torch.cat(
        [cell_second_moments(meshes[k]) for k in "ab"]
    )  # each triangle's own second moment
    L = (2 * (w * ((pts - centre).square().sum(-1) + J)).sum()).sqrt()
    torch.testing.assert_close(sc["L"], L, atol=0, rtol=1e-13)
    torch.testing.assert_close(sc["U"], U, atol=0, rtol=1e-13)
    torch.testing.assert_close(seen["g"][0][0, 0], m1 / m1.norm(), atol=1e-14, rtol=0)
    torch.testing.assert_close(sc["rho"], torch.tensor(RHO, dtype=D), atol=0, rtol=0)


def test_an_absent_source_is_refused_under_torch_compile_too(monkeypatch):
    """Whether the source's condition is given is structural, so its absence is the
    same ValueError in a traced forward (``is_tracing`` patched to True), not an
    error from inside the scales."""
    from physicsnemo.experimental.models.isla import model as model_mod

    m = _model(interior=False)
    meshes, gd, _ = _case()
    monkeypatch.setattr(model_mod, "is_tracing", lambda: True)
    with pytest.raises(ValueError, match="absent from this case"):
        _run(m, {k: v for k, v in meshes.items() if k != "inlet"}, gd)


def test_an_absent_or_zero_source_and_bad_declarations_are_refused():
    """A case without the source's condition, a source that is zero everywhere or whose
    velocities cancel, a non-positive density and the malformed declarations are each
    refused with an error that names them."""
    m = _model()
    meshes, gd, q = _case()
    with pytest.raises(ValueError, match="absent from this case"):
        _run(m, {k: v for k, v in meshes.items() if k != "inlet"}, gd, q)
    inlet = meshes["inlet"]
    zero = inlet.cell_data.clone()
    zero["bc_velocity"] = torch.zeros_like(zero["bc_velocity"])
    with pytest.raises(ValueError, match="zero everywhere"):
        _run(m, {**meshes, "inlet": inlet.with_data(cell_data=zero)}, gd, q)
    ### two equal cells with opposite velocities: U is defined, g is not
    pts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=D)
    nrm = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=D)
    vel = torch.tensor([[1.0, 2.0, 0.0], [-1.0, -2.0, 0.0]], dtype=D)
    data = {
        "bc_velocity": vel,
        "bc_tke": torch.ones(2, dtype=D),
        "bc_omega": torch.ones(2, dtype=D),
        "bc_gamma": torch.ones(2, dtype=D),
    }
    with pytest.raises(ValueError, match="cancel"):
        _run(
            m,
            {**meshes, "inlet": soup(pts, nrm, torch.ones(2, dtype=D), cell_data=data)},
            gd,
            q,
        )
    with pytest.raises(ValueError, match="'physics.density'\\) must be positive"):
        _run(m, meshes, {**gd, "physics.density": torch.tensor(0.0, dtype=D)}, q)
    bad = [
        (
            dict(globals_={**GLOBALS, "U_inf": {"rank": 1, "defines": "U"}}),
            "not both",
        ),
        (
            dict(
                schemas={
                    **SCHEMAS,
                    "inlet": {"bc_velocity": {"rank": 1, "units": {"U": 1}}},
                }
            ),
            r"\['U'\], which no field defines",
        ),
        (
            dict(globals_={"re_z": GLOBALS["re_z"]}),
            r"\['rho'\], which no field defines",
        ),
        (
            dict(outputs_={"p": {"rank": 0, "units": {"T": 1}}}),
            "which no field defines",
        ),
        (
            dict(schemas={**SCHEMAS, "wall": {"T": {"rank": 2, "defines": "E"}}}),
            "got rank 2",
        ),
        (
            dict(globals_={**GLOBALS, "s": {"rank": 0, "defines": "L"}}),
            "built-in length",
        ),
        (
            dict(globals_={**GLOBALS, "s": {"rank": 0, "defines": True}}),
            "non-empty string",
        ),
        (
            dict(schemas={**SCHEMAS, "wall": {"w": {"rank": 1, "defines": "V"}}}),
            "at most one scale is defined by rank-1 fields",
        ),
        (
            dict(globals_={**GLOBALS, "s": {"rank": 0, "defines": "U"}}),
            "defined by a rank-1 field and by the rank-0 global",
        ),
        (dict(outputs_={"p": {"rank": 0, "defines": "U"}}), "an output cannot"),
        (
            dict(schemas={**SCHEMAS, "wall": {"k": {"rank": 0, "defines": "U"}}}),
            "rank-1 field",
        ),
        (
            dict(schemas={**SCHEMAS, "wall": {"k": {"rank": 0, "defines": "rho"}}}),
            "rank-0 global",
        ),
        (
            dict(globals_={**GLOBALS, "rho2": {"rank": 0, "defines": "rho"}}),
            "two fields",
        ),
        (
            dict(
                globals_={
                    **GLOBALS,
                    "physics.density": {
                        "rank": 0,
                        "defines": "rho",
                        "units": {"U": 1},
                    },
                }
            ),
            "takes no units",
        ),
        (
            dict(output_stats={"velocity": {"mean": [1.0, 0.0, 0.0], "std": 2.0}}),
            "must be 0",
        ),
        (
            dict(output_stats={"other": {"mean": 0.0, "std": 2.0}}),
            "not a declared output",
        ),
        (dict(outputs_={"p": {"rank": 0, "scale": 0.0}}), "positive finite"),
        (
            dict(
                globals_={
                    **GLOBALS,
                    "physics.density": {"rank": 0, "defines": "rho", "scale": 2.0},
                }
            ),
            "no units and no scale",
        ),
        (
            dict(globals_={**GLOBALS, "d": {"rank": 1, "units": {"L": 1}}}),
            "direction only",
        ),
    ]
    for kw, match in bad:
        with pytest.raises((ValueError, TypeError), match=match):
            _model(**kw)


def test_malformed_scale_keys_and_a_zero_global_source_are_refused():
    """The declarations' ISLA-local keys are checked at construction with the field
    named (the units a mapping of scale names to finite exponents, ``output_stats`` a
    mapping with one positive std and a scalar's one mean, a rank-1 global that
    defines a scale taking no constant scale, one rank-1 global source at most), and
    a global source that is zero in a case leaves its scale undefined."""
    velocity = {"rank": 1, "units": {"U": 1}}
    bad = [
        (
            TypeError,
            dict(outputs_={"p": {"rank": 0, "units": ["U"]}}),
            r"output_schema\['p'\]: units must be a mapping",
        ),
        (
            TypeError,
            dict(outputs_={"p": {"rank": 0, "units": {1: 1}}}),
            "units names a scale by a string; got 1",
        ),
        (
            TypeError,
            dict(outputs_={"p": {"rank": 0, "units": {"U": float("inf")}}}),
            "the exponent of 'U' must be a finite number",
        ),
        (
            TypeError,
            dict(outputs_={"p": {"rank": 0, "units": {"U": True}}}),
            "the exponent of 'U' must be a finite number",
        ),
        (
            TypeError,
            dict(output_stats=[("velocity", 1.0)]),
            "output_stats must be a mapping",
        ),
        (
            ValueError,
            dict(output_stats={"pressure": {"mean": 0.0, "std": 0.0}}),
            r"output_stats\['pressure'\]: std must be one positive number",
        ),
        (
            ValueError,
            dict(output_stats={"pressure": {"mean": [0.0, 1.0], "std": 1.0}}),
            "a scalar's mean is one number",
        ),
    ]
    for error, kw, match in bad:
        with pytest.raises(error, match=match):
            _model(**kw)
    ### a rank-1 global source, alone (the inlet's velocity defines nothing)
    plain = {**SCHEMAS, "inlet": {"bc_gamma": {"rank": 0}}, "outlet": {}}
    sources = [
        (
            {"U_inf": {**velocity, "defines": "U", "scale": 2.0}},
            "a rank-1 global's magnitude is the scale 'U'; it takes no scale",
        ),
        (
            {"U_a": {**velocity, "defines": "U"}, "U_b": {**velocity, "defines": "U"}},
            r"at most one rank-1 global defines the scale 'U'; got \['U_a', 'U_b'\]",
        ),
    ]
    for extra, match in sources:
        with pytest.raises(ValueError, match=match):
            _model(schemas=plain, globals_={**GLOBALS, **extra})
    m = _model(
        schemas=plain, globals_={**GLOBALS, "U_inf": {**velocity, "defines": "U"}}
    )
    meshes, gd, q = _case(schemas=plain)
    _run(m, meshes, {**gd, "U_inf": torch.tensor([3.0, 0.0, 0.0], dtype=D)}, q)
    with pytest.raises(
        ValueError,
        match="the source 'U_inf' of the scale 'U' is zero: the scale is undefined",
    ):
        _run(m, meshes, {**gd, "U_inf": torch.zeros(3, dtype=D)}, q)


def test_a_rank_1_field_that_defines_a_scale_has_its_units():
    """A rank-1 field that defines U has the units {U: 1}: declared without units it
    gives the outputs of the declaration with them, bitwise, so similarity holds; other
    units are refused, on a boundary field or a global source."""
    meshes, gd, q = _case()
    base = _run(_model(), meshes, gd, q)

    def velocity(**spec):
        return {
            **SCHEMAS,
            "inlet": {
                **SCHEMAS["inlet"],
                "bc_velocity": {"rank": 1, **spec, "defines": "U"},
            },
        }

    got = _run(_model(schemas=velocity()), meshes, gd, q)
    assert all(torch.equal(got[k], base[k]) for k in base)
    plain = {**SCHEMAS, "inlet": {"bc_gamma": {"rank": 0}}, "outlet": {}}
    bad = [
        dict(schemas=velocity(units=u)) for u in ({"U": 2}, {"U": 1, "L": 1}, {"U": 0})
    ]
    bad.append(
        dict(
            schemas=plain,
            globals_={
                **GLOBALS,
                "U_inf": {"rank": 1, "units": {"L": 1}, "defines": "U"},
            },
        )
    )
    for kw in bad:
        with pytest.raises(
            ValueError, match="defines the scale 'U' has the units \\{'U': 1\\}"
        ):
            _model(**kw)


def test_the_scale_names_are_free():
    """The code names no scale but L: the same declaration with U and rho renamed S
    and m gives the same outputs for the same parameters, bitwise."""

    def renamed(spec):
        if isinstance(spec, dict):
            out = {}
            for k, v in spec.items():
                if k == "units":
                    v = {{"U": "S", "rho": "m"}.get(n, n): e for n, e in v.items()}
                elif k == "defines":
                    v = {"U": "S", "rho": "m"}[v]
                else:
                    v = renamed(v)
                out[k] = v
            return out
        return spec

    a = _model()
    b = _model(
        schemas=renamed(SCHEMAS), globals_=renamed(GLOBALS), outputs_=renamed(OUTPUTS)
    )
    b.load_state_dict(a.state_dict())
    assert b._rms_scale == "S" and b._defined_values == {"m": "physics.density"}
    meshes, gd, q = _case()
    ya, yb = _run(a, meshes, gd, q), _run(b, meshes, gd, q)
    for name in ya:
        assert torch.equal(yb[name], ya[name]), name


def test_output_stats_are_the_dataset_s_normalize_mesh_fields():
    """With the dataset's NormalizeMeshFields stats the model returns its physical
    outputs standardized by them (the targets' space, vector mean 0), the inverse gives
    the physical outputs back, and the checkpoint arguments keep the merged declaration."""
    norm = NormalizeMeshFields(
        association="point_data",
        fields={
            "velocity": {"type": "vector", "mean": [0, 0, 0], "std": 11.6},
            "pressure": {"type": "scalar", "mean": 3.0, "std": 1129.0},
            "nut": {"type": "scalar", "mean": 0, "std": 0.61},
        },
    )
    physical, standardized = _model(), _model(output_stats=norm.stats)
    meshes, gd, q = _case()
    y = _run(physical, meshes, gd, q)
    ystd = _run(standardized, meshes, gd, q)
    expected = norm(Mesh(points=q, point_data=y)).point_data
    for name in y:
        torch.testing.assert_close(
            ystd[name].to(expected[name].dtype), expected[name], atol=1e-6, rtol=1e-5
        )
    back = norm.inverse_td(
        Mesh(points=q, point_data={k: v.float() for k, v in ystd.items()}).point_data
    )
    for name in y:
        torch.testing.assert_close(
            back[name].double(), y[name], atol=1e-6 * y[name].abs().max(), rtol=1e-5
        )
    args = standardized._args["__args__"]
    assert args["output_stats"]["pressure"] == {"mean": 3.0, "std": 1129.0}
    assert args["boundary_schemas"]["inlet"]["bc_velocity"] == {
        "rank": 1,
        "units": {"U": 1},
        "defines": "U",
    }
    assert args["global_schema"]["physics.density"] == {"rank": 0, "defines": "rho"}
    rebuilt = ISLA(**args)
    assert (
        rebuilt._boundary_sources == [("inlet", "bc_velocity")]
        and rebuilt.output_stats == standardized.output_stats
    )


def test_a_global_source_adds_no_direction():
    """An airfoil in a freestream: the global U_inf is the source, so U = |U_inf| and g is U_inf's own
    direction (K unchanged); x -> s x, U_inf -> c U_inf with re_z unchanged leaves the
    dimensionless outputs (v / U, Cp) unchanged and scales nut by s c."""
    globals_ = {
        "U_inf": {"rank": 1, "units": {"U": 1}, "defines": "U"},
        "re_z": {"rank": 0},
    }
    outs = {
        "velocity": {"rank": 1},
        "pressure": {"rank": 0},
        "nut": {"rank": 0, "units": {"U": 1, "L": 1}},
    }
    torch.manual_seed(0)
    m = (
        ISLA(
            output_schema=outs,
            boundary_schemas={"airfoil": {}},
            global_schema=globals_,
            interior=True,
            n_spatial_dims=2,
            hidden=32,
            n_layers=2,
            n_slices=8,
        )
        .double()
        .eval()
    )
    m.tree_read_theta = 0.0  # the exact boundary read, as in _model
    m = live(m)
    assert m._n_global_vectors == 1 and m._n_global_scalars == 1
    seen = _spy(m)
    pts, nrm, w, _ = cloud(80, d=2, seed=3)
    mesh = soup(pts, nrm, w)
    q = torch.randn(20, 2, dtype=D, generator=torch.Generator().manual_seed(1)) * 2
    u_inf = torch.tensor([40.0, 3.0], dtype=D)
    gd = {"U_inf": u_inf, "re_z": torch.tensor(RE_Z, dtype=D)}
    base = _run(m, {"airfoil": mesh}, gd, q)
    torch.testing.assert_close(seen["scales"][0]["U"], u_inf.norm(), atol=0, rtol=1e-15)
    assert seen["g"][0] is None
    s, c = 0.3, 2.5
    moved = _run(
        m,
        {"airfoil": transform(mesh, scale=s)},
        {"U_inf": c * u_inf, "re_z": gd["re_z"]},
        s * q,
    )
    torch.testing.assert_close(
        moved["velocity"], base["velocity"], atol=1e-10, rtol=1e-9
    )
    torch.testing.assert_close(
        moved["pressure"], base["pressure"], atol=1e-10, rtol=1e-9
    )
    torch.testing.assert_close(
        moved["nut"],
        s * c * base["nut"],
        atol=1e-10 * base["nut"].abs().max(),
        rtol=1e-9,
    )


def test_an_output_s_scale_multiplies_the_head_s_prediction():
    """``scale: c`` is the typical size of the dimensionless output: the same
    parameters give c times the output, before output_stats, the vector covariant as
    before; the checkpoint arguments keep it with the declaration."""
    cs = (0.5, 0.2, 1e-4, 3e3)
    scaled = {name: {**f, "scale": c} for (name, f), c in zip(OUTPUTS.items(), cs)}
    a = _model()
    b = _model(outputs_=scaled, output_stats={"pressure": {"mean": 1.0, "std": 4.0}})
    b.load_state_dict(a.state_dict())
    meshes, gd, q = _case()
    ya, yb = _run(a, meshes, gd, q), _run(b, meshes, gd, q)
    for name, c in zip(OUTPUTS, cs):
        expected = c * ya[name] if name != "pressure" else (c * ya[name] - 1.0) / 4.0
        torch.testing.assert_close(
            yb[name], expected, atol=1e-12 * expected.abs().max(), rtol=1e-12
        )
    assert b._args["__args__"]["output_schema"]["nut"] == {
        "rank": 0,
        "units": {"U": 1, "L": 1},
        "scale": 1e-4,
    }


def test_an_input_s_scale_divides_it_after_its_units():
    """``scale: c`` on an input is the typical size of its dimensionless value: the
    model reads x / (c prod scale^exp), the same as the unscaled declaration given
    x / c, for a boundary scalar, a boundary vector and a rank-0 global."""
    scaled = {
        **SCHEMAS,
        "inlet": {
            **SCHEMAS["inlet"],
            "bc_omega": {**SCHEMAS["inlet"]["bc_omega"], "scale": 400.0},
            "bc_gamma": {"rank": 0, "scale": 0.5},
        },
        "outlet": {"bc_pressure": {**SCHEMAS["outlet"]["bc_pressure"], "scale": 0.1}},
    }
    globals_ = {**GLOBALS, "phi": {"rank": 0, "units": {"U": 1}, "scale": 3.0}}
    plain_globals = {**GLOBALS, "phi": {"rank": 0, "units": {"U": 1}}}
    a, b = _model(globals_=plain_globals), _model(schemas=scaled, globals_=globals_)
    b.load_state_dict(a.state_dict())
    meshes, gd, q = _case()
    gd = {**gd, "phi": torch.tensor(12.0, dtype=D)}
    divided = {}
    for name, mesh in meshes.items():
        data = mesh.cell_data.clone()
        for f, spec in scaled[name].items():
            if "scale" in spec:
                data[f] = data[f] / spec["scale"]
        divided[name] = mesh.with_data(cell_data=data)
    ya = _run(a, divided, {**gd, "phi": gd["phi"] / 3.0}, q)
    yb = _run(b, meshes, gd, q)
    for name in ya:
        torch.testing.assert_close(
            yb[name], ya[name], atol=1e-12 * ya[name].abs().max(), rtol=1e-11
        )
    assert b._args["__args__"]["boundary_schemas"]["inlet"]["bc_gamma"] == {
        "rank": 0,
        "scale": 0.5,
    }


### ---------------------------------------------------------------------------
### case_scales
### ---------------------------------------------------------------------------


def _scales_case(d: int, source: str, seed: int = 0):
    """A float32 case in ``d`` dimensions: ``inlet``, ``outlet`` and ``wall`` triangle
    or segment soups, the inlet's and the outlet's rank-1 velocities (``source``
    ``"boundary"``) or a global vector (``"global"``) defining U, and a rank-0 global
    defining rho, both globals in float64 (rounded to the meshes' float32 by the
    model). Returns the boundary schemas, the global schema, the meshes, the globals
    and the ``defines`` of :func:`case_scales` (the pairs listed out of ISLA's order)."""
    vel = {"rank": 1, "units": {"U": 1}, "defines": "U"}
    if source == "boundary":
        schemas = {"inlet": {"bc_velocity": vel}, "outlet": {"bc_u": vel}, "wall": {}}
        globals_ = {"rho": {"rank": 0, "defines": "rho"}}
        defines = {"U": [("outlet", "bc_u"), ("inlet", "bc_velocity")], "rho": "rho"}
    else:
        schemas = {"inlet": {}, "outlet": {}, "wall": {}}
        globals_ = {"U_inf": vel, "rho": {"rank": 0, "defines": "rho"}}
        defines = {"U": "U_inf", "rho": "rho"}
    g = torch.Generator().manual_seed(seed + 7)
    meshes = {}
    for k, (name, n) in enumerate({"inlet": 9, "outlet": 6, "wall": 23}.items()):
        pts, nrm, w, _ = cloud(n, d=d, seed=seed + k, dtype=torch.float32)
        data = {
            f: torch.randn(n, d, generator=g) + 4.0 for f in schemas[name]
        }  # rank-1 cell fields
        meshes[name] = soup(pts + 3.0 * k, nrm, w, cell_data=data or None)
    gd = {
        "U_inf": torch.randn(d, generator=g, dtype=D) * 9.0,
        "rho": torch.tensor(1.18, dtype=D),
    }
    return schemas, globals_, meshes, gd, defines


@pytest.mark.parametrize("source", ["boundary", "boundary_absent", "global"])
@pytest.mark.parametrize("interior", [False, True])
@pytest.mark.parametrize("d", [2, 3])
def test_case_scales_are_the_model_s_bitwise(d, interior, source):
    """On a float32 case, :func:`case_scales` returns the scales a forward pass
    computes (``ISLA._case_scales``) bitwise, as float64 scalars: L, U from pooled
    boundary sources (an inlet's velocity and an outlet's, listed out of order; or the
    inlet's alone, the case given without its outlet, whose pair both skip) or from a
    global vector, and rho from a rank-0 global; surface and interior, 2-D and 3-D."""
    schemas, globals_, meshes, gd, defines = _scales_case(
        d, "global" if source == "global" else "boundary"
    )
    if source == "boundary_absent":
        del meshes["outlet"]
    torch.manual_seed(0)
    m = ISLA(
        output_schema={"p": {"rank": 0, "units": {"U": 2, "rho": 1}}},
        boundary_schemas=schemas,
        global_schema=globals_,
        interior=interior,
        n_boundary_tokens=16 if interior else 200_000,
        n_spatial_dims=d,
        hidden=16,
        n_layers=1,
        n_slices=4,
    ).eval()
    seen = _spy(m)
    q = torch.randn(5, d, generator=torch.Generator().manual_seed(3)) * 4
    _run(m, meshes, {k: gd[k] for k in globals_}, q if interior else None)
    internal = seen["scales"][0]
    got = case_scales(meshes, defines=defines, global_data=gd)
    assert set(got) == {"L", "U", "rho"}
    for k, v in got.items():
        assert v.dtype == D and v.shape == () and torch.equal(v, internal[k]), k
    assert torch.equal(got["L"], seen["gauge"][0].reshape(()).double())


@pytest.mark.parametrize("interior", [False, True])
def test_a_scale_named_g_is_an_ordinary_scale(interior):
    """A scale may take any name but ``"L"``: one named ``"g"`` (gravity, say), the
    name of the direction the rank-1 sources derive, gives the outputs of the same
    scale named ``"U"`` bitwise, and :func:`case_scales` gives the model's scales."""

    def renamed(spec):
        if not isinstance(spec, dict):
            return spec
        out = {k: renamed(v) for k, v in spec.items()}
        if "units" in spec:
            out["units"] = {
                ("g" if n == "U" else n): e for n, e in spec["units"].items()
            }
        if spec.get("defines") == "U":
            out["defines"] = "g"
        return out

    meshes, gd, q = _case()
    q = q if interior else None
    base = _run(_model(interior), meshes, gd, q)
    m = _model(interior, schemas=renamed(SCHEMAS), outputs_=renamed(OUTPUTS))
    seen = _spy(m)
    got = _run(m, meshes, gd, q)
    assert all(torch.equal(got[k], base[k]) for k in base)
    defines = {"g": [("inlet", "bc_velocity")], "rho": "physics.density"}
    assert case_scales(meshes, defines=defines, global_data=gd) == seen["scales"][0]


@pytest.mark.parametrize("improper", [False, True])
@pytest.mark.parametrize("d", [2, 3])
def test_case_scales_are_invariant_and_l_scales_with_the_coordinates(d, improper):
    """Every scale is invariant under an orthogonal map (a rotation or a reflection)
    and a translation of the case, its vectors mapped with it; under x -> s x the
    length L becomes s L and the other scales stay."""
    _, _, meshes, gd, defines = _scales_case(d, "boundary")
    defines["V"] = "U_inf"
    base = case_scales(meshes, defines=defines, global_data=gd)
    R = orthogonal(d, seed=11, improper=improper)
    shift = torch.arange(1.0, d + 1, dtype=D) * 50.0
    moved = {}
    for k, m in meshes.items():
        t = transform(m, R=R, shift=shift)
        data = t.cell_data.clone()
        for f in m.cell_data.keys():
            if f in ("bc_velocity", "bc_u"):
                data[f] = (m.cell_data[f].double() @ R.T).to(m.cell_data[f].dtype)
        moved[k] = t.with_data(cell_data=data)
    gd_moved = {**gd, "U_inf": gd["U_inf"] @ R.T}
    out = case_scales(moved, defines=defines, global_data=gd_moved)
    for k in base:
        torch.testing.assert_close(out[k], base[k], atol=0, rtol=1e-5)
    for s in (1e-3, 7.0, 1e4):
        grown = case_scales(
            {k: transform(m, scale=s) for k, m in meshes.items()},
            defines=defines,
            global_data=gd,
        )
        torch.testing.assert_close(grown["L"], s * base["L"], atol=0, rtol=1e-5)
        for k in ("U", "V", "rho"):
            torch.testing.assert_close(grown[k], base[k], atol=0, rtol=1e-5)


def test_case_scales_refuse_an_undefined_scale():
    """No cells, no given boundary among a unit's pairs, an absent cell field or global
    field, a field or global of the wrong shape, sources zero everywhere, a zero global
    vector, a non-positive rank-0 global, an empty source list, a source that is not a
    sequence of string pairs and a field defining L are refused, each naming the unit
    and the input."""
    _, _, meshes, gd, _ = _scales_case(3, "boundary")
    inlet = meshes["inlet"]
    empty = Mesh(points=torch.zeros(0, 3), cells=torch.zeros(0, 3, dtype=torch.long))
    with pytest.raises(ValueError, match="'L' is undefined.*no cells"):
        case_scales({"inlet": empty})
    cases = [
        (
            {"U": [("outlet", "bc_u")]},
            {"inlet": inlet},
            None,
            r"defines\['U'\]: no boundary of the sources \[\('outlet', 'bc_u'\)\] is given",
        ),
        (
            {"U": ("inlet", "bc_velocity")},
            {"inlet": inlet},
            None,
            r"defines\['U'\] must be a global field's name or a sequence of "
            r"\(boundary, field\) string pairs",
        ),
        (
            {"U": [("inlet", 3)]},
            {"inlet": inlet},
            None,
            r"defines\['U'\] must be .* pairs",
        ),
        (
            {"U": "U_inf"},
            {"inlet": inlet},
            {"U_inf": torch.ones(2, 3)},
            r"defines\['U'\]: global_data\['U_inf'\] must hold one value \(rank 0\) "
            r"or 3 .* got shape \(2, 3\)",
        ),
        (
            {"rho": "rho"},
            {"inlet": inlet},
            {"rho": torch.tensor([1.2, 1.3])},
            r"defines\['rho'\]: global_data\['rho'\] must hold one value",
        ),
        (
            {"U": [("inlet", "bc_tke")]},
            {"inlet": inlet},
            None,
            r"defines\['U'\].*no cell field 'bc_tke'",
        ),
        ({"U": []}, {"inlet": inlet}, None, r"defines\['U'\] names no"),
        ({"L": "rho"}, {"inlet": inlet}, gd, r"defines\['L'\].*built-in"),
        (
            {"U": "U_inf"},
            {"inlet": inlet},
            None,
            r"defines\['U'\]: global_data\['U_inf'\] is not given",
        ),
        (
            {"U": "U_inf"},
            {"inlet": inlet},
            {"U_inf": torch.zeros(3)},
            r"defines\['U'\]: global_data\['U_inf'\] is zero",
        ),
        (
            {"rho": "rho"},
            {"inlet": inlet},
            {"rho": torch.tensor(-1.0)},
            r"defines\['rho'\]: global_data\['rho'\] must be positive",
        ),
    ]
    for defines, ms, g, match in cases:
        with pytest.raises(ValueError, match=match):
            case_scales(ms, defines=defines, global_data=g)
    scalar = inlet.with_data(cell_data={"bc_velocity": torch.ones(inlet.n_cells)})
    with pytest.raises(ValueError, match=r"defines\['U'\].*rank-1 cell field"):
        case_scales({"inlet": scalar}, defines={"U": [("inlet", "bc_velocity")]})
    zero = inlet.with_data(cell_data={"bc_velocity": torch.zeros(inlet.n_cells, 3)})
    with pytest.raises(ValueError, match=r"defines\['U'\].*zero everywhere"):
        case_scales({"inlet": zero}, defines={"U": [("inlet", "bc_velocity")]})


def test_case_scales_run_the_forward_pass_mesh_checks():
    """As ISLA's forward does, :func:`case_scales` refuses an empty mapping, a list, a
    value that is not a ``Mesh``, a mesh that is not a boundary (a tetrahedral
    ``Mesh[3, 3]``) and meshes of mixed dtypes, each naming the input."""
    _, _, meshes, _, _ = _scales_case(3, "boundary")
    inlet = meshes["inlet"]
    tets = Mesh(points=torch.eye(4, 3), cells=torch.tensor([[0, 1, 2, 3]]))
    wide = Mesh(points=inlet.points.double(), cells=inlet.cells)
    for value in ({}, [inlet]):
        with pytest.raises(
            ValueError, match="boundary_meshes must be a non-empty mapping"
        ):
            case_scales(value)
    with pytest.raises(
        TypeError, match=r"boundary_meshes\['inlet'\] must be a physicsnemo.mesh.Mesh"
    ):
        case_scales({"inlet": inlet.points})
    with pytest.raises(
        ValueError, match=r"boundary_meshes\['vol'\].*got a Mesh\[3, 3\]"
    ):
        case_scales({"inlet": inlet, "vol": tets})
    with pytest.raises(
        ValueError,
        match=r"boundary_meshes\['wide'\] has 3 spatial dimensions and torch.float64",
    ):
        case_scales({"inlet": inlet, "wide": wide})


def test_case_scales_read_a_global_by_dotted_name_or_nested():
    """A global field is found by its dotted name in a flat mapping, a nested mapping
    and a nested TensorDict alike, as ISLA's forward reads ``global_data``."""
    _, _, meshes, gd, _ = _scales_case(3, "global")
    v = gd["U_inf"]
    forms = [
        {"fs.U": v},
        {"fs": {"U": v}},
        TensorDict({"fs": {"U": v}}, batch_size=[]),
    ]
    got = [
        case_scales(meshes, defines={"U": "fs.U"}, global_data=g)["U"] for g in forms
    ]
    assert all(torch.equal(x, got[0]) for x in got)
    assert torch.equal(got[0], v.to(torch.float32).double().norm())
