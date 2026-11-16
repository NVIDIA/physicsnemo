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

"""ISLA's boundary conditions by name (``boundary_schemas``).

Each condition declares its own per-cell fields (an empty schema for a condition
without any); ``forward`` takes any non-empty subset of the declared names in any
order. The symmetry and remeshing contracts hold with every condition carrying its
own fields; a condition's identity is in its parameters (swapping two conditions'
names changes the outputs), not in a tag field; an absent condition is a condition
with no cells; and every condition's parameters receive a gradient in every pass.
"""

import pytest
import torch

from physicsnemo.experimental.models.isla import ISLA
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import EFFECTIVE_MEASURE_KEY
from test.experimental.models.isla._forced import in_another_thread_while_compiling
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
OUT = {"p": {"rank": 0}, "v": {"rank": 1}, "s": {"rank": 2, "symmetric": True}}
#: A set of conditions with different fields of every rank up to 2, and
#: two conditions without any.
SCHEMAS = {
    "inlet": {"u": {"rank": 1}, "k": {"rank": 0}},
    "outlet": {
        "p": {"rank": 0},
        "u_rev": {"rank": 1},
        "T": {"rank": 2, "symmetric": True},
    },
    "no_slip": {},
    "slip": {},
}
SIZES = {"inlet": 40, "outlet": 30, "no_slip": 120, "slip": 50}
#: The cells of each condition sit in their own region, so a condition is a part of
#: one boundary as in a dataset.
OFFSETS = {
    "inlet": (-6.0, 0, 0),
    "outlet": (6.0, 0, 0),
    "no_slip": (0, 0, 0),
    "slip": (0, 4.0, 0),
}


#: The same conditions with ISLA's own scales: the inlet's velocity is the source of U
#: and of the derived direction g, the fields and outputs carry units.
SCALED = {
    "inlet": {
        "u": {"rank": 1, "units": {"U": 1}, "defines": "U"},
        "k": {"rank": 0, "units": {"U": 2}},
    },
    "outlet": {
        "p": {"rank": 0, "units": {"U": 2}},
        "u_rev": {"rank": 1, "units": {"U": 1}},
        "T": {"rank": 2, "symmetric": True, "units": {"U": 2}},
    },
    "no_slip": {},
    "slip": {},
}
SCALED_OUT = {
    "p": {"rank": 0, "units": {"U": 2}},
    "v": {"rank": 1, "units": {"U": 1}},
    "s": {"rank": 2, "symmetric": True, "units": {"U": 1, "L": 1}},
}


def _interior(schemas):
    """The conditions without their rank-2 fields: the interior task takes
    no rank-2 boundary inputs (refused at construction), the surface task does."""
    return {
        k: {f: s for f, s in v.items() if s["rank"] < 2} for k, v in schemas.items()
    }


def _model(interior=False, schemas=SCHEMAS, output_schema=OUT, **kw):
    torch.manual_seed(0)
    return live(
        ISLA(
            output_schema=output_schema,
            boundary_schemas=_interior(schemas) if interior else schemas,
            global_schema={"g": {"rank": 1}, "re": {"rank": 0}},
            interior=interior,
            hidden=32,
            n_layers=2,
            n_slices=8,
            **kw,
        )
        .double()
        .eval()
    )


def _meshes(schemas=SCHEMAS, sizes=SIZES, seed=0) -> dict[str, Mesh]:
    g = torch.Generator().manual_seed(seed + 50)
    out = {}
    for k, (name, n) in enumerate(sizes.items()):
        pts, nrm, w, _ = cloud(n, seed=seed + k)
        data = {
            f: torch.randn(n, *(3,) * s["rank"], dtype=D, generator=g)
            for f, s in schemas[name].items()
        }
        if "T" in data:
            data["T"] = data["T"] + data["T"].transpose(-1, -2)
        out[name] = soup(
            pts + torch.tensor(OFFSETS.get(name, (0, 0, 0)), dtype=D),
            nrm,
            w,
            cell_data=data,
        )
    return out


GD = {"g": torch.tensor([0.6, 0.8, 0.0], dtype=D), "re": torch.tensor(0.4, dtype=D)}
Q = torch.randn(25, 3, dtype=D, generator=torch.Generator().manual_seed(9)) * 3


def _run(m, meshes, gd=GD, q=None):
    with torch.no_grad():
        return outputs(
            m(
                boundary_meshes=meshes,
                prediction_points=q,
                prediction_boundaries=None if q is not None else list(meshes),
                global_data=gd,
            )
        )


def _specs(m, name):
    return m.boundary_schemas[name]


@pytest.mark.parametrize("scaled", [False, True], ids=["unscaled", "scaled"])
@pytest.mark.parametrize("improper", [False, True], ids=["rotation", "reflection"])
@pytest.mark.parametrize("interior", [False, True], ids=["surface", "interior"])
def test_every_condition_s_fields_are_covariant(interior, improper, scaled):
    """Rotations and reflections with a translation: every condition's vector and
    tensor fields move with the geometry, and every output transforms by its rank;
    with ISLA's own scales the derived direction g moves with the source as well."""
    m = _model(
        interior, **(dict(schemas=SCALED, output_schema=SCALED_OUT) if scaled else {})
    )
    assert m._n_global_vectors == 1 + scaled
    meshes = _meshes(_interior(SCHEMAS) if interior else SCHEMAS)
    q = Q if interior else None
    base = _run(m, meshes, q=q)
    R = orthogonal(3, 3, improper)
    t = torch.tensor([1.5, -2.0, 0.7], dtype=D)
    moved = {k: transform(v, R, t, specs=_specs(m, k)) for k, v in meshes.items()}
    out = _run(
        m,
        moved,
        {"g": GD["g"] @ R.T, "re": GD["re"]},
        None if q is None else q @ R.T + t,
    )
    for name, spec in m.output_schema.items():
        assert torch.allclose(
            out[name], law(base[name], R, spec.rank), atol=1e-10, rtol=0.0
        ), name


def test_a_condition_s_cells_permute_with_its_outputs():
    """Permuting one condition's cells (and its fields) permutes that condition's
    surface outputs and leaves the others as they are."""
    m = _model()
    meshes = _meshes()
    base = _run(m, meshes)
    inlet = meshes["inlet"]
    perm = torch.randperm(inlet.n_cells, generator=torch.Generator().manual_seed(1))
    permuted = Mesh(
        points=inlet.points, cells=inlet.cells[perm], cell_data=inlet.cell_data[perm]
    )
    out = _run(m, {**meshes, "inlet": permuted})
    ### The tokens are the conditions in canonical order: inlet, no_slip, outlet, slip.
    n = inlet.n_cells
    for k in base:
        assert torch.allclose(out[k][:n], base[k][perm], atol=1e-10, rtol=0.0)
        assert torch.allclose(out[k][n:], base[k][n:], atol=1e-10, rtol=0.0)


def _split(mesh: Mesh) -> Mesh:
    """Every cell as two coincident half-measure copies carrying its fields: the
    same discrete boundary measure and the same data, re-represented."""
    data = mesh.cell_data.apply(
        lambda t: t.repeat_interleave(2, 0), batch_size=[2 * mesh.n_cells]
    )
    data[EFFECTIVE_MEASURE_KEY] = (
        mesh.cell_data[EFFECTIVE_MEASURE_KEY].repeat_interleave(2) / 2
    )
    return Mesh(
        points=mesh.points, cells=mesh.cells.repeat_interleave(2, 0), cell_data=data
    )


@pytest.mark.parametrize("name", list(SCHEMAS))
def test_remeshing_one_condition_changes_no_interior_prediction(name):
    """Splitting every cell of one condition into two half-measure copies with its
    fields (the budget above every cell count, so the draw keeps every cell) leaves
    the interior predictions unchanged, for a condition with fields and without. The
    invariance is exact for the interior task's exact read (theta = 0, every cell pair read); at
    the default opening angle the split boundary's forest differs and the Barnes-Hut
    read is an approximation."""
    m = _model(interior=True)
    m.tree_read_theta = 0.0
    meshes = _meshes()
    base = _run(m, meshes, q=Q)
    out = _run(m, {**meshes, name: _split(meshes[name])}, q=Q)
    for k in base:
        assert torch.allclose(out[k], base[k], atol=1e-10, rtol=0.0), k


@pytest.mark.parametrize("interior", [False, True], ids=["surface", "interior"])
def test_the_order_of_the_input_mapping_changes_nothing(interior):
    """The tokens are the conditions in canonical (sorted) order whatever the order
    of the mapping: the outputs are bitwise the same."""
    m = _model(interior)
    meshes = _meshes()
    q = Q if interior else None
    base = _run(m, meshes, q=q)
    out = _run(m, dict(reversed(list(meshes.items()))), q=q)
    assert all(torch.equal(out[k], base[k]) for k in base)


@pytest.mark.parametrize("interior", [False, True], ids=["surface", "interior"])
def test_swapping_two_conditions_names_changes_the_outputs(interior):
    """Identity is structural: two conditions without fields read different input
    layers, so the same cells under the other name give other outputs (no tag field
    tells the model which is which)."""
    m = _model(interior)
    meshes = _meshes()
    q = Q if interior else None
    base = _run(m, meshes, q=q)
    swapped = {**meshes, "no_slip": meshes["slip"], "slip": meshes["no_slip"]}
    out = _run(m, swapped, q=q)
    if not interior:
        ### The swapped tokens in canonical order are inlet, the slip cells, outlet,
        ### the no-slip cells: put them back in the order of the base tokens.
        n_in, n_s, n_out, n_ns = (
            SIZES["inlet"],
            SIZES["slip"],
            SIZES["outlet"],
            SIZES["no_slip"],
        )
        order = [*range(n_in), *range(n_in + n_s + n_out, n_in + n_s + n_out + n_ns)]
        order += [*range(n_in + n_s, n_in + n_s + n_out), *range(n_in, n_in + n_s)]
        out = {k: v[order] for k, v in out.items()}
    assert max(float((out[k] - base[k]).abs().max()) for k in base) > 1e-3
    assert not torch.equal(
        m.boundary_conditions["no_slip"].seed.weight,
        m.boundary_conditions["slip"].seed.weight,
    )


@pytest.mark.parametrize("interior", [False, True], ids=["surface", "interior"])
def test_an_absent_condition_is_a_condition_without_cells(interior):
    """Any non-empty subset of the declared names is a valid input: an absent
    condition contributes no tokens, exactly as one given with no cells, and the
    present conditions' tokens are the same cells in the same order."""
    m = _model(interior)
    meshes = _meshes()
    q = Q if interior else None
    subset = {k: meshes[k] for k in ("inlet", "no_slip")}
    out = _run(m, subset, q=q)
    empty = Mesh(points=meshes["slip"].points, cells=meshes["slip"].cells[:0])
    empty_outlet = _meshes(sizes={"outlet": 0})["outlet"]
    assert all(
        torch.equal(
            v, _run(m, {**subset, "slip": empty, "outlet": empty_outlet}, q=q)[k]
        )
        for k, v in out.items()
    )
    n = SIZES["inlet"] + SIZES["no_slip"]
    assert out["p"].shape == ((25,) if interior else (n,))
    full = _run(m, meshes, q=q)
    if interior:
        assert not torch.allclose(out["p"], full["p"], atol=1e-6)


def test_unknown_and_missing_names_are_refused():
    """An undeclared condition, an empty mapping and a mesh without its condition's
    fields are refused, each error naming the condition."""
    m = _model()
    meshes = _meshes()
    with pytest.raises(
        ValueError,
        match="\\['wall'\\], which are not declared boundary conditions; this model declares \\['inlet', 'no_slip', 'outlet', 'slip'\\]",
    ):
        m(boundary_meshes={**meshes, "wall": meshes["slip"]}, global_data=GD)
    with pytest.raises(ValueError, match="non-empty mapping"):
        m(boundary_meshes={}, global_data=GD)
    ### each mesh carries its own condition's fields
    with pytest.raises(
        ValueError,
        match="boundary_meshes\\['outlet'\\].cell_data is missing the declared fields \\['T', 'p', 'u_rev'\\]",
    ):
        m(boundary_meshes={"outlet": meshes["slip"]}, global_data=GD)


def test_the_eager_checks_hold_while_another_thread_compiles():
    """Since torch 2.13 ``torch.compiler.is_compiling()`` is True in every thread while
    any thread compiles; the input checks of an eager forward do not read it, so an
    undeclared condition and a mesh without its fields are still refused in a thread
    that runs while another compiles."""
    m = _model()
    meshes = _meshes()

    def errors():
        seen = []
        for boundary_meshes in (
            {**meshes, "wall": meshes["slip"]},
            {"outlet": meshes["slip"]},
        ):
            try:
                m(boundary_meshes=boundary_meshes, global_data=GD)
            except ValueError as e:
                seen.append(str(e))
        return seen

    seen = in_another_thread_while_compiling(errors)
    assert len(seen) == 2
    assert "which are not declared boundary conditions" in seen[0]
    assert "is missing the declared fields" in seen[1]


#: The conditions' parameters live in an nn.ModuleDict keyed by name, so a name that
#: is one of its attributes is refused too, naming the argument.
MODULE_DICT_NAMES = ["keys", "type", "update", "training", "items", "forward"]


@pytest.mark.parametrize(
    "schemas, match",
    [
        ({}, "non-empty mapping"),
        (None, "non-empty mapping"),
        ({"a.b": {}}, "without '.'"),
        ({"": {}}, "without '.'"),
    ]
    + [
        ({**SCHEMAS, n: {}}, f"boundary_schemas: the condition name '{n}'")
        for n in MODULE_DICT_NAMES
    ],
)
def test_the_declaration_is_validated(schemas, match):
    """An empty declaration, a name with a dot or no name, and a name that is an
    attribute of nn.ModuleDict are refused at construction."""
    with pytest.raises(ValueError, match=match):
        _model(schemas=schemas)


def test_the_declaration_is_canonical_and_per_condition():
    """Names in sorted order whatever the declaration's; each condition's input layer
    reads [1, n.g, its fields' invariants, re]; vector and tensor maps exist only for
    the conditions with such inputs and map onto the widest condition's channels."""
    m = _model(schemas=dict(reversed(list(SCHEMAS.items()))))
    assert m.boundary_names == ("inlet", "no_slip", "outlet", "slip")
    assert list(m.boundary_conditions) == list(m.boundary_names)
    widths = {k: c.seed.in_features for k, c in m.boundary_conditions.items()}
    ### inlet: u (2 frame cosines, norm) and k; outlet: p, u_rev (3), T (3 frame pairs, trace, norm)
    assert widths == {
        "inlet": 1 + 1 + 4 + 1,
        "no_slip": 3,
        "outlet": 1 + 1 + 9 + 1,
        "slip": 3,
    }
    c = m.boundary_conditions
    assert c["no_slip"].vector_map is None and c["slip"].tensor_map is None
    ### outlet: u_rev and T's two derived vectors (T n, T g); inlet: u
    assert m._n_boundary_vecs == 3 and tuple(c["outlet"].vector_map.shape) == (3, 3)
    assert (
        tuple(c["inlet"].vector_map.shape) == (3, 1) and c["inlet"].tensor_map is None
    )
    assert m._n_boundary_tens == 1 and tuple(c["outlet"].tensor_map.shape) == (1, 1)


def test_a_condition_s_fields_and_maps_change_the_outputs():
    """A condition's own fields reach the prediction, through the seed and through
    its vector map into the slice layers."""
    m = _model(interior=True)
    meshes = _meshes()
    base = _run(m, meshes, q=Q)
    inlet = meshes["inlet"]
    data = inlet.cell_data.clone()
    data["u"] = data["u"] * 2
    out = _run(m, {**meshes, "inlet": inlet.with_data(cell_data=data)}, q=Q)
    assert not torch.allclose(out["p"], base["p"], atol=1e-6)
    with torch.no_grad():
        m.boundary_conditions["inlet"].vector_map.mul_(-1.5)
    assert not torch.allclose(_run(m, meshes, q=Q)["p"], base["p"], atol=1e-6)


def test_every_condition_s_parameters_train_even_when_it_is_absent():
    """DDP needs every parameter in every pass: with the slip condition absent and the
    outlet given without cells, every parameter still receives a gradient. The
    interior task takes no rank-2 inputs, so its outlet declares none."""
    schemas = _interior(SCHEMAS)
    torch.manual_seed(0)
    m = ISLA(
        output_schema=OUT,
        boundary_schemas=schemas,
        global_schema={"g": {"rank": 1}, "re": {"rank": 0}},
        interior=True,
        hidden=32,
        n_layers=2,
        n_slices=8,
    )
    meshes = {k: v.to(torch.float32) for k, v in _meshes(schemas).items()}
    meshes["outlet"] = _meshes(schemas, sizes={"outlet": 0})["outlet"].to(torch.float32)
    del meshes["slip"]
    gd = {k: v.float() for k, v in GD.items()}
    out = outputs(
        m(
            boundary_meshes=meshes,
            prediction_points=points_mesh(Q.float()),
            global_data=gd,
        )
    )
    sum(v.square().mean() for v in out.values()).backward()
    assert not [n for n, p in m.named_parameters() if p.grad is None]
    assert all(torch.isfinite(v).all() for v in out.values())


@pytest.mark.slow  # 12 s on CPU: one compile per subset
def test_a_change_of_condition_subset_compiles_once_per_subset():
    """Under the whole-model compile (the surface task's), cases of one subset with
    other cell counts reuse a graph; another subset is another graph, compiled once,
    and the first subset's graph is reused when its cases come back."""
    torch._dynamo.reset()
    m = _model().float()
    compiled = torch.compile(m, backend="eager", fullgraph=True, dynamic=True)
    gd = {k: v.float() for k, v in GD.items()}

    def case(names, extra):
        sizes = {k: SIZES[k] + extra + 7 * i for i, k in enumerate(names)}
        meshes = {
            k: v.to(torch.float32) for k, v in _meshes(sizes=sizes, seed=extra).items()
        }
        return {
            "boundary_meshes": meshes,
            "prediction_boundaries": list(meshes),
            "global_data": gd,
        }

    with torch.no_grad():
        compiled(**case(SCHEMAS, 0))
        compiled(**case(["inlet", "outlet", "no_slip"], 0))
        with torch._dynamo.config.patch(error_on_recompile=True):
            compiled(**case(SCHEMAS, 3))
            compiled(**case(["inlet", "outlet", "no_slip"], 5))
