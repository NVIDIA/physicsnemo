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

r"""Release-readiness tests for ISLA (MODELS_IMPLEMENTATION.md MOD-008a/b/c).

- MOD-008a: the constructor and every public attribute, for the defaults in three
  and two dimensions and a configuration that sets every argument to a non-default
  value (:data:`PUBLIC_ATTRIBUTES` lists them; a new attribute fails until listed).
- MOD-008b: forward non-regression against golden outputs stored in ``data/``: the
  two small float32 configurations (``isla_default_v1.pth``,
  ``isla_dtt_v1.pth``), and in ``isla_release_v2.pth`` a float64 configuration per
  task at 1e-12, a rich surface model with scales, the interior task in two
  dimensions, the in-model token draw over about 2,000 cells, and ``query_chunks``
  (outputs and gradients); and the backbone golden.
- MOD-008c: committed reference checkpoints of both tasks (``isla_default_v1.mdlus``,
  ``isla_dtt_v1.mdlus``) through ``Module.from_checkpoint`` (attributes and
  forward), a checkpoint round trip over schemas, and the refusal of another version
  on every loading path.

**Regenerating the data.** Every file in ``data/`` is written by one command, and
only by it (a missing file fails; nothing is created on the fly)::

    python -m test.experimental.models.isla._release_cases FILE ... | all

with the named files (e.g. ``isla_release_v2.pth``), or ``all`` for every file.
Regenerate only what a change is meant to move, and say so in the commit: the
reference checkpoints stand for a checkpoint written by the code of their version
(regenerate them at a version bump), and ``isla_backbone_v1.pth`` only when the
backbone changes by decision (a forward hook on ``norm_out`` under
``CONFIGS["default"]`` on ``_make_inputs``, through ``forward``).
``_release_cases.WRITERS`` writes the files of this test.
"""

import dataclasses
import inspect
import io
import json
import logging
import re
import warnings
import zipfile
from pathlib import Path

import pytest
import torch

from physicsnemo.core.module import Module
from physicsnemo.experimental.models.isla import ISLA
from test.experimental.models.isla import _model_cases as C
from test.experimental.models.isla._marks import DEVICES
from test.experimental.models.isla._meshes import outputs
from test.experimental.models.isla._release_cases import (
    CHECKPOINTS,
    CONFIGS,
    DATA_DIR,
    IO,
    RELEASE_V2,
    V2_CONFIGS,
    _backbone_golden,
    _checkpoint_output,
    _chunks_golden,
    _fill,
    _instantiate_model,
    _make_inputs,
    _v2_outputs,
)

REGENERATE = "python -m test.experimental.models.isla._release_cases"
D, F = torch.float64, torch.float32


def _load_reference(file_name: str, key: str | None = None) -> dict:
    """A committed golden (``data/<file_name>``, its entry ``key`` if given); a missing
    one fails with the command that writes it."""
    path = DATA_DIR / file_name
    if not path.exists():
        pytest.fail(f"{path} is missing: regenerate it with `{REGENERATE} {file_name}`")
    data = torch.load(path, weights_only=True)
    return data if key is None else data[key]


def _assert_dict_close(out: dict, ref: dict, atol: float, rtol: float) -> None:
    assert list(out.keys()) == list(ref.keys())
    for k in out.keys():
        torch.testing.assert_close(
            out[k], ref[k].to(out[k].device), atol=atol, rtol=rtol
        )


def _assert_scaled_close(out: dict, ref: dict, tol: float) -> None:
    """Every field within ``tol`` of its golden's largest magnitude (rtol 0: the
    error of an output is set by the field's scale, not by each element's)."""
    assert list(out.keys()) == list(ref.keys())
    for k, r in ref.items():
        r = r.to(out[k].device)
        torch.testing.assert_close(
            out[k],
            r,
            atol=tol * float(r.abs().max()),
            rtol=0.0,
            msg=lambda m: f"{k}: {m}",
        )


# ---------------------------------------------------------------------------
# MOD-008a: constructor and public attributes
# ---------------------------------------------------------------------------

#: Every public attribute an ISLA has (``vars(model)``, where submodules, parameters
#: and buffers never sit). A new one fails ``test_isla_constructor_attributes`` until
#: it is listed here and its value asserted there; one that is gone fails there too.
PUBLIC_ATTRIBUTES = (
    "training",
    "meta",
    "logger",
    "compile_blocks",
    "n_spatial_dims",
    "output_schema",
    "boundary_schemas",
    "global_schema",
    "output_stats",
    "interior",
    "n_boundary_tokens",
    "tree_read_theta",
    "tree_read_leaf_size",
    "query_chunk_pairs",
    "mlp_ratio",
    "routing_temperature",
    "block_activation_budget",
    "hidden",
    "n_layers",
    "n_slices",
    "boundary_names",
    "qt",
    "kernel_basis",
    "kernel_reads",
    "condition_log_shares",  # None with one declared condition, else a submodule
)

#: The head's width (its coefficients) per configuration of ``ATTRIBUTE_CONFIGS``. The
#: defaults have K = 1 global direction: m = 2K + 3 = 5 true basis vectors, so p costs
#: 1 and v 5. The custom configuration has K = 3 (g1, g2 and the direction the inlet's
#: velocity defines) in two dimensions: m = 9 true basis vectors and K + 2 = 5
#: pseudovectors, so a vector costs 9, a pseudovector 5, a symmetric rank-2 tensor
#: m(m+1)/2 + 1 = 46 and a general one m^2 + 1 = 82.
HEAD_WIDTHS = {"with_defaults": 6, "with_defaults_2d": 6, "with_custom_args": 144}

#: Every constructor argument at a value other than its default: an interior task in
#: two dimensions (rank-2 boundary inputs are refused there) with three conditions, a
#: rank-1 field defining the velocity scale, a rank-0 global defining a density, a
#: dimensionless global, dotted, odd and rank-2 outputs, compiled blocks, a token
#: budget, the boundary read's and the blocks' knobs and output_stats.
CUSTOM = dict(
    output_schema={
        "p": {"rank": 0, "units": {"U": 2, "rho": 1}},
        "q.s": {"rank": 0},
        "u": {"rank": 1, "units": {"U": 1}},
        "w": {"rank": 1, "parity": "odd"},
        "stress": {"rank": 2, "symmetric": True},
        "t": {"rank": 2},
    },
    boundary_schemas={
        "wall": {"trace": {"rank": 0}},
        "inlet": {"velocity": {"rank": 1, "units": {"U": 1}, "defines": "U"}},
        "outlet": {},
    },
    global_schema={
        "g1": {"rank": 1},
        "g2": {"rank": 1},
        "rho": {"rank": 0, "defines": "rho"},
        "re": {"rank": 0},
    },
    interior=True,
    n_boundary_tokens=1000,
    tree_read_theta=0.5,
    tree_read_leaf_size=8,
    query_chunk_pairs=1 << 20,
    hidden=64,
    n_layers=3,
    n_slices=16,
    mlp_ratio=2,
    routing_temperature=5.0,
    compile_blocks=True,
    n_spatial_dims=2,
    output_stats={"p": {"mean": 0.5, "std": 2.0}},
)

ATTRIBUTE_CONFIGS = {
    "with_defaults": IO,
    "with_defaults_2d": dict(IO, n_spatial_dims=2),
    "with_custom_args": CUSTOM,
}


def _ranks(schema) -> dict:
    return {k: (s.rank, s.symmetric, s.parity) for k, s in schema.items()}


def _expected_attributes(config: str) -> dict:
    """The expected value of every public attribute (a callable: a predicate)."""
    common = dict(
        training=True,
        meta=ISLA.MetaData(),
        logger=lambda v: isinstance(v, logging.Logger),
        block_activation_budget=1.0,
    )
    if config != "with_custom_args":
        return dict(
            common,
            compile_blocks=False,
            n_spatial_dims=3 if config == "with_defaults" else 2,
            output_schema=lambda v: (
                _ranks(v) == {"p": (0, False, "even"), "v": (1, False, "even")}
            ),
            boundary_schemas={"wall": {}},
            global_schema=lambda v: _ranks(v) == {"g": (1, False, "even")},
            output_stats=None,
            interior=False,
            n_boundary_tokens=200_000,
            tree_read_theta=1.0,
            tree_read_leaf_size=4,
            query_chunk_pairs=2**28,
            mlp_ratio=4,
            routing_temperature=10.0,
            hidden=512,  # the class defaults
            n_layers=12,
            n_slices=128,
            boundary_names=("wall",),
            qt=None,
            kernel_basis=None,
            kernel_reads=None,
            condition_log_shares=None,
        )
    return dict(
        common,
        compile_blocks=True,
        n_spatial_dims=2,
        output_schema=lambda v: (
            _ranks(v)
            == {
                "p": (0, False, "even"),
                "q.s": (0, False, "even"),
                "u": (1, False, "even"),
                "w": (1, False, "odd"),
                "stress": (2, True, "even"),
                "t": (2, False, "even"),
            }
        ),
        boundary_schemas=lambda v: (
            list(v) == ["inlet", "outlet", "wall"]  # canonical order
            and {k: _ranks(s) for k, s in v.items()}
            == {
                "inlet": {"velocity": (1, False, "even")},
                "outlet": {},
                "wall": {"trace": (0, False, "even")},
            }
        ),
        global_schema=lambda v: list(v) == ["g1", "g2", "rho", "re"],
        output_stats={"p": {"mean": 0.5, "std": 2.0}},
        interior=True,
        n_boundary_tokens=1000,
        tree_read_theta=0.5,
        tree_read_leaf_size=8,
        query_chunk_pairs=1 << 20,
        mlp_ratio=2,
        routing_temperature=5.0,
        hidden=64,
        n_layers=3,
        n_slices=16,
        boundary_names=("inlet", "outlet", "wall"),
    )


def _public_attributes(model) -> dict:
    """``vars`` of the model without its private names, with the knobs it holds as
    checked properties (``ISLA._overridable_args``)."""
    knobs = {k: getattr(model, k) for k in ISLA._overridable_args}
    return {**{k: v for k, v in vars(model).items() if not k.startswith("_")}, **knobs}


def test_the_custom_configuration_sets_every_argument():
    """``with_custom_args`` gives every constructor argument, each at a value other
    than its default."""
    params = inspect.signature(ISLA.__init__).parameters
    assert set(CUSTOM) == set(params) - {"self"}
    for name, value in CUSTOM.items():
        assert value != params[name].default, name


@pytest.mark.parametrize("config", list(ATTRIBUTE_CONFIGS))
def test_isla_constructor_attributes(config):
    """Every public attribute of the model (``vars``, not a submodule or a parameter)
    is in :data:`PUBLIC_ATTRIBUTES` and has its expected value, the default
    configuration has exactly the listed ones (no stale entry), and the modules the
    configuration implies are built, the head at its width (:data:`HEAD_WIDTHS`)."""
    kw = ATTRIBUTE_CONFIGS[config]
    m = ISLA(**kw)
    found = _public_attributes(m)
    assert set(found) <= set(PUBLIC_ATTRIBUTES), "unlisted: " + str(
        sorted(set(found) - set(PUBLIC_ATTRIBUTES))
    )
    expected = _expected_attributes(config)
    public = set(expected)
    assert set(found) == public, sorted(set(found) ^ public)
    if config == "with_defaults":
        assert set(PUBLIC_ATTRIBUTES) == public, sorted(set(PUBLIC_ATTRIBUTES) ^ public)
    for name, want in expected.items():
        value = getattr(m, name)
        assert want(value) if callable(want) else value == want, (name, value)
    assert isinstance(m, Module) and m.reference_kernels is False
    hidden, n_layers = kw.get("hidden", 512), kw.get("n_layers", 12)
    assert len(m.blocks) == n_layers and m.norm_out.normalized_shape == (hidden,)
    assert m.head.out_features == HEAD_WIDTHS[config]
    assert list(m.boundary_conditions) == list(m.boundary_names)
    if m.interior:
        assert m.qt is not None and len(m.kernel_reads) == n_layers
        assert m.kernel_basis is not None


#: The tunable knobs and their defaults: a checkpoint that stores none of them is the
#: same model at the defaults.
KNOB_DEFAULTS = dict(
    tree_read_theta=1.0,
    tree_read_leaf_size=4,
    query_chunk_pairs=2**28,
    mlp_ratio=4,
    routing_temperature=10.0,
)
#: Every knob away from its default, which the interior round trip sets.
KNOBS = dict(
    tree_read_theta=0.5,
    tree_read_leaf_size=8,
    query_chunk_pairs=1000,
    mlp_ratio=2,
    routing_temperature=5.0,
)


def test_isla_knobs_have_their_defaults_and_are_validated():
    """Each knob has its default; a bad value is a ValueError naming it;
    the boundary read's knobs belong to the interior task, so a surface task takes
    only their defaults (as it does ``n_boundary_tokens``'); theta 0 is allowed."""
    params = inspect.signature(ISLA.__init__).parameters
    assert {k: params[k].default for k in KNOB_DEFAULTS} == KNOB_DEFAULTS
    assert all(KNOBS[k] != KNOB_DEFAULTS[k] for k in KNOB_DEFAULTS)
    small = dict(hidden=16, n_layers=1, n_slices=8, **IO)
    bad = dict(
        tree_read_theta=(-0.5, float("nan"), float("inf"), True, "1"),
        tree_read_leaf_size=(0, -4, 2.5, True),
        query_chunk_pairs=(0, 1.5, False),
        mlp_ratio=(0, 2.0, True),
        routing_temperature=(0.0, -1.0, float("inf"), float("nan"), True),
        hidden=(0, 16.5, True),
        n_layers=(0, 1.0),
        n_slices=(0, -8, 8.0),
        n_spatial_dims=(0, 2.9, True),
    )
    for name, values in bad.items():
        for value in values:
            with pytest.raises(
                ValueError, match=f"^{name} must be a finite|^{name} must be a positive"
            ):
                ISLA(**{**small, "interior": True, name: value})
    for name in ("tree_read_theta", "tree_read_leaf_size", "query_chunk_pairs"):
        with pytest.raises(ValueError, match=f"^{name} sets the interior task's"):
            ISLA(**small, **{name: 2})
        ISLA(**small, **{name: KNOB_DEFAULTS[name]})
    m = ISLA(**small, interior=True, tree_read_theta=0)
    assert m.tree_read_theta == 0.0 and isinstance(m.tree_read_theta, float)
    surface = ISLA(**small, mlp_ratio=2, routing_temperature=0.5)
    assert (surface.mlp_ratio, surface.routing_temperature) == (2, 0.5)


def test_isla_forward_knobs_are_checked_on_assignment_and_overridable(tmp_path):
    """The knobs the forward pass reads (``n_boundary_tokens``, ``tree_read_theta``,
    ``query_chunk_pairs``) refuse on assignment the values the constructor refuses,
    with its message, and ``Module.from_checkpoint`` overrides them through the
    constructor's checks."""
    m = ISLA(hidden=16, n_layers=1, n_slices=8, interior=True, **IO)
    bad = dict(
        n_boundary_tokens=(0, 2.5, True),
        tree_read_theta=(-1.0, float("nan"), True),
        query_chunk_pairs=(0, 1.5, False),
    )
    for name, values in bad.items():
        for value in values:
            with pytest.raises(ValueError, match=f"^{name} must be a"):
                setattr(m, name, value)
    m.tree_read_theta, m.query_chunk_pairs, m.n_boundary_tokens = 0, 2**20, 1000.0
    assert (m.tree_read_theta, m.query_chunk_pairs, m.n_boundary_tokens) == (
        0.0,
        2**20,
        1000,
    )
    assert isinstance(m.tree_read_theta, float) and isinstance(m.n_boundary_tokens, int)
    ### A surface task refuses any value but the default, on assignment as in the
    ### constructor, and takes the default.
    s = ISLA(hidden=16, n_layers=1, n_slices=8, **IO)
    for name, value in (
        ("n_boundary_tokens", 5),
        ("tree_read_theta", 0.3),
        ("query_chunk_pairs", 7),
    ):
        with pytest.raises(ValueError, match=r"a surface task \(interior=False\)"):
            setattr(s, name, value)
        setattr(s, name, getattr(s, name))
    path = tmp_path / "isla.mdlus"
    m.save(str(path))
    over = dict(n_boundary_tokens=500, tree_read_theta=0.5, query_chunk_pairs=2**10)
    loaded = Module.from_checkpoint(str(path), override_args=over)
    assert {k: getattr(loaded, k) for k in over} == over
    with pytest.raises(
        ValueError, match="^query_chunk_pairs must be a positive integer"
    ):
        Module.from_checkpoint(str(path), override_args={"query_chunk_pairs": 0})


# ---------------------------------------------------------------------------
# MOD-008b: forward non-regression against stored golden outputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("config", list(CONFIGS))
def test_isla_forward_non_regression(device, config):
    """The outputs of ``CONFIGS[config]`` against ``isla_<config>_v1.pth``. Regenerate:
    ``python -m test.experimental.models.isla._release_cases
    isla_<config>_v1.pth``."""
    model = _instantiate_model(**CONFIGS[config]).to(device)
    inputs = _make_inputs(device, query=(config != "default"))
    with torch.no_grad():
        out = outputs(model(**inputs))
    n_out = 64 if config == "default" else 40
    assert out["p"].shape == (n_out,) and out["v"].shape == (n_out, 3)
    ref = _load_reference(f"isla_{config}_v1.pth", "output")
    tol = 1e-5 if device == "cpu" else 1e-4
    _assert_dict_close(out, ref, atol=tol, rtol=tol)


#: Tolerances relative to each field's largest magnitude. CPU (reference kernels):
#: float64 1e-12, which holds across BLAS builds, CPU architectures and thread counts
#: (measured at most 6.2e-16 on Grace against the x86 golden); float32
#: 1e-5 (measured 1.4e-6). CUDA, kernels auto (the fused geometry region, the fast
#: forest, walk and geometry search, the fused read; TF32 off) against the CPU golden:
#: float64 1e-12 and float32 1e-4, for the reordered sums of the fused kernels and of
#: CUDA's atomics (measured 7.7e-16 and 1.6e-6 on GB300).
V2_TOL = {("cpu", D): 1e-12, ("cpu", F): 1e-5, ("cuda", D): 1e-12, ("cuda", F): 1e-4}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("config", list(V2_CONFIGS))
def test_isla_forward_non_regression_v2(device, config, monkeypatch):
    """The outputs of each configuration of :data:`V2_CONFIGS` against
    ``isla_release_v2.pth`` (recorded on one CPU thread with the reference kernels),
    at :data:`V2_TOL`. ``dtt_draw_f32`` pins the draw of 512 of its cells
    (its hits keep 1e-9 off the cells' boundaries).
    Regenerate: ``python -m test.experimental.models.isla._release_cases
    isla_release_v2.pth``."""
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    out = _v2_outputs(config, device)
    kind = "cpu" if device == "cpu" else "cuda"
    _assert_scaled_close(
        out,
        _load_reference(RELEASE_V2, config)["output"],
        V2_TOL[(kind, V2_CONFIGS[config][1])],
    )


@pytest.mark.parametrize("device", DEVICES)
def test_isla_query_chunks_non_regression(device, monkeypatch):
    """``query_chunks`` (every public method has a golden): the outputs of three
    ranges of the interior model's points, its pair counts (exact), and every parameter's
    gradient after the chunks' backward passes and ``chunks.backward()`` (each as a
    projection on a fixed random direction, within the float64 bound of
    :data:`V2_TOL` of the largest sum of magnitudes of a projection's terms).
    Regenerate: ``python -m test.experimental.models.isla._release_cases
    isla_release_v2.pth``."""
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    out = _chunks_golden(device)
    ref = _load_reference(RELEASE_V2, "query_chunks")
    tol = V2_TOL[("cpu" if device == "cpu" else "cuda", D)]
    assert torch.equal(out.pop("pair_counts"), ref["pair_counts"])
    proj, scale = out.pop("grad_projections"), out.pop("grad_scales")
    _assert_scaled_close(
        out, {k: v for k, v in ref.items() if k.startswith("range")}, tol
    )
    bound = tol * float(ref["grad_scales"].max())
    torch.testing.assert_close(proj, ref["grad_projections"], atol=bound, rtol=0.0)
    torch.testing.assert_close(scale, ref["grad_scales"], atol=bound, rtol=0.0)


def test_isla_backbone_matches_the_recorded_backbone():
    """With no per-token inputs, the pre-head token state (``norm_out`` output) of
    the surface model is the recorded one, for the same parameters and inputs.
    ``data/isla_backbone_v1.pth`` was recorded with a forward hook on ``norm_out``
    (seed 0 parameters, seed 1234 inputs, as here) on an x86 CPU, through
    ``forward``. Compared at float32 tolerance: the values are bitwise on the
    recording architecture and differ in the last bits across CPU architectures
    (BLAS reduction order). Regenerate only when the backbone changes by decision:
    ``python -m test.experimental.models.isla._release_cases
    isla_backbone_v1.pth``."""
    ref = _load_reference("isla_backbone_v1.pth", "default")
    torch.testing.assert_close(
        _backbone_golden(), ref["norm_out"], atol=1e-6, rtol=1e-5
    )


# ---------------------------------------------------------------------------
# MOD-008c: checkpoints: the committed references, round trips, other versions
# ---------------------------------------------------------------------------


def _stored_args(path) -> dict:
    """The constructor arguments a ``.mdlus`` stores (its ``args.json``)."""
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("args.json"))["__args__"]


def _comparable(model) -> dict:
    """The public attributes in a comparable form (the logger by its name)."""
    out = {}
    for k, v in _public_attributes(model).items():
        if k == "logger":
            v = v.name
        out[k] = v
    return out


def test_seeded_fills_leave_other_parameters_bitwise_when_one_is_removed():
    """The release writer's fill (:func:`_fill`) and the model cases'
    (``_model_cases._params``) draw each parameter from a generator keyed by the seed and
    the parameter's name: a model with one parameter fewer (here an early one, the token
    embedding's bias) gets every other parameter bitwise the same values, so a
    state-dict change moves only the parameters it changes."""
    full = ISLA(**CONFIGS["dtt"])
    cut = ISLA(**CONFIGS["dtt"])
    cut.embed[1].register_parameter("bias", None)
    a = dict(_fill(full, 7).named_parameters())
    b = dict(_fill(cut, 7).named_parameters())
    assert set(a) - set(b) == {"embed.1.bias"}
    assert all(torch.equal(a[k], b[k]) for k in b)
    ga, gb = C._params(full, 7), C._params(cut, 7)
    assert set(ga) - set(gb) == {"param.embed.1.bias"}
    assert all(torch.equal(ga[k], gb[k]) for k in gb)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("file_name", list(CHECKPOINTS))
def test_isla_reference_checkpoint_matches_golden(device, file_name):
    """A committed reference checkpoint (of the current version) loads through
    ``Module.from_checkpoint`` without a warning, as the model of its configuration
    (every public attribute but the execution choices, which a loaded model starts at
    "auto"), and keeps its outputs (``<name>_checkpoint_output_v1.pth``). Regenerate
    (at a version bump): ``python -m test.experimental.models.isla._release_cases
    <name>_v1.mdlus <name>_checkpoint_output_v1.pth``."""
    path = DATA_DIR / file_name
    if not path.exists():
        pytest.fail(f"{path} is missing: regenerate it with `{REGENERATE} {file_name}`")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = Module.from_checkpoint(str(path)).eval()
    assert not [w for w in caught if "checkpoint" in str(w.message)]
    assert isinstance(model, ISLA)
    cfg, _ = CHECKPOINTS[file_name]
    fresh = ISLA(**cfg).eval()
    assert _comparable(model) == _comparable(fresh)
    assert (
        model._geo_kernel == model._tree_read_kernel == model._geometry_kernel == "auto"
    )
    assert len(model.blocks) == cfg["n_layers"]
    assert model.norm_out.normalized_shape == (cfg["hidden"],)
    out = _checkpoint_output(file_name, device)
    ref = _load_reference(
        file_name.replace("_v1.mdlus", "_checkpoint_output_v1.pth"), "output"
    )
    tol = 1e-5 if device == "cpu" else 1e-4
    _assert_dict_close(out, ref, atol=tol, rtol=tol)


#: The schemas a checkpoint must keep (each a model case of ``_model_cases.py``):
#: outputs of every rank, symmetry and parity with two global directions; dotted
#: names of fields, globals and outputs; the interior task with every knob away from
#: its default; a model with scales and output_stats; and the interior task in two
#: dimensions.
_HEADS_OUTPUTS = {
    "s": {"rank": 0},
    "v": {"rank": 1},
    "pv": {"rank": 1, "parity": "odd"},
    "t": {"rank": 2},
    "ts": {"rank": 2, "symmetric": True},
    "pt": {"rank": 2, "parity": "odd"},
    "t3": {"rank": 3},
    "t3o": {"rank": 3, "parity": "odd"},
    "t4s": {"rank": 4, "symmetric": True},
}
ROUND_TRIPS = {
    "heads": C._ModelSpec(
        dict(
            hidden=16,
            n_layers=1,
            n_slices=8,
            output_schema=_HEADS_OUTPUTS,
            boundary_schemas={"wall": {}},
            global_schema={"g0": {"rank": 1}, "g1": {"rank": 1}},
        ),
        {"wall": lambda: C._soup_cells(48, 3, 91)},
        prediction_boundaries=("wall",),
    ),
    "names": C._ModelSpec(
        dict(
            hidden=16,
            n_layers=1,
            n_slices=8,
            output_schema={
                "surface.pressure": {"rank": 0},
                "surface.wall_shear": {"rank": 1},
                "solid.stress": {"rank": 2, "symmetric": True},
                "nut": {"rank": 0},
            },
            global_schema={
                "conditions.freestream": {"rank": 1},
                "conditions.re": {"rank": 0},
            },
            boundary_schemas={
                "wall": {"bc.trace": {"rank": 0}, "bc.traction": {"rank": 1}}
            },
        ),
        {"wall": lambda: C._soup_cells(48, 3, 92)},
        prediction_boundaries=("wall",),
    ),
    "interior": dataclasses.replace(
        C._MODEL_SPECS["dtt"](D), cfg={**C._MODEL_SPECS["dtt"](D).cfg, **KNOBS}
    ),
    "scaled": C._MODEL_SPECS["rich_surface"](D),
    "two_dimensional": C._MODEL_SPECS["dtt_2d"](D),
}


@pytest.mark.parametrize("schema", list(ROUND_TRIPS))
def test_isla_checkpoint_round_trip(tmp_path, schema):
    """``save`` and ``Module.from_checkpoint`` keep the model: the checkpoint names
    every constructor argument (``Module.__new__`` stores the bound defaults, so it
    never depends on the class defaults), the loaded model has every public
    attribute of the saved one, and its outputs on the case are bitwise the saved
    model's (on the CPU; serialization is device-independent, and the reference
    checkpoints' tests run on every device)."""
    spec = ROUND_TRIPS[schema]
    model, kw = C._model_setup(spec, C._model_build(spec)(), "cpu", "reference")
    ### from_checkpoint builds the model in float32 and loads the weights into it:
    ### float32-representable parameters make the float64 round trip exact.
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.float())
    path = tmp_path / f"isla_{schema}.mdlus"
    model.save(path)
    assert set(_stored_args(path)) == set(
        inspect.signature(ISLA.__init__).parameters
    ) - {"self"}
    loaded = Module.from_checkpoint(str(path)).to(dtype=spec.dtype).eval()
    assert isinstance(loaded, ISLA)
    ### reference_kernels is an execution choice the checkpoint does not store: the
    ### loaded model starts with every region at "auto", so it takes the saved model's choices.
    assert (
        loaded._geo_kernel
        == loaded._tree_read_kernel
        == loaded._geometry_kernel
        == "auto"
    )
    C._model_kernels(loaded, "reference")
    assert _comparable(loaded) == _comparable(model)
    ### the stored order of the schemas, which the dict comparison ignores (only the
    ### outputs' order changes the outputs)
    assert [
        list(s)
        for s in (
            loaded.output_schema,
            loaded.global_schema,
            *loaded.boundary_schemas.values(),
        )
    ] == [
        list(s)
        for s in (
            model.output_schema,
            model.global_schema,
            *model.boundary_schemas.values(),
        )
    ]
    with torch.no_grad():
        a, b = outputs(loaded(**kw)), outputs(model(**kw))
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a), schema


#: A version other than this model's, of a checkpoint's metadata; the state-dict
#: versions torch records in its weights (``ISLA._version``) are one below and one above
#: the model's.
OTHER_RELEASE = "another"


def _rewritten(src: Path, dst: Path, version=None, state_dict=None) -> Path:
    """A copy of the checkpoint ``src`` with its metadata version and/or its weights
    (``model.pt``) replaced."""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "metadata.json" and version is not None:
                meta = json.loads(data)
                meta["mdlus_file_version"] = version
                data = json.dumps(meta).encode()
            if item.filename == "model.pt" and state_dict is not None:
                buf = io.BytesIO()
                torch.save(state_dict, buf)
                data = buf.getvalue()
            zout.writestr(item, data)
    return dst


@pytest.mark.parametrize(
    "other", [ISLA._version - 1, ISLA._version + 1], ids=["older", "newer"]
)
@pytest.mark.parametrize("file_name", list(CHECKPOINTS))
def test_isla_checkpoint_of_another_version_is_refused(tmp_path, file_name, other):
    """ISLA keeps no upgrade path. A reference checkpoint relabelled as written by
    another version (its metadata version and its weights' state-dict version) is
    refused by ``Module.from_checkpoint`` with an error naming both versions, and by
    ``Module.load`` into a current model with the state-dict version named; with only
    its weights' state-dict version other (the metadata current), ``from_checkpoint``
    refuses it too, and so do ``load_state_dict`` and a DDP-style wrapper. The
    checkpoint as committed then loads into the same model, its weights bitwise."""
    current = ISLA.__model_checkpoint_version__
    cfg, _ = CHECKPOINTS[file_name]
    src = DATA_DIR / file_name
    with zipfile.ZipFile(src) as z:
        sd = torch.load(io.BytesIO(z.read("model.pt")), weights_only=True)
    assert sd._metadata[""]["version"] == ISLA._version
    sd._metadata[""]["version"] = other
    old = _rewritten(src, tmp_path / "old.mdlus", OTHER_RELEASE, sd)
    with pytest.raises(
        IOError,
        match=re.escape(
            f"version {OTHER_RELEASE} is not compatible with current version {current}"
        ),
    ):
        Module.from_checkpoint(str(old))
    sd_version = (
        rf"state-dict version {other}; ISLA {re.escape(current)} "
        rf"loads version {ISLA._version}"
    )
    fresh = ISLA(**cfg)
    with pytest.raises(RuntimeError, match=sd_version):
        fresh.load(str(old), strict=True)
    weights_only = _rewritten(src, tmp_path / "weights.mdlus", None, sd)
    with pytest.raises(RuntimeError, match=sd_version):
        Module.from_checkpoint(str(weights_only))
    with pytest.raises(RuntimeError, match="no upgrade path"):
        fresh.load_state_dict(sd, strict=False)
    wrapped = torch.nn.ModuleDict({"module": fresh}).state_dict()
    wrapped._metadata["module"]["version"] = other
    with pytest.raises(RuntimeError, match=sd_version):
        torch.nn.ModuleDict({"module": fresh}).load_state_dict(wrapped)
    fresh.load(str(src))  # the current checkpoint loads
    _assert_dict_close(fresh.state_dict(), sd, atol=0.0, rtol=0.0)


def test_isla_state_dict_of_reordered_fields_is_refused(tmp_path):
    """The weights follow the schemas' declaration order, so a state dict records it:
    one loads into a model that declares the same output fields, or global or boundary
    fields of one rank, in another order only with a RuntimeError naming the saved
    order (``load_state_dict`` and ``Module.load``). Renamed fields, fields of
    different ranks interleaved otherwise, and the order as saved load."""
    small = dict(hidden=16, n_layers=1, n_slices=4)
    s, v = {"rank": 0}, {"rank": 1}
    out, glob, bnd = (
        {"p": s, "q": s},
        {"a": v, "b": v, "c": s},
        {"t": s, "u": s, "w": v},
    )

    def model(output_schema=out, global_schema=glob, wall=bnd):
        return ISLA(
            output_schema=output_schema,
            global_schema=global_schema,
            boundary_schemas={"wall": wall},
            **small,
        )

    saved = model()
    sd = saved.state_dict()
    assert sd._metadata[""]["fields"] == {
        "output_schema": ["p", "q"],
        "global_schema rank-1": ["a", "b"],
        "global_schema rank-0": ["c"],
        "boundary_schemas['wall'] rank-0": ["t", "u"],
        "boundary_schemas['wall'] rank-1": ["w"],
    }
    swapped = {
        "output_schema": ({"q": s, "p": s}, "output_schema"),
        "global_schema": ({"b": v, "a": v, "c": s}, "global_schema rank-1"),
        "wall": ({"u": s, "t": s, "w": v}, "boundary_schemas['wall'] rank-0"),
    }
    for arg, (schema, group) in swapped.items():
        stored = sd._metadata[""]["fields"][group]
        with pytest.raises(
            RuntimeError, match=re.escape(f"the {group} fields in the order {stored!r}")
        ):
            model(**{arg: schema}).load_state_dict(sd)
    path = tmp_path / "isla.mdlus"
    saved.save(str(path))
    with pytest.raises(RuntimeError, match="would load into the wrong fields"):
        model(output_schema=swapped["output_schema"][0]).load(str(path))
    model(output_schema={"pressure": s, "q": s}).load_state_dict(sd)
    model(
        global_schema={"c": s, "a": v, "b": v}, wall={"w": v, "t": s, "u": s}
    ).load_state_dict(sd)
    model().load(str(path))


def test_isla_state_dict_without_metadata_loads_with_a_warning(caplog):
    """A state dict copied into a plain ``dict`` has lost torch's metadata, so its
    state-dict version and its fields' order cannot be checked: it loads, with a
    warning that says so."""
    kw = dict(
        output_schema={"p": {"rank": 0}},
        boundary_schemas={"wall": {}},
        hidden=16,
        n_layers=1,
        n_slices=4,
    )
    sd = ISLA(**kw).state_dict()
    with caplog.at_level(
        logging.WARNING, logger="physicsnemo.experimental.models.isla.model"
    ):
        ISLA(**kw).load_state_dict(sd)
        assert not caplog.records
        ISLA(**kw).load_state_dict(dict(sd))
    assert [r.getMessage() for r in caplog.records] == [
        "ISLA: this state dict has no metadata (torch's state_dict()._metadata, which a copy into a "
        "plain dict or torch.distributed.checkpoint's full state dict drops), so neither its "
        "state-dict version nor its fields' order is checked"
    ]


def test_isla_unknown_constructor_argument_is_a_type_error():
    """An argument ISLA does not take is a TypeError naming it (no silent ``**kwargs``),
    in either task."""
    name = "nonsense"
    for interior in (False, True):
        with pytest.raises(TypeError, match=f"unexpected keyword argument '{name}'"):
            ISLA(
                hidden=16, n_layers=1, n_slices=8, interior=interior, **IO, **{name: 1}
            )
