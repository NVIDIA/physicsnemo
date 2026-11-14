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

"""The configurations and inputs that the reference files of ``data/`` were recorded
with, shared with the tests that compare against those files, and the writers of the
reference outputs and checkpoints (:data:`WRITERS`).

Regenerate reference files on the CPU with::

    python -m test.experimental.models.isla._release_cases FILE ... | all

Regenerate only what a change is meant to move, and say so in the commit.
"""

import dataclasses
import sys
from pathlib import Path

import torch

from physicsnemo.core.module import Module
from physicsnemo.experimental.models.isla import ISLA
from test.experimental.models.isla import _model_cases as C
from test.experimental.models.isla._meshes import outputs, soup
from test.experimental.models.isla._model_cases import param_generator

DATA_DIR = Path(__file__).parent / "data"
D, F = torch.float64, torch.float32

#: Every configuration declares one scalar and one vector output, one boundary condition
#: without fields and one global direction.
IO = dict(
    output_schema={"p": {"rank": 0}, "v": {"rank": 1}},
    boundary_schemas={"wall": {}},
    global_schema={"g": {"rank": 1}},
)

#: One small configuration in its two forms: a surface task and an interior task
#: (the constructor's ``interior`` flag).
CONFIGS = {
    "default": dict(hidden=32, n_layers=2, n_slices=8, **IO),
    "dtt": dict(hidden=32, n_layers=2, n_slices=8, interior=True, **IO),
}


def _fill(model: torch.nn.Module, seed: int) -> torch.nn.Module:
    """Every parameter of ``model`` set to ``randn * 0.1`` from its own generator
    (:func:`param_generator`); returns the model."""
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(
                torch.randn(p.shape, generator=param_generator(seed, n), dtype=p.dtype)
                * 0.1
            )
    return model


def _instantiate_model(seed: int = 0, **kwargs) -> ISLA:
    """Build an ISLA with reproducible random parameters (MOD-008b pattern; each
    parameter from its own name-keyed generator, :func:`_fill`), forced to the eager
    geometry region (the one the reference outputs were recorded with)."""
    model = _fill(ISLA(**kwargs), seed)
    model._geo_kernel = "eager"
    return model.eval()


def _make_inputs(device, query: bool = False, seed: int = 1234) -> dict:
    """Deterministic mesh inputs: one case, a 64-cell boundary soup with unit normals
    and positive effective measures, one global direction, and 40 prediction points
    for the interior task."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = 64
    pts = torch.randn(n, 3, generator=gen) * torch.tensor([3.0, 2.0, 1.0])
    nrm = torch.nn.functional.normalize(torch.randn(n, 3, generator=gen), dim=-1)
    w = torch.rand(n, generator=gen) + 0.5
    g = torch.nn.functional.normalize(torch.randn(3, generator=gen), dim=-1)
    inputs = dict(
        boundary_meshes={"wall": soup(pts, nrm, w).to(device)},
        global_data={"g": g.to(device)},
    )
    if query:
        inputs["prediction_points"] = (torch.randn(40, 3, generator=gen) * 2.0).to(
            device
        )
    else:
        inputs["prediction_boundaries"] = ["wall"]
    return inputs


#: The committed reference checkpoints: the file, its configuration and the seed of
#: its parameters. The interior model's is narrower (hidden 16) to keep ``data/`` within its
#: budget.
CHECKPOINTS = {
    "isla_default_v1.mdlus": (CONFIGS["default"], 7),
    "isla_dtt_v1.mdlus": (dict(CONFIGS["dtt"], hidden=16), 8),
}


def _forward_golden(config: str) -> dict:
    """What ``isla_<config>_v1.pth`` stores: the CPU outputs of ``CONFIGS[config]``."""
    model = _instantiate_model(**CONFIGS[config])
    with torch.no_grad():
        return outputs(model(**_make_inputs("cpu", query=(config != "default"))))


#: The reference outputs of ``isla_release_v2.pth``: a model case of
#: ``_model_cases.py`` (its configuration, inputs from correctly rounded operations,
#: the same bits on every platform, and random parameters) and its dtype, with the
#: meshes of :data:`V2_MESHES`. In evaluation mode; the draw at its fixed start, at
#: the default tree-read angle.
RELEASE_V2 = "isla_release_v2.pth"
V2_CONFIGS = {
    "surface_f64": ("surface", D),
    "dtt_f64": ("dtt", D),
    "rich_surface_f32": ("rich_surface", F),
    "dtt_2d_f64": ("dtt_2d", D),
    "dtt_draw_f32": ("dtt_draw", F),  # the in-model draw
}

#: The release cases' curved meshes, from the exact builders of ``_model_cases.py``
#: (the shared builders' float32 icosphere and circle change with the CPU; the shell's
#: points are exact).
V2_MESHES = {
    "dtt": {"wall": lambda: C._exact_sphere_cells(2)},
    "dtt_2d": {"wall": C._exact_circle_cells},
    "dtt_draw": {
        "wall": lambda: C._exact_sphere_cells(3),
        "farfield": lambda: C._shell_cells(3, (4.0, 3.0, 2.5)),
    },
}


def _v2_case(name: str, device: str):
    case, dtype = V2_CONFIGS[name]
    spec = C._MODEL_SPECS[case](dtype)
    if case in V2_MESHES:
        spec = dataclasses.replace(spec, meshes=V2_MESHES[case])
    variant = "reference" if device == "cpu" else "auto"
    return C._model_setup(spec, C._model_build(spec)(), device, variant), dtype


def _v2_outputs(name: str, device: str) -> dict:
    (model, kw), _ = _v2_case(name, device)
    with torch.no_grad():
        return outputs(model(**kw))


#: query_chunks' golden: the interior model in float64 over three ranges of its 60 points.
CHUNKS = ((0, 20), (20, 45), (45, 60))


def _chunks_golden(device: str) -> dict:
    """``query_chunks`` of ``dtt_f64``: each range's outputs (a fixed cotangent on
    each), then, after ``chunks.backward()``, every parameter's gradient as its
    projection on a fixed random direction and the sum of the magnitudes of the
    projection's terms (the scale its rounding is set by)."""
    (model, kw), _ = _v2_case("dtt_f64", device)
    chunks = model.query_chunks(**kw)
    out = {"pair_counts": chunks.pair_counts.cpu()}
    for i, (a, b) in enumerate(CHUNKS):
        fields = outputs(chunks(a, b))
        C._cotangent_loss(fields, 90 + 10 * i).backward()
        out.update({f"range{i}.{k}": t.detach().cpu() for k, t in fields.items()})
    chunks.backward()
    gen = torch.Generator().manual_seed(95)
    proj, scale = [], []
    for p in model.parameters():
        r = torch.randn(p.shape, generator=gen, dtype=D)
        g = p.grad.detach().cpu().double()
        proj.append((g * r).sum())
        scale.append((g * r).abs().sum())
    out["grad_projections"], out["grad_scales"] = torch.stack(proj), torch.stack(scale)
    return out


def _backbone_golden() -> torch.Tensor:
    """The pre-head token state (``norm_out``'s output) of ``CONFIGS["default"]`` on
    ``_make_inputs`` (a point-soup mesh) through ``forward``, on the CPU."""
    model = _instantiate_model(**CONFIGS["default"])
    captured = {}
    model.norm_out.register_forward_hook(
        lambda mod, i, o: captured.setdefault("x", o.detach().clone())
    )
    with torch.no_grad():
        model(**_make_inputs("cpu"))
    return captured["x"]


def _checkpoint_output(file_name: str, device) -> dict:
    """The forward pass of a committed reference checkpoint on ``_make_inputs``."""
    model = Module.from_checkpoint(str(DATA_DIR / file_name)).to(device).eval()
    with torch.no_grad():
        return outputs(model(**_make_inputs(device, query=model.interior)))


def _save_output(file_name: str, output: dict) -> None:
    torch.save(
        {"output": {k: v.detach().cpu() for k, v in output.items()}},
        DATA_DIR / file_name,
    )


def _write_checkpoint(file_name: str) -> None:
    cfg, seed = CHECKPOINTS[file_name]
    _instantiate_model(seed=seed, **cfg).save(DATA_DIR / file_name)


def _write_v2() -> None:
    golden = {name: {"output": _v2_outputs(name, "cpu")} for name in V2_CONFIGS}
    golden["query_chunks"] = _chunks_golden("cpu")
    torch.save(golden, DATA_DIR / RELEASE_V2)


#: Every file of ``data/`` test_isla_release.py reads, and how it is written (on the CPU).
WRITERS = {
    "isla_default_v1.pth": lambda: _save_output(
        "isla_default_v1.pth", _forward_golden("default")
    ),
    "isla_dtt_v1.pth": lambda: _save_output("isla_dtt_v1.pth", _forward_golden("dtt")),
    "isla_backbone_v1.pth": lambda: torch.save(
        {"default": {"norm_out": _backbone_golden()}}, DATA_DIR / "isla_backbone_v1.pth"
    ),
    "isla_default_v1.mdlus": lambda: _write_checkpoint("isla_default_v1.mdlus"),
    "isla_default_checkpoint_output_v1.pth": lambda: _save_output(
        "isla_default_checkpoint_output_v1.pth",
        _checkpoint_output("isla_default_v1.mdlus", "cpu"),
    ),
    "isla_dtt_v1.mdlus": lambda: _write_checkpoint("isla_dtt_v1.mdlus"),
    "isla_dtt_checkpoint_output_v1.pth": lambda: _save_output(
        "isla_dtt_checkpoint_output_v1.pth",
        _checkpoint_output("isla_dtt_v1.mdlus", "cpu"),
    ),
    RELEASE_V2: _write_v2,
}


def regenerate(names=None, log=print) -> None:
    """Write the named files of :data:`WRITERS` (all by default) in their order (a
    checkpoint before its output golden), on the CPU."""
    for name in WRITERS:
        if names is None or name in names:
            WRITERS[name]()
            log(f"wrote {DATA_DIR / name} ({(DATA_DIR / name).stat().st_size} bytes)")


if __name__ == "__main__":
    wanted = sys.argv[1:]
    if not wanted or (wanted != ["all"] and not set(wanted) <= set(WRITERS)):
        sys.exit(f"name the files to write ({', '.join(WRITERS)}) or `all`")
    torch.set_num_threads(1)  # the reference files are recorded on one CPU thread
    regenerate(None if wanted == ["all"] else set(wanted))
