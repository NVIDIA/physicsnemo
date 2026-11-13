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

"""Small model cases shared by the release tests.

Each case is an ISLA configuration, its boundary meshes and its inputs, built on the
CPU from seeded ``torch.Generator`` draws. ``_release_cases.py`` records the reference
outputs of these cases in ``data/`` and ``test_isla_release.py`` compares against
them, with curved meshes from the exact builders (``_exact_*``): correctly rounded
operations only (float64 draws, ``+ - * /`` and numpy's ``sqrt``, one cast to float32),
so that the inputs are the same bits on every platform.
"""

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch

from physicsnemo.experimental.models.isla import ISLA

D = torch.float64


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def param_generator(seed: int, name: str) -> torch.Generator:
    """The CPU generator of one parameter's seeded fill, seeded by the base ``seed``
    and the parameter's qualified ``name`` (a SHA-256 of both, stable across processes),
    so that adding, removing or reordering other parameters leaves its values bitwise.
    Shared with the release writer (``_release_cases._fill``)."""
    key = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return torch.Generator(device="cpu").manual_seed(
        int.from_bytes(key[:8], "little") >> 1
    )


def _to(inputs: dict, device, keys=None) -> dict:
    return {k: v.to(device) for k, v in inputs.items() if keys is None or k in keys}


def _directions(g: torch.Generator, *shape: int) -> torch.Tensor:
    """Unit vectors ``shape`` (last axis the dimension) without a square root: the
    inverse stereographic projection ``(2x, |x|^2 - 1) / (|x|^2 + 1)`` of Gaussian
    ``x`` in one dimension fewer (unit to rounding)."""
    x = torch.randn(*shape[:-1], shape[-1] - 1, generator=g, dtype=D)
    s = x[..., 0] * x[..., 0]
    for k in range(1, x.shape[-1]):
        s = s + x[..., k] * x[..., k]
    return torch.cat([2 * x, (s - 1)[..., None]], -1) / (s + 1)[..., None]


def _unit(x: torch.Tensor) -> torch.Tensor:
    """``x`` normalized along its last axis (orientation decisions, the icosphere)."""
    sq = x[..., 0] * x[..., 0]
    for k in range(1, x.shape[-1]):
        sq = sq + x[..., k] * x[..., k]
    return x / sq.sqrt()[..., None]


def _icosphere(radius: float, subdivisions: int):
    """``sphere_icosahedral.load(radius, subdivisions)``: its cells, its float32 points
    to an ulp, projected by :func:`_unit` (its ``torch.norm`` is not correctly rounded:
    torch 2.13's moved the points by an ulp from 2.12's on Arm CPUs)."""
    from physicsnemo.mesh.primitives.surfaces import icosahedron_surface

    ico = icosahedron_surface.load()
    ### Its vertices before the norm, (0, +-1, +-phi) cyclically (1 / |v| = 0.53,
    ### phi / |v| = 0.85).
    phi = (1.0 + 5.0**0.5) / 2.0
    raw = ico.points.sign() * torch.where(ico.points.abs() > 0.7, phi, 1.0)
    m = ico.with_points(_unit(raw)).subdivide(levels=subdivisions, filter="linear")
    return m.with_points(_unit(m.points) * radius)


def _cloud(n, d=3, seed=0, *, dtype=D, scale=None):
    """``_meshes.cloud`` (the same draws), with its unit vectors normalized by
    :func:`_unit` and float32 clouds drawn in float64 and rounded once."""
    g = _gen(seed)
    scale = (
        torch.arange(d, 0, -1, dtype=D)
        if scale is None
        else torch.as_tensor(scale, dtype=D)
    )
    pts = torch.randn(n, d, generator=g, dtype=D) * scale
    nrm = _directions(g, n, d)
    w = torch.rand(n, generator=g, dtype=D) + 0.5
    drv = _directions(g, d)
    return tuple(t.to(dtype) for t in (pts, nrm, w, drv))


def _cotangent(t: torch.Tensor, seed: int) -> torch.Tensor:
    """A fixed random cotangent for ``t`` (drawn in float64 on the CPU)."""
    return torch.randn(t.shape, generator=_gen(seed), dtype=D).to(t.device, t.dtype)


def _params(module: torch.nn.Module, seed: int, std: float = 0.3) -> dict:
    """Random values for every parameter of ``module`` (``param.<name>``), float64,
    each from its own generator keyed by ``seed`` and the parameter's name, so that a
    change to one parameter leaves every other one's values bitwise."""
    return {
        f"param.{n}": torch.randn(p.shape, generator=param_generator(seed, n), dtype=D)
        * std
        for n, p in module.named_parameters()
    }


def _oriented(points, cells, outward):
    from physicsnemo.mesh import Mesh

    m = Mesh(points=points, cells=cells)
    cells = cells.clone()
    flip = (m.cell_normals * outward(m.cell_centroids)).sum(-1) < 0
    cells[flip, -2:] = cells[flip, -2:].flip(-1)
    return points, cells


def _near_wall(dim, n, seed, band, radius=1.0):
    g = _gen(seed)
    dirs = _directions(g, n, dim)
    return dirs * (radius + band * (2 * torch.rand(n, 1, dtype=D, generator=g) - 1))


def _cube(levels=0):
    from physicsnemo.mesh import Mesh
    from physicsnemo.mesh.primitives.surfaces import cube_surface

    m = cube_surface.load()
    m = Mesh(points=m.points.double(), cells=m.cells)
    return m.subdivide(levels=levels, filter="linear") if levels else m


_MODEL_IO = dict(
    output_schema={"p": {"rank": 0}, "v": {"rank": 1}},
    boundary_schemas={"wall": {}},
    global_schema={"g": {"rank": 1}},
)


_MODEL_SMALL = dict(hidden=32, n_layers=2, n_slices=8)


#: A surface model with every kind of input and output: boundary fields of rank 0, 1
#: and 2 on two conditions, a rank-1 field that defines the velocity scale (so the
#: derived direction g joins the global one), a rank-0 global that defines a density,
#: a dimensionless global, outputs of rank 0, 1 and 2 (symmetric and general), odd
#: outputs of rank 1 and 2, units, an output constant, and output_stats.
_RICH = dict(
    output_schema={
        "p": {"rank": 0, "units": {"U": 2, "rho": 1}},
        "v": {"rank": 1, "units": {"U": 1}},
        "tau": {"rank": 2, "symmetric": True, "units": {"U": 2, "rho": 1}},
        "grad_v": {"rank": 2, "units": {"U": 1, "L": -1}},
        "omega": {"rank": 1, "parity": "odd", "units": {"U": 1, "L": -1}},
        "spin": {"rank": 2, "parity": "odd"},
        "nut": {"rank": 0, "units": {"U": 1, "L": 1}, "scale": 1e-3},
    },
    boundary_schemas={
        "wall": {
            "trace": {"rank": 0},
            "traction": {"rank": 1, "units": {"U": 1}, "defines": "U"},
            "stress": {"rank": 2, "symmetric": True, "units": {"U": 2, "rho": 1}},
        },
        "outlet": {"p_out": {"rank": 0, "units": {"U": 2, "rho": 1}}},
    },
    global_schema={
        "g": {"rank": 1},
        "rho": {"rank": 0, "defines": "rho"},
        "re": {"rank": 0},
    },
    output_stats={"p": {"mean": 0.3, "std": 2.0}, "omega": {"mean": 0.0, "std": 1.5}},
)


@dataclass(frozen=True)
class _ModelSpec:
    """One model case: the constructor's arguments, the boundary meshes (condition name
    -> maker of ``(points, cells)`` in float64), the surface task's prediction
    boundaries or the interior task's number of prediction points, and the dtype."""

    cfg: dict
    meshes: dict
    dtype: torch.dtype = D
    prediction_boundaries: tuple = ()
    n_queries: int = 0
    d: int = 3
    seed: int = 60


def _soup_cells(n, d, seed, offset=0.0):
    """``n`` random simplices (a triangle soup in 3-D, segments in 2-D) about an
    anisotropic cloud: each vertex its point plus a Gaussian offset of 0.15."""
    pts, _, _, _ = _cloud(n, d=d, seed=seed)
    corners = 0.15 * torch.randn(n, d, d, generator=_gen(seed + 1), dtype=D)
    verts = (pts[:, None] + offset + corners).reshape(-1, d)
    return verts, torch.arange(n * d).reshape(n, d)


def _body(x: torch.Tensor) -> torch.Tensor:
    """A smooth, orientation-preserving deformation of the unit sphere (or circle)
    into a lopsided body with distinct axes, off the origin (``+ - *`` only)."""
    if x.shape[-1] == 2:
        u, v = x[:, 0], x[:, 1]
        return torch.stack([1.4 * u + 0.2 * v * v + 0.3, 0.8 * v + 0.1 * u - 0.2], -1)
    u, v, w = x[:, 0], x[:, 1], x[:, 2]
    return torch.stack(
        [1.5 * u + 0.2 * v * v + 0.3, v + 0.1 * w - 0.2, 0.6 * w + 0.15 * u * u + 0.1],
        -1,
    )


def _sphere_cells(subdivisions):
    """An icosphere deformed by :func:`_body`, outward."""
    m = _icosphere(1.0, subdivisions)
    p, c = _oriented(m.points.double(), m.cells, lambda x: x)
    return _body(p), c


def _shell_cells(levels, half):
    """The box surface ``[-half_k, half_k]`` (each a multiple of 1/2), outward."""
    m = _cube(levels)
    return m.points * (2 * torch.tensor(half, dtype=D)), m.cells


def _circle_cells(n=48):
    """A circle polyline deformed by :func:`_body`, outward."""
    from physicsnemo.mesh.primitives.curves import circle_2d

    m = circle_2d.load(n_points=n)
    p, c = _oriented(m.points.double(), m.cells, lambda x: x)
    return _body(p), c


### The release cases' meshes (``_release_cases.py``), the same bits on every CPU:
### float64 ``+ - * /`` and numpy's square root only. torch's CPU sqrt goes through
### MKL's vector math, which is not correctly rounded and changes with the CPU.


def _exact_unit(x: torch.Tensor) -> torch.Tensor:
    """:func:`_unit` of float64 ``x`` with a correctly rounded square root."""
    sq = x[..., 0] * x[..., 0]
    for k in range(1, x.shape[-1]):
        sq = sq + x[..., k] * x[..., k]
    return x / torch.from_numpy(np.sqrt(sq.numpy()))[..., None]


#: The icosahedron: its vertices ``(0, +-1, +-phi)`` cyclically and its faces.
_PHI = (1.0 + math.sqrt(5.0)) / 2.0
_ICO_POINTS = [
    (-1, _PHI, 0), (1, _PHI, 0), (-1, -_PHI, 0), (1, -_PHI, 0),
    (0, -1, _PHI), (0, 1, _PHI), (0, -1, -_PHI), (0, 1, -_PHI),
    (_PHI, 0, -1), (_PHI, 0, 1), (-_PHI, 0, -1), (-_PHI, 0, 1),
]  # fmt: skip
_ICO_CELLS = [
    (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
    (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
    (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
    (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
]  # fmt: skip


def _exact_sphere_cells(subdivisions):
    """:func:`_sphere_cells` from :data:`_ICO_POINTS`: each triangle split into four at
    its edges' midpoints ``subdivisions`` times, then projected by :func:`_exact_unit`."""
    p = torch.tensor(_ICO_POINTS, dtype=D)
    c = torch.tensor(_ICO_CELLS)
    for _ in range(subdivisions):
        e = torch.cat([c[:, [0, 1]], c[:, [1, 2]], c[:, [2, 0]]]).sort(-1).values
        edges, inv = e.unique(dim=0, return_inverse=True)
        (a, b, d), (ab, bd, da) = c.T, len(p) + inv.view(3, -1)
        p = torch.cat([p, (p[edges[:, 0]] + p[edges[:, 1]]) * 0.5])
        children = ((a, ab, da), (ab, b, bd), (da, bd, d), (ab, bd, da))
        c = torch.cat([torch.stack(t, -1) for t in children])
    p, c = _oriented(_exact_unit(p), c, lambda x: x)
    return _body(p), c


def _exact_circle_cells(n=48):
    """:func:`_circle_cells` from the rational parametrization ``(1 - t^2, 2 t) / (1 + t^2)``:
    ``t`` on an even grid of ``[0, tan(pi / 8)]`` gives ``n / 8`` points of each octant
    (the second reflected, the other quadrants rotated)."""
    k = n // 8
    t = torch.arange(k + 1, dtype=D) * ((math.sqrt(2.0) - 1.0) / k)
    s = 1 + t * t
    x, y = (1 - t * t) / s, 2 * t / s
    q = torch.cat([torch.stack([x, y], -1)[:k], torch.stack([y, x], -1)[1:].flip(0)])
    for _ in range(3):
        q = torch.cat([q, torch.stack([-q[-2 * k :, 1], q[-2 * k :, 0]], -1)])
    c = torch.stack([torch.arange(n), torch.arange(1, n + 1) % n], -1)
    p, c = _oriented(q, c, lambda x: x)
    return _body(p), c


def _model_build(spec: _ModelSpec):
    def build():
        from test.experimental.models.isla._meshes import live

        g = _gen(spec.seed)
        x = {}
        model = ISLA(**spec.cfg)
        schemas = model.boundary_schemas
        for name, maker in spec.meshes.items():
            p, c = maker()
            x[f"in.{name}.points"], x[f"in.{name}.cells"] = p, c
            for f, fs in schemas[name].items():
                t = torch.randn(c.shape[0], *[spec.d] * fs.rank, generator=g, dtype=D)
                x[f"in.{name}.{f}"] = t + t.transpose(-1, -2) if fs.symmetric else t
        for f, fs in model.global_schema.items():
            if fs.rank == 1:
                x[f"in.global.{f}"] = _directions(g, spec.d)
            else:
                x[f"in.global.{f}"] = 1.0 + torch.rand((), generator=g, dtype=D)
        if spec.n_queries:
            x["in.queries"] = _near_wall(
                spec.d, spec.n_queries, spec.seed + 1, 0.8, 2.0
            )
        params = _params(model, spec.seed + 2, std=0.1)
        if model.kernel_reads is not None:
            ### (in float64: a float32 randn draws different values on x86 and on Arm)
            live(model.double(), spec.seed + 3)
            for n, p in model.named_parameters():
                if n.startswith("kernel_reads.") and n.endswith(".out.weight"):
                    params[f"param.{n}"] = p.detach().double()
        ### Every parameter in one flat input.
        x["in.params"] = torch.cat([v.reshape(-1) for v in params.values()])
        return x

    return build


def _model_kernels(model, variant: str):
    """The reference of every piece (CPU, and the CUDA reference variant), or every
    piece's fast path where it can run (``auto``)."""
    ref = variant == "reference"
    model.reference_kernels = ref


def _model_setup(spec: _ModelSpec, inputs, device, variant):
    """The loaded model and the arguments of its forward pass."""
    x = _to(inputs, device)
    dt = spec.dtype
    model = ISLA(**spec.cfg).to(device, dt).eval()
    with torch.no_grad():
        flat, i = x["in.params"], 0
        for p in model.parameters():
            p.copy_(flat[i : i + p.numel()].view_as(p))
            i += p.numel()
    _model_kernels(model, variant)
    return model, model_inputs(spec, x, device, model)


def _nested(fields: dict) -> dict:
    """``{"a.b": t}`` as ``{"a": {"b": t}}`` (the schemas' dotted names as groups)."""
    out = {}
    for name, t in fields.items():
        *groups, leaf = name.split(".")
        node = out
        for g in groups:
            node = node.setdefault(g, {})
        node[leaf] = t
    return out


def model_inputs(spec: _ModelSpec, inputs: dict, device, model) -> dict:
    """The keyword arguments of the forward pass of ``model`` (an ISLA of the case's
    configuration) on a model case's inputs (``build()``), on ``device`` in the
    case's dtype."""
    from physicsnemo.mesh import Mesh

    x, dt, schemas = _to(inputs, device), spec.dtype, model
    meshes = {}
    for name in spec.meshes:
        data = {f: x[f"in.{name}.{f}"].to(dt) for f in schemas.boundary_schemas[name]}
        meshes[name] = Mesh(
            points=x[f"in.{name}.points"].to(dt),
            cells=x[f"in.{name}.cells"],
            cell_data=_nested(data),
        )
    kw = dict(
        boundary_meshes=meshes,
        global_data=_nested(
            {f: x[f"in.global.{f}"].to(dt) for f in schemas.global_schema}
        ),
    )
    if spec.n_queries:
        kw["prediction_points"] = x["in.queries"].to(dt)
    else:
        kw["prediction_boundaries"] = list(spec.prediction_boundaries)
    return kw


def _cotangent_loss(fields: dict, seed: int) -> torch.Tensor:
    return sum(
        (t * _cotangent(t, seed + i)).sum()
        for i, (_, t) in enumerate(sorted(fields.items()))
    )


_MODEL_SPECS = {
    "surface": lambda dt: _ModelSpec(
        dict(**_MODEL_SMALL, **_MODEL_IO),
        {"wall": lambda: _soup_cells(150, 3, 61)},
        dt,
        prediction_boundaries=("wall",),
    ),
    "rich_surface": lambda dt: _ModelSpec(
        dict(**_MODEL_SMALL, **_RICH),
        {
            "wall": lambda: _soup_cells(120, 3, 62),
            "outlet": lambda: _soup_cells(40, 3, 63, offset=6.0),
        },
        dt,
        prediction_boundaries=("wall",),
    ),
    "dtt": lambda dt: _ModelSpec(
        dict(**_MODEL_SMALL, **_MODEL_IO, interior=True),
        {"wall": lambda: _sphere_cells(2)},
        dt,
        n_queries=60,
    ),
    "dtt_2d": lambda dt: _ModelSpec(
        dict(**_MODEL_SMALL, **_MODEL_IO, interior=True, n_spatial_dims=2),
        {"wall": _circle_cells},
        dt,
        n_queries=60,
        d=2,
    ),
    ### About 2,000 cells over two conditions, 512 tokens drawn (at the fixed start).
    "dtt_draw": lambda dt: _ModelSpec(
        dict(
            **_MODEL_SMALL,
            output_schema=_MODEL_IO["output_schema"],
            global_schema=_MODEL_IO["global_schema"],
            boundary_schemas={"wall": {}, "farfield": {}},
            interior=True,
            n_boundary_tokens=512,
        ),
        {
            "wall": lambda: _sphere_cells(3),
            "farfield": lambda: _shell_cells(3, (4.0, 3.0, 2.5)),
        },
        dt,
        n_queries=60,
    ),
}
