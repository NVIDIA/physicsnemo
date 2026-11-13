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

"""ISLA's own scales: the ISLA-local keys of the field declarations, and the per-case scales.

Next to its rank spec, a field of a boundary condition's schema, of ``global_schema``
or of ``output_schema`` may carry the keys of :data:`SCALE_KEYS`: ``units``,
``defines: <name>`` and ``scale``. The ISLA module docstring, *Scales*, says what
each does. A rank-0 global that defines a scale is consumed and does not enter as an
input. :func:`split_scale_keys` pops these keys before
:meth:`~physicsnemo.mesh.fields.FieldSchema.parse`, so the shared
:class:`~physicsnemo.mesh.fields.RankSpec` grammar is untouched. Private to
:mod:`physicsnemo.experimental.models.isla`, except :func:`case_scales`, which the
package exports."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import torch
from tensordict import TensorDict

from physicsnemo.experimental.models.isla._geometry import (
    cell_second_moments,
    rms_pairwise_distance,
)
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import cell_measures
from physicsnemo.mesh.fields import FieldSchema, RankSpec

#: The built-in scale, the length unit of the boundary.
LENGTH = "L"
#: The key with which a field defines a scale (its name, the key's value).
DEFINES = "defines"
#: The ISLA-local keys of a field declaration.
SCALE_KEYS = ("units", DEFINES, "scale")


def _check_local(kind: str, name: str, local: Mapping) -> dict:
    """One field's ISLA-local keys, checked and made JSON-safe.
    :func:`parse_scales` checks, against every field, that its ``units`` name
    defined scales."""
    where = f"{kind}[{name!r}]"
    out = {}
    if "units" in local:
        units = local["units"]
        if not isinstance(units, Mapping):
            raise TypeError(
                f"{where}: units must be a mapping of scale name to exponent; got {units!r}"
            )
        for k, e in units.items():
            if not isinstance(k, str):
                raise TypeError(f"{where}: units names a scale by a string; got {k!r}")
            if isinstance(e, bool) or not isinstance(e, Real) or not math.isfinite(e):
                raise TypeError(
                    f"{where}: the exponent of {k!r} must be a finite number; got {e!r}"
                )
        out["units"] = {
            k: (int(e) if float(e).is_integer() else float(e))
            for k, e in units.items()
            if e
        }
    if DEFINES in local:
        defined = local[DEFINES]
        if not isinstance(defined, str) or not defined:
            raise TypeError(
                f"{where}: {DEFINES} names a scale by a non-empty string; got {defined!r}"
            )
        if defined == LENGTH:
            raise ValueError(
                f"{where}: {LENGTH!r} is the built-in length unit; a field cannot define it"
            )
        if kind == "output_schema":
            raise ValueError(f"{where}: an output cannot define a scale")
        out[DEFINES] = defined
    if "scale" in local:
        c = local["scale"]
        if (
            isinstance(c, bool)
            or not isinstance(c, Real)
            or not (math.isfinite(c) and c > 0)
        ):
            raise ValueError(
                f"{where}: scale must be a positive finite number; got {c!r}"
            )
        out["scale"] = float(c)
    return out


def split_scale_keys(kind: str, specs):
    """``specs`` without the ISLA-local keys, and those keys by dotted field name.

    A field is a :class:`~physicsnemo.mesh.fields.RankSpec` or a mapping with a
    non-mapping ``rank``. Any other mapping is a group, whose fields take its name as
    a prefix (the grammar's nested and dotted forms give the same dotted names).
    Everything else is left for :meth:`~physicsnemo.mesh.fields.FieldSchema.parse`
    to accept or refuse."""
    extras: dict[str, dict] = {}
    if not isinstance(specs, Mapping):
        return specs, extras

    def walk(group: Mapping, prefix: str) -> dict:
        out = {}
        for name, value in group.items():
            dotted = f"{prefix}.{name}" if prefix else name
            if isinstance(value, Mapping) and not isinstance(value, RankSpec):
                if "rank" in value and not isinstance(value["rank"], Mapping):
                    local = {k: value[k] for k in SCALE_KEYS if k in value}
                    if local:
                        extras[str(dotted)] = _check_local(kind, str(dotted), local)
                    value = {k: v for k, v in value.items() if k not in SCALE_KEYS}
                else:
                    value = walk(value, dotted)
            out[name] = value
        return out

    return walk(specs, ""), extras


def merged(schema, extras: Mapping):
    """The declaration a checkpoint stores: the parsed ``schema`` (``None`` when
    empty) with each field's ISLA-local keys merged back."""
    if not extras:
        return schema or None
    return {name: {**spec, **extras.get(name, {})} for name, spec in schema.items()}


def check_output_stats(stats, output_schema) -> dict | None:
    """``output_stats`` (``{name: {"mean", "std"}}``, the format of
    ``NormalizeMeshFields.stats``; a ``type`` key is ignored) as JSON-safe floats.
    Each name must be a declared output and each std positive. The mean of a field
    of rank 1 or more must be zero, since a nonzero mean breaks rotation covariance."""
    if stats is None:
        return None
    if not isinstance(stats, Mapping):
        raise TypeError(
            f"output_stats must be a mapping of output name to {{mean, std}}; got {stats!r}"
        )
    out = {}
    for name, s in stats.items():
        if name not in output_schema:
            raise ValueError(
                f"output_stats[{name!r}]: not a declared output ({list(output_schema)!r})"
            )
        mean = torch.as_tensor(s["mean"], dtype=torch.float64)
        std = torch.as_tensor(s["std"], dtype=torch.float64)
        if std.numel() != 1 or not float(std) > 0:
            raise ValueError(
                f"output_stats[{name!r}]: std must be one positive number; got {s['std']!r}"
            )
        if output_schema[name].rank == 0:
            if mean.numel() != 1:
                raise ValueError(
                    f"output_stats[{name!r}]: a scalar's mean is one number; got {s['mean']!r}"
                )
        elif bool((mean != 0).any()):
            raise ValueError(
                f"output_stats[{name!r}]: the mean of a field of rank {output_schema[name].rank} must be 0 "
                f"(a nonzero mean breaks rotation covariance); got {s['mean']!r}"
            )
        out[name] = {
            "mean": float(mean.reshape(-1)[0]) if mean.numel() else 0.0,
            "std": float(std),
        }
    return out


@dataclass(frozen=True)
class DeclaredScales:
    """The scales the declarations define (:func:`parse_scales`), as the model uses
    them. Per field, the units and the constant ``scale`` of the outputs, the rank-0
    globals and each condition's fields. The rank-1 sources of the one RMS scale (the
    boundary's ``(condition, field)`` pairs or one global) and its name. The rank-0
    globals whose values are scales (scale -> field). The checked ``output_stats``.
    Whether any scale is declared (``scaled``), and whether the head runs in the
    parameters' precision under autocast (``fp32_head``)."""

    output_units: dict
    output_scale: dict
    global_units: dict
    global_scale: dict
    boundary_units: dict
    boundary_scale: dict
    boundary_sources: list
    global_sources: list
    defined_values: dict
    rms_scale: str | None
    output_stats: dict | None
    scaled: bool
    fp32_head: bool


def parse_scales(
    out_extras,
    b_extras,
    g_extras,
    output_stats,
    output_schema,
    boundary_schemas,
    global_schema,
) -> DeclaredScales:
    """The opt-in scales (the ISLA module docstring, *Scales*) from the declarations'
    ISLA-local keys (``out_extras``, ``g_extras`` and ``b_extras`` by condition, from
    :func:`split_scale_keys`). Checks them against the parsed schemas: which ranks
    may define a scale or carry units and a constant, at most one RMS scale (from
    boundary fields or one global), and every unit a defined scale.
    :func:`check_output_stats` checks ``output_stats``."""
    decl = {"output_schema": out_extras, "global_schema": g_extras}
    decl.update({f"boundary_schemas[{n!r}]": e for n, e in b_extras.items()})
    schemas = {"output_schema": output_schema, "global_schema": global_schema}
    schemas.update({f"boundary_schemas[{n!r}]": s for n, s in boundary_schemas.items()})
    used = set()
    for kind, extras in decl.items():
        for name, local in extras.items():
            rank = schemas[kind][name].rank
            defined = local.get(DEFINES)
            if defined is not None and not (
                rank == 1 or (rank == 0 and kind == "global_schema")
            ):
                raise ValueError(
                    f"{kind}[{name!r}]: a scale is defined by a rank-1 field or a rank-0 global; got rank {rank}"
                )
            if defined is not None and rank == 1 and local.get("units", {defined: 1}) != {defined: 1}:
                raise ValueError(
                    f"{kind}[{name!r}]: a rank-1 field that defines the scale {defined!r} has the units "
                    f"{{{defined!r}: 1}}, declared or not; got {local['units']!r}"
                )
            if (
                defined is not None
                and rank == 0
                and ("units" in local or "scale" in local)
            ):
                raise ValueError(
                    f"{kind}[{name!r}]: the value of the scale {defined!r} is consumed, not an input; "
                    "it takes no units and no scale"
                )
            if kind == "global_schema" and rank == 1 and defined is None:
                if "units" in local or "scale" in local:
                    raise ValueError(
                        f"{kind}[{name!r}]: a rank-1 global that defines no scale enters as a "
                        "direction only, so units and a scale have no effect on it; declare its magnitude as a "
                        "rank-0 global"
                    )
            if kind == "global_schema" and rank == 1 and "scale" in local:
                raise ValueError(
                    f"{kind}[{name!r}]: a rank-1 global's magnitude is the scale {defined!r}; it takes no scale"
                )
            used |= set(local.get("units", {}))
    output_units = {n: e["units"] for n, e in out_extras.items() if e.get("units")}
    output_scale = {n: e["scale"] for n, e in out_extras.items() if "scale" in e}
    global_units = {
        n: e["units"]
        for n, e in g_extras.items()
        if e.get("units") and DEFINES not in e and global_schema[n].rank == 0
    }
    global_scale = {n: e["scale"] for n, e in g_extras.items() if "scale" in e}
    ### A rank-1 field that defines a scale S has the units {S: 1} (its RMS magnitude is S).
    boundary_units = {
        c: {
            n: e.get("units") or {e[DEFINES]: 1}
            for n, e in extras.items()
            if e.get("units") or DEFINES in e
        }
        for c, extras in b_extras.items()
    }
    boundary_scale = {
        c: {n: e["scale"] for n, e in extras.items() if "scale" in e}
        for c, extras in b_extras.items()
    }
    ### The rank-1 sources of the one RMS scale (boundary fields, pooled, or one
    ### global) and the rank-0 globals whose values are scales (consumed).
    boundary_sources = [
        (c, n)
        for c, extras in b_extras.items()
        for n, e in extras.items()
        if DEFINES in e
    ]
    global_sources = [
        n for n, e in g_extras.items() if DEFINES in e and global_schema[n].rank == 1
    ]
    defined_values = {}
    for n, e in g_extras.items():
        if DEFINES in e and global_schema[n].rank == 0:
            if e[DEFINES] in defined_values:
                raise ValueError(
                    f"global_schema: two fields define the scale {e[DEFINES]!r}"
                )
            defined_values[e[DEFINES]] = n
    rms = {b_extras[c][n][DEFINES] for c, n in boundary_sources}
    rms |= {g_extras[n][DEFINES] for n in global_sources}
    if len(rms) > 1:
        raise ValueError(
            f"at most one scale is defined by rank-1 fields; got {sorted(rms)!r}"
        )
    rms_scale = next(iter(rms), None)
    if rms_scale in defined_values:
        raise ValueError(
            f"the scale {rms_scale!r} is defined by a rank-1 field and by the rank-0 global "
            f"{defined_values[rms_scale]!r}"
        )
    if boundary_sources and global_sources:
        raise ValueError(
            f"the sources of the scale {rms_scale!r} are rank-1 boundary fields or one rank-1 global, "
            f"not both: got {boundary_sources!r} and {global_sources!r}"
        )
    if len(global_sources) > 1:
        raise ValueError(
            f"at most one rank-1 global defines the scale {rms_scale!r}; got {global_sources!r}"
        )
    declared = {LENGTH, *defined_values, *rms}
    undefined = sorted(used - declared)
    if undefined:
        raise ValueError(
            f"units name {undefined!r}, which no field defines; the scales are {sorted(declared)!r}"
        )
    has_source = bool(boundary_sources or global_sources)
    output_stats = check_output_stats(output_stats, output_schema)
    has_input_scale = bool(global_scale or any(boundary_scale.values()))
    scaled = bool(used or has_source or defined_values or has_input_scale)
    ### Under autocast, the head and the redimensionalization run in the parameters'
    ### precision whenever a physical output is assembled or standardized. The bf16
    ### trunk bounds the outputs' precision either way. The fp32 head keeps bf16's
    ### rounding of the head's own product and of a nonzero output mean (such as
    ### mean / std = 0.3) out of the outputs.
    fp32_head = scaled or output_stats is not None or bool(output_scale)
    return DeclaredScales(
        output_units=output_units,
        output_scale=output_scale,
        global_units=global_units,
        global_scale=global_scale,
        boundary_units=boundary_units,
        boundary_scale=boundary_scale,
        boundary_sources=boundary_sources,
        global_sources=global_sources,
        defined_values=defined_values,
        rms_scale=rms_scale,
        output_stats=output_stats,
        scaled=scaled,
        fp32_head=fp32_head,
    )


def scale_factor(units: Mapping | None, scales: Mapping):
    """``prod_k scales[k] ** units[k]`` (``None`` for a dimensionless field)."""
    if not units:
        return None
    factor = None
    for k, e in units.items():
        term = scales[k] ** e
        factor = term if factor is None else factor * term
    return factor


def rms_moments(sources):
    """``(sum a, sum a |v|^2, sum a v)`` in float64 over ``sources``, pairs of a
    per-cell vector field ``v (n, d)`` and the cells' measures ``a (n,)``."""
    s0 = s2 = s1 = 0.0
    for v, a in sources:
        v, a = v.double(), a.double()
        s0 = s0 + a.sum()
        s2 = s2 + (a * v.square().sum(-1)).sum()
        s1 = s1 + (a[:, None] * v).sum(0)
    return s0, s2, s1


def length_unit(meshes: Sequence[Mesh], measures: Sequence[torch.Tensor]) -> torch.Tensor:
    """The length unit of ``meshes``, shape ``(1, 1, 1)`` in their dtype: the RMS
    pairwise distance of their piecewise-flat cells
    (:func:`~physicsnemo.experimental.models.isla._geometry.rms_pairwise_distance`).
    Each cell enters with its centroid, its measure (``measures``, one ``(n_cells,)``
    per mesh) and its own second moment, the meshes in the order given."""
    return rms_pairwise_distance(
        torch.cat([m.cell_centroids for m in meshes])[None],
        torch.cat(list(measures))[None],
        torch.cat([cell_second_moments(m) for m in meshes])[None],
    )


def case_scales(
    boundary_meshes: Mapping[str, Mesh],
    *,
    defines: Mapping[str, Sequence[tuple[str, str]] | str] | None = None,
    global_data: TensorDict | Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    r"""A case's scales, as :class:`~physicsnemo.experimental.models.isla.ISLA`
    computes them from its boundary and its global inputs.

    A model needs these scales to make a case's fields dimensionless. They depend
    only on the posed problem: the boundary and the fields that define a scale.
    Another model, or a preprocessing step that makes its inputs dimensionless,
    gets from this function the numbers ISLA uses. It is as experimental as ISLA,
    and moves with it out of ``physicsnemo.experimental``: until then a recipe, a
    script or an experimental model can import it, but ``physicsnemo.models``,
    ``physicsnemo.datapipes`` and ``physicsnemo.mesh`` cannot, since the layer
    contract of ``.importlinter`` keeps them from importing
    ``physicsnemo.experimental``.

    Parameters
    ----------
    boundary_meshes : Mapping[str, Mesh]
        The case's boundaries by name, at full resolution, as ISLA's ``forward``
        takes them.
    defines : Mapping[str, Sequence[tuple[str, str]] | str], optional, default=None
        The scales to compute besides ``"L"``, by unit name (any name, such as
        ``"U"`` or ``"rho"``). These are ISLA's ``defines`` keys (a field declared
        with ``defines: <unit>``). A sequence of ``(boundary, field)`` string pairs
        names rank-1 cell fields ``(n_cells, d)`` of ``boundary_meshes``. The scale
        is then their measure-weighted RMS magnitude
        :math:`\sqrt{\sum_i a_i |v_i|^2 / \sum_i a_i}`, pooled over the cells of
        every pair whose boundary is given. As in ISLA, a pair whose boundary is
        absent is skipped. A string names a field of ``global_data``. The scale is
        then the norm of a rank-1 field, or the value of a rank-0 one.
    global_data : TensorDict or Mapping[str, torch.Tensor], optional, default=None
        The case's global fields, by dotted name or nested, for the units that
        name one. A rank-1 field holds ``n_spatial_dims`` values, a rank-0 one a
        single value.

    Returns
    -------
    dict[str, torch.Tensor]
        Float64 scalars by unit name: ``"L"`` and each unit of ``defines``.
        ``"L"`` is the length unit, the measure-weighted RMS pairwise distance of
        every given boundary's cells. Each cell enters with its own second moment,
        so ``"L"`` is exact for the piecewise-flat boundary. The scales are bitwise
        those that an ISLA declaring the same boundaries and ``defines`` computes
        on any case it accepts. As in ISLA, the boundaries are taken in sorted
        order, the pairs of a unit by boundary in that order and then as listed,
        the length in the meshes' dtype, and a global field rounded to the meshes'
        dtype first. Every scale is invariant under rotations, reflections and
        translations of the case, and ``"L"`` scales with its coordinates.

    Raises
    ------
    TypeError
        When a value of ``boundary_meshes`` is not a ``Mesh``.
    ValueError
        When ``boundary_meshes`` is not a non-empty mapping. When a mesh is not a
        boundary (``Mesh[d - 1, d]``), the meshes differ in ``d`` or in dtype, or
        their points are not floating; the message names the mesh. When a unit's
        source is neither a string nor a sequence of ``(boundary, field)`` string
        pairs, or when a scale is undefined: no cell, no given boundary among a
        unit's pairs, a missing cell field or global field, a field of the wrong
        shape, sources that are zero everywhere, or a non-positive global value;
        the message names the unit and the input. These checks read the values on
        the host.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.isla import case_scales
    >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
    >>> body = sphere_icosahedral.load(subdivisions=1)
    >>> s = case_scales(
    ...     {"body": body},
    ...     defines={"U": "U_inf"},
    ...     global_data={"U_inf": torch.tensor([3.0, 4.0, 0.0])},
    ... )
    >>> sorted(s), s["U"].item(), s["L"].dtype
    (['L', 'U'], 5.0, torch.float64)
    """
    _check_boundary_meshes(boundary_meshes)
    names = sorted(boundary_meshes)
    meshes = [boundary_meshes[n] for n in names]
    if not sum(m.n_cells for m in meshes):
        raise ValueError(
            f"the length unit {LENGTH!r} is undefined: boundary_meshes "
            f"({names!r}) have no cells"
        )
    measures = [cell_measures(m) for m in meshes]
    like = meshes[0].points
    length = length_unit(meshes, measures).reshape(()).double()
    if not bool(torch.isfinite(length) & (length > 0)):
        raise ValueError(
            f"the length unit {LENGTH!r} of boundary_meshes ({names!r}) is undefined "
            f"(got {length.item()}): their cells have no measure or a non-finite point"
        )
    scales = {LENGTH: length}
    present = dict(zip(names, zip(meshes, measures)))
    for unit, source in (defines or {}).items():
        if unit == LENGTH:
            raise ValueError(
                f"defines[{unit!r}]: {LENGTH!r} is the built-in length unit; a field cannot define it"
            )
        if isinstance(source, str):
            scales[unit] = _global_scale(unit, source, global_data, like)
        else:
            scales[unit] = _rms_scale(unit, source, present)
    return scales


def _check_boundary_meshes(boundary_meshes) -> None:
    """The mesh checks of ISLA's ``forward`` that need no model: a non-empty mapping
    of ``Mesh``, each the boundary of its domain (``Mesh[d - 1, d]``), all with one
    ``d`` and one floating dtype."""
    if not isinstance(boundary_meshes, Mapping) or not boundary_meshes:
        raise ValueError("boundary_meshes must be a non-empty mapping of name to Mesh")
    ref = None
    for name, mesh in boundary_meshes.items():
        if not isinstance(mesh, Mesh):
            raise TypeError(
                f"boundary_meshes[{name!r}] must be a physicsnemo.mesh.Mesh, got {type(mesh).__name__}"
            )
        d, dtype = mesh.n_spatial_dims, mesh.points.dtype
        if mesh.n_manifold_dims != d - 1:
            raise ValueError(
                f"boundary_meshes[{name!r}]: the boundary of a {d}-dimensional domain is a "
                f"Mesh[{d - 1}, {d}] (its cells carry the normals); got a "
                f"Mesh[{mesh.n_manifold_dims}, {d}]"
            )
        if not dtype.is_floating_point:
            raise ValueError(f"boundary_meshes[{name!r}] has {dtype} points; they must be floating")
        if ref is None:
            ref = (name, d, dtype)
        elif (d, dtype) != ref[1:]:
            raise ValueError(
                f"boundary_meshes[{name!r}] has {d} spatial dimensions and {dtype} points, "
                f"boundary_meshes[{ref[0]!r}] {ref[1]} and {ref[2]}: every boundary must have the "
                f"same number of spatial dimensions and dtype"
            )


def _rms_scale(unit: str, source, present: Mapping) -> torch.Tensor:
    """The measure-weighted RMS magnitude of the cell fields ``source`` (pairs of a
    boundary of ``present``, name to ``(mesh, measures)``, and a field), pooled in
    ISLA's order: by boundary name, then as listed."""

    def is_pair(p) -> bool:
        return isinstance(p, Sequence) and not isinstance(p, str) and len(p) == 2 and all(isinstance(x, str) for x in p)

    if isinstance(source, str) or not isinstance(source, Sequence) or not all(is_pair(p) for p in source):
        raise ValueError(
            f"defines[{unit!r}] must be a global field's name or a sequence of (boundary, field) "
            f"string pairs; got {source!r}"
        )
    pairs = sorted(((b, f) for b, f in source), key=lambda p: p[0])
    if not pairs:
        raise ValueError(f"defines[{unit!r}] names no (boundary, field) pair")
    ### ISLA pools the pairs of the given boundaries and refuses only when none is given.
    given = [(b, f) for b, f in pairs if b in present]
    if not given:
        raise ValueError(
            f"defines[{unit!r}]: no boundary of the sources {pairs!r} is given; the boundaries "
            f"are {sorted(present)!r}"
        )
    fields = []
    for b, f in given:
        where = f"defines[{unit!r}]: boundary_meshes[{b!r}]"
        mesh, measure = present[b]
        v = mesh.cell_data.get(FieldSchema.key(f), None)
        if not isinstance(v, torch.Tensor):
            raise ValueError(f"{where} has no cell field {f!r}")
        if v.ndim != 2 or v.shape != (mesh.n_cells, mesh.n_spatial_dims):
            raise ValueError(
                f"{where}.cell_data[{f!r}] must be a rank-1 cell field of shape "
                f"{(mesh.n_cells, mesh.n_spatial_dims)}; got {tuple(v.shape)}"
            )
        fields.append((v, measure))
    s0, s2, _ = rms_moments(fields)
    if not (bool(s0 > 0) and bool(s2 > 0)):
        raise ValueError(
            f"defines[{unit!r}]: the sources {given!r} have no cells, no measure or are zero "
            "everywhere: the scale is undefined"
        )
    return (s2 / s0).sqrt()


def _global_scale(unit: str, name: str, global_data, like: torch.Tensor) -> torch.Tensor:
    """The norm of the rank-1 global field ``name`` of ``global_data``, or the value of
    a rank-0 one, rounded to ``like``'s dtype and then taken in float64."""
    where = f"defines[{unit!r}]: global_data[{name!r}]"
    if isinstance(global_data, Mapping) and not isinstance(global_data, TensorDict):
        ### A plain mapping, nested or by dotted names, as ISLA's forward reads it.
        td = TensorDict({}, batch_size=[])
        for k, value in global_data.items():
            td.set(FieldSchema.key(k) if isinstance(k, str) else k, value)
        global_data = td
    v = global_data.get(FieldSchema.key(name), None) if isinstance(global_data, TensorDict) else None
    if not isinstance(v, torch.Tensor):
        raise ValueError(f"{where} is not given")
    d = like.shape[-1]
    if v.numel() not in (1, d):
        raise ValueError(
            f"{where} must hold one value (rank 0) or {d} (rank 1, the meshes' n_spatial_dims) for one "
            f"case; got shape {tuple(v.shape)}"
        )
    v = v.to(like.device).reshape(-1).to(like.dtype).double()
    if v.numel() == 1:
        value = v.reshape(())
        if not bool(value > 0):
            raise ValueError(f"{where} must be positive; got {value.item()}")
        return value
    value = v.norm()
    if not bool(value > 0):
        raise ValueError(f"{where} is zero: the scale is undefined")
    return value
