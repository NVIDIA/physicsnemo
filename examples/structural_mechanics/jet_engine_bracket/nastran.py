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

r"""Reader for the OptiStruct decks of the jet engine bracket datasets (SimJEB, DeepJEB).

Only the cards these decks use are read: ``GRID``, ``CTETRA`` (4 or 10 nodes), ``MAT1``,
``RBE2`` (the clamped bolt interfaces), ``RBE3`` (the loaded clevis interface), ``SPC``,
``SPCADD``, ``LOAD``, ``FORCE`` and ``MOMENT``, in small-field format with continuations.
Every deck has four labelled static subcases: vertical, horizontal and diagonal forces and a
torsional moment, all applied at the ``RBE3`` reference node.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

#: Load cases, in the order used everywhere in this example
LOADS = ("ver", "hor", "dia", "tor")
_LABELS = {"vertical": "ver", "horizontal": "hor", "diagonal": "dia", "torsion": "tor"}


def require(condition: bool, message: str) -> None:
    r"""Raise ``ValueError(message)`` unless ``condition`` holds."""
    if not condition:
        raise ValueError(message)


def number(text: str) -> float:
    r"""Parse a Nastran real, including the exponent shorthand ``1.5-3`` for ``1.5e-3``."""
    text = text.strip().replace("D", "E").replace("d", "e")
    if "e" not in text.lower():
        text = re.sub(r"(?<=[0-9.])([+-][0-9]+)$", r"E\1", text)
    value = float(text)
    require(np.isfinite(value), f"Non-finite number {text!r}")
    return value


def field_cards(path: str | Path):
    r"""Yield ``(card, fields)`` for every small-field bulk data card, continuations appended."""
    bulk, current = False, None
    with Path(path).open(errors="ignore") as stream:
        for raw in stream:
            line = raw.rstrip("\r\n")
            if line.strip() == "BEGIN BULK":
                bulk = True
                continue
            if not bulk or not line.strip() or line.lstrip().startswith("$"):
                continue
            if line.strip() == "ENDDATA":
                break
            kind = line[:8].strip()
            if not kind or kind.startswith("+"):
                require(current is not None, "Continuation without a card")
                current[1].extend(line[i : i + 8].strip() for i in range(8, 72, 8))
                continue
            if current is not None:
                yield current
            current = (kind, [line[i : i + 8].strip() for i in range(8, 72, 8)])
    if current is not None:
        yield current


def parse_deck(path: str | Path) -> dict:
    r"""Mesh, material, interfaces and the four load cases of a bracket deck.

    Parameters
    ----------
    path : str | pathlib.Path
        The ``.fem`` deck.

    Returns
    -------
    dict
        ``grids`` (node id to coordinates in mm), ``tets`` (:math:`(T, 4)` or :math:`(T, 10)`
        node ids), ``young_modulus`` (MPa) and ``poisson_ratio``, ``supports`` and ``loaded``
        (node ids tied to the ``RBE2`` and ``RBE3`` interfaces), ``references`` (their
        independent nodes, which are not mesh nodes) and ``conditions``: per load case its
        ``type`` (``FORCE`` or ``MOMENT``), ``vector`` and ``reference_position``.
    """
    cases, case = {}, None
    with Path(path).open(errors="ignore") as stream:
        for line in stream:
            text = line.strip()
            if text == "BEGIN BULK":
                break
            if text.startswith("SUBCASE"):
                case = int(text.split()[1])
                cases[case] = {}
            elif case is not None and text.startswith("LABEL"):
                cases[case]["label"] = (
                    re.split(r"[=\s]", text, maxsplit=1)[1].strip(" =").lower()
                )
            elif case is not None and re.match(r"^(SPC|LOAD)\s*=", text):
                key, value = text.split("=", 1)
                cases[case][key.strip()] = int(value)
    grids, tets, materials = {}, [], {}
    supports, support_refs, loaded, load_refs = set(), set(), set(), set()
    spc, spcadd, combinations, point_loads = {}, {}, {}, {}
    for kind, fields in field_cards(path):
        if kind == "GRID":
            require(
                fields[1] in ("", "0") and fields[5] in ("", "0"),
                "Non-basic GRID coordinate system",
            )
            grids[int(fields[0])] = [number(v) for v in fields[2:5]]
        elif kind == "CTETRA":
            nodes = [int(v) for v in fields[2:12] if v]
            require(len(nodes) in (4, 10), "Expected a 4- or 10-node CTETRA")
            tets.append(nodes)
        elif kind == "MAT1":
            materials[int(fields[0])] = (number(fields[1]), number(fields[3]))
        elif kind == "RBE2":
            require(fields[2] == "123456", "Unsupported RBE2 degrees of freedom")
            support_refs.add(int(fields[1]))
            supports.update(int(v) for v in fields[3:] if v)
        elif kind == "RBE3":
            load_refs.add(int(fields[2]))
            loaded.update(int(v) for v in fields[6:] if v.isdigit())
        elif kind == "SPC":
            for start in (1, 4):
                if fields[start]:
                    value = number(fields[start + 2]) if fields[start + 2] else 0.0
                    spc.setdefault(int(fields[0]), []).append(
                        (int(fields[start]), fields[start + 1], value)
                    )
        elif kind == "SPCADD":
            spcadd[int(fields[0])] = [int(v) for v in fields[1:] if v]
        elif kind == "LOAD":
            values = [v for v in fields if v]
            parts = [
                (number(values[i]), int(values[i + 1]))
                for i in range(2, len(values) - 1, 2)
            ]
            combinations[int(values[0])] = (number(values[1]), parts)
        elif kind in ("FORCE", "MOMENT"):
            require(fields[2] in ("", "0"), "Non-basic load coordinate system")
            vector = number(fields[3]) * np.array([number(v) for v in fields[4:7]])
            point_loads.setdefault(int(fields[0]), []).append(
                (kind, int(fields[1]), vector)
            )
    require(
        len(tets) > 0 and len({len(t) for t in tets}) == 1, "Expected one CTETRA type"
    )
    require(len(materials) == 1, "Expected one isotropic material")
    require(
        support_refs and len(load_refs) == 1 and not supports & loaded,
        "Unexpected interface layout",
    )
    conditions = {}
    for subcase in cases.values():
        load = _LABELS.get(subcase.get("label", ""))
        if load is None:
            continue  # e.g. a normal-modes subcase
        sets = spcadd.get(subcase["SPC"], [subcase["SPC"]])
        clamps = [c for s in sets for c in spc.get(s, [])]
        require(
            {c[0] for c in clamps} == support_refs
            and all(c[1] == "123456" and c[2] == 0 for c in clamps),
            "The SPCs must clamp the RBE2 references",
        )
        scale, parts = combinations.get(
            subcase["LOAD"], (1.0, [(1.0, subcase["LOAD"])])
        )
        active = [
            (k, node, scale * f * vec)
            for f, sid in parts
            for k, node, vec in point_loads.get(sid, [])
        ]
        require(
            len(active) == 1 and active[0][1] in load_refs,
            f"Expected one load at the RBE3 reference ({load})",
        )
        kind, node, vector = active[0]
        require(
            kind == ("MOMENT" if load == "tor" else "FORCE"),
            f"Unexpected load type for {load}",
        )
        conditions[load] = {
            "type": kind,
            "vector": vector,
            "reference_position": np.asarray(grids[node]),
        }
    require(set(conditions) == set(LOADS), "Missing load cases")
    young_modulus, poisson_ratio = next(iter(materials.values()))
    return {
        "grids": grids,
        "tets": np.asarray(tets, dtype=np.int64),
        "young_modulus": young_modulus,
        "poisson_ratio": poisson_ratio,
        "supports": supports,
        "loaded": loaded,
        "references": support_refs | load_refs,
        "conditions": conditions,
    }
