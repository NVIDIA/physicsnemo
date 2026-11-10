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

r"""On-disk layout of the converted bracket datasets, shared by the converters and the datapipe.

One :class:`~physicsnemo.mesh.DomainMesh` per design and load case,
``<root>/samples/shape_<id>/<load>.pdmsh``:

- ``interior``: every mesh node (float32 mm) with the tetrahedra (corner nodes) as cells, and
  per node ``region_id`` (0 interior, 1 free surface, 2 clamped bolt interfaces, 3 loaded clevis
  interface), ``displacement`` (mm) and ``von_mises`` (MPa);
- ``boundaries["surface"]``: the boundary triangles of the tetrahedral mesh, with ``region_id``
  and ``interior_index`` (the index of each surface node among the interior nodes);
- ``global_data``: ``shape_id``, ``load_id`` and ``load_onehot`` (order of :data:`LOADS`),
  ``force`` and ``moment`` (N, N mm), ``load_reference_position`` (mm), ``young_modulus`` (MPa)
  and ``poisson_ratio``.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from nastran import LOADS

from physicsnemo.mesh import DomainMesh, Mesh


def sample_path(root: str | Path, shape: int, load: str) -> Path:
    r"""Path of the sample of design ``shape`` under load case ``load``."""
    return Path(root) / "samples" / f"shape_{int(shape)}" / f"{load}.pdmsh"


def load_split(path: str | Path) -> dict[str, list[int]]:
    r"""Design ids per split (``train``, ``val``, ``test``) of a split file in ``splits/``."""
    data = json.loads(Path(path).read_text())
    return {k: [int(v) for v in data[k]] for k in ("train", "val", "test")}


def boundary_triangles(tets: np.ndarray) -> np.ndarray:
    r"""Faces of a tetrahedral mesh that belong to one tetrahedron only, :math:`(F, 3)` node indices.

    For 10-node tetrahedra (Nastran order: corners, then the mid-edge nodes of the edges
    12, 23, 31, 14, 24, 34) every boundary face is split into four triangles through its
    mid-edge nodes, so that every node of the surface is a vertex.
    """
    if tets.shape[1] == 4:
        faces = np.concatenate(
            [
                tets[:, [0, 1, 2]],
                tets[:, [0, 1, 3]],
                tets[:, [0, 2, 3]],
                tets[:, [1, 2, 3]],
            ]
        )
        _, first, counts = np.unique(
            np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True
        )
        if counts.max() > 2:
            raise ValueError("Non-manifold tetrahedron face")
        return faces[first[counts == 1]]
    # 10-node: each face as (corner a, corner b, corner c, mid ab, mid bc, mid ca)
    six = np.concatenate(
        [
            tets[:, [0, 1, 2, 4, 5, 6]],
            tets[:, [0, 1, 3, 4, 8, 7]],
            tets[:, [0, 2, 3, 6, 9, 7]],
            tets[:, [1, 2, 3, 5, 9, 8]],
        ]
    )
    _, first, counts = np.unique(
        np.sort(six[:, :3], axis=1), axis=0, return_index=True, return_counts=True
    )
    if counts.max() > 2:
        raise ValueError("Non-manifold tetrahedron face")
    b = six[first[counts == 1]]
    return np.concatenate(
        [b[:, [0, 3, 5]], b[:, [3, 1, 4]], b[:, [5, 4, 2]], b[:, [3, 4, 5]]]
    )


def save_design(
    root: str | Path,
    shape: int,
    points: np.ndarray,
    tets: np.ndarray,
    region_id: np.ndarray,
    fields: dict[str, dict[str, np.ndarray]],
    deck: dict,
) -> None:
    r"""Write the four samples of one design (atomically: into a staging folder, then renamed).

    Parameters
    ----------
    root : str | pathlib.Path
        Output dataset folder.
    shape : int
        Design id.
    points : numpy.ndarray
        Node coordinates of shape :math:`(N, 3)` in mm.
    tets : numpy.ndarray
        Tetrahedra of shape :math:`(T, 4)` or :math:`(T, 10)`, as indices into ``points``.
    region_id : numpy.ndarray
        Region of every node, shape :math:`(N,)`.
    fields : dict[str, dict[str, numpy.ndarray]]
        Per load case ``displacement`` :math:`(N, 3)` in mm and ``von_mises`` :math:`(N,)` in MPa.
    deck : dict
        Output of :func:`nastran.parse_deck`.
    """
    f32 = lambda v: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))  # noqa: E731
    i32 = lambda v: torch.from_numpy(np.ascontiguousarray(v, dtype=np.int32))  # noqa: E731
    triangles = boundary_triangles(tets)
    surface_nodes = np.unique(triangles)
    local = np.full(len(points), -1, dtype=np.int64)
    local[surface_nodes] = np.arange(len(surface_nodes))
    destination = Path(root) / "samples" / f"shape_{int(shape)}"
    stage = destination.with_name(destination.name + ".partial")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    for index, load in enumerate(LOADS):
        condition = deck["conditions"][load]
        interior = Mesh(
            points=f32(points),
            cells=torch.from_numpy(np.ascontiguousarray(tets[:, :4], dtype=np.int64)),
            point_data={
                "region_id": i32(region_id),
                "displacement": f32(fields[load]["displacement"]),
                "von_mises": f32(fields[load]["von_mises"]),
            },
        )
        surface = Mesh(
            points=f32(points[surface_nodes]),
            cells=torch.from_numpy(local[triangles]),
            point_data={
                "region_id": i32(region_id[surface_nodes]),
                "interior_index": torch.from_numpy(surface_nodes),
            },
        )
        vector = torch.as_tensor(condition["vector"], dtype=torch.float64)
        zero = torch.zeros(3, dtype=torch.float64)
        global_data = {
            "shape_id": torch.tensor(int(shape)),
            "load_id": torch.tensor(index),
            "load_onehot": torch.eye(len(LOADS), dtype=torch.float64)[index],
            "force": vector if condition["type"] == "FORCE" else zero,
            "moment": vector if condition["type"] == "MOMENT" else zero,
            "load_reference_position": torch.as_tensor(
                condition["reference_position"], dtype=torch.float64
            ),
            "young_modulus": torch.tensor(
                float(deck["young_modulus"]), dtype=torch.float64
            ),
            "poisson_ratio": torch.tensor(
                float(deck["poisson_ratio"]), dtype=torch.float64
            ),
        }
        DomainMesh(
            interior=interior, boundaries={"surface": surface}, global_data=global_data
        ).save(stage / f"{load}.pdmsh")
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(stage, destination)
