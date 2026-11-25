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

r"""Bracket samples as GeoTransolver inputs and normalized targets.

Every mesh node is a query point. Per node the model sees its coordinates (in units of
``coordinate_scale_mm``) and a one-hot of its region (interior, free surface, clamped bolt
interface, loaded clevis interface); the geometry context is every node voxel-downsampled at
``token_voxel_mm`` (one node per voxel), so it describes the solid, not only its surface; the
global context is the one-hot of the load case. Targets are the displacement and the von Mises
stress divided by per-load-case scales (:func:`load_scales`), so that the four load cases have
comparable magnitudes.

With ``sdf=True`` a sample also carries the surface triangle mesh, consistently oriented outward,
for the signed-distance stencil of
:class:`~physicsnemo.experimental.models.geotransolver_sdf.GeoTransolverSDF` (the model finds the
neighbours of every node itself, on the GPU).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from bracket_io import sample_path
from nastran import LOADS

from physicsnemo.mesh import DomainMesh

#: Number of node regions (see ``bracket_io.py``)
REGIONS = 4
#: Channels of the model output: displacement (3) and von Mises stress (1)
OUTPUTS = 4


def load_scales(path: str | Path) -> dict[str, dict[str, float]]:
    r"""Per-load-case normalization scales ``{"displacement": {load: mm}, "von_mises": {load: MPa}}``."""
    data = json.loads(Path(path).read_text())
    return {
        f: {load: float(data[f][load]) for load in LOADS}
        for f in ("displacement", "von_mises")
    }


def voxel_first(points: np.ndarray, voxel: float) -> np.ndarray:
    r"""Indices of the first point in every voxel of edge ``voxel``, in point order."""
    keys = np.floor(points / voxel).astype(np.int64)
    return np.sort(np.unique(keys, axis=0, return_index=True)[1])


def outward_faces(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    r"""Surface triangles :math:`(F, 3)` reoriented consistently and outward.

    The sign of the signed distance comes from the triangle normals. Orientation is propagated
    across shared (manifold) edges from a seed face; each connected component keeps the orientation
    of the majority of its faces, so that an inner shell keeps facing into its cavity, and the whole
    surface is turned over if its enclosed volume is negative. Unchanged when already consistent and
    outward.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order, connected_components

    f, v = np.asarray(faces, dtype=np.int64), np.asarray(points, dtype=np.float64)
    n = len(f)
    half = np.concatenate(
        [f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]
    )  # (3F, 2) directed edges
    owner = np.tile(np.arange(n), 3)
    _, edge, count = np.unique(
        np.sort(half, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    edge = edge.reshape(-1)
    order = np.argsort(edge, kind="stable")
    pairs = order[np.repeat(count, count) == 2].reshape(
        -1, 2
    )  # half-edges of manifold edges
    a, b = owner[pairs[:, 0]], owner[pairs[:, 1]]
    parity = (half[pairs[:, 0]] == half[pairs[:, 1]]).all(
        1
    )  # same direction: one face flipped
    graph = coo_matrix((parity.astype(np.int8) + 1, (a, b)), shape=(n, n)).tocsr()
    graph = graph + graph.T
    flip = np.zeros(n, dtype=bool)
    components, label = connected_components(graph, directed=False)
    for c in range(components):
        seed = int(np.flatnonzero(label == c)[0])
        visit, pred = breadth_first_order(
            graph, seed, directed=False, return_predecessors=True
        )
        up = np.where(pred < 0, np.arange(n), pred)  # pointer jumping to the seed
        step = np.zeros(n, dtype=bool)
        step[visit[1:]] = np.asarray(graph[visit[1:], up[visit[1:]]]).reshape(-1) == 2
        while (up[visit] != seed).any():
            step, up = step ^ step[up], up[up]
        flip[visit] = (
            step[visit] if 2 * step[visit].sum() <= len(visit) else ~step[visit]
        )
    g = np.where(flip[:, None], f[:, [0, 2, 1]], f)
    if (
        v[g[:, 0]] * np.cross(v[g[:, 1]], v[g[:, 2]])
    ).sum() < 0:  # enclosed volume: outward
        flip = ~flip
    return np.where(flip[:, None], f[:, [0, 2, 1]], f)


def sample_inputs(
    domain: DomainMesh,
    scales: dict[str, dict[str, float]],
    coordinate_scale_mm: float = 100.0,
    token_voxel_mm: float = 2.0,
    sdf: bool = False,
) -> dict[str, torch.Tensor]:
    r"""Model inputs and normalized targets of one sample.

    Parameters
    ----------
    domain : physicsnemo.mesh.DomainMesh
        A sample in the layout of ``bracket_io.py``.
    scales : dict[str, dict[str, float]]
        Per-load-case scales from :func:`load_scales`.
    coordinate_scale_mm : float, optional, default=100.0
        Length unit of the model coordinates, in mm.
    token_voxel_mm : float, optional, default=2.0
        Voxel edge of the geometry context, in mm.
    sdf : bool, optional, default=False
        Also return the outward-facing surface of :class:`GeoTransolverSDF` (see the module docstring).

    Returns
    -------
    dict[str, torch.Tensor]
        ``local_embedding`` :math:`(N, 3 + 4)`, ``local_positions`` :math:`(N, 3)`, ``geometry``
        :math:`(M, 3)`, ``global_embedding`` :math:`(1, 4)`, ``target`` :math:`(N, 4)` (normalized
        displacement and von Mises stress), ``scale`` :math:`(4,)` (multiplies ``target`` back to
        mm and MPa), ``shape_id`` and ``load_id``.
    """
    load_id = int(domain.global_data["load_id"])
    load = LOADS[load_id]
    points = domain.interior.points.float() / coordinate_scale_mm  # (N, 3)
    region = torch.nn.functional.one_hot(
        domain.interior.point_data["region_id"].long(), REGIONS
    ).float()  # (N, 4)
    keep = voxel_first(points.numpy(), token_voxel_mm / coordinate_scale_mm)
    scale = torch.tensor(
        [scales["displacement"][load]] * 3 + [scales["von_mises"][load]]
    )
    target = (
        torch.cat(
            [
                domain.interior.point_data["displacement"].float(),
                domain.interior.point_data["von_mises"].float()[:, None],
            ],
            dim=1,
        )
        / scale
    )
    sample = {
        "local_embedding": torch.cat([points, region], dim=1),
        "local_positions": points,
        "geometry": points[torch.from_numpy(keep)],
        "global_embedding": domain.global_data["load_onehot"].float()[None],
        "target": target,
        "scale": scale,
        "shape_id": torch.tensor(int(domain.global_data["shape_id"])),
        "load_id": torch.tensor(load_id),
    }
    if sdf:
        surface = domain.boundaries["surface"]
        sample.update(
            surface_vertices=surface.points.float() / coordinate_scale_mm,
            surface_faces=torch.from_numpy(
                outward_faces(surface.points.double().numpy(), surface.cells.numpy())
            ),
        )
    return sample


class BracketDataset(torch.utils.data.Dataset):
    r"""Every (design, load case) of a converted bracket dataset, as :func:`sample_inputs`.

    Parameters
    ----------
    root : str | pathlib.Path
        Converted dataset (``convert_deepjeb.py`` or ``convert_simjeb.py``).
    shapes : list[int]
        Design ids.
    scales : dict[str, dict[str, float]]
        Per-load-case scales from :func:`load_scales`.
    loads : tuple[str, ...], optional, default=LOADS
        Load cases.
    coordinate_scale_mm : float, optional, default=100.0
        Length unit of the model coordinates, in mm.
    token_voxel_mm : float, optional, default=2.0
        Voxel edge of the geometry context, in mm.
    sdf : bool, optional, default=False
        Also return the outward-facing surface of :class:`GeoTransolverSDF` (:func:`sample_inputs`).
    """

    def __init__(
        self,
        root: str | Path,
        shapes: list[int],
        scales: dict[str, dict[str, float]],
        loads: tuple[str, ...] = LOADS,
        coordinate_scale_mm: float = 100.0,
        token_voxel_mm: float = 2.0,
        sdf: bool = False,
    ) -> None:
        self.root, self.scales, self.sdf = Path(root), scales, bool(sdf)
        self.coordinate_scale_mm, self.token_voxel_mm = (
            coordinate_scale_mm,
            token_voxel_mm,
        )
        self.items = [(int(s), load) for s in shapes for load in loads]
        missing = [
            sample_path(self.root, s, load)
            for s, load in self.items
            if not sample_path(self.root, s, load).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} samples missing under {self.root}, e.g. {missing[0]}"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shape, load = self.items[index]
        domain = DomainMesh.load(sample_path(self.root, shape, load))
        return sample_inputs(
            domain, self.scales, self.coordinate_scale_mm, self.token_voxel_mm, self.sdf
        )


def model_inputs(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    r"""The model's keyword arguments from a batch: those of
    :class:`~physicsnemo.models.geotransolver.GeoTransolver`, or of
    :class:`~physicsnemo.experimental.models.geotransolver_sdf.GeoTransolverSDF` when the batch
    carries the surface (``sdf=True``)."""
    if "surface_vertices" in batch:
        keys = (
            "local_embedding",
            "geometry",
            "global_embedding",
            "surface_vertices",
            "surface_faces",
        )
    else:
        keys = ("local_embedding", "local_positions", "geometry", "global_embedding")
    return {k: batch[k].to(device, non_blocking=True) for k in keys}
