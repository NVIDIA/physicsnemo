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

r"""Shared fixtures-as-functions of the ``"local"`` mesh-layout tests."""

import torch

from physicsnemo.datapipes.transforms.mesh import (
    CenterMesh,
    ComputeSurfaceNormals,
    DropMeshFields,
    MeshToDomainMesh,
    NormalizeMeshFields,
    RandomRotateMesh,
    RandomTranslateMesh,
    SubsampleMesh,
)
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.io import to_zarr

NX, NY = 24, 12  # 2 * 23 * 11 = 506 cells: uneven over 3 and 4 ranks
WINDOW = 331  # odd: uneven over every world size > 1


def grid_mesh(seed: int = 7, offset: float = 100.0) -> Mesh:
    r"""Jittered triangulated grid whose cells share vertices across every seam."""
    g = torch.Generator().manual_seed(seed)
    x, y = torch.meshgrid(
        torch.arange(NX, dtype=torch.float32),
        torch.arange(NY, dtype=torch.float32),
        indexing="ij",
    )
    z = 0.1 * torch.randn(NX, NY, generator=g)
    points = torch.stack([x, y, z], dim=-1).reshape(-1, 3) + offset
    vid = torch.arange(NX * NY).reshape(NX, NY)
    a, b = vid[:-1, :-1].reshape(-1), vid[1:, :-1].reshape(-1)
    c, d = vid[1:, 1:].reshape(-1), vid[:-1, 1:].reshape(-1)
    cells = torch.cat([torch.stack([a, b, c], 1), torch.stack([a, c, d], 1)])
    # Interleave the two triangles of each quad, then shuffle the blocks.
    cells = cells.reshape(2, -1, 3).transpose(0, 1).reshape(-1, 2, 3)
    cells = cells[torch.randperm(cells.shape[0], generator=g)].reshape(-1, 3)
    mesh = Mesh(points=points, cells=cells)
    mesh.point_data["velocity"] = torch.randn(mesh.n_points, 3, generator=g)
    mesh.cell_data["pressure"] = torch.randn(mesh.n_cells, generator=g)
    mesh.cell_data["wss"] = torch.randn(mesh.n_cells, 3, generator=g)
    mesh.global_data["Re"] = torch.tensor(1.0e6)
    mesh.global_data["U_inf"] = torch.tensor([1.0, 0.0, 0.0])
    return mesh


def write_sample(mesh: Mesh, root, name: str, fmt: str) -> None:
    r"""Store ``mesh`` as ``root/name.<fmt>`` (``pmsh`` or ``zarr``)."""
    if fmt == "pmsh":
        mesh.save(root / f"{name}.pmsh")
    else:
        to_zarr(mesh, root / f"{name}.zarr")


def recipe_chain(
    *,
    center: bool = True,
    area_weighted: bool = False,
    augment: bool = False,
    window: int | None = WINDOW,
    store_center_as: str | None = None,
):
    r"""Recipe-style chain: drop, center, augment, normalize, normals, subsample, domain."""
    fields = {
        "pressure": {"type": "scalar", "mean": 0.5, "std": 2.0},
        "wss": {"type": "vector", "mean": [1.0, 0.0, 0.0], "std": 0.5},
    }
    chain = [DropMeshFields(global_data=["Re"])]
    if center:
        chain.append(
            CenterMesh(
                use_area_weighting=area_weighted, store_center_as=store_center_as
            )
        )
    if augment:
        chain += [
            RandomRotateMesh(
                axes=["z"],
                mode="axis_aligned",
                transform_cell_data=True,
                transform_global_data=True,
            ),
            RandomTranslateMesh(
                distribution=torch.distributions.Uniform(
                    torch.tensor([-1.0, -1.0, 0.0]),
                    torch.tensor([1.0, 1.0, 1e-6]),
                    validate_args=False,
                )
            ),
        ]
    chain += [
        NormalizeMeshFields(association="cell_data", fields=fields),
        ComputeSurfaceNormals(store_as="cell_data", field_name="normals"),
        # A no-op on the sample when the reader already drew the window.
        SubsampleMesh(n_cells=window or 10**6),
        MeshToDomainMesh(
            cell_data_targets=["pressure", "wss"], interior_points="cell_centroids"
        ),
    ]
    return chain
