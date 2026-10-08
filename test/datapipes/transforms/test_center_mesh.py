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

"""Focused tests for the mesh centering transform."""

import torch

from physicsnemo.datapipes.transforms.mesh import CenterMesh
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.calculus.measure import scale_measures


def test_center_mesh_optionally_stores_subtracted_center():
    mesh = Mesh(
        points=torch.tensor(
            [
                [2.0, 1.0, -1.0],
                [4.0, 3.0, 1.0],
            ]
        ),
        global_data={"case_id": torch.tensor(7)},
    )

    centered = CenterMesh(
        use_area_weighting=False,
        store_center_as="center",
    )(mesh)

    torch.testing.assert_close(centered.points.mean(dim=0), torch.zeros(3))
    torch.testing.assert_close(
        centered.global_data["center"],
        torch.tensor([3.0, 2.0, 0.0]),
    )
    assert "center" not in mesh.global_data


def test_center_mesh_stores_domain_center_at_domain_level():
    domain = DomainMesh(
        interior=Mesh(
            points=torch.tensor(
                [
                    [2.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                ]
            )
        ),
        boundaries={
            "wall": Mesh(
                points=torch.tensor(
                    [
                        [10.0, 0.0, 0.0],
                        [12.0, 0.0, 0.0],
                    ]
                )
            )
        },
    )

    centered = CenterMesh(
        use_area_weighting=False,
        store_center_as="center",
    ).apply_to_domain(domain)

    torch.testing.assert_close(
        centered.global_data["center"], torch.tensor([3.0, 0.0, 0.0])
    )
    torch.testing.assert_close(centered.interior.points.mean(dim=0), torch.zeros(3))
    torch.testing.assert_close(
        centered.boundaries["wall"].points,
        torch.tensor(
            [
                [7.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
            ]
        ),
    )
    assert "center" not in centered.interior.global_data


def test_center_mesh_default_does_not_add_metadata():
    mesh = Mesh(points=torch.tensor([[2.0, 0.0], [4.0, 0.0]]))

    centered = CenterMesh(use_area_weighting=False)(mesh)

    assert not centered.global_data.keys()


def _triangulated_patch(n_u: int, n_v: int) -> Mesh:
    """Planar patch in z=0, quadratically spaced in x so cell areas vary."""
    s = torch.linspace(0.0, 1.0, n_u) ** 2
    t = torch.linspace(0.0, 1.0, n_v)
    xy = torch.stack(torch.meshgrid(s, t, indexing="ij"), dim=-1).reshape(-1, 2)
    points = torch.cat([xy, torch.zeros(len(xy), 1)], dim=-1)
    cells = []
    for i in range(n_u - 1):
        for j in range(n_v - 1):
            a = i * n_v + j
            b = (i + 1) * n_v + j
            cells.append([a, b, a + 1])
            cells.append([b, b + 1, a + 1])
    return Mesh(points=points, cells=torch.tensor(cells, dtype=torch.int64))


def _biased_subsample(mesh: Mesh, bias: int) -> Mesh:
    """Compacted cell subsample: every cell of the front half (centroid x below
    the median), each back-half cell independently with probability
    ``1 / bias``; each kept cell carries its inverse inclusion probability as
    a measure weight (exact HT)."""
    x = mesh.cell_centroids[:, 0]
    front = x < x.median()
    draw = torch.rand(mesh.n_cells, generator=torch.Generator().manual_seed(0))
    keep = front | (draw < 1.0 / bias)
    cells = mesh.cells[keep]
    used, cells = torch.unique(cells, return_inverse=True)
    sub = Mesh(points=mesh.points[used], cells=cells)
    scale_measures(sub, torch.where(front[keep], 1.0, float(bias)).to(sub.points.dtype))
    return sub


def _center(mesh: Mesh, **kwargs) -> torch.Tensor:
    return CenterMesh(store_center_as="center", **kwargs)(mesh).global_data["center"]


def test_center_mesh_measure_weighting_recovers_full_centroid_from_biased_sample():
    """HT-weighted centering is a density-invariant frame; raw areas are not.

    A 10:1 front/back subsample carrying exact Horvitz-Thompson weights is
    centered, with ``use_measure_weighting``, where the full mesh's
    area-weighted centroid is (to the sample's quadrature error, ~0.4% of the
    extent here); the plain point mean and the raw-area centroid of the same
    sample both sit ~25% of the extent toward the dense half.
    """
    full = _triangulated_patch(241, 81)
    sub = _biased_subsample(full, bias=10)
    full_center = _center(full, use_area_weighting=True)
    extent = full.points[:, 0].max() - full.points[:, 0].min()

    ht = _center(sub, use_area_weighting=True, use_measure_weighting=True)
    area_only = _center(sub, use_area_weighting=True)
    plain = _center(sub, use_area_weighting=False)

    tol = 0.01 * extent
    assert torch.linalg.vector_norm(ht - full_center) < tol
    assert torch.linalg.vector_norm(area_only - full_center) > 5 * tol
    assert torch.linalg.vector_norm(plain - full_center) > 5 * tol


def test_center_mesh_measure_weighting_off_is_unchanged():
    """Default flag: raw-area centroid even when measure weights are recorded."""
    sub = _biased_subsample(_triangulated_patch(31, 11), bias=10)
    expected = (sub.cell_centroids * sub.cell_areas[:, None]).sum(
        0
    ) / sub.cell_areas.sum()

    assert torch.equal(_center(sub, use_area_weighting=True), expected)
    assert "use_measure_weighting" not in repr(CenterMesh())
    assert "use_measure_weighting=True" in repr(CenterMesh(use_measure_weighting=True))
