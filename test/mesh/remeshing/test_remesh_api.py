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

"""Device-independent tests for the public Mesh remeshing API."""

import inspect
import math

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.mesh.remeshing import remesh
from physicsnemo.mesh.remeshing._remeshing import _close_and_orient


def _fillet_box(n_fillet: int = 64, radius: float = 0.05) -> Mesh:
    """A closed unit box like a CAD export: flat sides of one quad, finely
    sampled vertical fillets, and caps fanned from their centers."""
    outline = []
    for k, (x, y) in enumerate([(1, 1), (-1, 1), (-1, -1), (1, -1)]):
        angle = torch.linspace(0, math.pi / 2, n_fillet) + k * math.pi / 2
        center = torch.tensor([x, y]) * (0.5 - radius)
        outline.append(center + radius * torch.stack([angle.cos(), angle.sin()], 1))
    outline = torch.cat(outline)
    n = outline.shape[0]
    bottom = torch.nn.functional.pad(outline, (0, 1))
    top = torch.nn.functional.pad(outline, (0, 1), value=1.0)
    centers = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    i = torch.arange(n)
    j = (i + 1) % n
    cells = torch.cat(
        [
            torch.stack([i, j, j + n], 1),
            torch.stack([i, j + n, i + n], 1),
            torch.stack([torch.full_like(i, 2 * n), j, i], 1),
            torch.stack([torch.full_like(i, 2 * n + 1), i + n, j + n], 1),
        ]
    )
    return Mesh(points=torch.cat([bottom, top, centers]), cells=cells)


def _assert_even_closed_and_outward(output: Mesh, max_area_ratio: float) -> None:
    areas = output.cell_areas.double()
    p1, p99 = torch.quantile(areas, areas.new_tensor([0.01, 0.99]))
    assert float(p99 / p1) < max_area_ratio
    # Closed and consistently wound: every directed edge appears exactly once,
    # and its reverse also appears
    cells = output.cells
    directed = torch.cat([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]])
    unique_directed = torch.unique(directed, dim=0)
    assert unique_directed.shape[0] == directed.shape[0]
    starts, ends = unique_directed.T
    n_points = output.n_points
    assert torch.isin(ends * n_points + starts, starts * n_points + ends).all()
    vertices = output.points.double()[cells]
    signed_volume = (
        vertices[:, 0] * torch.linalg.cross(vertices[:, 1], vertices[:, 2])
    ).sum() / 6
    assert float(signed_volume) > 0.9


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_remesh_evens_out_large_faces_and_slivers(device):
    source = _fillet_box().to(device)
    areas = source.cell_areas
    assert float(areas.max() / areas.min()) > 1000

    output = remesh(source, 250)

    assert 200 <= output.n_points <= 260
    _assert_even_closed_and_outward(output, max_area_ratio=10.0)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_remesh_changes_a_mesh_at_its_own_point_count(device):
    source = _fillet_box().to(device)
    source.point_data["affine"] = source.points @ source.points.new_tensor(
        [0.3, -0.7, 1.1]
    )
    source.point_data["resolution"] = 1.0 + (source.points[:, 2] > 0.5).float()

    output = remesh(
        source,
        source.n_points,
        transfer_point_data="affine",
        resolution_field="resolution",
    )

    _assert_even_closed_and_outward(output, max_area_ratio=30.0)
    torch.testing.assert_close(
        output.point_data["affine"],
        output.points @ output.points.new_tensor([0.3, -0.7, 1.1]),
        rtol=1e-4,
        atol=1e-4,
    )
    upper_fraction = (output.points[:, 2] > 0.5).float().mean()
    assert float(upper_fraction) > 0.6


def test_remesh_repair_fills_holes_and_cuts_out_bad_faces():
    # A remesh of a closed sphere with one vertex's faces missing and a face
    # doubled the other way round, whose edges then have three faces each
    source = sphere_icosahedral.load(subdivisions=1)
    coefficients = source.points.new_tensor([0.3, -0.7, 1.1])
    cells = source.cells
    keep = ~(cells == 0).any(dim=1)
    broken = torch.cat([cells[keep], cells[keep][:1, [0, 2, 1]]])
    output = Mesh(points=source.points, cells=broken)
    output.point_data["affine"] = source.points @ coefficients
    source_faces = torch.zeros(source.n_points, dtype=torch.int64)
    source_faces[cells.flatten()] = torch.arange(source.n_cells).repeat_interleave(3)

    repaired = _close_and_orient(output, source, source_faces)

    # The cut's fill adds a centroid, and the missing vertex is dropped
    is_new = torch.cdist(repaired.points, source.points).min(dim=1).values > 1e-6
    assert int(is_new.sum()) >= 1
    assert not (repaired.points == source.points[0]).all(dim=1).any()
    _assert_even_closed_and_outward(repaired, max_area_ratio=100.0)
    # A filled-in centroid takes its boundary's mean, which is affine here
    torch.testing.assert_close(
        repaired.point_data["affine"], repaired.points @ coefficients
    )


def test_remesh_keeps_a_strip_thinner_than_the_cluster_spacing():
    # Refined, the strip has no face spanning three of the four clusters
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0e-7, 0.0], [0.0, 1.0e-7, 0.0]],
        dtype=torch.float64,
    )
    source = Mesh(points=points, cells=torch.tensor([[0, 1, 2], [0, 2, 3]]))

    output = remesh(source, 4, max_iterations=0)

    assert output.n_points == 4
    assert output.n_cells == 2


def test_remesh_public_signatures():
    remesh_signature = inspect.signature(remesh)
    assert tuple(remesh_signature.parameters) == (
        "mesh",
        "n_clusters",
        "max_iterations",
        "transfer_point_data",
        "resolution_field",
    )
    assert Mesh.remesh is remesh
    mesh_remesh_signature = inspect.signature(Mesh.remesh)
    assert mesh_remesh_signature == remesh_signature
    assert remesh_signature.parameters["max_iterations"].default == 4
    assert remesh_signature.parameters["transfer_point_data"].default is False
    assert remesh_signature.parameters["resolution_field"].default is None


def test_remesh_runs_on_cpu():
    source = sphere_icosahedral.load(subdivisions=2)
    output = remesh(
        source,
        48,
        max_iterations=1,
    )

    assert 3 <= output.n_points <= 48
    assert output.points.shape[1] == 3
    assert output.cells.ndim == 2 and output.cells.shape[1] == 3
    assert output.points.device.type == "cpu"
    assert output.points.dtype == source.points.dtype
    assert output.cells.dtype == torch.int64
    report = output.validate(check_manifoldness=True)
    assert report["valid"]
    assert report["is_manifold"]


def test_mesh_remesh_runs_on_cpu():
    source = sphere_icosahedral.load(subdivisions=1)
    output = source.remesh(24, max_iterations=1)

    assert 3 <= output.n_points <= 24
    assert output.points.device.type == "cpu"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_remesh_preserves_reduced_precision(dtype):
    source = sphere_icosahedral.load(subdivisions=1).to(dtype)

    output = remesh(source, 24, max_iterations=1)

    assert output.points.dtype == dtype
    assert torch.isfinite(output.points).all()
    assert output.validate(check_manifoldness=True)["valid"]


def test_cpu_remesh_rejects_unsafe_geometry():
    nonfinite = sphere_icosahedral.load(subdivisions=1)
    nonfinite.points[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        remesh(nonfinite, 24)

    invalid_cells = sphere_icosahedral.load(subdivisions=1)
    invalid_cells.cells[0, 0] = invalid_cells.n_points
    with pytest.raises(ValueError, match="indices"):
        remesh(invalid_cells, 24)


def test_remesh_rejects_non_surface_mesh():
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    cells = torch.tensor([[0, 1], [1, 2]])
    source = Mesh(points=points, cells=cells)

    with pytest.raises(NotImplementedError, match="2D triangle surface"):
        remesh(source, 3)
