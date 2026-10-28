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

"""Backend choice and custom neighbor graphs of the mesh LSQ gradients."""

import pytest
import torch

import physicsnemo.mesh.calculus._lsq_gradient as lsq_kernels
import physicsnemo.mesh.calculus._lsq_gradient._torch_impl as lsq_torch
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus import (
    compute_gradient_cells_lsq,
    compute_gradient_points_lsq,
    compute_point_derivatives,
)
from physicsnemo.mesh.neighbors import Adjacency
from physicsnemo.mesh.primitives.planar import unit_square
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.mesh.primitives.volumes import cube_volume
from test.conftest import requires_module

pytestmark = requires_module("warp")


def _point_cloud(n: int = 64, k: int = 6) -> tuple[Mesh, Adjacency]:
    generator = torch.Generator().manual_seed(0)
    points = torch.rand(n, 3, generator=generator)
    indices = torch.randint(0, n, (n * k,), generator=generator)
    return Mesh(points=points), Adjacency(
        offsets=torch.arange(0, (n + 1) * k, k), indices=indices
    )


MESHES = {
    "volume": lambda: cube_volume.load(subdivisions=2),
    "planar": lambda: unit_square.load(subdivisions=2),
    "surface": lambda: sphere_icosahedral.load(subdivisions=1),
}


@pytest.fixture
def backends(monkeypatch):
    """Record which backend each LSQ solve runs on."""
    calls = []
    torch_solve, warp_solve = (
        lsq_torch._mesh_lsq_gradients_torch,
        lsq_kernels.mesh_lsq_gradient,
    )

    def spy_torch(*args, **kwargs):
        calls.append("torch")
        return torch_solve(*args, **kwargs)

    def spy_warp(*args, **kwargs):
        calls.append(kwargs["implementation"])
        return warp_solve(*args, **kwargs)

    monkeypatch.setattr(lsq_torch, "_mesh_lsq_gradients_torch", spy_torch)
    monkeypatch.setattr(lsq_kernels, "mesh_lsq_gradient", spy_warp)
    return calls


@pytest.mark.parametrize(
    ("name", "dtype", "expected"),
    [
        ("volume", torch.float32, "warp"),
        ("planar", torch.float32, "warp"),
        ("volume", torch.float64, "torch"),
        ("surface", torch.float32, "torch"),
    ],
)
def test_none_picks_warp_where_it_is_as_accurate(backends, name, dtype, expected):
    """Warp for float32 meshes whose cells fill the space; PyTorch for float64
    and for surfaces in 3D, whose extrinsic systems are ill-conditioned."""
    mesh = MESHES[name]()
    mesh = Mesh(points=mesh.points.to(dtype), cells=mesh.cells)
    values = mesh.points.sum(dim=1)
    compute_gradient_points_lsq(mesh, values)
    compute_gradient_cells_lsq(mesh, mesh.cell_centroids.sum(dim=1))
    assert backends == [expected, expected]


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_implementation_overrides_the_choice(backends, implementation):
    mesh = MESHES["surface"]() if implementation == "warp" else MESHES["volume"]()
    compute_gradient_points_lsq(mesh, mesh.points[:, 0], implementation=implementation)
    compute_gradient_cells_lsq(
        mesh, mesh.cell_centroids[:, 0], implementation=implementation
    )
    assert backends == [implementation, implementation]


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_custom_adjacency_on_a_point_cloud(implementation):
    """A point cloud differentiates on the neighbor graph it is given."""
    cloud, adjacency = _point_cloud()
    values = torch.sin(cloud.points).sum(dim=1)
    gradient = compute_gradient_points_lsq(
        cloud, values, adjacency=adjacency, implementation=implementation
    )
    expected = lsq_kernels.mesh_lsq_gradient(
        cloud.points,
        values,
        adjacency.offsets,
        adjacency.indices,
        implementation=implementation,
    )
    torch.testing.assert_close(gradient, expected, rtol=0, atol=0)


@pytest.mark.parametrize("implementation", [None, "torch", "warp"])
def test_directed_adjacency(implementation):
    """Neighbor lists may be directed: a point that lists none gets zero."""
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    adjacency = Adjacency(
        offsets=torch.tensor([0, 3, 3, 3, 3]), indices=torch.tensor([1, 2, 3])
    )
    gradient = compute_gradient_points_lsq(
        Mesh(points=points),
        points @ torch.tensor([1.0, 2.0, 3.0]),
        adjacency=adjacency,
        implementation=implementation,
    )
    torch.testing.assert_close(gradient[0], torch.tensor([1.0, 2.0, 3.0]))
    assert not gradient[1:].any()


def test_fields_of_one_dtype_share_one_warp_call(backends):
    """compute_point_derivatives stacks fields into one Warp call."""
    mesh = MESHES["volume"]()
    mesh.point_data["s"] = mesh.points.sum(dim=1)
    mesh.point_data["v"] = 2 * mesh.points
    result = compute_point_derivatives(mesh, keys=["s", "v"], gradient_type="extrinsic")
    assert backends == ["warp"]
    for key in ("s", "v"):
        expected = compute_gradient_points_lsq(
            mesh, mesh.point_data[key], implementation="warp"
        )
        torch.testing.assert_close(result.point_data[f"{key}_gradient"], expected)


def test_second_derivatives_through_warp_match_torch():
    """Differentiating twice through the Warp backend matches PyTorch."""
    mesh = MESHES["volume"]()

    def second_derivatives(implementation):
        points = mesh.points.clone().requires_grad_()
        values = torch.sin(mesh.points).sum(dim=1).requires_grad_()
        gradient = compute_gradient_points_lsq(
            Mesh(points=points, cells=mesh.cells), values, implementation=implementation
        )
        first = torch.autograd.grad(
            gradient.square().sum(), (points, values), create_graph=True
        )
        return torch.autograd.grad(
            sum(d.square().sum() for d in first), (points, values)
        )

    for warp, reference in zip(second_derivatives("warp"), second_derivatives("torch")):
        torch.testing.assert_close(
            warp, reference, rtol=1e-4, atol=1e-4 * reference.abs().max().item()
        )


def test_invalid_backend_requests():
    mesh = MESHES["surface"]()
    with pytest.raises(ValueError, match="Must be 'torch', 'warp', or None"):
        compute_gradient_points_lsq(mesh, mesh.points[:, 0], implementation="cuda")
    with pytest.raises(ValueError, match="supports only intrinsic=False"):
        compute_gradient_points_lsq(
            mesh, mesh.points[:, 0], intrinsic=True, implementation="warp"
        )
