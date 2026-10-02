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
"""A transported material fan is not a reinterpreted projected polygon."""

import pytest
import torch

from physicsnemo.experimental.models.meshtransolver import SurfaceContactGraphBuilder
from physicsnemo.nn.functional.neighbors.surface_contact import (
    closest_point_facet,
    closest_point_triangle,
    supported_quad_mask,
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


@pytest.mark.parametrize("scale", [1e-10, 1.0, 1e10])
@pytest.mark.parametrize("collapsed", [False, True])
def test_collapsed_triangle_is_segment_or_point_with_finite_gradients(
    device, scale, collapsed
):
    tri = torch.tensor([[[0.0, 0, 0], [2, 0, 0], [1, 0, 0]]], device=device) * scale
    if collapsed:
        tri.zero_()
    tri.requires_grad_()
    p = tri.new_tensor([[0.7, 0.4, 0.3]]) * scale
    p.requires_grad_()
    with pytest.raises(ValueError, match="degenerate"):
        closest_point_triangle(p, tri)
    result = closest_point_triangle(p, tri, degenerate="edges")
    expected = p.new_tensor([[0.0, 0, 0]] if collapsed else [[0.7, 0, 0]])
    torch.testing.assert_close(result.closest / scale, expected)
    torch.testing.assert_close(result.barycentric.sum(-1), p.new_ones(1))
    assert (result.barycentric >= 0).all()
    assert (result.normal == 0).all()
    (result.distance / scale).sum().backward()
    assert torch.isfinite(p.grad).all() and torch.isfinite(tri.grad).all()


@pytest.mark.parametrize("quad", [False, True])
def test_material_equals_strict_on_supported_geometry_values_and_gradients(
    device, quad
):
    vertices = [[[0.0, 0, 0], [2, 0, 0.1], [1.8, 2, 0.3], [0, 2, -0.1]]]
    f = torch.tensor(vertices, dtype=torch.float64, device=device)
    if not quad:
        f = f[:, :3]
    f.requires_grad_()
    p = f.new_tensor([[1.2, 0.4, 0.6]], requires_grad=True)
    strict = closest_point_facet(p, f)
    material = closest_point_facet(p, f, material_fan=True)
    for field in ("closest", "barycentric", "distance", "normal"):
        torch.testing.assert_close(
            getattr(strict, field), getattr(material, field), atol=0, rtol=0
        )
    a = torch.autograd.grad(strict.distance.sum(), (p, f), retain_graph=True)
    b = torch.autograd.grad(material.distance.sum(), (p, f))
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_folded_fan_matches_explicit_material_triangles_and_gradcheck(device):
    # Its projected boundary is a bow tie, but the transported material
    # triangles are individually nondegenerate and define a two-sided surface.
    f = torch.tensor(
        [[[0.0, 0, 0], [2, 2, 0.4], [2, 0, 0], [0, 2, 0.1]]],
        dtype=torch.float64,
        device=device,
    )
    assert not supported_quad_mask(f).all()
    p = f.new_tensor([[1.6, 0.31, 0.61]], requires_grad=True)
    f.requires_grad_()
    with pytest.raises(ValueError, match="unsupported"):
        closest_point_facet(p, f)
    result = closest_point_facet(p, f, material_fan=True)
    center = f.mean(1)
    pieces = [
        closest_point_triangle(p, torch.stack((center, f[:, i], f[:, (i + 1) % 4]), 1))
        for i in range(4)
    ]
    expected = min(pieces, key=lambda item: float(item.distance.detach()))
    torch.testing.assert_close(result.closest, expected.closest)
    torch.testing.assert_close(result.barycentric.sum(-1), p.new_ones(1))
    assert (result.barycentric >= 0).all()
    assert torch.autograd.gradcheck(
        lambda q, v: closest_point_facet(q, v, material_fan=True).distance,
        (p, f),
        atol=1e-5,
        rtol=1e-4,
    )
    rotation = p.new_tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    transformed = closest_point_facet(
        p @ rotation + 7, f @ rotation + 7, material_fan=True
    )
    torch.testing.assert_close(transformed.closest, result.closest @ rotation + 7)
    torch.testing.assert_close(transformed.barycentric, result.barycentric)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_forecast_fold_and_current_collapse_retain_pair_and_live_gradients(
    device, implementation
):
    reference = torch.tensor(
        [[0.0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0], [1, 0.1, 0.3]], device=device
    )
    faces = torch.tensor([[0, 1, 2, 3]], device=device)
    x = reference.clone()
    x[:4, 1] = 0  # collapse material facet to a segment; do not drop it
    x.requires_grad_()
    v = torch.zeros_like(x, requires_grad=True)
    with torch.no_grad():
        v[4, 2] = -100
    builder = SurfaceContactGraphBuilder(
        implementation=implementation, material_fan=True, prediction_horizon=0.005
    )
    graph = builder(
        x,
        faces=faces,
        shell_thickness=x.new_ones(5),
        velocities=v,
        reference_positions=reference,
    )
    assert graph.edge_index.shape == (2, 1)
    assert graph.edge_index[1, 0] == 4
    torch.testing.assert_close(graph.source_weights.sum(-1), x.new_ones(1))
    loss = graph.edge_features.square().sum() + graph.edge_weights.sum()
    loss.backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(v.grad).all()
    assert x.grad.abs().sum() > 0 and v.grad.abs().sum() > 0
    # The input-quality guard has NOT been removed.
    with pytest.raises(ValueError, match="reference facet IDs"):
        builder(
            x,
            faces=faces,
            shell_thickness=x.new_ones(5),
            velocities=v,
            reference_positions=x.detach(),
        )
    with pytest.raises(ValueError, match="reference_positions"):
        builder(x, faces=faces, shell_thickness=x.new_ones(5), velocities=v)


def test_material_forecast_fold_keeps_two_anchor_contact(device):
    x = torch.tensor(
        [[0.0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0], [1.5, 0.2, 0.4]], device=device
    )
    v = torch.zeros_like(x)
    v[1, 1] = 500  # unsupported projected forecast polygon
    faces = torch.tensor([[0, 1, 2, 3]], device=device)
    args = dict(faces=faces, velocities=v, shell_thickness=x.new_zeros(5))
    with pytest.raises(ValueError, match="forecast facet IDs"):
        SurfaceContactGraphBuilder(implementation="torch", prediction_horizon=0.005)(
            x, **args
        )
    graph = SurfaceContactGraphBuilder(
        implementation="torch", prediction_horizon=0.005, material_fan=True
    )(x, **args, reference_positions=x)
    assert graph.edge_index.shape == (2, 1)
    assert torch.isfinite(graph.edge_features).all()


def test_material_rejects_nonfinite_geometry_and_degenerate_reference_triangle(device):
    ref = torch.tensor([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0]], device=device)
    builder = SurfaceContactGraphBuilder(implementation="torch", material_fan=True)
    args = dict(
        faces=torch.tensor([[0, 1, 2]], device=device),
        velocities=torch.zeros_like(ref),
        shell_thickness=ref.new_zeros(4),
        reference_positions=ref,
    )
    with pytest.raises(ValueError, match="degenerate"):
        builder(ref, **args)
    ref[2, 1] = 1
    live = ref.clone()
    live[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        builder(live, **args)
