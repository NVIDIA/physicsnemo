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

"""Tests for the implicit-function building blocks."""

import math

import pytest
import torch

from physicsnemo.mesh.generate import (
    project_to_zero_set,
    sdf_box,
    sdf_difference,
    sdf_intersection,
    sdf_polygon_2d,
    sdf_sphere,
    sdf_union,
)


def test_sphere_exact_distance_any_dim():
    for d in (2, 3, 4):
        phi = sdf_sphere([0.0] * d, 0.5)
        x = torch.zeros(1, d, dtype=torch.float64)
        assert torch.isclose(phi(x), torch.tensor([-0.5], dtype=torch.float64))
        x[0, 0] = 2.0
        assert torch.isclose(phi(x), torch.tensor([1.5], dtype=torch.float64))


def test_box_exact_inside_and_outside():
    phi = sdf_box([-1.0, -2.0], [1.0, 2.0])
    pts = torch.tensor(
        [[0.0, 0.0], [0.9, 0.0], [2.0, 0.0], [2.0, 3.0]], dtype=torch.float64
    )
    vals = phi(pts)
    assert torch.isclose(vals[0], torch.tensor(-1.0, dtype=torch.float64))
    assert torch.isclose(vals[1], torch.tensor(-0.1, dtype=torch.float64))
    assert torch.isclose(vals[2], torch.tensor(1.0, dtype=torch.float64))
    # Outside a corner: Euclidean distance to it.
    assert torch.isclose(vals[3], torch.tensor(math.sqrt(2.0), dtype=torch.float64))


def test_csg_signs():
    a = sdf_sphere([0.0, 0.0], 1.0)
    b = sdf_sphere([1.0, 0.0], 1.0)
    x = torch.tensor([[0.5, 0.0]], dtype=torch.float64)  # inside both
    assert float(sdf_union(a, b)(x)) < 0
    assert float(sdf_intersection(a, b)(x)) < 0
    assert float(sdf_difference(a, b)(x)) > 0  # removed by b
    y = torch.tensor([[-0.5, 0.0]], dtype=torch.float64)  # only in a
    assert float(sdf_difference(a, b)(y)) < 0


def test_polygon_sign_and_distance():
    square = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    phi = sdf_polygon_2d(square)
    pts = torch.tensor(
        [[0.5, 0.5], [0.5, -0.5], [1.5, 0.5], [0.5, 0.25]], dtype=torch.float64
    )
    vals = phi(pts)
    assert torch.isclose(vals[0], torch.tensor(-0.5, dtype=torch.float64))
    assert torch.isclose(vals[1], torch.tensor(0.5, dtype=torch.float64))
    assert torch.isclose(vals[2], torch.tensor(0.5, dtype=torch.float64))
    assert torch.isclose(vals[3], torch.tensor(-0.25, dtype=torch.float64))


def test_polygon_tolerates_duplicate_vertex():
    """A repeated consecutive vertex (common in real polygon data) makes a
    zero-length segment; the point-to-segment projection must not divide
    0/0 there -- one NaN would poison the min over segments for EVERY
    query point."""
    square = [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    phi = sdf_polygon_2d(square)
    pts = torch.tensor([[0.5, 0.5], [0.5, -0.5], [1.5, 0.5]], dtype=torch.float64)
    vals = phi(pts)
    assert bool(torch.isfinite(vals).all())
    ref = sdf_polygon_2d(square[:1] + square[2:])(pts)
    assert torch.allclose(vals, ref)


def test_polygon_orientation_invariance():
    square_ccw = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    square_cw = list(reversed(square_ccw))
    x = torch.tensor([[0.5, 0.5], [2.0, 0.5]], dtype=torch.float64)
    assert torch.allclose(sdf_polygon_2d(square_ccw)(x), sdf_polygon_2d(square_cw)(x))


def test_projection_lands_on_zero_set():
    phi = sdf_sphere([0.0, 0.0, 0.0], 0.75)
    g = torch.Generator().manual_seed(0)
    x = torch.rand(256, 3, generator=g, dtype=torch.float64) * 2 - 1
    proj = project_to_zero_set(phi, x, iters=4)
    assert float(phi(proj).abs().max()) < 1e-12


def test_projection_survives_gradient_kinks():
    """Points exactly on a polygon edge (sqrt(0) kink) must not go NaN."""
    square = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    phi = sdf_polygon_2d(square)
    x = torch.tensor(
        [[0.5, 0.0], [0.0, 0.0], [0.5, 0.5]], dtype=torch.float64
    )  # on edge, on vertex, center
    proj = project_to_zero_set(phi, x, iters=3)
    assert bool(torch.isfinite(proj).all())
    assert float(phi(proj).abs().max()) < 1e-9


def test_projection_gradient_flow():
    """project_to_zero_set detaches: it is for generation, not autograd.

    (The differentiable path is refit_mesh_to_implicit; this test pins the
    contract so a future change is deliberate.)"""
    r = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    phi = lambda x: x.norm(dim=-1) - r  # noqa: E731
    x = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    proj = project_to_zero_set(phi, x)
    assert not proj.requires_grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_follows_query(dtype):
    phi = sdf_union(sdf_sphere([0.0, 0.0], 0.5), sdf_box([-1.0, -1.0], [0.0, 0.0]))
    x = torch.zeros(4, 2, dtype=dtype)
    assert phi(x).dtype == dtype


def test_polygon_honors_batch_dims():
    """The (..., d) implicit-function contract must hold for the polygon."""
    square = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    phi = sdf_polygon_2d(square)
    g = torch.Generator().manual_seed(0)
    x = torch.rand(3, 5, 7, 2, generator=g, dtype=torch.float64) * 2 - 0.5
    batched = phi(x)
    assert batched.shape == (3, 5, 7)
    flat = phi(x.reshape(-1, 2))
    assert torch.equal(batched.reshape(-1), flat)
