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

"""Tests for the canonical frame of a weighted point cloud (``spatial._frame``)."""

import pytest
import torch

from physicsnemo.mesh.spatial._frame import canonical_frame, frame_coordinates

D = torch.float64


def _skewed_cloud(n=4000, seed=0, device="cpu"):
    """An anisotropic cloud, skewed along every axis, with positive weights."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, generator=g, dtype=D) * torch.tensor([3.0, 2.0, 1.0], dtype=D)
    x = x + 0.3 * x.square()  # a positive third moment along each axis
    w = torch.rand(n, generator=g, dtype=D) + 0.5
    return x.to(device), w.to(device)


def _orthogonal(seed, improper):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=D))
    q = q * torch.sign(torch.diagonal(r))
    if (torch.linalg.det(q) < 0) != improper:
        q[:, 0] = -q[:, 0]
    return q


def _weighted_moments(x, w, frame):
    w = w / w.sum()
    proj = frame_coordinates(x - (w[:, None] * x).sum(0), frame)
    return (w[:, None] * proj.square()).sum(0), (w[:, None] * proj.pow(3)).sum(0)


def test_the_frame_is_orthonormal(device):
    x, w = _skewed_cloud(device=device)
    frame = canonical_frame(x, w)
    eye = torch.eye(3, dtype=D, device=device)
    torch.testing.assert_close(frame.T @ frame, eye, atol=1e-12, rtol=0)


def test_the_axes_have_ascending_variance_and_positive_skew(device):
    """The columns are in ascending weighted variance, each signed so that the
    weighted third moment along it is positive."""
    x, w = _skewed_cloud(device=device)
    m2, m3 = _weighted_moments(x, w, canonical_frame(x, w))
    assert bool((m2[1:] > m2[:-1]).all())
    assert bool((m3 > 0).all())


@pytest.mark.parametrize("seed, improper", [(1, False), (2, True), (3, False)])
def test_the_frame_moves_with_the_cloud(device, seed, improper):
    """Under x -> k R x + t (a rotation or a reflection) and rescaled weights, the
    frame is R times the frame, and the centered coordinates scale by k."""
    x, w = _skewed_cloud(device=device)
    R = _orthogonal(seed, improper).to(device)
    k, t = 2.5, torch.tensor([3.0, -1.0, 2.0], dtype=D, device=device)
    y = k * x @ R.T + t
    frame_x, frame_y = canonical_frame(x, w), canonical_frame(y, 3 * w)
    torch.testing.assert_close(frame_y, R @ frame_x, atol=1e-9, rtol=0)
    coords_x = frame_coordinates(x - x.mean(0), frame_x)
    coords_y = frame_coordinates(y - y.mean(0), frame_y)
    torch.testing.assert_close(coords_y, k * coords_x, atol=1e-9, rtol=0)


def test_a_symmetric_body_takes_its_signs_from_the_lab_axes(device):
    """On a grid symmetric about its center every third moment vanishes, so each
    axis's sign follows the lab axis it overlaps most: the frame is a permutation
    with positive entries, the same for the body and its mirror image."""
    axes = [torch.linspace(-s, s, n, dtype=D) for s, n in ((3, 7), (2, 5), (1, 3))]
    x = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    x, w = x.to(device), torch.ones(x.shape[0], dtype=D, device=device)
    expected = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=D, device=device
    )
    torch.testing.assert_close(canonical_frame(x, w), expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(canonical_frame(-x, w), expected, atol=1e-12, rtol=0)


def test_float32_points_give_a_float32_frame(device):
    x, w = _skewed_cloud(device=device)
    frame = canonical_frame(x.float(), w.float())
    assert frame.dtype == torch.float32
    torch.testing.assert_close(frame, canonical_frame(x, w).float(), atol=1e-5, rtol=0)


def test_the_coordinates_are_the_product_with_the_frame(device):
    x, w = _skewed_cloud(device=device)
    frame = canonical_frame(x, w)
    torch.testing.assert_close(
        frame_coordinates(x, frame), x @ frame, atol=1e-12, rtol=1e-12
    )
