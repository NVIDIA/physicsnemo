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

import pytest
import torch

from physicsnemo.experimental.nn import SignedDistanceStencil

pytest.importorskip("warp")


def _cube(device):
    r"""Triangulated cube :math:`[-0.5, 0.5]^3` with outward-facing triangles."""
    v = torch.tensor(
        [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)], dtype=torch.float32, device=device
    )
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    f = torch.tensor([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))], dtype=torch.int64, device=device)
    return v, f


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU"))])
def device(request):
    return torch.device(request.param)


@pytest.mark.parametrize("spacings, stencil", [((0.1,), 3), ((0.1, 0.4), 3), ((0.2,), 5)])
def test_constructor(spacings, stencil):
    layer = SignedDistanceStencil(spacings=spacings, out_features=8, stencil=stencil)
    assert layer.num_probes == len(spacings) * stencil**3
    assert layer.mlp[0].in_features == layer.num_probes
    assert layer.spacings == tuple(spacings)
    assert torch.count_nonzero(layer.out.weight) == 0 and torch.count_nonzero(layer.out.bias) == 0


@pytest.mark.parametrize("kwargs", [dict(spacings=()), dict(spacings=(0.0,)), dict(spacings=(0.1,), stencil=4), dict(spacings=(0.1,), stencil=1)])
def test_constructor_rejects(kwargs):
    with pytest.raises(ValueError):
        SignedDistanceStencil(out_features=4, **kwargs)


def test_probe_values_cube(device):
    v, f = _cube(device)
    layer = SignedDistanceStencil(spacings=(0.2,), out_features=4).to(device)
    values = layer.probe_values(torch.zeros(1, 1, 3, device=device), v[None], f[None])[0, 0]
    expected = (layer.offsets.abs().amax(dim=1) - 0.5) / 0.2  # inside a cube: distance to the nearest face
    torch.testing.assert_close(values, expected.to(values.dtype), atol=1e-4, rtol=0)
    assert torch.all(values < 0)


def test_clip(device):
    v, f = _cube(device)
    layer = SignedDistanceStencil(spacings=(0.01,), out_features=4, clip=4.0).to(device)
    values = layer.probe_values(torch.zeros(1, 1, 3, device=device), v[None], f[None])
    assert torch.all(values == -4.0)


def test_sign_methods_agree_on_a_closed_surface(device):
    v, f = _cube(device)
    points = torch.tensor([[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.3, -0.2, 0.45]]], device=device)
    normals = SignedDistanceStencil(spacings=(0.2,), out_features=4).to(device)
    winding = SignedDistanceStencil(spacings=(0.2,), out_features=4, use_sign_winding_number=True).to(device)
    torch.testing.assert_close(normals.probe_values(points, v[None], f[None]), winding.probe_values(points, v[None], f[None]))


def test_reversed_triangles_change_the_sign(device):
    # both sign methods need outward triangles: reversed, a point inside is no longer negative
    v, f = _cube(device)
    points = torch.tensor([[[0.3, -0.2, 0.45]]], device=device)
    for winding in (False, True):
        layer = SignedDistanceStencil(spacings=(0.02,), out_features=4, use_sign_winding_number=winding).to(device)
        centre = layer.num_probes // 2  # the probe at the point itself
        assert layer.probe_values(points, v[None], f[None])[0, 0, centre] < 0
        assert layer.probe_values(points, v[None], f.flip(-1)[None])[0, 0, centre] > 0


def test_forward_starts_at_zero(device):
    v, f = _cube(device)
    layer = SignedDistanceStencil(spacings=(0.1, 0.3), out_features=6).to(device)
    out = layer(torch.rand(2, 5, 3, device=device) - 0.5, v[None].expand(2, -1, -1), f[None].expand(2, -1, -1))
    assert out.shape == (2, 5, 6)
    assert torch.all(out == 0)


def test_forward_rejects_shapes():
    layer = SignedDistanceStencil(spacings=(0.1,), out_features=2)
    v, f = _cube("cpu")
    with pytest.raises(ValueError):
        layer(torch.zeros(1, 4, 2), v[None], f[None])
    with pytest.raises(ValueError):
        layer(torch.zeros(1, 4, 3), v, f[None])
