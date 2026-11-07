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

from physicsnemo.experimental.nn import RadiusPooledFeatures


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU"),
        ),
    ]
)
def device(request):
    return torch.device(request.param)


def _layer(**kwargs):
    args = dict(
        radii=(0.1, 0.3), caps=(4, 8), voxel_sizes=(0.01, 0.03), hidden=6, query_cap=16
    )
    args.update(kwargs)
    return RadiusPooledFeatures(**args)


def test_constructor():
    layer = _layer()
    assert len(layer.mlps) == 2
    assert layer.radii == (0.1, 0.3) and layer.caps == (4, 8) and layer.query_cap == 16


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(caps=(4,)),
        dict(radii=(0.0, 0.3)),
        dict(query_cap=4),
        dict(voxel_sizes=(0.01, -1.0)),
    ],
)
def test_constructor_rejects(kwargs):
    with pytest.raises(ValueError):
        _layer(**kwargs)


def test_shapes(device):
    layer = _layer().to(device)
    out = layer(
        torch.rand(2, 200, 3, device=device), torch.rand(2, 7, 3, device=device)
    )
    assert len(out) == 2 and all(o.shape == (2, 7, 6) for o in out)


def test_no_neighbours_gives_zeros(device):
    layer = _layer().to(device)
    candidates = torch.rand(1, 50, 3, device=device)
    queries = candidates.new_full((1, 3, 3), 10.0)  # far from every candidate
    for o in layer(candidates, queries):
        assert torch.all(o == 0)


def test_neighbours_within_radius_and_valid(device):
    layer = _layer().to(device)
    candidates = torch.rand(1, 300, 3, device=device)
    queries = torch.rand(1, 5, 3, device=device)
    for (coords, valid), radius, cap in zip(
        layer.neighbours(candidates, queries), layer.radii, layer.caps
    ):
        assert coords.shape == (1, 5, cap, 3) and valid.shape == (1, 5, cap)
        distances = (coords - queries[:, :, None]).norm(dim=-1)
        assert torch.all(distances[valid] <= radius + 1e-5)


def test_query_on_a_candidate_counts(device):
    # a query that coincides with a candidate finds it (index 0 is not mistaken for an empty slot)
    layer = _layer(radii=(0.05,), caps=(1,), voxel_sizes=(0.01,), query_cap=1).to(
        device
    )
    candidates = torch.tensor([[[0.0, 0.0, 0.0]]], device=device)
    ((coords, valid),) = layer.neighbours(candidates, candidates.clone())
    assert bool(valid.all())
    torch.testing.assert_close(coords[0, 0, 0], candidates[0, 0])


def test_eval_is_deterministic_and_train_draws(device):
    layer = _layer(radii=(0.5,), caps=(2,), voxel_sizes=(0.001,), query_cap=16).to(
        device
    )
    candidates = torch.rand(1, 400, 3, device=device)
    queries = torch.rand(1, 20, 3, device=device)
    layer.eval()
    a, b = layer(candidates, queries), layer(candidates, queries)
    torch.testing.assert_close(a[0], b[0])
    layer.train()
    draws = [layer.neighbours(candidates, queries)[0][0] for _ in range(3)]
    assert not all(torch.equal(draws[0], d) for d in draws[1:])


def test_forward_is_pool_of_the_drawn_neighbours(device):
    layer = _layer().to(device).eval()
    candidates, queries = (
        torch.rand(1, 300, 3, device=device),
        torch.rand(1, 9, 3, device=device),
    )
    for a, b in zip(
        layer(candidates, queries),
        layer.pool(layer.neighbours(candidates, queries), queries),
    ):
        torch.testing.assert_close(a, b)


def test_forward_rejects_shapes():
    layer = _layer()
    with pytest.raises(ValueError):
        layer(torch.zeros(1, 10, 2), torch.zeros(1, 3, 3))
    with pytest.raises(ValueError):
        layer(torch.zeros(1, 10, 3), torch.zeros(2, 3, 3))


def test_empty_candidates_and_queries(device):
    layer = _layer().to(device).eval()
    feats = layer(
        torch.zeros(1, 0, 3, device=device), torch.rand(1, 4, 3, device=device)
    )
    assert all(h.shape == (1, 4, 6) and torch.all(h == 0) for h in feats)
    feats = layer(
        torch.rand(1, 50, 3, device=device), torch.zeros(1, 0, 3, device=device)
    )
    assert all(h.shape == (1, 0, 6) for h in feats)


def test_public_methods_reject_shapes():
    layer = _layer()
    with pytest.raises(ValueError):
        layer.neighbours(torch.zeros(1, 10, 2), torch.zeros(1, 3, 3))
    drawn = layer.neighbours(torch.rand(1, 10, 3), torch.rand(1, 3, 3))
    with pytest.raises(ValueError):
        layer.pool(drawn[:1], torch.rand(1, 3, 3))
    with pytest.raises(ValueError):
        layer.pool(drawn, torch.rand(1, 4, 3))
