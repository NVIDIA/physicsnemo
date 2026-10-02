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

"""Search uses explicit Torch autograd, not Warp-owned tensor gradients."""

import pytest
import torch
import warp as wp
from torch.utils.checkpoint import checkpoint

from physicsnemo.nn.functional.neighbors.contact_search._warp_impl import (
    contact_search_impl,
)
from physicsnemo.nn.functional.neighbors.radius_search._warp_impl import (
    radius_search_impl,
)


def _search(points, queries, kind, max_points):
    if kind == "radius":
        return radius_search_impl(points, queries, 2.0, max_points, False, True)
    offsets = torch.zeros(
        queries.shape[-2] + 1, dtype=torch.int32, device=points.device
    )
    excluded = torch.empty(0, dtype=torch.int32, device=points.device)
    return contact_search_impl(
        points, queries, 2.0, excluded, offsets, max_points, False, True
    )


@pytest.mark.parametrize("kind", ["radius", "contact"])
@pytest.mark.parametrize("max_points", [None, 3])
@pytest.mark.parametrize("batched", [False, True])
def test_search_does_not_allocate_warp_gradients(
    device, monkeypatch, kind, max_points, batched
):
    """Reproduce the implicit Warp allocation on grad-enabled batched inputs."""
    shape = (2, 6, 3) if batched else (6, 3)
    points = (
        torch.linspace(0.0, 0.5, steps=torch.tensor(shape).prod().item(), device=device)
        .reshape(shape)
        .requires_grad_()
    )
    queries = (points.detach() + 0.01).requires_grad_()
    wraps = []
    original = wp.from_torch

    def checked_wrap(tensor, *args, **kwargs):
        wrapped = original(tensor, *args, **kwargs)
        grad_pointer = wrapped.grad
        assert not grad_pointer, "Search allocated a redundant Warp gradient buffer"
        wraps.append(tensor.shape)
        return wrapped

    monkeypatch.setattr(wp, "from_torch", checked_wrap)
    _, gathered, _, _ = _search(points, queries, kind, max_points)
    assert wraps
    assert points.grad is None and queries.grad is None
    gathered.square().sum().backward()
    assert points.grad is not None and torch.isfinite(points.grad).all()
    assert queries.grad is None  # Discrete neighbor selection has no derivative.


@pytest.mark.parametrize("kind", ["radius", "contact"])
@pytest.mark.parametrize("max_points", [None, 3])
@pytest.mark.parametrize("batched", [False, True])
def test_search_backward_matches_indexed_gather(device, kind, max_points, batched):
    shape = (2, 6, 3) if batched else (6, 3)
    points = (
        torch.linspace(0.0, 0.5, steps=torch.tensor(shape).prod().item(), device=device)
        .reshape(shape)
        .requires_grad_()
    )
    queries = points.detach() + 0.01
    indices, gathered, _, counts = _search(points, queries, kind, max_points)
    reference = points.detach().clone().requires_grad_()
    if max_points is None:
        expected = (
            reference[indices[0].long(), indices[2].long()]
            if batched
            else reference[indices[1].long()]
        )
    else:
        valid = torch.arange(max_points, device=device) < counts.unsqueeze(-1)
        if batched:
            batch = torch.arange(shape[0], device=device)[:, None, None]
            expected = reference[batch, indices.clamp_min(0).long()]
        else:
            expected = reference[indices.clamp_min(0).long()]
        expected = expected * valid.unsqueeze(-1)
    weights = torch.linspace(0.1, 1.0, gathered.numel(), device=device).reshape_as(
        gathered
    )
    torch.testing.assert_close(gathered, expected)
    (gathered * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(points.grad, reference.grad)


@pytest.mark.parametrize("kind", ["radius", "contact"])
def test_checkpointed_four_step_search_preserves_bptt(device, kind):
    initial = torch.linspace(0.0, 0.5, 18, device=device).reshape(1, 6, 3)

    def step(x):
        _, neighbors, _, _ = _search(x, x, kind, 3)
        return x + 0.05 * neighbors.mean(dim=-2)

    def rollout(use_checkpoint):
        leaf = initial.clone().requires_grad_()
        state = leaf
        for _ in range(4):
            state = (
                checkpoint(step, state, use_reentrant=False)
                if use_checkpoint
                else step(state)
            )
        state.square().sum().backward()
        return state.detach(), leaf.grad

    eager, eager_grad = rollout(False)
    recomputed, recomputed_grad = rollout(True)
    torch.testing.assert_close(recomputed, eager)
    torch.testing.assert_close(recomputed_grad, eager_grad)
    assert torch.isfinite(recomputed_grad).all() and recomputed_grad.abs().sum() > 0
