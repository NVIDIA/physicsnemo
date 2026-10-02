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

"""Deterministic selected-coordinate gradients, including repeated neighbors."""

import pytest
import torch

from physicsnemo.nn.functional.neighbors.radius_search._warp_impl import (
    _deterministic_point_gradients,
    apply_grad_to_points,
    radius_search_impl,
)


@pytest.fixture
def deterministic():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.fixture(params=["cpu", "cuda"])
def target_device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return request.param


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("limited", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_point_gradient_layouts(deterministic, target_device, batched, limited, empty):
    batches, points, queries, capacity = (2 if batched else 1), 4, 3, 4
    shape = [batches, points, 3] if batched else [points, 3]
    indices = torch.tensor(
        [[1, 1, -999, -999], [2, -999, -999, -999], [-999] * 4],
        device=target_device,
        dtype=torch.int32,
    )
    counts = torch.tensor([2, 1, 0], device=target_device, dtype=torch.int32)
    if batched:
        indices = indices.repeat(batches, 1, 1)
        counts = counts.repeat(batches, 1)
    gradients = torch.arange(
        indices.numel() * 3, device=target_device, dtype=torch.float32
    ).reshape(*indices.shape, 3)
    valid = torch.arange(capacity, device=target_device) < counts.unsqueeze(-1)
    gradients[~valid] = float("nan")
    if empty:
        counts.zero_()
        valid.zero_()
    reference = gradients.new_zeros(shape)
    for b in range(batches):
        for q in range(queries):
            count = int(counts[b, q] if batched else counts[q])
            for k in range(count):
                idx = int(indices[b, q, k] if batched else indices[q, k])
                if batched:
                    reference[b, idx] += gradients[b, q, k]
                else:
                    reference[idx] += gradients[q, k]
    if not limited:
        coordinates = valid.nonzero().T
        indices = torch.cat((coordinates[:-1], indices[valid][None]), dim=0).to(
            torch.int32
        )
        gradients = gradients[valid]
    # Exercise noncontiguous gradient views without altering values.
    gradients = gradients.transpose(-1, -2).contiguous().transpose(-1, -2)
    actual = apply_grad_to_points(
        indices, counts, gradients, shape, capacity if limited else None
    )
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("limited", [False, True])
def test_collision_backward_is_bitwise_repeatable(
    deterministic, target_device, limited
):
    generator = torch.Generator(device=target_device).manual_seed(826)
    q, k, n = 8192, 4, 7
    indices = torch.randint(
        n, (q, k), generator=generator, device=target_device, dtype=torch.int32
    )
    counts = torch.full((q,), k, device=target_device, dtype=torch.int32)
    gradients = torch.randn(q, k, 3, generator=generator, device=target_device)
    if not limited:
        indices = torch.stack(
            (
                torch.arange(q, device=target_device).repeat_interleave(k),
                indices.flatten(),
            )
        ).to(torch.int32)
        gradients = gradients.reshape(-1, 3)
    reference = apply_grad_to_points(
        indices, counts, gradients, [n, 3], k if limited else None
    )
    for _ in range(12):
        actual = apply_grad_to_points(
            indices, counts, gradients, [n, 3], k if limited else None
        )
        assert torch.equal(actual, reference)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("limited", [False, True])
def test_public_warp_backward_matches_selected_indices(
    deterministic, target_device, batched, limited
):
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [4.0, 0.0, 0.0]], device=target_device
    )
    queries = torch.tensor(
        [[0.02, 0.0, 0.0], [0.05, 0.0, 0.0], [10.0, 0.0, 0.0]], device=target_device
    )
    if batched:
        points = points.repeat(2, 1, 1)
        queries = queries.repeat(2, 1, 1)
    points.requires_grad_()
    indices, selected, _, counts = radius_search_impl(
        points, queries, 0.5, 4 if limited else None, False, True
    )
    weights = torch.arange(
        selected.numel(), device=target_device, dtype=selected.dtype
    ).reshape_as(selected)
    expected = _deterministic_point_gradients(
        indices, counts, weights, list(points.shape), 4 if limited else None
    )
    for _ in range(3):
        actual = torch.autograd.grad(selected, points, weights, retain_graph=True)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_selected_point_gradient_supports_higher_order(deterministic):
    indices = torch.tensor([[0, 1], [1, 1]], dtype=torch.int32)
    counts = torch.tensor([2, 2], dtype=torch.int32)
    gradients = torch.randn(2, 2, 3, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda g: _deterministic_point_gradients(indices, counts, g, [2, 3], 2),
        (gradients,),
    )
