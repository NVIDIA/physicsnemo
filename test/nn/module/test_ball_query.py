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

from physicsnemo.nn.module.ball_query import BQWarp
from test.conftest import requires_module


@requires_module("warp")
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("reverse_mapping", [True, False])
def test_bqwarp_batch_gt_1(device: str, batch_size: int, reverse_mapping: bool):
    """BQWarp should accept batch_size > 1 and return correctly shaped outputs."""
    torch.manual_seed(42)
    n_points = 30
    n_grid = 20
    radius = 1.5
    neighbors_in_radius = 5

    bq = BQWarp(radius=radius, neighbors_in_radius=neighbors_in_radius)

    x = torch.randn(batch_size, n_points, 3, device=device)
    p_grid = torch.randn(batch_size, n_grid, 3, device=device)

    mapping, outputs = bq(x, p_grid, reverse_mapping=reverse_mapping)

    if reverse_mapping:
        expected_query_count = n_grid
    else:
        expected_query_count = n_points

    assert mapping.shape == (batch_size, expected_query_count, neighbors_in_radius)
    assert outputs.shape == (batch_size, expected_query_count, neighbors_in_radius, 3)


@requires_module("warp")
def test_bqwarp_batch_1_unchanged(device: str):
    """BQWarp with B=1 should produce the same results as before the batching change."""
    torch.manual_seed(42)
    n_points = 30
    n_grid = 20
    radius = 1.5
    neighbors_in_radius = 5

    bq = BQWarp(radius=radius, neighbors_in_radius=neighbors_in_radius)

    x = torch.randn(1, n_points, 3, device=device)
    p_grid = torch.randn(1, n_grid, 3, device=device)

    mapping, outputs = bq(x, p_grid, reverse_mapping=True)
    assert mapping.shape == (1, n_grid, neighbors_in_radius)
    assert outputs.shape == (1, n_grid, neighbors_in_radius, 3)


@requires_module("warp")
def test_bqwarp_compile(device: str):
    """BQWarp should work under torch.compile with deterministic shapes."""
    if "cuda" in device:
        pytest.skip("Skipping BQWarp torch.compile on CUDA")
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available")

    torch.manual_seed(42)
    bq = BQWarp(radius=1.5, neighbors_in_radius=5)
    x = torch.randn(2, 20, 3, device=device)
    p_grid = torch.randn(2, 15, 3, device=device)

    eager_map, eager_out = bq(x, p_grid)
    compiled_bq = torch.compile(bq, fullgraph=True)
    comp_map, comp_out = compiled_bq(x, p_grid)

    torch.testing.assert_close(eager_map, comp_map, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(eager_out, comp_out, atol=1e-6, rtol=1e-6)
