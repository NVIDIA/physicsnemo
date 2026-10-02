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

import torch

from physicsnemo.models.geotransolver.context_projector import (
    ContextProjector,
    GlobalContextBuilder,
)

# =============================================================================
# ContextProjector Tests
# =============================================================================


def test_context_projector_forward(device):
    """Test ContextProjector forward pass."""
    torch.manual_seed(42)

    dim = 64
    heads = 4
    dim_head = 16
    slice_num = 8
    batch_size = 2
    n_tokens = 100

    projector = ContextProjector(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,
        slice_num=slice_num,
        use_te=False,
        plus=False,
    ).to(device)

    x = torch.randn(batch_size, n_tokens, dim).to(device)

    slice_tokens = projector(x)

    # Output shape: [Batch, Heads, Slice_num, dim_head]
    assert slice_tokens.shape == (batch_size, heads, slice_num, dim_head)
    assert not torch.isnan(slice_tokens).any()


def test_global_context_builder_can_return_live_geometry_context(device):
    builder = GlobalContextBuilder(
        functional_dims=(3,),
        geometry_dim=3,
        n_hidden=32,
        n_head=4,
        slice_num=8,
        use_te=False,
    ).to(device)
    local = (torch.randn(1, 12, 3, device=device),)
    geometry = torch.randn(1, 12, 3, device=device, requires_grad=True)

    _, _, detached = builder.build_context(local, None, geometry)
    assert detached is not None
    assert not detached.requires_grad

    _, _, live = builder.build_context(
        local, None, geometry, detach_geometry_context=False
    )
    assert live is not None
    assert live.requires_grad
    live.square().mean().backward()
    assert geometry.grad is not None
    assert torch.isfinite(geometry.grad).all()
