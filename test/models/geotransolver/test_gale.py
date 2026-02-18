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

from physicsnemo.experimental.models.geotransolver.gale import (
    GALE,
    GALE_block,
)

# =============================================================================
# GALE (Geometry-Aware Latent Embeddings) Attention Tests
# =============================================================================


def test_gale_forward_basic(device):
    """Test GALE attention layer forward pass without context."""
    torch.manual_seed(42)

    dim = 64
    heads = 4
    dim_head = 16
    slice_num = 8
    batch_size = 2
    n_tokens = 100

    gale = GALE(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,
        slice_num=slice_num,
        use_te=False,
        plus=False,
        context_dim=dim_head,  # Must match dim_head for cross attention
    ).to(device)

    # Single input tensor wrapped in tuple
    x = torch.randn(batch_size, n_tokens, dim).to(device)

    outputs = gale((x,), context=None)

    assert len(outputs) == 1
    assert outputs[0].shape == (batch_size, n_tokens, dim)
    assert not torch.isnan(outputs[0]).any()


def test_gale_forward_with_context(device):
    """Test GALE attention layer forward pass with cross-attention context."""
    torch.manual_seed(42)

    dim = 64
    heads = 4
    dim_head = 16
    slice_num = 8
    batch_size = 2
    n_tokens = 100
    context_tokens = 32
    context_dim = dim_head

    gale = GALE(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,
        slice_num=slice_num,
        use_te=False,
        plus=False,
        context_dim=context_dim,
    ).to(device)

    x = torch.randn(batch_size, n_tokens, dim).to(device)
    context = torch.randn(batch_size, heads, context_tokens, context_dim).to(device)

    outputs = gale((x,), context=context)

    assert len(outputs) == 1
    assert outputs[0].shape == (batch_size, n_tokens, dim)
    assert not torch.isnan(outputs[0]).any()


def test_gale_forward_multiple_inputs(device):
    """Test GALE attention layer with multiple input tensors."""
    torch.manual_seed(42)

    dim = 64
    heads = 4
    dim_head = 16
    slice_num = 8
    batch_size = 2
    n_tokens_1 = 100
    n_tokens_2 = 150
    context_dim = dim_head

    gale = GALE(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,
        slice_num=slice_num,
        use_te=False,
        plus=False,
        context_dim=context_dim,
    ).to(device)

    x1 = torch.randn(batch_size, n_tokens_1, dim).to(device)
    x2 = torch.randn(batch_size, n_tokens_2, dim).to(device)

    outputs = gale((x1, x2), context=None)

    assert len(outputs) == 2
    assert outputs[0].shape == (batch_size, n_tokens_1, dim)
    assert outputs[1].shape == (batch_size, n_tokens_2, dim)
    assert not torch.isnan(outputs[0]).any()
    assert not torch.isnan(outputs[1]).any()


# =============================================================================
# GALE_block Tests
# =============================================================================


def test_gale_block_forward(device):
    """Test GALE_block transformer block forward pass."""
    torch.manual_seed(42)

    hidden_dim = 64
    n_head = 4
    batch_size = 2
    n_tokens = 100
    slice_num = 8
    context_dim = hidden_dim // n_head

    block = GALE_block(
        num_heads=n_head,
        hidden_dim=hidden_dim,
        dropout=0.0,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=slice_num,
        use_te=False,
        plus=False,
        context_dim=context_dim,
    ).to(device)

    x = torch.randn(batch_size, n_tokens, hidden_dim).to(device)
    context = torch.randn(batch_size, n_head, slice_num, context_dim).to(device)

    outputs = block((x,), global_context=context)

    assert len(outputs) == 1
    assert outputs[0].shape == (batch_size, n_tokens, hidden_dim)
    assert not torch.isnan(outputs[0]).any()


def test_gale_block_multiple_inputs(device):
    """Test GALE_block with multiple input tensors."""
    torch.manual_seed(42)

    hidden_dim = 64
    n_head = 4
    batch_size = 2
    n_tokens_1 = 100
    n_tokens_2 = 150
    slice_num = 8
    context_dim = hidden_dim // n_head

    block = GALE_block(
        num_heads=n_head,
        hidden_dim=hidden_dim,
        dropout=0.0,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=slice_num,
        use_te=False,
        plus=False,
        context_dim=context_dim,
    ).to(device)

    x1 = torch.randn(batch_size, n_tokens_1, hidden_dim).to(device)
    x2 = torch.randn(batch_size, n_tokens_2, hidden_dim).to(device)
    context = torch.randn(batch_size, n_head, slice_num, context_dim).to(device)

    outputs = block((x1, x2), global_context=context)

    assert len(outputs) == 2
    assert outputs[0].shape == (batch_size, n_tokens_1, hidden_dim)
    assert outputs[1].shape == (batch_size, n_tokens_2, hidden_dim)
