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

"""Integration: apply_lora + wrap_mlp on a real TE GeoTransolver (GPU+TE).

Validates targeting + the fused-MLP path on the actual target model: attention
te.Linear layers wrap as LoRA_te_Linear, and each block's fused te.LayerNormMLP
wraps as the LoRA_te_LayerNormMLP residual.
"""

import pytest
import torch

from test.conftest import requires_module  # noqa: E402

_ATTN = r"blocks\.\d+\.Attn\.(in_project_x|in_project_fx|qkv_project|out_linear|cross_[qkv])"


@requires_module("transformer_engine")
@requires_module("warp")
@requires_module("jaxtyping")
@requires_module("tensordict")
def test_geotransolver_attention_and_wrap_mlp():
    if not torch.cuda.is_available():
        pytest.skip("TE GeoTransolver requires CUDA.")

    from physicsnemo.experimental.models.geotransolver.geotransolver import (
        GeoTransolver,
    )
    from physicsnemo.experimental.peft import (
        LoRAConfig,
        apply_lora,
        is_lora_layer,
        split_params_for_optimizer,
    )
    from physicsnemo.experimental.peft.lora import (
        LoRA_te_LayerNormMLP,
        LoRA_te_Linear,
    )

    torch.manual_seed(0)
    model = GeoTransolver(
        functional_dim=6,
        out_dim=4,
        geometry_dim=3,
        global_dim=3,
        n_layers=2,
        n_hidden=64,
        dropout=0.0,
        n_head=8,
        act="gelu",
        mlp_ratio=4,
        slice_num=16,
        use_te=True,
        plus=False,
        include_local_features=False,
    ).cuda()

    res = apply_lora(model, LoRAConfig(rank=8, target_pattern=_ATTN, wrap_mlp=True))

    # Each block's fused FFN became the residual wrapper.
    assert isinstance(model.blocks[0].ln_mlp1, LoRA_te_LayerNormMLP)
    assert isinstance(model.blocks[1].ln_mlp1, LoRA_te_LayerNormMLP)
    # Attention projections became TE Linear LoRA wrappers.
    assert isinstance(model.blocks[0].Attn.qkv_project, LoRA_te_Linear)
    # Sanity on counts: 2 blocks * (>=5 attn te.Linear + 1 fused MLP).
    assert res.n_wrapped >= 2 * 6
    assert res.n_trainable > 0 and res.n_frozen > res.n_trainable

    # Optimizer routing: LoRA params are collected separately (→ AdamW, not Muon).
    groups = split_params_for_optimizer(model)
    assert len(groups["lora"]) == 2 * res.n_wrapped  # lora_A + lora_B per wrapped layer
    assert all(p.requires_grad for p in groups["lora"])

    # Forward + backward runs and produces gradients on a LoRA param.
    b, n = 2, 64
    local = torch.randn(b, n, 6, device="cuda")
    pos = local[:, :, :3]
    geom = torch.randn(b, 80, 3, device="cuda")
    glob = torch.randn(b, 5, 3, device="cuda")
    out = model(local, local_positions=pos, global_embedding=glob, geometry=geom)
    out.pow(2).mean().backward()
    some_lora = next(m for m in model.modules() if is_lora_layer(m))
    assert some_lora.lora_A.grad is not None
