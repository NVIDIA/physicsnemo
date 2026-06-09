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

"""Generate committed reference artifacts for the Point-Transformer attention blocks.

These ``.mdlus`` checkpoints and ``.pth`` input/output references are consumed by
``test_reference_regression`` in
``test/nn/module/test_point_transformer_attention.py`` (MOD-008b/c). Generation
is seeded and runs on CPU (torch ``LayerNorm``) so the golden is deterministic;
small dimensions keep the committed fixtures lean.

Run from the repo root after an intentional behavior change::

    python test/nn/module/data/generate_point_transformer_attention_references.py
"""

import os

import torch

from physicsnemo.nn import LocalPointTransformerBlock, LocalTokenCrossAttentionBlock

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def _generate_self_block() -> None:
    torch.manual_seed(20240611)
    block = LocalPointTransformerBlock(
        dim=16,
        num_heads=2,
        neighbor_k=4,
        dilation=2,
        mlp_ratio=2,
        dropout=0.0,
        conditioning_dim=4,
    ).eval()
    feats = torch.randn(24, 16)
    coords = torch.randn(24, 3)
    cond = torch.randn(4)
    with torch.no_grad():
        out = block(feats, coords, cond=cond)
    block.save(os.path.join(DATA_DIR, "local_point_transformer_block_v1.mdlus"))
    torch.save(
        {"feats": feats, "coords": coords, "cond": cond, "out": out},
        os.path.join(DATA_DIR, "local_point_transformer_block_v1.pth"),
    )


def _generate_cross_block() -> None:
    torch.manual_seed(20240612)
    block = LocalTokenCrossAttentionBlock(
        dim=16,
        num_heads=2,
        neighbor_k=4,
        mlp_ratio=2,
        dropout=0.0,
        conditioning_dim=4,
    ).eval()
    qf = torch.randn(20, 16)
    qc = torch.randn(20, 3)
    cf = torch.randn(14, 16)
    cc = torch.randn(14, 3)
    cond = torch.randn(4)
    with torch.no_grad():
        out = block(qf, qc, cf, cc, cond=cond)
    block.save(os.path.join(DATA_DIR, "local_token_cross_attention_block_v1.mdlus"))
    torch.save(
        {"qf": qf, "qc": qc, "cf": cf, "cc": cc, "cond": cond, "out": out},
        os.path.join(DATA_DIR, "local_token_cross_attention_block_v1.pth"),
    )


if __name__ == "__main__":
    _generate_self_block()
    _generate_cross_block()
    print(f"wrote reference artifacts to {DATA_DIR}")
