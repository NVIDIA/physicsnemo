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

"""Tests for diffusion conditioning wrappers."""

import torch
from tensordict import TensorDict

from physicsnemo.diffusion.utils import ConcatConditionWrapper
from physicsnemo.models.diffusion_unets import SongUNet


def _make_songunet() -> SongUNet:
    return SongUNet(
        img_resolution=8,
        in_channels=4,
        out_channels=3,
        label_dim=4,
        model_channels=8,
        channel_mult=[1],
        channel_mult_emb=1,
        num_blocks=1,
        attn_resolutions=[],
        dropout=0.0,
        use_apex_gn=False,
    )


def _make_dit():
    from physicsnemo.experimental.models.dit import DiT

    return DiT(
        input_size=8,
        patch_size=4,
        in_channels=4,
        out_channels=3,
        condition_dim=4,
        hidden_size=16,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        attention_backend="timm",
        layernorm_backend="torch",
        dit_initialization=False,
    )


def test_concat_wrapper_songunet_tensordict_example():
    model = _make_songunet()
    wrapper = ConcatConditionWrapper(model)
    x = torch.randn(2, 3, 8, 8)
    t = torch.rand(2)
    condition = TensorDict(
        {
            "cond_concat": torch.randn(2, 1, 8, 8),
            "cond_vec": torch.randn(2, 4),
        },
        batch_size=[2],
    )

    out = wrapper(x, t, condition)
    assert out.shape == (2, 3, 8, 8)


def test_concat_wrapper_dit_tensordict_example():
    model = _make_dit()
    wrapper = ConcatConditionWrapper(model)
    x = torch.randn(2, 3, 8, 8)
    t = torch.rand(2)
    condition = TensorDict(
        {
            "cond_concat": torch.randn(2, 1, 8, 8),
            "cond_vec": torch.randn(2, 4),
        },
        batch_size=[2],
    )

    out = wrapper(x, t, condition)
    assert out.shape == (2, 3, 8, 8)


def test_concat_wrapper_plain_tensor_as_image_condition():
    model = _make_dit()
    wrapper = ConcatConditionWrapper(model)
    x = torch.randn(2, 3, 8, 8)
    t = torch.rand(2)
    cond_image = torch.randn(2, 1, 8, 8)

    out = wrapper(x, t, cond_image)
    assert out.shape == (2, 3, 8, 8)
