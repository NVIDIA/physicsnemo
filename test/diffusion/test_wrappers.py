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

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.diffusion.utils import ConcatConditionWrapper
from physicsnemo.models.diffusion_unets import (
    DhariwalUNet,
    SongUNetPosEmbd,
    SongUNetPosLtEmbd,
)
from physicsnemo.models.diffusion_unets import (
    SongUNet as LegacySongUNet,
)

from .helpers import ConditionedConv2dPredictor


class CustomSongUNet(LegacySongUNet):
    """User subclass retaining the legacy calling convention."""


class SongUNet(ConditionedConv2dPredictor):
    """Unrelated PhysicsNeMo model sharing a legacy model's name."""


WRAPPER_MODELS = (
    LegacySongUNet,
    SongUNetPosEmbd,
    SongUNetPosLtEmbd,
    DhariwalUNet,
    CustomSongUNet,
    ConditionedConv2dPredictor,
    SongUNet,
)


def _make_wrapper_inputs(model_class):
    if issubclass(model_class, ConditionedConv2dPredictor):
        model = model_class()
    else:
        kwargs = {}
        in_channels = 4
        if issubclass(model_class, SongUNetPosEmbd):
            kwargs["N_grid_channels"] = 0
        if issubclass(model_class, SongUNetPosLtEmbd):
            kwargs["lead_time_channels"] = 2
            in_channels += 2
        model = model_class(
            img_resolution=8,
            in_channels=in_channels,
            out_channels=3,
            label_dim=4,
            model_channels=64,
            channel_mult=[1],
            channel_mult_emb=1,
            num_blocks=1,
            attn_resolutions=[],
            dropout=0.0,
            **kwargs,
        )
    model.eval()
    x = torch.randn(2, 3, 8, 8)
    t = torch.rand(2)
    condition = TensorDict(
        {"cond_concat": torch.randn(2, 1, 8, 8), "cond_vec": torch.randn(2, 4)},
        batch_size=[2],
    )
    model_kwargs = {}
    if issubclass(model_class, SongUNetPosLtEmbd):
        model_kwargs["lead_time_label"] = torch.tensor([0, 1])
    return model, x, t, condition, model_kwargs


def _make_songunet() -> LegacySongUNet:
    return LegacySongUNet(
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
    from physicsnemo.models.dit import DiT

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


@pytest.mark.parametrize("model_class", WRAPPER_MODELS, ids=lambda cls: cls.__name__)
class TestConsistency:
    def test_matches_direct_call(self, model_class):
        model, x, t, condition, model_kwargs = _make_wrapper_inputs(model_class)
        wrapper = ConcatConditionWrapper(model)
        concatenated = torch.cat([x, condition["cond_concat"]], dim=1)
        if issubclass(model_class, ConditionedConv2dPredictor):
            expected = model(concatenated, t, condition=condition["cond_vec"])
        else:
            expected = model(
                concatenated,
                noise_labels=t,
                class_labels=condition["cond_vec"],
                **model_kwargs,
            )
        torch.testing.assert_close(wrapper(x, t, condition, **model_kwargs), expected)


@pytest.mark.parametrize("model_class", WRAPPER_MODELS, ids=lambda cls: cls.__name__)
class TestCompile:
    def test_fullgraph_forward_backward(self, model_class):
        model, x, t, condition, model_kwargs = _make_wrapper_inputs(model_class)
        wrapper = ConcatConditionWrapper(model)
        x.requires_grad_()
        eager = wrapper(x, t, condition, **model_kwargs)
        expected_grad = torch.autograd.grad(eager.sum(), x)[0]
        compiled = torch.compile(wrapper, backend="aot_eager", fullgraph=True)
        with torch._dynamo.config.patch(error_on_recompile=True):
            for _ in range(2):
                actual = compiled(x, t, condition, **model_kwargs)
                torch.testing.assert_close(actual, eager)
                actual_grad = torch.autograd.grad(actual.sum(), x)[0]
                torch.testing.assert_close(actual_grad, expected_grad)
