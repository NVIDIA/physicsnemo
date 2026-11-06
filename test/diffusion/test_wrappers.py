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
from jaxtyping import Float, Shaped
from tensordict import TensorDict

from physicsnemo.core import Module
from physicsnemo.diffusion import DiffusionExtraOutputs, DiffusionExtraTimes
from physicsnemo.diffusion.utils import ConcatConditionWrapper
from physicsnemo.models.diffusion_unets import SongUNet


class CustomExtraTimes(DiffusionExtraTimes):
    r"""Additional time fields that callers can supply independently."""

    test_t: Float[torch.Tensor, " B"] | None = None


class CustomExtraOutputs(DiffusionExtraOutputs):
    test_pred: Float[torch.Tensor, " B *dims"] | None = None


class StructuredDiffusionModel(Module):
    r"""Linear model with optional destination times and learned loss weights."""

    def __init__(self, output: str = "tuple"):
        super().__init__()
        self.output = output
        self.gain = torch.nn.Parameter(torch.tensor(0.75))

    def forward(
        self,
        x: Float[torch.Tensor, " B *dims"],
        t: Float[torch.Tensor, " B"]
        | tuple[Float[torch.Tensor, " B"], Shaped[DiffusionExtraTimes, " B"]],
        condition: TensorDict | None = None,
        *,
        offset: float = 0.0,
    ) -> (
        Float[torch.Tensor, " B *dims"]
        | tuple[Float[torch.Tensor, " B *dims"], Shaped[DiffusionExtraOutputs, " B"]]
    ):
        r"""Return a prediction, optionally paired with its loss weight."""
        t, extra_times = t if isinstance(t, tuple) else (t, None)
        t_bc = t.reshape(-1, *([1] * (x.ndim - 1)))  # (B, 1, ...)
        prediction = self.gain * x + t_bc + offset
        if extra_times is not None:
            for time in extra_times.to_tensordict().values():
                prediction = prediction + time.reshape(-1, *([1] * (x.ndim - 1)))
        if condition is not None:
            prediction = prediction + condition["shift"]
        if self.output == "tensor":
            return prediction
        if self.output == "custom":
            return prediction, CustomExtraOutputs(
                test_pred=prediction.square(), batch_size=[x.shape[0]]
            )
        return prediction, DiffusionExtraOutputs(
            logvar=self.gain * t, batch_size=[x.shape[0]]
        )


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


@pytest.mark.parametrize("fields,output", [(("r",), "tuple"), (("test_t",), "custom")])
@pytest.mark.parametrize("compile_model", [False, True])
def test_concat_wrapper_extra_fields(fields, output, compile_model, device):
    model = StructuredDiffusionModel(output).to(device)
    wrapper = ConcatConditionWrapper(model)
    x = torch.randn(2, 3, 5, 7, device=device, requires_grad=True)
    t = torch.tensor([0.4, 0.8], device=device, requires_grad=True)
    condition = torch.ones(2, 1, 5, 7, device=device, requires_grad=True)
    inputs = [x, t, condition, model.gain]
    time = t
    if fields is not None:
        values = {
            name: (t.detach() / (index + 2)).requires_grad_(True)
            for index, name in enumerate(fields)
        }
        inputs.extend(values.values())
        time = (t, CustomExtraTimes(**values, batch_size=[2]))
    expected = model(torch.cat((x, condition), dim=1), time)
    if compile_model:
        wrapper = torch.compile(wrapper, backend="aot_eager", fullgraph=True)
    actual = wrapper(x, time, condition)
    actual, extras = actual
    expected, expected_extras = expected
    torch.testing.assert_close(
        extras.to_tensordict().to_dict(), expected_extras.to_tensordict().to_dict()
    )
    actual = actual.sum() + sum(
        value.sum() for value in extras.to_tensordict().values()
    )
    expected = expected.sum() + sum(
        value.sum() for value in expected_extras.to_tensordict().values()
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), inputs),
        torch.autograd.grad(expected.sum(), inputs),
    )
