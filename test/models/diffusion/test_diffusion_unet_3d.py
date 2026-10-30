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

"""Tests for DiffusionUNet3D."""

from typing import Any

import pytest
import torch
import torch._dynamo
from tensordict import TensorDict

from physicsnemo.models import DiffusionUNet3D
from test.models.diffusion._helpers import (
    GLOBAL_SEED,
    compare_outputs,
    instantiate_model_deterministic,
    load_or_create_checkpoint,
    load_or_create_reference,
)

_CPU_TOLERANCES = {"atol": 1e-3, "rtol": 1e-3}
_GPU_TOLERANCES = {"atol": 1e-2, "rtol": 5e-2}

ARCH_CONFIGS = (
    (
        "unconditional",
        {
            "x_channels": 2,
            "num_levels": 2,
            "model_channels": 8,
            "channel_mult": [1, 2],
            "num_blocks": 1,
            "dropout": 0.0,
        },
        (2, 2, 4, 8, 8),
    ),
    (
        "conditional_attention",
        {
            "x_channels": 2,
            "vol_cond_channels": 2,
            "vec_cond_dim": 4,
            "num_levels": 2,
            "model_channels": 8,
            "channel_mult": [1, 2],
            "num_blocks": 1,
            "attention_levels": [1],
            "dropout": 0.0,
        },
        (2, 2, 4, 8, 8),
    ),
    (
        "ncsnpp",
        {
            "x_channels": 2,
            "vol_cond_channels": 1,
            "vec_cond_dim": 2,
            "num_levels": 3,
            "model_channels": 8,
            "channel_mult": [1, 2, 2],
            "num_blocks": 1,
            "attention_levels": [2],
            "embedding_type": "fourier",
            "channel_mult_noise": 2,
            "encoder_type": "residual",
            "decoder_type": "skip",
            "resample_filter": [1, 3, 3, 1],
            "bottleneck_attention": False,
            "activation": "gelu",
            "dropout": 0.0,
        },
        (2, 2, 4, 8, 8),
    ),
)


@pytest.fixture
def tolerances(device):
    return _CPU_TOLERANCES if device == "cpu" else _GPU_TOLERANCES


def _generate_batch_data(
    model_kwargs: dict[str, Any],
    x_shape: tuple[int, int, int, int, int],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, TensorDict | None]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(GLOBAL_SEED)
    batch_size = x_shape[0]
    x = torch.randn(*x_shape, generator=gen, device="cpu").to(device)
    t = (torch.rand(batch_size, generator=gen) * 0.5 + 0.4).to(device)

    condition_data = {}
    if model_kwargs.get("vec_cond_dim", 0):
        condition_data["vector"] = torch.randn(
            batch_size,
            model_kwargs["vec_cond_dim"],
            generator=gen,
        ).to(device)
    if model_kwargs.get("vol_cond_channels", 0):
        condition_data["volume"] = torch.randn(
            batch_size,
            model_kwargs["vol_cond_channels"],
            *x_shape[2:],
            generator=gen,
        ).to(device)

    condition = (
        TensorDict(condition_data, batch_size=[batch_size]) if condition_data else None
    )
    return x, t, condition


class TestConstructor:
    """Tests model construction and public attributes."""

    @pytest.mark.parametrize(
        "config_name,model_kwargs,x_shape",
        ARCH_CONFIGS,
        ids=[config[0] for config in ARCH_CONFIGS],
    )
    def test_attributes(self, config_name, model_kwargs, x_shape):
        model = DiffusionUNet3D(**model_kwargs)

        assert model.x_channels == model_kwargs["x_channels"]
        assert model.vol_cond_channels == model_kwargs.get("vol_cond_channels", 0)
        assert model.vec_cond_dim == model_kwargs.get("vec_cond_dim", 0)
        assert model.num_levels == model_kwargs["num_levels"]
        assert model.embedding_type == model_kwargs.get("embedding_type", "positional")
        assert model.checkpoint_level == model_kwargs.get("checkpoint_level", 0)
        assert model.emb_channels == model_kwargs["model_channels"] * model_kwargs.get(
            "channel_mult_emb", 4
        )
        assert all(
            module.amp_mode is True
            for module in model.modules()
            if hasattr(module, "amp_mode")
        )

    def test_default_attributes(self):
        model = DiffusionUNet3D(x_channels=2)

        assert model.x_channels == 2
        assert model.vol_cond_channels == 0
        assert model.vec_cond_dim == 0
        assert model.num_levels == 4
        assert model.embedding_type == "positional"
        assert model.checkpoint_level == 0
        assert model.emb_channels == 128 * 4
        assert all(
            module.amp_mode is True
            for module in model.modules()
            if hasattr(module, "amp_mode")
        )


@pytest.mark.parametrize(
    "config_name,model_kwargs,x_shape",
    ARCH_CONFIGS,
    ids=[config[0] for config in ARCH_CONFIGS],
)
class TestNonRegression:
    """Tests model outputs against saved reference data."""

    def test_forward(
        self,
        deterministic_settings,
        config_name,
        model_kwargs,
        x_shape,
        device,
        tolerances,
    ):
        model = instantiate_model_deterministic(
            DiffusionUNet3D, seed=0, **model_kwargs
        ).to(device)
        x, t, condition = _generate_batch_data(model_kwargs, x_shape, device)
        out = model(x, t, condition=condition)

        ref = load_or_create_reference(
            f"diffusion_unet_3d_{config_name}_forward.pth",
            lambda: {"out": out.cpu()},
        )
        compare_outputs(out, ref["out"], **tolerances)

    def test_forward_from_checkpoint(
        self,
        deterministic_settings,
        config_name,
        model_kwargs,
        x_shape,
        device,
        tolerances,
    ):
        def create_fn():
            return instantiate_model_deterministic(
                DiffusionUNet3D, seed=0, **model_kwargs
            )

        model = load_or_create_checkpoint(
            f"diffusion_unet_3d_{config_name}.mdlus",
            create_fn,
        ).to(device)
        x, t, condition = _generate_batch_data(model_kwargs, x_shape, device)
        out = model(x, t, condition=condition)

        ref = load_or_create_reference(
            f"diffusion_unet_3d_{config_name}_forward.pth",
            lambda: {"out": out.cpu()},
        )
        compare_outputs(out, ref["out"], **tolerances)


@pytest.mark.parametrize(
    "config_name,model_kwargs,x_shape",
    ARCH_CONFIGS,
    ids=[config[0] for config in ARCH_CONFIGS],
)
class TestCompile:
    """Tests model compatibility with torch.compile."""

    @pytest.mark.usefixtures("nop_compile")
    def test_forward(
        self,
        deterministic_settings,
        config_name,
        model_kwargs,
        x_shape,
        device,
    ):
        torch._dynamo.config.error_on_recompile = True
        model = instantiate_model_deterministic(
            DiffusionUNet3D, seed=0, **model_kwargs
        ).to(device)
        model.eval()
        x, t, condition = _generate_batch_data(model_kwargs, x_shape, device)
        compiled = torch.compile(model, fullgraph=True)

        with torch.no_grad():
            eager_out = model(x, t, condition=condition)
            compiled_out = compiled(x, t, condition=condition)
            repeated_out = compiled(x, t, condition=condition)

        torch.testing.assert_close(compiled_out, eager_out)
        torch.testing.assert_close(repeated_out, compiled_out)


@pytest.mark.parametrize(
    "config_name,model_kwargs,x_shape",
    ARCH_CONFIGS,
    ids=[config[0] for config in ARCH_CONFIGS],
)
class TestGradientFlow:
    """Tests gradients with respect to inputs and model parameters."""

    def test_forward(self, config_name, model_kwargs, x_shape, device):
        model = instantiate_model_deterministic(
            DiffusionUNet3D, seed=0, **model_kwargs
        ).to(device)
        x, t, condition = _generate_batch_data(model_kwargs, x_shape, device)
        x.requires_grad_(True)

        model(x, t, condition=condition).sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        parameter_grads = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        assert all(grad is not None for grad in parameter_grads)
        assert all(torch.isfinite(grad).all() for grad in parameter_grads)
