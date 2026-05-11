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

from typing import Any, Dict, Tuple

import pytest
import torch
import torch._dynamo
from tensordict import TensorDict

from physicsnemo.experimental.models.diffusion_unets import DiffusionUNet3D
from test.models.diffusion._helpers import (
    GLOBAL_SEED,
    compare_outputs,
    instantiate_model_deterministic,
    load_or_create_checkpoint,
    load_or_create_reference,
)

# Loose GPU tolerances are needed here because attention via SDPA returns
# meaningfully different values on CPU vs GPU (and across GPU architectures),
# and the test models are initialized with purely-random weights and inputs.
# Scoped to this file so the looseness doesn't leak into sibling tests.
_CPU_TOLERANCES = {"atol": 1e-3, "rtol": 1e-3}
_GPU_TOLERANCES = {"atol": 1e-2, "rtol": 5e-2}


@pytest.fixture
def tolerances(device):
    return _CPU_TOLERANCES if device == "cpu" else _GPU_TOLERANCES


# =============================================================================
# Architecture configurations
# =============================================================================

# (name, arch_kwargs, x_shape) — minimal sizes that exercise every code path
ARCH_CONFIGS: Tuple[Tuple[str, Dict[str, Any], Tuple[int, int, int, int, int]], ...] = (
    (
        "default",
        dict(
            x_channels=2,
            num_levels=2,
            model_channels=8,
            channel_mult=[1, 2],
            num_blocks=1,
            dropout=0.0,
        ),
        (2, 2, 4, 8, 8),
    ),
    (
        "conditional",
        dict(
            x_channels=2,
            vol_cond_channels=2,
            vec_cond_dim=4,
            num_levels=2,
            model_channels=8,
            channel_mult=[1, 2],
            num_blocks=1,
            attention_levels=[1],
            dropout=0.0,
        ),
        (2, 2, 4, 8, 8),
    ),
    (
        "advanced",
        dict(
            x_channels=2,
            vol_cond_channels=1,
            vec_cond_dim=2,
            num_levels=3,
            model_channels=8,
            channel_mult=[1, 2, 2],
            num_blocks=1,
            attention_levels=[2],
            embedding_type="fourier",
            channel_mult_noise=2,
            encoder_type="residual",
            decoder_type="skip",
            resample_filter=[1, 3, 3, 1],
            bottleneck_attention=False,
            activation="gelu",
            dropout=0.0,
        ),
        (2, 2, 4, 8, 8),
    ),
)


def _generate_batch_data(
    arch_kwargs: Dict[str, Any],
    x_shape: Tuple[int, int, int, int, int],
    seed: int,
    device: str,
) -> Dict[str, Any]:
    """Generate a deterministic (x, t, condition) tuple for the architecture."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    B = x_shape[0]
    x = torch.randn(*x_shape, generator=gen).to(device)
    t = (torch.rand(B, generator=gen) * 0.5 + 0.4).to(device)

    cond_entries: Dict[str, torch.Tensor] = {}
    if arch_kwargs.get("vec_cond_dim", 0) > 0:
        cond_entries["vector"] = torch.randn(
            B, arch_kwargs["vec_cond_dim"], generator=gen
        ).to(device)
    if arch_kwargs.get("vol_cond_channels", 0) > 0:
        cond_entries["volume"] = torch.randn(
            B,
            arch_kwargs["vol_cond_channels"],
            *x_shape[2:],
            generator=gen,
        ).to(device)

    condition = (
        TensorDict(cond_entries, batch_size=[B]).to(device) if cond_entries else None
    )
    return {"x": x, "t": t, "condition": condition}


# =============================================================================
# Constructor tests (default + invalid cases, not parametrized)
# =============================================================================


class TestConstructor:
    """Constructor / attribute tests not tied to a specific architecture."""

    def test_default_attributes(self, device):
        """Default-construction values match the documented defaults."""
        model = DiffusionUNet3D(x_channels=4).to(device)
        assert model.x_channels == 4
        assert model.vol_cond_channels == 0
        assert model.vec_cond_dim == 0
        assert model.embedding_type == "positional"
        assert model.num_levels == 4
        assert model.checkpoint_level == 0
        assert model.emb_channels == 128 * 4  # model_channels * channel_mult_emb
        assert isinstance(model, DiffusionUNet3D)

    def test_invalid_channel_mult_raises(self):
        with pytest.raises(ValueError, match="channel_mult"):
            DiffusionUNet3D(x_channels=2, num_levels=4, channel_mult=[1, 2, 3])

    def test_invalid_attention_level_raises(self):
        with pytest.raises(ValueError, match="attention_levels"):
            DiffusionUNet3D(
                x_channels=2,
                num_levels=2,
                channel_mult=[1, 2],
                attention_levels=[5],
            )

    def test_zero_embedding_with_condition_raises(self):
        with pytest.raises(ValueError, match="embedding_type='zero'"):
            DiffusionUNet3D(
                x_channels=2,
                vec_cond_dim=4,
                num_levels=2,
                channel_mult=[1, 2],
                embedding_type="zero",
            )


# =============================================================================
# Architecture tests (class-level parametrize over ARCH_CONFIGS)
# =============================================================================


@pytest.mark.parametrize(
    "arch_name,arch_kwargs,x_shape",
    ARCH_CONFIGS,
    ids=[c[0] for c in ARCH_CONFIGS],
)
class TestArchitecture:
    """Tests parameterized across every architecture configuration."""

    def test_attributes_match_kwargs(self, arch_name, arch_kwargs, x_shape, device):
        """Every public attribute reflects the kwargs (or its documented default)."""
        model = DiffusionUNet3D(**arch_kwargs).to(device)

        assert model.x_channels == arch_kwargs["x_channels"]
        assert model.vol_cond_channels == arch_kwargs.get("vol_cond_channels", 0)
        assert model.vec_cond_dim == arch_kwargs.get("vec_cond_dim", 0)
        assert model.num_levels == arch_kwargs["num_levels"]
        assert model.embedding_type == arch_kwargs.get("embedding_type", "positional")
        assert model.checkpoint_level == arch_kwargs.get("checkpoint_level", 0)

        expected_emb = arch_kwargs.get("model_channels", 128) * arch_kwargs.get(
            "channel_mult_emb", 4
        )
        assert model.emb_channels == expected_emb

    def test_forward_non_regression(
        self,
        deterministic_settings,
        arch_name,
        arch_kwargs,
        x_shape,
        device,
        tolerances,
    ):
        """Forward output matches a saved reference."""
        model = instantiate_model_deterministic(
            DiffusionUNet3D, seed=0, **arch_kwargs
        ).to(device)
        data = _generate_batch_data(arch_kwargs, x_shape, GLOBAL_SEED, device)
        out = model(data["x"], data["t"], condition=data["condition"])

        ref_file = f"diffusion_unet_3d_{arch_name}_forward.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)

    def test_forward_from_checkpoint(
        self,
        deterministic_settings,
        arch_name,
        arch_kwargs,
        x_shape,
        device,
        tolerances,
    ):
        """Forward output from a loaded checkpoint matches the same reference."""

        def create_fn():
            return instantiate_model_deterministic(
                DiffusionUNet3D, seed=0, **arch_kwargs
            )

        ckpt_file = f"diffusion_unet_3d_{arch_name}.mdlus"
        model = load_or_create_checkpoint(ckpt_file, create_fn).to(device)
        data = _generate_batch_data(arch_kwargs, x_shape, GLOBAL_SEED, device)
        out = model(data["x"], data["t"], condition=data["condition"])

        ref_file = f"diffusion_unet_3d_{arch_name}_forward.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)

    def test_forward_output(self, arch_name, arch_kwargs, x_shape, device):
        """Forward returns a tensor of the expected shape and dtype."""
        model = instantiate_model_deterministic(
            DiffusionUNet3D, seed=0, **arch_kwargs
        ).to(device)
        data = _generate_batch_data(arch_kwargs, x_shape, GLOBAL_SEED, device)
        out = model(data["x"], data["t"], condition=data["condition"])
        assert out.shape == x_shape
        assert out.dtype == data["x"].dtype

    def test_gradient_flow(self, arch_name, arch_kwargs, x_shape, device):
        """Gradients flow back through the model."""
        model = instantiate_model_deterministic(
            DiffusionUNet3D, seed=0, **arch_kwargs
        ).to(device)
        data = _generate_batch_data(arch_kwargs, x_shape, GLOBAL_SEED, device)
        x = data["x"].clone().requires_grad_(True)
        out = model(x, data["t"], condition=data["condition"])
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    @pytest.mark.usefixtures("nop_compile")
    def test_compile(
        self,
        deterministic_settings,
        arch_name,
        arch_kwargs,
        x_shape,
        device,
    ):
        """Compiled forward matches eager and graph is reused on second call."""
        torch._dynamo.config.error_on_recompile = True

        # eval mode disables dropout so eager and compiled paths are deterministic
        model = (
            instantiate_model_deterministic(DiffusionUNet3D, seed=0, **arch_kwargs)
            .to(device)
            .eval()
        )
        data = _generate_batch_data(arch_kwargs, x_shape, GLOBAL_SEED, device)
        x, t, cond = data["x"], data["t"], data["condition"]

        compiled = torch.compile(model, fullgraph=True)

        with torch.no_grad():
            out_eager = model(x, t, condition=cond)
            out_compiled = compiled(x, t, condition=cond)
        torch.testing.assert_close(out_eager, out_compiled)

        with torch.no_grad():
            out_compiled_2 = compiled(x, t, condition=cond)
        torch.testing.assert_close(out_compiled, out_compiled_2)
