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

"""Tests for the 3D building blocks (GroupNorm3D, Conv3D, UNetAttention3D, UNetBlock3D)."""

from typing import Any, Dict, Tuple

import pytest
import torch
import torch._dynamo

from physicsnemo.experimental.nn import (
    Conv3D,
    GroupNorm3D,
    UNetAttention3D,
    UNetBlock3D,
)
from test.models.diffusion._helpers import (
    GLOBAL_SEED,
    compare_outputs,
    instantiate_model_deterministic,
    load_or_create_checkpoint,
    load_or_create_reference,
    make_input,
)

# Loose GPU tolerances are needed here because attention via SDPA returns
# meaningfully different values on CPU vs GPU (and across GPU architectures),
# and the test blocks are initialized with purely-random weights and inputs.
# Scoped to this file so the looseness doesn't leak into sibling tests.
_CPU_TOLERANCES = {"atol": 1e-3, "rtol": 1e-3}
_GPU_TOLERANCES = {"atol": 1e-2, "rtol": 5e-2}


@pytest.fixture
def tolerances(device):
    return _CPU_TOLERANCES if device == "cpu" else _GPU_TOLERANCES


# =============================================================================
# GroupNorm3D
# =============================================================================

# (name, kwargs)
GROUPNORM_CONFIGS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("default", dict(num_channels=32)),
    ("custom_groups", dict(num_channels=16, num_groups=8, eps=1e-6)),
    ("min_per_group", dict(num_channels=8, num_groups=32)),
)


@pytest.mark.parametrize(
    "config_name,kwargs",
    GROUPNORM_CONFIGS,
    ids=[c[0] for c in GROUPNORM_CONFIGS],
)
class TestGroupNorm3D:
    """Tests for GroupNorm3D, parameterized over configurations."""

    def test_attributes_match_kwargs(self, config_name, kwargs, device):
        gn = GroupNorm3D(**kwargs).to(device)
        # num_groups is capped to keep at least min_channels_per_group=4 channels per group
        expected_num_groups = min(
            kwargs.get("num_groups", 32),
            kwargs["num_channels"] // 4,
        )
        assert gn.num_groups == expected_num_groups
        assert gn.eps == kwargs.get("eps", 1e-5)
        assert gn.weight.shape == (kwargs["num_channels"],)
        assert gn.bias.shape == (kwargs["num_channels"],)

    def test_forward_non_regression(
        self,
        deterministic_settings,
        config_name,
        kwargs,
        device,
        tolerances,
    ):
        gn = instantiate_model_deterministic(GroupNorm3D, seed=0, **kwargs).to(device)
        x = make_input(
            (2, kwargs["num_channels"], 4, 8, 8), seed=GLOBAL_SEED, device=device
        )
        out = gn(x)
        ref_file = f"groupnorm3d_{config_name}_forward.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)

    def test_forward_from_checkpoint(
        self,
        deterministic_settings,
        config_name,
        kwargs,
        device,
        tolerances,
    ):
        def create_fn():
            return instantiate_model_deterministic(GroupNorm3D, seed=0, **kwargs)

        ckpt_file = f"groupnorm3d_{config_name}.mdlus"
        gn = load_or_create_checkpoint(ckpt_file, create_fn).to(device)
        x = make_input(
            (2, kwargs["num_channels"], 4, 8, 8), seed=GLOBAL_SEED, device=device
        )
        out = gn(x)
        ref_file = f"groupnorm3d_{config_name}_forward.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)

    def test_gradient_flow(self, config_name, kwargs, device):
        gn = GroupNorm3D(**kwargs).to(device)
        x = torch.randn(
            2, kwargs["num_channels"], 4, 8, 8, device=device, requires_grad=True
        )
        gn(x).sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    @pytest.mark.usefixtures("nop_compile")
    def test_compile(self, deterministic_settings, config_name, kwargs, device):
        torch._dynamo.config.error_on_recompile = True
        gn = GroupNorm3D(**kwargs).to(device).eval()
        x = make_input(
            (2, kwargs["num_channels"], 4, 8, 8), seed=GLOBAL_SEED, device=device
        )
        compiled = torch.compile(gn, fullgraph=True)
        with torch.no_grad():
            out_eager = gn(x)
            out_compiled = compiled(x)
        torch.testing.assert_close(out_eager, out_compiled)
        with torch.no_grad():
            out_compiled_2 = compiled(x)
        torch.testing.assert_close(out_compiled, out_compiled_2)


# =============================================================================
# Conv3D
# =============================================================================

# (name, kwargs)
CONV3D_CONFIGS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    (
        "plain",
        dict(in_channels=4, out_channels=8, kernel=3),
    ),
    (
        "down",
        dict(in_channels=4, out_channels=8, kernel=3, down=True),
    ),
    (
        "up_ncsnpp",
        dict(
            in_channels=4,
            out_channels=8,
            kernel=3,
            up=True,
            resample_filter=[1, 3, 3, 1],
        ),
    ),
    (
        "no_bias_xavier",
        dict(
            in_channels=4,
            out_channels=8,
            kernel=3,
            bias=False,
            init_mode="xavier_uniform",
        ),
    ),
)


class TestConv3DErrors:
    """Constructor validation errors (not parametrized over configs)."""

    def test_up_down_both_raises(self):
        with pytest.raises(ValueError, match="up.*down"):
            Conv3D(in_channels=4, out_channels=8, kernel=3, up=True, down=True)

    def test_invalid_resample_filter_raises(self):
        with pytest.raises(ValueError, match="resample_filter"):
            Conv3D(
                in_channels=4,
                out_channels=8,
                kernel=3,
                down=True,
                resample_filter=[],
            )
        with pytest.raises(ValueError, match="resample_filter"):
            Conv3D(
                in_channels=4,
                out_channels=8,
                kernel=3,
                down=True,
                resample_filter=[1, 0],
            )


@pytest.mark.parametrize(
    "config_name,kwargs",
    CONV3D_CONFIGS,
    ids=[c[0] for c in CONV3D_CONFIGS],
)
class TestConv3D:
    """Tests for Conv3D, parameterized over configurations."""

    def test_attributes_match_kwargs(self, config_name, kwargs, device):
        conv = Conv3D(**kwargs).to(device)
        assert conv.in_channels == kwargs["in_channels"]
        assert conv.out_channels == kwargs["out_channels"]
        assert conv.up == kwargs.get("up", False)
        assert conv.down == kwargs.get("down", False)
        if kwargs["kernel"] > 0:
            assert conv.weight is not None
            assert conv.weight.shape == (
                kwargs["out_channels"],
                kwargs["in_channels"],
                kwargs["kernel"],
                kwargs["kernel"],
                kwargs["kernel"],
            )
        if kwargs.get("bias", True) and kwargs["kernel"] > 0:
            assert conv.bias is not None
            assert conv.bias.shape == (kwargs["out_channels"],)
        else:
            assert conv.bias is None

    def test_forward_non_regression(
        self,
        deterministic_settings,
        config_name,
        kwargs,
        device,
        tolerances,
    ):
        conv = instantiate_model_deterministic(Conv3D, seed=0, **kwargs).to(device)
        x = make_input(
            (2, kwargs["in_channels"], 4, 8, 8), seed=GLOBAL_SEED, device=device
        )
        out = conv(x)
        ref_file = f"conv3d_{config_name}_forward.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)

    def test_forward_from_checkpoint(
        self,
        deterministic_settings,
        config_name,
        kwargs,
        device,
        tolerances,
    ):
        def create_fn():
            return instantiate_model_deterministic(Conv3D, seed=0, **kwargs)

        ckpt_file = f"conv3d_{config_name}.mdlus"
        conv = load_or_create_checkpoint(ckpt_file, create_fn).to(device)
        x = make_input(
            (2, kwargs["in_channels"], 4, 8, 8), seed=GLOBAL_SEED, device=device
        )
        out = conv(x)
        ref_file = f"conv3d_{config_name}_forward.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)

    def test_gradient_flow(self, config_name, kwargs, device):
        conv = Conv3D(**kwargs).to(device)
        x = torch.randn(
            2, kwargs["in_channels"], 4, 8, 8, device=device, requires_grad=True
        )
        conv(x).sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    @pytest.mark.usefixtures("nop_compile")
    def test_compile(self, deterministic_settings, config_name, kwargs, device):
        torch._dynamo.config.error_on_recompile = True
        conv = Conv3D(**kwargs).to(device).eval()
        x = make_input(
            (2, kwargs["in_channels"], 4, 8, 8), seed=GLOBAL_SEED, device=device
        )
        compiled = torch.compile(conv, fullgraph=True)
        with torch.no_grad():
            out_eager = conv(x)
            out_compiled = compiled(x)
        torch.testing.assert_close(out_eager, out_compiled)
        with torch.no_grad():
            out_compiled_2 = compiled(x)
        torch.testing.assert_close(out_compiled, out_compiled_2)


# =============================================================================
# UNetAttention3D
# =============================================================================

# (name, kwargs)
ATTENTION_CONFIGS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("single_head", dict(out_channels=16, num_heads=1)),
    ("multi_head", dict(out_channels=16, num_heads=4)),
    ("custom_eps", dict(out_channels=8, num_heads=2, eps=1e-6)),
)


class TestUNetAttention3DErrors:
    """Constructor validation errors (not parametrized over configs)."""

    def test_invalid_num_heads_raises(self):
        with pytest.raises(ValueError, match="num_heads"):
            UNetAttention3D(out_channels=16, num_heads=0)
        with pytest.raises(ValueError, match="num_heads"):
            UNetAttention3D(out_channels=16, num_heads=-1)

    def test_indivisible_channels_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            UNetAttention3D(out_channels=15, num_heads=4)


@pytest.mark.parametrize(
    "config_name,kwargs",
    ATTENTION_CONFIGS,
    ids=[c[0] for c in ATTENTION_CONFIGS],
)
class TestUNetAttention3D:
    """Tests for UNetAttention3D, parameterized over configurations."""

    def test_attributes_match_kwargs(self, config_name, kwargs, device):
        attn = UNetAttention3D(**kwargs).to(device)
        assert attn.num_heads == kwargs["num_heads"]

    def test_forward_non_regression(
        self,
        deterministic_settings,
        config_name,
        kwargs,
        device,
        tolerances,
    ):
        attn = instantiate_model_deterministic(UNetAttention3D, seed=0, **kwargs).to(
            device
        )
        x = make_input(
            (2, kwargs["out_channels"], 4, 8, 8), seed=GLOBAL_SEED, device=device
        )
        out = attn(x)
        ref_file = f"unet_attention_3d_{config_name}_forward.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)

    def test_forward_from_checkpoint(
        self,
        deterministic_settings,
        config_name,
        kwargs,
        device,
        tolerances,
    ):
        def create_fn():
            return instantiate_model_deterministic(UNetAttention3D, seed=0, **kwargs)

        ckpt_file = f"unet_attention_3d_{config_name}.mdlus"
        attn = load_or_create_checkpoint(ckpt_file, create_fn).to(device)
        x = make_input(
            (2, kwargs["out_channels"], 4, 8, 8), seed=GLOBAL_SEED, device=device
        )
        out = attn(x)
        ref_file = f"unet_attention_3d_{config_name}_forward.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)

    def test_gradient_flow(self, config_name, kwargs, device):
        attn = UNetAttention3D(**kwargs).to(device)
        x = torch.randn(
            2, kwargs["out_channels"], 4, 8, 8, device=device, requires_grad=True
        )
        attn(x).sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    @pytest.mark.usefixtures("nop_compile")
    def test_compile(self, deterministic_settings, config_name, kwargs, device):
        torch._dynamo.config.error_on_recompile = True
        attn = UNetAttention3D(**kwargs).to(device).eval()
        x = make_input(
            (2, kwargs["out_channels"], 4, 8, 8), seed=GLOBAL_SEED, device=device
        )
        compiled = torch.compile(attn, fullgraph=True)
        with torch.no_grad():
            out_eager = attn(x)
            out_compiled = compiled(x)
        torch.testing.assert_close(out_eager, out_compiled)
        with torch.no_grad():
            out_compiled_2 = compiled(x)
        torch.testing.assert_close(out_compiled, out_compiled_2)


# =============================================================================
# UNetBlock3D
# =============================================================================

# (name, kwargs, x_shape, emb_shape)
BLOCK_CONFIGS: Tuple[
    Tuple[str, Dict[str, Any], Tuple[int, ...], Tuple[int, ...]], ...
] = (
    (
        "plain",
        dict(in_channels=8, out_channels=16, emb_channels=32),
        (2, 8, 4, 8, 8),
        (2, 32),
    ),
    (
        "attention_multi_head",
        dict(
            in_channels=8,
            out_channels=16,
            emb_channels=32,
            attention=True,
            num_heads=4,
        ),
        (2, 8, 4, 8, 8),
        (2, 32),
    ),
    (
        "down_adaptive",
        dict(
            in_channels=8,
            out_channels=16,
            emb_channels=32,
            down=True,
            adaptive_scale=True,
        ),
        (2, 8, 4, 8, 8),
        (2, 32),
    ),
    (
        "up_gelu",
        dict(
            in_channels=8,
            out_channels=16,
            emb_channels=32,
            up=True,
            activation="gelu",
            resample_filter=[1, 3, 3, 1],
        ),
        (2, 8, 4, 8, 8),
        (2, 32),
    ),
)


@pytest.mark.parametrize(
    "config_name,kwargs,x_shape,emb_shape",
    BLOCK_CONFIGS,
    ids=[c[0] for c in BLOCK_CONFIGS],
)
class TestUNetBlock3D:
    """Tests for UNetBlock3D, parameterized over configurations."""

    def test_attributes_match_kwargs(
        self, config_name, kwargs, x_shape, emb_shape, device
    ):
        block = UNetBlock3D(**kwargs).to(device)
        assert block.in_channels == kwargs["in_channels"]
        assert block.out_channels == kwargs["out_channels"]
        assert block.emb_channels == kwargs["emb_channels"]
        assert block.attention == kwargs.get("attention", False)
        assert block.dropout == kwargs.get("dropout", 0.0)
        assert block.skip_scale == kwargs.get("skip_scale", 1.0)
        assert block.adaptive_scale == kwargs.get("adaptive_scale", True)
        if kwargs.get("attention", False):
            assert hasattr(block, "attn")
        else:
            assert not hasattr(block, "attn")

    def test_forward_non_regression(
        self,
        deterministic_settings,
        config_name,
        kwargs,
        x_shape,
        emb_shape,
        device,
        tolerances,
    ):
        block = instantiate_model_deterministic(UNetBlock3D, seed=0, **kwargs).to(
            device
        )
        x = make_input(x_shape, seed=GLOBAL_SEED, device=device)
        emb = make_input(emb_shape, seed=GLOBAL_SEED + 1, device=device)
        out = block(x, emb)
        ref_file = f"unet_block_3d_{config_name}_forward.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)

    def test_forward_from_checkpoint(
        self,
        deterministic_settings,
        config_name,
        kwargs,
        x_shape,
        emb_shape,
        device,
        tolerances,
    ):
        def create_fn():
            return instantiate_model_deterministic(UNetBlock3D, seed=0, **kwargs)

        ckpt_file = f"unet_block_3d_{config_name}.mdlus"
        block = load_or_create_checkpoint(ckpt_file, create_fn).to(device)
        x = make_input(x_shape, seed=GLOBAL_SEED, device=device)
        emb = make_input(emb_shape, seed=GLOBAL_SEED + 1, device=device)
        out = block(x, emb)
        ref_file = f"unet_block_3d_{config_name}_forward.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)

    def test_gradient_flow(self, config_name, kwargs, x_shape, emb_shape, device):
        block = UNetBlock3D(**kwargs).to(device)
        x = torch.randn(*x_shape, device=device, requires_grad=True)
        emb = torch.randn(*emb_shape, device=device)
        block(x, emb).sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    @pytest.mark.usefixtures("nop_compile")
    def test_compile(
        self,
        deterministic_settings,
        config_name,
        kwargs,
        x_shape,
        emb_shape,
        device,
    ):
        torch._dynamo.config.error_on_recompile = True
        block = (
            instantiate_model_deterministic(UNetBlock3D, seed=0, **kwargs)
            .to(device)
            .eval()
        )
        x = make_input(x_shape, seed=GLOBAL_SEED, device=device)
        emb = make_input(emb_shape, seed=GLOBAL_SEED + 1, device=device)
        compiled = torch.compile(block, fullgraph=True)
        with torch.no_grad():
            out_eager = block(x, emb)
            out_compiled = compiled(x, emb)
        torch.testing.assert_close(out_eager, out_compiled)
        with torch.no_grad():
            out_compiled_2 = compiled(x, emb)
        torch.testing.assert_close(out_compiled, out_compiled_2)
