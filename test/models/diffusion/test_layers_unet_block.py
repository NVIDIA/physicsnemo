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

from pathlib import Path
from typing import Dict, Tuple

import pytest
import torch

import physicsnemo
from physicsnemo.nn import UNetBlock

# import common  # noqa: E402


def _err(x: torch.Tensor, y: torch.Tensor) -> str:
    abs_err = torch.amax(torch.abs(x - y))
    rel_err = torch.amax(torch.abs(x - y) / (torch.abs(y) + 1e-4))
    return f"max_abs_err: {abs_err}, max_rel_err: {rel_err}"


def _instantiate_model(cls, seed: int = 0, **kwargs):
    """
    Helper function to instantiate a model with reproducible random parameters.
    """
    model: physicsnemo.core.Module = cls(**kwargs)
    gen: torch.Generator = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(
                torch.randn(
                    param.shape,
                    generator=gen,
                    dtype=param.dtype,
                )
            )
    return model


class UNetBlockModule(physicsnemo.core.Module):
    """
    A wrapper around UNetBlock with attention that has a factory method to
    create a model with reproducible random parameters.
    """

    _overridable_args: set[str] = {"use_apex_gn", "fused_conv_bias"}

    def __init__(
        self,
        arch_type: str = "unet_block_type_1",
        use_apex_gn: bool = False,
        fused_conv_bias: bool = False,
    ):
        super().__init__()
        C_in, Ne = 16, 8
        C_out = C_in * 2
        # Default parameters (no attention)
        if arch_type == "unet_block_type_1":
            self.unet_block = UNetBlock(
                in_channels=C_in,
                out_channels=C_out,
                emb_channels=Ne,
                use_apex_gn=use_apex_gn,
                fused_conv_bias=fused_conv_bias,
            )
        # Attention with 2 heads
        elif arch_type == "unet_block_type_2":
            self.unet_block = UNetBlock(
                in_channels=C_in,
                out_channels=C_out,
                emb_channels=Ne,
                attention=True,
                num_heads=2,
                channels_per_head=16,
                use_apex_gn=use_apex_gn,
                fused_conv_bias=fused_conv_bias,
            )
        # Attention with single head and skip_scale != 1.0
        elif arch_type == "unet_block_type_3":
            self.unet_block = UNetBlock(
                in_channels=C_in,
                out_channels=C_out,
                emb_channels=Ne,
                attention=True,
                channels_per_head=C_out,
                skip_scale=0.5,
                use_apex_gn=use_apex_gn,
                fused_conv_bias=fused_conv_bias,
            )

    factory: classmethod = classmethod(_instantiate_model)

    def forward(self, x, emb):
        return self.unet_block(x, emb)


def generate_data(device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Helper function to generate data for the test.
    """
    torch.manual_seed(0)
    B, C_in, H, W, Ne = 4, 16, 24, 24, 8
    x: torch.Tensor = torch.randn(B, C_in, H, W).to(device)
    emb: torch.Tensor = torch.randn(B, Ne).to(device)
    return x, emb


@pytest.mark.parametrize("use_apex_gn", [False, True], ids=["no-apexgn", "apexgn"])
@pytest.mark.parametrize("fused_conv_bias", [False, True], ids=["non_fused", "fused"])
@pytest.mark.parametrize(
    "arch_type",
    ["unet_block_type_1", "unet_block_type_2", "unet_block_type_3"],
    ids=["arch1", "arch2", "arch3"],
)
def test_unet_block_non_regression(
    arch_type, apex_device, use_apex_gn, fused_conv_bias
):
    """
    Test that UNetBlock can be instantiated and compare the output with a
    reference output generated with v1.0.1.
    """
    model: UNetBlockModule = UNetBlockModule.factory(
        arch_type=arch_type,
        use_apex_gn=use_apex_gn,
        fused_conv_bias=fused_conv_bias,
    ).to(apex_device)

    # Check that the model is instantiated correctly
    if arch_type == "unet_block_type_1":
        assert model.unet_block.in_channels == 16
        assert model.unet_block.out_channels == 32
        assert model.unet_block.emb_channels == 8
        assert model.unet_block.attention is False
        assert model.unet_block.num_heads == 0
        assert model.unet_block.dropout == 0.0
        assert model.unet_block.skip_scale == 1.0
    elif arch_type == "unet_block_type_2":
        assert model.unet_block.in_channels == 16
        assert model.unet_block.out_channels == 32
        assert model.unet_block.emb_channels == 8
        assert model.unet_block.attention is True
        assert model.unet_block.num_heads == 2
        assert model.unet_block.dropout == 0.0
        assert model.unet_block.skip_scale == 1.0
    elif arch_type == "unet_block_type_3":
        assert model.unet_block.in_channels == 16
        assert model.unet_block.out_channels == 32
        assert model.unet_block.emb_channels == 8
        assert model.unet_block.attention is True
        assert model.unet_block.num_heads == 1
        assert model.unet_block.dropout == 0.0
        assert model.unet_block.skip_scale == 0.5

    # Load reference data
    script_dir = Path(__file__).parent
    file_name: str = str(
        script_dir / Path(f"data/output_diffusion_{arch_type}-v1.0.1.pth")
    )
    loaded_data: Dict[str, torch.Tensor] = torch.load(file_name)
    x, emb = loaded_data["x"].to(apex_device), loaded_data["emb"].to(apex_device)
    out_ref = loaded_data["out"].to(apex_device)
    out: torch.Tensor = model(x, emb)

    # NOTE: this test needs very large tolerances to pass (seems hardware
    # dependent) because of the attention mechanism.
    if arch_type in ["unet_block_type_2", "unet_block_type_3"]:
        if apex_device == "cpu":
            atol, rtol = 0.005, 1e-3
        elif apex_device == "cuda:0":
            atol, rtol = 5.0, 1e-2
    else:
        atol, rtol = 1e-3, 1e-3

    if arch_type == "unet_block_type_1":
        pytest.xfail("Accuracy issue with apex and unet_block_type_1")

    assert torch.allclose(out, out_ref, atol=atol, rtol=rtol), _err(out, out_ref)


@pytest.mark.parametrize("use_apex_gn", [False, True], ids=["no-apexgn", "apexgn"])
@pytest.mark.parametrize("fused_conv_bias", [False, True], ids=["non_fused", "fused"])
@pytest.mark.parametrize(
    "arch_type",
    ["unet_block_type_1", "unet_block_type_2", "unet_block_type_3"],
    ids=["arch1", "arch2", "arch3"],
)
def test_unet_block_non_regression_from_checkpoint(
    apex_device, use_apex_gn, fused_conv_bias, arch_type
):
    """
    Tests loading and non-regression of a checkpoint generated with the
    UNetBlock class with v1.0.1. Also tests the API to override ``use_apex_gn``
    and ``fused_conv_bias`` when loading the checkpoint.
    """
    script_dir = Path(__file__).parent
    file_name: str = str(
        script_dir / Path(f"data/checkpoint_diffusion_{arch_type}-v1.0.1.mdlus")
    )
    model: physicsnemo.core.Module = physicsnemo.core.Module.from_checkpoint(
        file_name=file_name,
        override_args={
            "use_apex_gn": use_apex_gn,
            "fused_conv_bias": fused_conv_bias,
        },
    ).to(apex_device)

    # Check that the model is instantiated correctly
    if arch_type == "unet_block_type_1":
        assert model.unet_block.in_channels == 16
        assert model.unet_block.out_channels == 32
        assert model.unet_block.emb_channels == 8
        assert model.unet_block.attention is False
        assert model.unet_block.num_heads == 0
        assert model.unet_block.dropout == 0.0
        assert model.unet_block.skip_scale == 1.0
    elif arch_type == "unet_block_type_2":
        assert model.unet_block.in_channels == 16
        assert model.unet_block.out_channels == 32
        assert model.unet_block.emb_channels == 8
        assert model.unet_block.attention is True
        assert model.unet_block.num_heads == 2
        assert model.unet_block.dropout == 0.0
        assert model.unet_block.skip_scale == 1.0
    elif arch_type == "unet_block_type_3":
        assert model.unet_block.in_channels == 16
        assert model.unet_block.out_channels == 32
        assert model.unet_block.emb_channels == 8
        assert model.unet_block.attention is True
        assert model.unet_block.num_heads == 1
        assert model.unet_block.dropout == 0.0
        assert model.unet_block.skip_scale == 0.5

    # Load reference data
    file_name: str = str(
        script_dir / Path(f"data/output_diffusion_{arch_type}-v1.0.1.pth")
    )
    loaded_data: Dict[str, torch.Tensor] = torch.load(file_name)
    x, emb = loaded_data["x"].to(apex_device), loaded_data["emb"].to(apex_device)
    out_ref = loaded_data["out"].to(apex_device)
    out: torch.Tensor = model(x, emb)

    # NOTE: this test needs very large tolerances to pass (seems hardware
    # dependent) because of the attention mechanism.
    if arch_type in ["unet_block_type_2", "unet_block_type_3"]:
        if apex_device == "cpu":
            atol, rtol = 0.005, 1e-3
        elif apex_device == "cuda:0":
            atol, rtol = 5.0, 1e-3
    else:
        atol, rtol = 1e-3, 1e-3

    if arch_type == "unet_block_type_1":
        pytest.xfail("Accuracy issue with apex and unet_block_type_1")

    assert torch.allclose(out, out_ref, atol=atol, rtol=rtol), _err(out, out_ref)


# ---------------------------------------------------------------------------
#   FOR CHECKPOINT AND DATA GENERATION WITH v1.0.1
# ---------------------------------------------------------------------------


# # For checkpoint generation with v1.0.1
# class UNetBlockModule(physicsnemo.Module):
#     """
#     A wrapper around UNetBlock with attention that has a factory method to
#     create a model with reproducible random parameters.
#     """

#     def __init__(
#         self,
#         arch_type: str = "unet_block_type_1",
#     ):
#         super().__init__()
#         C_in, Ne = 16, 8
#         C_out = C_in * 2
#         if arch_type == "unet_block_type_1":
#             self.unet_block = UNetBlock(
#                 in_channels=C_in,
#                 out_channels=C_out,
#                 emb_channels=Ne,
#             )
#         elif arch_type == "unet_block_type_2":
#             self.unet_block = UNetBlock(
#                 in_channels=C_in,
#                 out_channels=C_out,
#                 emb_channels=Ne,
#                 attention=True,
#                 num_heads=2,
#                 channels_per_head=16,
#             )
#         elif arch_type == "unet_block_type_3":
#             self.unet_block = UNetBlock(
#                 in_channels=C_in,
#                 out_channels=C_out,
#                 emb_channels=Ne,
#                 attention=True,
#                 channels_per_head=C_out,
#                 skip_scale=0.5,
#             )

#     factory: classmethod = classmethod(_instantiate_model)

#     def forward(self, x, emb):
#         return self.unet_block(x, emb)


# # For data generation with v1.0.1
# @pytest.mark.parametrize(
#     "arch_type",
#     ["unet_block_type_1", "unet_block_type_2", "unet_block_type_3"],
#     ids=["arch1", "arch2", "arch3"],
# )
# @pytest.mark.parametrize("device", ["cpu"])
# def test_unet_block_generate_data(device, arch_type):
#     """
#     Test that UNetBlock can be instantiated and compare the output with a
#     reference output generated with v1.0.1.
#     """

#     model: UNetBlockModule = UNetBlockModule.factory(arch_type=arch_type).to(device)

#     # Check that the model is instantiated correctly
#     if arch_type == "unet_block_type_1":
#         assert model.unet_block.in_channels == 16
#         assert model.unet_block.out_channels == 32
#         assert model.unet_block.emb_channels == 8
#         # assert model.unet_block.attention is False, err_msg
#         assert model.unet_block.num_heads == 0
#         assert model.unet_block.dropout == 0.0
#         assert model.unet_block.skip_scale == 1.0
#     elif arch_type == "unet_block_type_2":
#         assert model.unet_block.in_channels == 16
#         assert model.unet_block.out_channels == 32
#         assert model.unet_block.emb_channels == 8
#         # assert model.unet_block.attention is True, err_msg
#         assert model.unet_block.num_heads == 2
#         assert model.unet_block.dropout == 0.0
#         assert model.unet_block.skip_scale == 1.0
#     elif arch_type == "unet_block_type_3":
#         assert model.unet_block.in_channels == 16
#         assert model.unet_block.out_channels == 32
#         assert model.unet_block.emb_channels == 8
#         # assert model.unet_block.attention is True, err_msg
#         assert model.unet_block.num_heads == 1
#         assert model.unet_block.dropout == 0.0
#         assert model.unet_block.skip_scale == 0.5

#     model.save(f"checkpoint_diffusion_{arch_type}-v1.0.1.mdlus")

#     x, emb = generate_data(device)
#     out: torch.Tensor = model(x, emb)

#     torch.save(
#         {"x": x, "emb": emb, "out": out}, f"output_diffusion_{arch_type}-v1.0.1.pth"
#     )
