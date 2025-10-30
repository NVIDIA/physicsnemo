# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

import os
import sys
from pathlib import Path
from typing import Dict

import pytest
import torch

import physicsnemo
from physicsnemo.models.diffusion.layers import Attention

script_path: str = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(script_path), ".."))

# import common  # noqa: E402


def _err(x: torch.Tensor, y: torch.Tensor) -> str:
    abs_err = torch.amax(torch.abs(x - y))
    rel_err = torch.amax(torch.abs(x - y) / (torch.abs(y) + 1e-4))
    return f"max_abs_err: {abs_err}, max_rel_err: {rel_err}"


def _instantiate_model(cls, seed: int = 0, **kwargs):
    """
    Helper function to instantiate a model with reproducible random parameters.
    """
    model: physicsnemo.Module = cls(**kwargs)
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


class AttentionModule(physicsnemo.Module):
    """
    A wrapper around Attention that has a factory method to
    create a model with reproducible random parameters.
    """

    _overridable_args: set[str] = {"use_apex_gn", "fused_conv_bias"}

    def __init__(
        self,
        arch_type: str = "attention_type_1",
        use_apex_gn: bool = False,
        fused_conv_bias: bool = False,
    ):
        super().__init__()
        C = 32
        # Single head
        if arch_type == "attention_type_1":
            self.attention = Attention(
                out_channels=C,
                num_heads=1,
                use_apex_gn=use_apex_gn,
                fused_conv_bias=fused_conv_bias,
            )
        # Multi-head
        elif arch_type == "attention_type_2":
            self.attention = Attention(
                out_channels=C,
                num_heads=8,
                use_apex_gn=use_apex_gn,
                fused_conv_bias=fused_conv_bias,
            )

    factory: classmethod = classmethod(_instantiate_model)

    def forward(self, x):
        return self.attention(x)


def generate_data(device: str) -> torch.Tensor:
    """
    Helper function to generate data for the test.
    """
    torch.manual_seed(0)
    B, C, H, W = 4, 32, 24, 16
    x: torch.Tensor = torch.randn(B, C, H, W).to(device)
    return x


@pytest.mark.parametrize(
    ("device", "use_apex_gn"),
    [
        ("cuda:0", False),
        ("cuda:0", True),
        ("cpu", False),
    ],
    ids=["gpu", "gpu-apexgn", "cpu"],
)
@pytest.mark.parametrize("fused_conv_bias", [False, True], ids=["non_fused", "fused"])
@pytest.mark.parametrize(
    "arch_type",
    ["attention_type_1", "attention_type_2"],
    ids=["arch1", "arch2"],
)
def test_attention_non_regression(arch_type, device, use_apex_gn, fused_conv_bias):
    """
    Test that Attention can be instantiated and compare the output with a
    reference output.
    """

    model: AttentionModule = AttentionModule.factory(
        arch_type=arch_type,
        use_apex_gn=use_apex_gn,
        fused_conv_bias=fused_conv_bias,
    ).to(device)

    # Check that the model is instantiated correctly
    if arch_type == "attention_type_1":
        assert model.attention.num_heads == 1
    elif arch_type == "attention_type_2":
        assert model.attention.num_heads == 8

    # Load reference data
    file_name: str = str(
        Path(__file__).parents[1].resolve()
        / Path("data")
        / Path(f"output_diffusion_{arch_type}.pth")
    )
    loaded_data: Dict[str, torch.Tensor] = torch.load(file_name)
    x, out_ref = loaded_data["x"].to(device), loaded_data["out"].to(device)
    out: torch.Tensor = model(x)

    # NOTE: this test needs very large tolerances to pass (seems hardware
    # dependent)
    if device == "cpu":
        atol, rtol = 0.005, 1e-3
    elif device == "cuda:0":
        atol, rtol = 5.0, 1e-3
    assert torch.allclose(out, out_ref, atol=atol, rtol=rtol), _err(out, out_ref)


@pytest.mark.parametrize(
    ("device", "use_apex_gn"),
    [
        ("cuda:0", False),
        ("cuda:0", True),
        ("cpu", False),
    ],
    ids=["gpu", "gpu-apexgn", "cpu"],
)
@pytest.mark.parametrize("fused_conv_bias", [False, True], ids=["non_fused", "fused"])
@pytest.mark.parametrize(
    "arch_type",
    ["attention_type_1", "attention_type_2"],
    ids=["arch1", "arch2"],
)
def test_attention_non_regression_from_checkpoint(
    device, use_apex_gn, fused_conv_bias, arch_type
):
    """
    Tests loading and non-regression of a checkpoint generated with the
    Attention class. Also tests the API to override ``use_apex_gn``
    and ``fused_conv_bias`` when loading the checkpoint.
    """

    file_name: str = str(
        Path(__file__).parents[1].resolve()
        / Path("data")
        / Path(f"checkpoint_diffusion_{arch_type}.mdlus")
    )

    model: physicsnemo.Module = physicsnemo.Module.from_checkpoint(
        file_name=file_name,
        override_args={
            "use_apex_gn": use_apex_gn,
            "fused_conv_bias": fused_conv_bias,
        },
    ).to(device)

    # Check that the model is instantiated correctly
    if arch_type == "attention_type_1":
        assert model.attention.num_heads == 1
    elif arch_type == "attention_type_2":
        assert model.attention.num_heads == 8

    # Load reference data
    file_name: str = str(
        Path(__file__).parents[1].resolve()
        / Path("data")
        / Path(f"output_diffusion_{arch_type}.pth")
    )
    loaded_data: Dict[str, torch.Tensor] = torch.load(file_name)
    x, out_ref = loaded_data["x"].to(device), loaded_data["out"].to(device)
    out: torch.Tensor = model(x)

    if device == "cpu":
        atol, rtol = 0.005, 1e-3
    elif device == "cuda:0":
        atol, rtol = 5.0, 1e-3
    assert torch.allclose(out, out_ref, atol=atol, rtol=rtol), _err(out, out_ref)


# ---------------------------------------------------------------------------
#   FOR CHECKPOINT AND DATA GENERATION
# ---------------------------------------------------------------------------


# @pytest.mark.parametrize(
#     "arch_type",
#     ["attention_type_1", "attention_type_2"],
#     ids=["arch1", "arch2"],
# )
# @pytest.mark.parametrize("device", ["cpu"])
# def test_attention_generate_data(device, arch_type):
#     """
#     Test that just generates data for the attention test.
#     """

#     model: AttentionModule = AttentionModule.factory(arch_type=arch_type).to(device)

#     # Check that the model is instantiated correctly
#     if arch_type == "attention_type_1":
#         assert model.attention.num_heads == 1
#     elif arch_type == "attention_type_2":
#         assert model.attention.num_heads == 8

#     model.save(f"checkpoint_diffusion_{arch_type}.mdlus")

#     x = generate_data(device)
#     out: torch.Tensor = model(x)

#     torch.save({"x": x, "out": out}, f"output_diffusion_{arch_type}.pth")
