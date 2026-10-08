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

"""Tests for U-Net layers."""

import pytest

from physicsnemo.core import Module
from physicsnemo.nn import UNetBlock3D
from test.nn.module._helpers import (
    LayerTestFixtures,
    compare_outputs,
    instantiate_module_deterministic,
    load_or_create_checkpoint,
    load_or_create_reference,
    make_inputs,
)

UNET_BLOCK_3D_CONFIGS = (
    (
        "default",
        {"in_channels": 8, "out_channels": 16, "emb_channels": 32},
        ((2, 8, 4, 8, 8), (2, 32)),
    ),
    (
        "attention",
        {
            "in_channels": 8,
            "out_channels": 16,
            "emb_channels": 32,
            "attention": True,
            "num_heads": 4,
            "down": True,
            "adaptive_scale": False,
            "activation": "gelu",
        },
        ((2, 8, 4, 8, 8), (2, 32)),
    ),
)


@pytest.mark.parametrize(
    "config_name,module_kwargs,input_shapes",
    UNET_BLOCK_3D_CONFIGS,
    ids=[config[0] for config in UNET_BLOCK_3D_CONFIGS],
)
class TestConstructor:
    """Tests layer construction and public attributes."""

    def test_attributes(
        self,
        config_name,
        module_kwargs,
        input_shapes,
    ):
        module = UNetBlock3D(**module_kwargs)

        assert isinstance(module, Module)
        assert module.in_channels == module_kwargs["in_channels"]
        assert module.out_channels == module_kwargs["out_channels"]
        assert module.emb_channels == module_kwargs["emb_channels"]
        assert module.attention == module_kwargs.get("attention", False)
        assert module.dropout == module_kwargs.get("dropout", 0.0)
        assert module.skip_scale == module_kwargs.get("skip_scale", 1.0)
        assert module.adaptive_scale == module_kwargs.get("adaptive_scale", True)
        assert all(
            nested_module.amp_mode is True
            for nested_module in module.modules()
            if hasattr(nested_module, "amp_mode")
        )


@pytest.mark.parametrize(
    "config_name,module_kwargs,input_shapes",
    UNET_BLOCK_3D_CONFIGS,
    ids=[config[0] for config in UNET_BLOCK_3D_CONFIGS],
)
class TestNonRegression(LayerTestFixtures):
    """Tests layer outputs against saved reference data."""

    def test_forward(
        self,
        deterministic_settings,
        config_name,
        module_kwargs,
        input_shapes,
        device,
        tolerances,
    ):
        module = instantiate_module_deterministic(
            UNetBlock3D, seed=0, **module_kwargs
        ).to(device)
        inputs = make_inputs(input_shapes, device)
        out = module(*inputs)

        reference = load_or_create_reference(
            f"unet_block_3d_{config_name}_forward.pth",
            lambda: {"out": out.cpu()},
        )
        compare_outputs(out, reference["out"], **tolerances)

    def test_forward_from_checkpoint(
        self,
        deterministic_settings,
        config_name,
        module_kwargs,
        input_shapes,
        device,
        tolerances,
    ):
        def create_fn():
            return instantiate_module_deterministic(
                UNetBlock3D, seed=0, **module_kwargs
            )

        module = load_or_create_checkpoint(
            f"unet_block_3d_{config_name}.mdlus",
            create_fn,
        ).to(device)
        inputs = make_inputs(input_shapes, device)
        out = module(*inputs)

        reference = load_or_create_reference(
            f"unet_block_3d_{config_name}_forward.pth",
            lambda: {"out": out.cpu()},
        )
        compare_outputs(out, reference["out"], **tolerances)
