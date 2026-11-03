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

"""Tests for group normalization layers."""

import pytest

from physicsnemo.core import Module
from physicsnemo.nn import GroupNorm3D
from test.nn.module._helpers import (
    LayerTestFixtures,
    compare_outputs,
    instantiate_module_deterministic,
    load_or_create_checkpoint,
    load_or_create_reference,
    make_inputs,
)

GROUP_NORM_3D_CONFIGS = (
    (
        "default",
        {"num_channels": 32},
        ((2, 32, 4, 8, 8),),
    ),
    (
        "custom",
        {"num_channels": 16, "num_groups": 8, "eps": 1e-6},
        ((2, 16, 4, 8, 8),),
    ),
)


@pytest.mark.parametrize(
    "config_name,module_kwargs,input_shapes",
    GROUP_NORM_3D_CONFIGS,
    ids=[config[0] for config in GROUP_NORM_3D_CONFIGS],
)
class TestConstructor:
    """Tests layer construction and public attributes."""

    def test_attributes(
        self,
        config_name,
        module_kwargs,
        input_shapes,
    ):
        module = GroupNorm3D(**module_kwargs)

        assert isinstance(module, Module)
        assert module.num_groups == min(
            module_kwargs.get("num_groups", 32),
            module_kwargs["num_channels"]
            // module_kwargs.get("min_channels_per_group", 4),
        )
        assert module.eps == module_kwargs.get("eps", 1e-5)


@pytest.mark.parametrize(
    "config_name,module_kwargs,input_shapes",
    GROUP_NORM_3D_CONFIGS,
    ids=[config[0] for config in GROUP_NORM_3D_CONFIGS],
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
            GroupNorm3D, seed=0, **module_kwargs
        ).to(device)
        inputs = make_inputs(input_shapes, device)
        out = module(*inputs)

        reference = load_or_create_reference(
            f"group_norm_3d_{config_name}_forward.pth",
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
                GroupNorm3D, seed=0, **module_kwargs
            )

        module = load_or_create_checkpoint(
            f"group_norm_3d_{config_name}.mdlus",
            create_fn,
        ).to(device)
        inputs = make_inputs(input_shapes, device)
        out = module(*inputs)

        reference = load_or_create_reference(
            f"group_norm_3d_{config_name}_forward.pth",
            lambda: {"out": out.cpu()},
        )
        compare_outputs(out, reference["out"], **tolerances)
