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

import pytest
import torch

from physicsnemo.nn.functional import line_integral_convolution
from physicsnemo.nn.functional.rendering import LineIntegralConvolution
from test.conftest import requires_module


@requires_module("warp")
def test_line_integral_convolution_warp(device: str):
    coords = torch.linspace(-1.0, 1.0, 8, device=device)
    x, y, z = torch.meshgrid(coords, coords, coords, indexing="ij")
    vector_field = torch.stack([-y, x, 0.2 * torch.ones_like(z)], dim=-1)
    seed = torch.linspace(0.0, 1.0, 8, device=device).reshape(8, 1, 1)
    seed = seed.expand(8, 8, 8).contiguous()

    lic = line_integral_convolution(
        vector_field,
        seed,
        step_size=0.4,
        num_steps=4,
        implementation="warp",
    )

    assert lic.shape == (8, 8, 8)
    assert float(lic.min()) >= 0.0
    assert float(lic.max()) <= 1.0
    assert "warp" in LineIntegralConvolution.available_implementations()


@requires_module("warp")
def test_line_integral_convolution_make_inputs_forward(device: str):
    label, args, kwargs = next(
        iter(LineIntegralConvolution.make_inputs_forward(device))
    )
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    output = LineIntegralConvolution.dispatch(*args, implementation="warp", **kwargs)
    assert output is not None


@requires_module("warp")
def test_line_integral_convolution_error_handling(device: str):
    vector_field = torch.zeros(4, 4, 4, 3, device=device)
    seed = torch.zeros(4, 4, 4, device=device)

    with pytest.raises(ValueError, match="num_steps"):
        line_integral_convolution(
            vector_field, seed, num_steps=0, implementation="warp"
        )
