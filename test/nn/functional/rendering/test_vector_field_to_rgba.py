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

from physicsnemo.nn.functional import vector_field_to_rgba
from physicsnemo.nn.functional.rendering import VectorFieldToRGBA
from test.conftest import requires_module


def _vector_field(device: torch.device | str, grid_n: int = 6) -> torch.Tensor:
    coords = torch.linspace(-1.0, 1.0, grid_n, device=device)
    x, y, z = torch.meshgrid(coords, coords, coords, indexing="ij")
    return torch.stack([-y, x, 0.25 * torch.ones_like(z)], dim=-1)


@requires_module("warp")
def test_vector_field_to_rgba_warp(device: str):
    vector_field = _vector_field(device)
    lic_field = torch.ones(6, 6, 6, device=device)

    rgba_volume = vector_field_to_rgba(
        vector_field,
        lic_field,
        0.0,
        1.5,
        max_opacity=0.75,
        lic_threshold=0.25,
        implementation="warp",
    )

    assert rgba_volume.shape == (6, 6, 6, 4)
    assert rgba_volume.dtype == torch.uint8
    assert int(rgba_volume[..., 3].max()) > 0
    assert "warp" in VectorFieldToRGBA.available_implementations()
    assert "torch" in VectorFieldToRGBA.available_implementations()


def test_vector_field_to_rgba_torch():
    vector_field = _vector_field("cpu")
    lic_field = torch.ones(6, 6, 6)

    rgba_volume = vector_field_to_rgba(
        vector_field,
        lic_field,
        0.0,
        1.5,
        max_opacity=0.75,
        lic_threshold=0.25,
        implementation="torch",
    )

    assert rgba_volume.shape == (6, 6, 6, 4)
    assert rgba_volume.dtype == torch.uint8
    assert int(rgba_volume[..., 3].max()) > 0


@requires_module("warp")
def test_vector_field_to_rgba_backend_forward_parity(device: str):
    vector_field = _vector_field(device, grid_n=5)
    lic_field = torch.linspace(0.0, 1.0, 5 * 5 * 5, device=device).reshape(5, 5, 5)

    rgba_warp = vector_field_to_rgba(
        vector_field,
        lic_field,
        0.0,
        1.75,
        max_opacity=0.65,
        lic_threshold=0.3,
        implementation="warp",
    )
    rgba_torch = vector_field_to_rgba(
        vector_field,
        lic_field,
        0.0,
        1.75,
        max_opacity=0.65,
        lic_threshold=0.3,
        implementation="torch",
    )

    VectorFieldToRGBA.compare_forward(rgba_warp, rgba_torch)


def test_vector_field_to_rgba_make_inputs_forward():
    label, args, kwargs = next(iter(VectorFieldToRGBA.make_inputs_forward("cpu")))
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    output = VectorFieldToRGBA.dispatch(*args, implementation="torch", **kwargs)
    assert output is not None


def test_vector_field_to_rgba_compare_forward_contract():
    reference = torch.zeros(2, 2, 2, 4, dtype=torch.uint8)
    output = reference.clone()
    VectorFieldToRGBA.compare_forward(output, reference)


def test_vector_field_to_rgba_error_handling():
    vector_field = torch.zeros(4, 4, 4, 2)
    with pytest.raises(ValueError, match="vector_field"):
        vector_field_to_rgba(
            vector_field,
            torch.zeros(4, 4, 4),
            0.0,
            1.0,
            implementation="torch",
        )
