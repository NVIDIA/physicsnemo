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

"""Check field decoding and the terms of the training objective."""

import math

import pytest
import torch
from objectives import FieldDecoder, pressure_jump_loss


@pytest.fixture
def decoder():
    """Return a decoder with two pressure modes and two temperature modes."""
    basis = {
        "pressure_modes": [[0.1, 0.1, 0.1, 0.1], [0.02, -0.02, 0.03, -0.03]],
        "temperature_modes": [[0.02, 0.02, 0.02, 0.02], [0.01, -0.01, 0.01, -0.01]],
        "pressure_log_mean": [math.log(1e5)] * 4,
        "temperature_log_mean": [math.log(300.0)] * 4,
        "pressure_eigenvalues": [9.0, 1.0],
        "temperature_eigenvalues": [9.0, 1.0],
        "edges": [[0, 1], [1, 2], [2, 3]],
        "pressure_jump_scale": 2.0,
    }
    stats = {
        "gas_constant": 287.0,
        "fields": {"mean": [1.2, 1e5, 300.0], "std": [0.2, 2e4, 20.0]},
    }
    return FieldDecoder(basis, stats)


def test_decoded_fields_are_positive_and_satisfy_the_gas_law(decoder):
    """Density equals pressure over gas constant times temperature."""
    coefficients = torch.tensor([[0.3, -0.5, 0.7, 0.2], [-4.0, 3.0, -2.0, 1.0]])
    physical = decoder.physical(coefficients)
    assert physical.shape == (2, 4, 3)
    assert (physical > 0).all()
    torch.testing.assert_close(
        physical[..., 1], 287.0 * physical[..., 0] * physical[..., 2]
    )
    torch.testing.assert_close(
        decoder(coefficients) * decoder.field_std + decoder.field_mean, physical
    )
    assert not list(decoder.parameters())


def test_jump_loss_ignores_offsets_and_localizes_errors():
    """A constant offset has no jump error and a step error touches two edges."""
    target = torch.tensor([[0.0, 2.0, 2.0, 2.0]])
    edges = torch.tensor([[0, 1], [1, 2], [2, 3]])
    shifted = pressure_jump_loss(target + 5, target, edges, scale=1.0)
    torch.testing.assert_close(shifted, torch.zeros(1))
    displaced = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    torch.testing.assert_close(
        pressure_jump_loss(displaced, target, edges, scale=2.0), torch.tensor([4 / 3])
    )
