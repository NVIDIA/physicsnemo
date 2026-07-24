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

from physicsnemo.nn.functional import mesh_lsq_divergence
from physicsnemo.nn.functional.derivatives import MeshLSQDivergence
from test.conftest import requires_module
from test.nn.functional.derivatives._mesh_vector_calculus_test_utils import (
    check_backend_backward_parity,
    check_backend_forward_parity,
    make_lsq_case,
)


@pytest.mark.parametrize("n_dims", [2, 3])
def test_mesh_lsq_divergence_affine_field(device: str, n_dims: int):
    points, offsets, indices = make_lsq_case(device, n_dims=n_dims)
    coeffs = torch.arange(1, n_dims + 1, device=points.device, dtype=points.dtype)
    vector_field = points * coeffs.view(1, -1)

    output = mesh_lsq_divergence(
        points,
        vector_field,
        offsets,
        indices,
        implementation="torch",
    )

    expected = torch.full_like(output, coeffs.sum())
    torch.testing.assert_close(output, expected, atol=4e-3, rtol=4e-3)


@requires_module("warp")
def test_mesh_lsq_divergence_backend_forward_parity(device: str):
    check_backend_forward_parity(device, MeshLSQDivergence)


@requires_module("warp")
def test_mesh_lsq_divergence_backend_backward_parity(device: str):
    check_backend_backward_parity(device, MeshLSQDivergence, (0, 1))
