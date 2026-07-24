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

import torch

from physicsnemo.nn.functional import mesh_lsq_laplacian
from physicsnemo.nn.functional.derivatives import MeshLSQLaplacian
from test.conftest import requires_module
from test.nn.functional.derivatives._mesh_vector_calculus_test_utils import (
    check_backend_backward_parity,
    check_backend_forward_parity,
    make_lsq_case,
)


def test_mesh_lsq_laplacian_constant_is_zero(device: str):
    points, offsets, indices = make_lsq_case(device, n_dims=3)
    values = torch.ones((points.shape[0],), dtype=points.dtype, device=points.device)

    output = mesh_lsq_laplacian(
        points,
        values,
        offsets,
        indices,
        implementation="torch",
    )

    torch.testing.assert_close(output, torch.zeros_like(output), atol=1e-6, rtol=1e-6)


@requires_module("warp")
def test_mesh_lsq_laplacian_backend_forward_parity(device: str):
    check_backend_forward_parity(device, MeshLSQLaplacian)


@requires_module("warp")
def test_mesh_lsq_laplacian_backend_backward_parity(device: str):
    check_backend_backward_parity(device, MeshLSQLaplacian, (0, 1))
