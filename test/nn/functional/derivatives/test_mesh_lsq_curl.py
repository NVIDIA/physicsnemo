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

from physicsnemo.nn.functional import mesh_lsq_curl
from physicsnemo.nn.functional.derivatives import MeshLSQCurl
from test.conftest import requires_module
from test.nn.functional.derivatives._mesh_vector_calculus_test_utils import (
    check_backend_backward_parity,
    check_backend_forward_parity,
    make_lsq_case,
)


@pytest.mark.parametrize("n_dims", [2, 3])
def test_mesh_lsq_curl_affine_rotation(device: str, n_dims: int):
    points, offsets, indices = make_lsq_case(device, n_dims=n_dims)
    if n_dims == 2:
        vector_field = torch.stack((-points[:, 1], points[:, 0]), dim=-1)
        expected = torch.full((points.shape[0],), 2.0, device=points.device)
    else:
        vector_field = torch.stack(
            (-points[:, 1], points[:, 0], points[:, 2]),
            dim=-1,
        )
        expected = torch.zeros_like(points)
        expected[:, 2] = 2.0

    output = mesh_lsq_curl(
        points,
        vector_field,
        offsets,
        indices,
        implementation="torch",
    )

    torch.testing.assert_close(output, expected, atol=4e-3, rtol=4e-3)


@requires_module("warp")
def test_mesh_lsq_curl_backend_forward_parity(device: str):
    check_backend_forward_parity(device, MeshLSQCurl)


@requires_module("warp")
def test_mesh_lsq_curl_backend_backward_parity(device: str):
    check_backend_backward_parity(device, MeshLSQCurl, (0, 1))
