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

import physicsnemo.metrics.general.ensemble_metrics as em


@pytest.mark.parametrize("dim", [1, 2, -1])
@pytest.mark.parametrize("input_shape", [(4, 6, 5), (6, 6, 5)])
def test_variance_dim(device, input_shape, dim, rtol: float = 1e-4, atol: float = 1e-4):
    # The ensemble dimension is not the leading one here. Variance has to take
    # the sample count from that dimension and subtract a mean that broadcasts
    # along it. The square shape is the case that used to run without an error
    # and return a wrong value, the other shape used to raise a broadcast error.
    x = torch.randn(input_shape, device=device)
    remaining_shape = list(input_shape)
    del remaining_shape[dim]
    expected_n = input_shape[dim]

    V = em.Variance(remaining_shape, device=device)
    var = V(x, dim=dim)
    assert var.shape == tuple(remaining_shape)
    assert V.n.item() == expected_n
    assert torch.allclose(var, torch.var(x, dim=dim), rtol=rtol, atol=atol)
    assert torch.allclose(V.mean, torch.mean(x, dim=dim), rtol=rtol, atol=atol)

    # The standalone update rule has to accept the same batch dimension.
    _sum, _sum2, _n = em._update_var(V.sum, V.sum2, V.n, x, batch_dim=dim)
    doubled = torch.cat((x, x), dim=dim)
    assert _n.item() == 2 * expected_n
    assert torch.allclose(_sum / _n, doubled.mean(dim=dim), rtol=rtol, atol=atol)
    assert torch.allclose(
        _sum2 / (_n - 1.0), doubled.var(dim=dim), rtol=rtol, atol=atol
    )
