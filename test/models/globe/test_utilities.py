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

from physicsnemo.nn.functional.equivariant_ops import legendre_polynomials


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
def test_legendre_polynomials_values(device):
    """Test known values of Legendre polynomials."""
    x = torch.tensor([-1.0, 0.0, 0.5, 1.0], device=device)
    polys = legendre_polynomials(x, 5)

    assert len(polys) == 5
    assert torch.allclose(polys[0], torch.ones_like(x))
    assert torch.allclose(polys[1], x)

    # Direct verification of a few specific values
    assert torch.allclose(
        polys[2][1], torch.tensor(-0.5, device=device)
    )  # P_2(0) = -0.5
    assert torch.allclose(
        polys[3][2], torch.tensor(-0.4375, device=device)
    )  # P_3(0.5) = -0.4375
    assert torch.allclose(polys[4][3], torch.tensor(1.0, device=device))  # P_4(1) = 1


def test_legendre_polynomials_empty():
    """Test edge case with n=0."""
    x = torch.tensor([0.5])
    assert legendre_polynomials(x, 0) == []


def test_legendre_polynomials_orthogonality():
    """Test orthogonality: ∫P_m(x)P_n(x)dx = 0 for m ≠ n."""
    x = torch.linspace(-1, 1, 1000)
    dx = 2.0 / len(x)
    polys = legendre_polynomials(x, 5)

    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            integral = torch.sum(polys[i] * polys[j]) * dx
            assert abs(integral) < 5e-3


def test_legendre_polynomials_shapes():
    """Test that output shapes match input shapes."""
    for shape in [(10,), (5, 3), (2, 4, 6)]:
        x = torch.randn(shape)
        polys = legendre_polynomials(x, 4)
        assert all(p.shape == shape for p in polys)


def test_legendre_polynomials_gradient():
    """Test gradient flow through Legendre polynomials."""
    x = torch.tensor([0.5, -0.3, 0.8], requires_grad=True)
    polys = legendre_polynomials(x, 4)

    # P_0 is constant, so skip it
    for i in range(1, len(polys)):
        if x.grad is not None:
            x.grad.zero_()

        polys[i].sum().backward(retain_graph=True)
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
