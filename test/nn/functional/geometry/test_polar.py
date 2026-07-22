# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the internal differentiable proper-polar helper."""

import pytest
import torch

from physicsnemo.nn.functional.geometry.deform._polar import proper_rotation


def _proper_matrix(
    num_dims: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    seed: int = 5101,
) -> torch.Tensor:
    """Construct a deterministic proper orthogonal matrix."""

    generator = torch.Generator(device=device).manual_seed(seed)
    matrix = torch.randn(
        (num_dims, num_dims), generator=generator, dtype=dtype, device=device
    )
    orthogonal, _ = torch.linalg.qr(matrix)
    final_sign = torch.where(
        torch.linalg.det(orthogonal) < 0,
        -torch.ones((), dtype=dtype, device=device),
        torch.ones((), dtype=dtype, device=device),
    )
    correction = torch.cat(
        (
            torch.ones(num_dims - 1, dtype=dtype, device=device),
            final_sign.unsqueeze(0),
        )
    )
    return orthogonal * correction.unsqueeze(0)


def _raw_svd_rotation(covariance: torch.Tensor) -> torch.Tensor:
    """Reference forward using ordinary SVD autograd."""

    u, _, vh = torch.linalg.svd(covariance, full_matrices=False)
    unconstrained = u @ vh
    final_sign = torch.where(
        torch.linalg.det(unconstrained) < 0,
        -torch.ones_like(unconstrained[..., 0, 0]),
        torch.ones_like(unconstrained[..., 0, 0]),
    )
    correction = torch.cat(
        (torch.ones_like(u[..., 0, :-1]), final_sign.unsqueeze(-1)), dim=-1
    )
    return (u * correction.unsqueeze(-2)) @ vh


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("num_dims", [2, 3, 4])
def test_returns_batched_proper_orthogonal_matrices(device, dtype, num_dims):
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(5201 + num_dims)
    covariance = torch.randn(
        (2, 3, num_dims, num_dims),
        generator=generator,
        dtype=dtype,
        device=device,
    )

    rotation = proper_rotation(covariance)

    identity = torch.eye(num_dims, dtype=dtype, device=device).expand(2, 3, -1, -1)
    tolerance = 2.0e-5 if dtype == torch.float32 else 1.0e-12
    assert rotation.shape == covariance.shape
    assert rotation.dtype == dtype
    assert rotation.device == device
    torch.testing.assert_close(
        rotation.mT @ rotation, identity, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(
        torch.linalg.det(rotation),
        torch.ones((2, 3), dtype=dtype, device=device),
        atol=tolerance,
        rtol=tolerance,
    )


def test_forward_matches_proper_svd_for_reflection_covariances():
    covariance = torch.tensor(
        [
            [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, -1.0]],
            [[1.0, 0.2, -0.3], [0.4, -2.0, 0.1], [0.2, 0.5, 1.5]],
        ],
        dtype=torch.float64,
    )

    actual = proper_rotation(covariance)
    expected = _raw_svd_rotation(covariance)

    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        torch.linalg.det(actual), torch.ones(2, dtype=torch.float64)
    )


def test_vjp_matches_raw_svd_autograd_for_full_rank_covariance():
    left = _proper_matrix(3, seed=5301)
    right = _proper_matrix(3, seed=5302)
    singular_values = torch.diag(torch.tensor([3.0, 1.7, 0.6], dtype=torch.float64))
    covariance = (left @ singular_values @ right.mT).repeat(2, 1, 1)
    covariance[1] = covariance[1] + torch.tensor(
        [[0.1, -0.05, 0.02], [0.03, 0.04, -0.01], [-0.02, 0.06, 0.08]],
        dtype=torch.float64,
    )
    cotangent = torch.randn(
        covariance.shape,
        generator=torch.Generator().manual_seed(5303),
        dtype=torch.float64,
    )
    custom_input = covariance.clone().requires_grad_()
    reference_input = covariance.clone().requires_grad_()

    custom_gradient = torch.autograd.grad(
        (proper_rotation(custom_input) * cotangent).sum(), custom_input
    )[0]
    reference_gradient = torch.autograd.grad(
        (_raw_svd_rotation(reference_input) * cotangent).sum(), reference_input
    )[0]

    torch.testing.assert_close(
        custom_gradient, reference_gradient, atol=2e-11, rtol=2e-10
    )


def test_rank_two_planar_vjp_matches_finite_differences():
    left = _proper_matrix(3, seed=5401)
    right = _proper_matrix(3, seed=5402)
    covariance = (
        left @ torch.diag(torch.tensor([3.0, 1.0, 0.0], dtype=torch.float64)) @ right.mT
    )
    cotangent = torch.randn(
        covariance.shape,
        generator=torch.Generator().manual_seed(5403),
        dtype=torch.float64,
    )
    differentiable_covariance = covariance.clone().requires_grad_()
    analytic = torch.autograd.grad(
        (proper_rotation(differentiable_covariance) * cotangent).sum(),
        differentiable_covariance,
    )[0]

    step = 1.0e-6
    numeric = torch.empty_like(covariance)
    for row in range(3):
        for column in range(3):
            perturbation = torch.zeros_like(covariance)
            perturbation[row, column] = step
            positive = (proper_rotation(covariance + perturbation) * cotangent).sum()
            negative = (proper_rotation(covariance - perturbation) * cotangent).sum()
            numeric[row, column] = (positive - negative) / (2 * step)

    assert torch.isfinite(analytic).all()
    torch.testing.assert_close(analytic, numeric, atol=2e-8, rtol=2e-7)


def test_rank_one_nonunique_case_returns_finite_minimum_norm_subgradient():
    covariance = torch.diag(
        torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64)
    ).requires_grad_()
    cotangent = torch.tensor(
        [[0.2, -0.4, 0.7], [0.3, 0.1, -0.2], [-0.5, 0.8, 0.6]],
        dtype=torch.float64,
    )

    gradient = torch.autograd.grad(
        (proper_rotation(covariance) * cotangent).sum(), covariance
    )[0]

    assert torch.isfinite(gradient).all()


def test_proper_rotation_is_left_and_right_equivariant():
    generator = torch.Generator().manual_seed(5501)
    covariance = torch.randn((4, 3, 3), generator=generator, dtype=torch.float64)
    left = _proper_matrix(3, seed=5502)
    right = _proper_matrix(3, seed=5503)

    transformed = proper_rotation(left @ covariance @ right.mT)
    expected = left @ proper_rotation(covariance) @ right.mT

    torch.testing.assert_close(transformed, expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize(
    ("covariance", "error", "match"),
    [
        (torch.ones(3), ValueError, "must have shape"),
        (torch.ones(2, 3), ValueError, "square matrices"),
        (torch.ones(1, 1), ValueError, "dimension must be at least 2"),
        (torch.ones(2, 2, dtype=torch.float16), TypeError, "must have dtype"),
        (torch.ones(2, 2, dtype=torch.int64), TypeError, "must have dtype"),
    ],
)
def test_input_validation(covariance, error, match):
    with pytest.raises(error, match=match):
        proper_rotation(covariance)


def test_torch_compile_fullgraph_forward_and_backward():
    covariance = torch.tensor(
        [
            [[2.0, 0.2, -0.1], [0.1, 1.3, 0.4], [-0.3, 0.2, 0.7]],
            [[1.2, -0.1, 0.3], [0.2, 1.7, -0.2], [0.1, 0.4, 0.9]],
        ],
        requires_grad=True,
    )
    cotangent = torch.tensor(
        [
            [[0.1, -0.2, 0.3], [0.5, 0.4, -0.1], [-0.3, 0.2, 0.6]],
            [[-0.2, 0.1, 0.4], [0.3, -0.5, 0.2], [0.6, 0.1, -0.4]],
        ]
    )
    compiled = torch.compile(proper_rotation, fullgraph=True, backend="eager")

    rotation = compiled(covariance)
    gradient = torch.autograd.grad((rotation * cotangent).sum(), covariance)[0]
    eager_input = covariance.detach().clone().requires_grad_()
    eager_rotation = proper_rotation(eager_input)
    eager_gradient = torch.autograd.grad(
        (eager_rotation * cotangent).sum(), eager_input
    )[0]

    torch.testing.assert_close(rotation, eager_rotation)
    torch.testing.assert_close(gradient, eager_gradient)
