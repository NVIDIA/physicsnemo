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

"""Pure-Torch similarity Procrustes registration."""

import torch
from jaxtyping import Float
from torch.autograd.function import once_differentiable

from ._procrustes_common import solve_procrustes


class _FirstOrderOnly(torch.autograd.Function):
    """Pass a Torch tensor through while rejecting graph-building backward."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        """Return ``tensor`` unchanged."""

        return tensor

    @staticmethod
    def backward(ctx, tensor_gradient: torch.Tensor) -> tuple[torch.Tensor]:
        """Return a first derivative, but never a partial higher derivative."""

        if torch.is_grad_enabled():
            raise RuntimeError(
                "procrustes supports first-order reverse-mode derivatives only"
            )
        return (tensor_gradient,)


class _ProperOrthogonalProcrustes(torch.autograd.Function):
    """Project a square matrix onto ``SO(D)`` with a stable first derivative.

    The usual SVD expression for the Kabsch rotation has the correct forward
    value, but PyTorch's generic SVD backward is undefined when singular values
    repeat.  The closest proper rotation remains differentiable in many of
    those cases (for example, an isotropic full-rank covariance).  Its
    derivative is instead obtained from the Sylvester equation for the polar
    factor, whose denominators are sums of signed singular values.

    Only first-order derivatives are part of this internal operation's
    contract.  A zero off-diagonal denominator still represents a genuinely
    non-unique proper rotation, such as a tied reflection-correction axis.
    """

    @staticmethod
    def forward(
        ctx,
        covariance: Float[torch.Tensor, "batch num_dims num_dims"],
    ) -> Float[torch.Tensor, "batch num_dims num_dims"]:
        """Return the determinant-one Kabsch rotation."""
        left, singular_values, right_t = torch.linalg.svd(
            covariance, full_matrices=False
        )
        right = right_t.transpose(-2, -1)

        unconstrained_rotation = right @ left.transpose(-2, -1)
        determinant = torch.linalg.det(unconstrained_rotation)
        final_sign = torch.where(
            determinant < 0,
            -torch.ones_like(determinant),
            torch.ones_like(determinant),
        )
        correction = torch.cat(
            (torch.ones_like(singular_values[..., :-1]), final_sign.unsqueeze(-1)),
            dim=-1,
        )
        rotation = (right * correction.unsqueeze(-2)) @ left.transpose(-2, -1)

        ctx.save_for_backward(rotation, right, singular_values * correction)
        return rotation

    @staticmethod
    @once_differentiable
    def backward(
        ctx,
        rotation_gradient: Float[torch.Tensor, "batch num_dims num_dims"],
    ) -> tuple[Float[torch.Tensor, "batch num_dims num_dims"]]:
        """Apply the analytic vector-Jacobian product for the polar factor."""
        rotation, right, signed_singular_values = ctx.saved_tensors

        generator = 0.5 * (
            rotation_gradient @ rotation.transpose(-2, -1)
            - rotation @ rotation_gradient.transpose(-2, -1)
        )
        generator_basis = right.transpose(-2, -1) @ generator @ right
        denominator = signed_singular_values.unsqueeze(-1) + (
            signed_singular_values.unsqueeze(-2)
        )

        # The generator is skew-symmetric, so its diagonal is identically zero.
        # Replacing diagonal denominators avoids 0 / 0 for a valid rank-(D-1)
        # covariance while leaving genuinely non-unique off-diagonal cases
        # visible as non-finite gradients.
        covariance_dims = signed_singular_values.shape[-1]
        off_diagonal = ~torch.eye(
            covariance_dims,
            dtype=torch.bool,
            device=signed_singular_values.device,
        )
        safe_denominator = torch.where(
            off_diagonal, denominator, torch.ones_like(denominator)
        )
        solution_basis = torch.where(
            off_diagonal,
            generator_basis / safe_denominator,
            torch.zeros_like(generator_basis),
        )
        solution = right @ solution_basis @ right.transpose(-2, -1)
        covariance_gradient = -2.0 * rotation.transpose(-2, -1) @ solution
        return (covariance_gradient,)


def procrustes_torch(
    source: Float[torch.Tensor, "batch num_points num_dims"],
    target: Float[torch.Tensor, "batch num_points num_dims"],
    scale: bool = True,
) -> tuple[
    Float[torch.Tensor, "batch num_dims num_dims"],
    Float[torch.Tensor, "batch num_dims"],
    Float[torch.Tensor, " batch"],
]:
    """Solve normalized rank-three Procrustes registration with Torch."""

    # The projector has an analytic first-order VJP. Guard the complete Torch
    # computation so autograd cannot silently combine its detached saved state
    # with partial second derivatives through the surrounding tensor algebra.
    source = _FirstOrderOnly.apply(source)
    target = _FirstOrderOnly.apply(target)
    return solve_procrustes(
        source,
        target,
        scale,
        _ProperOrthogonalProcrustes.apply,
    )
