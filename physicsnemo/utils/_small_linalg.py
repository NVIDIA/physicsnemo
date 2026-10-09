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

"""Closed-form determinants and inverses of batches of small matrices.

Batched ``torch.linalg`` calls factorize every tiny matrix separately, which is
slow on GPUs and a loop of LAPACK calls on CPUs. Meshes have one such matrix
per cell, so these helpers use closed forms for matrices up to 3x3. They use
only elementwise products and sums, so they need no CUDA synchronization and
support autograd and reduced-precision dtypes.
"""

import torch
from jaxtyping import Float


def small_det(
    matrices: Float[torch.Tensor, "... n n"],
) -> Float[torch.Tensor, "..."]:
    """Determinant of each small square matrix, in closed form for n <= 3.

    Batched ``torch.linalg.det`` factorizes every tiny matrix separately: slow
    on GPUs, and a loop of LAPACK calls on CPUs.
    """
    n = matrices.shape[-1]
    m = matrices
    if n == 1:
        return m[..., 0, 0]
    if n == 2:
        return m[..., 0, 0] * m[..., 1, 1] - m[..., 0, 1] * m[..., 1, 0]
    if n == 3:  # scalar triple product of the rows
        return (m[..., 0, :] * torch.linalg.cross(m[..., 1, :], m[..., 2, :])).sum(-1)
    return torch.linalg.det(matrices)


def small_inverse(
    matrices: Float[torch.Tensor, "... n n"],
) -> Float[torch.Tensor, "... n n"]:
    """Inverse of each small invertible matrix, by its adjugate for n <= 3.

    The caller guarantees invertibility, so no error checks (which would
    synchronize CUDA callers) are made.
    """
    n = matrices.shape[-1]
    m = matrices
    if n == 1:
        return 1.0 / m
    if n == 2:
        adjugate = torch.stack(
            [
                torch.stack([m[..., 1, 1], -m[..., 0, 1]], dim=-1),
                torch.stack([-m[..., 1, 0], m[..., 0, 0]], dim=-1),
            ],
            dim=-2,
        )
        det = small_det(matrices)
    elif n == 3:
        # Row i of the cofactor matrix is the cross product of rows i + 1 and
        # i + 2 (mod 3); the adjugate is its transpose. Expanding along row 0
        # gives the determinant from the same cofactors.
        cofactors = torch.linalg.cross(m.roll(-1, dims=-2), m.roll(-2, dims=-2), dim=-1)
        adjugate = cofactors.transpose(-1, -2)
        det = (m[..., 0, :] * cofactors[..., 0, :]).sum(-1)
    else:
        return torch.linalg.inv_ex(matrices, check_errors=False).inverse
    return adjugate / det[..., None, None]
