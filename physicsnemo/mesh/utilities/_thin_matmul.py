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

"""Products of many short rows with one small matrix, without a GEMM."""

import torch
from jaxtyping import Float


def thin_matmul(
    rows: Float[torch.Tensor, "... n_in"],
    matrix: Float[torch.Tensor, "n_in n_out"],
) -> Float[torch.Tensor, "... n_out"]:
    """``rows @ matrix``, computed as ``n_in`` scaled copies of the matrix rows.

    Point coordinates and per-point vectors meet a matrix of spatial size in
    products of millions of rows by two or three columns. A GEMM is a poor
    fit for that shape on CUDA:

    - With TF32 allowed (``torch.backends.cuda.matmul.allow_tf32``, which NGC
      PyTorch containers set through ``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1``), a
      float32 GEMM rounds its inputs to about three significant digits. That
      suits network layers and not coordinates: on GB300 the rotated points
      are off by up to 5e-4 of the largest coordinate, against 9e-8 as columns.
    - A float64 GEMM this thin runs far below memory bandwidth on GPUs with few
      float64 units: on GB300 at 10M float32 points, the product in float64 with
      both casts takes 10.3 ms, against 0.25 ms as columns.

    The columns are several kernels where a float32 GEMM is one. On the same
    GPU they take 0.06 to 0.07 ms against 0.02 to 0.03 ms up to 100,000 points,
    and 0.25 ms against 0.77 ms at 10M points. They accumulate in place, so like
    the GEMM they hold no temporaries beyond the result.

    Parameters
    ----------
    rows : torch.Tensor
        The rows, shape ``(..., n_in)``.
    matrix : torch.Tensor
        The matrix, shape ``(n_in, n_out)``.

    Returns
    -------
    torch.Tensor
        ``rows @ matrix``, shape ``(..., n_out)``.

    Raises
    ------
    ValueError
        If the length of the rows differs from the number of matrix rows.
    """
    if rows.shape[-1] != matrix.shape[0]:
        raise ValueError(
            f"rows of length {rows.shape[-1]} cannot multiply a matrix of shape "
            f"{tuple(matrix.shape)}"
        )
    dtype = torch.promote_types(rows.dtype, matrix.dtype)
    if rows.shape[-1] == 0:
        return rows.new_zeros((*rows.shape[:-1], matrix.shape[1]), dtype=dtype)
    ### float16 and bfloat16 sum in float32: a partial sum can overflow float16
    ### where the product does not
    work = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    rows, matrix = rows.to(work), matrix.to(work)
    out = rows[..., :1] * matrix[0]
    for k in range(1, rows.shape[-1]):
        out.addcmul_(rows[..., k : k + 1], matrix[k])
    return out.to(dtype)
