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

"""Row gathers for differentiable mesh geometry."""

import torch
from jaxtyping import Int, Shaped


def gather_rows(
    values: Shaped[torch.Tensor, "n_rows ..."],
    indices: Int[torch.Tensor, "..."],
) -> Shaped[torch.Tensor, "..."]:
    """Gather rows of ``values``, the same as ``values[indices]``.

    Uses ``index_select``, whose backward on CUDA is an ``index_add_`` with
    atomics, instead of advanced indexing, whose backward is a sort-based
    ``index_put_``: several times faster and with less memory for the gathers
    of cell vertices from points. The forward values are identical. Because
    the CUDA backward accumulates with atomics, gradients are not bitwise
    reproducible across runs unless ``torch.use_deterministic_algorithms(True)``
    is set, as for the ``scatter_add_`` paths of ``physicsnemo.mesh``; in that
    mode both gathers cost the same.

    Parameters
    ----------
    values : torch.Tensor
        Tensor of shape ``(n_rows, *trailing)``.
    indices : torch.Tensor
        Integer row indices of any shape.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(*indices.shape, *trailing)``.

    Examples
    --------
    >>> points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    >>> cells = torch.tensor([[0, 1, 2], [2, 1, 0]])
    >>> torch.equal(gather_rows(points, cells), points[cells])
    True
    """
    return values.index_select(0, indices.reshape(-1)).view(
        *indices.shape, *values.shape[1:]
    )
