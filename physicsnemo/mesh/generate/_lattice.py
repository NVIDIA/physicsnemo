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

"""Kuhn (Freudenthal) simplicial lattices in arbitrary dimension."""

from itertools import permutations

import torch


def kuhn_lattice(lo, hi, h: float, device="cpu", dtype=torch.float64):
    """Simplicial lattice covering the box ``[lo, hi]``, any dimension.

    Each hypercube of edge ~``h`` is split into ``d!`` simplices, one per
    axis permutation (walk low corner -> high corner taking axis steps in
    permutation order). All cells are positively oriented by construction
    (odd permutations get a two-vertex swap).

    Returns ``(points (n_points, d), cells (n_cells, d+1) int64)``.
    """
    lo_t = torch.as_tensor(lo, dtype=dtype, device=device)
    hi_t = torch.as_tensor(hi, dtype=dtype, device=device)
    d = lo_t.shape[0]
    n = torch.ceil((hi_t - lo_t) / h).long().clamp(min=1)

    axes = [
        lo_t[i]
        + (hi_t[i] - lo_t[i])
        * torch.arange(n[i] + 1, device=device, dtype=dtype)
        / n[i]
        for i in range(d)
    ]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    points = grid.reshape(-1, d)

    vshape = (n + 1).tolist()
    strides = torch.ones(d, dtype=torch.int64, device=device)
    for i in range(d - 2, -1, -1):
        strides[i] = strides[i + 1] * vshape[i + 1]

    corner_axes = [torch.arange(n[i], device=device) for i in range(d)]
    corners = torch.stack(torch.meshgrid(*corner_axes, indexing="ij"), dim=-1).reshape(
        -1, d
    )

    eye = torch.eye(d, dtype=torch.int64, device=device)
    blocks = []
    for perm in permutations(range(d)):
        steps = eye[list(perm)]
        offsets = torch.cat(
            [
                torch.zeros(1, d, dtype=torch.int64, device=device),
                torch.cumsum(steps, dim=0),
            ]
        )
        multi = corners[:, None, :] + offsets[None, :, :]
        cells = (multi * strides).sum(-1)
        inversions = sum(
            1 for i in range(d) for j in range(i + 1, d) if perm[i] > perm[j]
        )
        if inversions % 2 == 1:
            cells = cells[:, [1, 0] + list(range(2, d + 1))]
        blocks.append(cells)

    return points, torch.cat(blocks, dim=0)
