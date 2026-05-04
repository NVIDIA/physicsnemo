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


def validate_inputs(points: torch.Tensor, queries: torch.Tensor):
    """Validate and normalize inputs to (B, N, 3) shape. Returns (points, queries, was_unbatched)."""
    if points.ndim == 2 and queries.ndim == 2:
        return points.unsqueeze(0), queries.unsqueeze(0), True
    elif points.ndim == 3 and queries.ndim == 3:
        if points.shape[0] != queries.shape[0]:
            raise ValueError(
                f"Batch dimensions must match: points has {points.shape[0]}, "
                f"queries has {queries.shape[0]}"
            )
        return points, queries, False
    else:
        raise ValueError(
            f"points and queries must be 2D (N, 3) or 3D (B, N, 3), "
            f"got {points.ndim}D and {queries.ndim}D"
        )


def format_returns(
    indices: torch.Tensor,
    points: torch.Tensor,
    distances: torch.Tensor,
    return_dists: bool,
    return_points: bool,
):
    """Select which tensors to include in the radius search return tuple.

    Always includes ``indices``. Appends ``points`` if ``return_points`` is True,
    and ``distances`` if ``return_dists`` is True.
    """
    if return_points:
        if return_dists:
            return indices, points, distances
        return indices, points

    if return_dists:
        return indices, distances

    return indices
