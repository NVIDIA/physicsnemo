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


def validate_inputs(points: torch.Tensor, num_samples: int):
    """Validate and normalize FPS inputs to ``(B, N, D)``.

    Returns ``(points, was_unbatched)`` where ``points`` is the rank-3 view and
    ``was_unbatched`` records whether a batch axis was added (so callers can
    strip it from the output).
    """
    if points.ndim == 2:
        points = points.unsqueeze(0)
        was_unbatched = True
    elif points.ndim == 3:
        was_unbatched = False
    else:
        raise ValueError(
            f"points must be 2D (N, D) or 3D (B, N, D), got {points.ndim}D"
        )
    n = int(points.shape[1])
    point_dim = int(points.shape[2])
    if point_dim < 1:
        raise ValueError(
            f"points must have a coordinate dimension D >= 1, got D={point_dim}"
        )
    if not isinstance(num_samples, int) or isinstance(num_samples, bool):
        raise ValueError(
            f"num_samples must be an int, got {type(num_samples).__name__}"
        )
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    if num_samples > n:
        raise ValueError(
            f"num_samples ({num_samples}) cannot exceed the number of points ({n})"
        )
    return points, was_unbatched
