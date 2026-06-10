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

from .utils import validate_inputs


@torch.no_grad()
def farthest_point_sampling(
    points: torch.Tensor,
    num_samples: int,
    random_start: bool = False,
) -> torch.Tensor:
    """Pure-PyTorch greedy farthest-point sampling.

    Iteratively selects the point whose minimum distance to the already
    selected set is largest. The running selection index is kept on-device
    (no per-iteration host synchronization), and the batch axis is processed
    in parallel.

    Parameters
    ----------
    points : torch.Tensor
        Coordinates of shape ``(N, D)`` or ``(B, N, D)``.
    num_samples : int
        Number of points to select (``1 <= num_samples <= N``).
    random_start : bool, optional
        If ``True``, start from a random point per cloud; otherwise start from
        index 0. Default ``False``.

    Returns
    -------
    torch.Tensor
        Selected indices of shape ``(num_samples,)`` (unbatched) or
        ``(B, num_samples)`` (batched), dtype ``int64``.
    """
    points, was_unbatched = validate_inputs(points, num_samples)
    # Compute in float32 to match the Warp backend and to avoid low-precision
    # distance accumulation for fp16/bf16 inputs.
    points = points.to(torch.float32)
    batch_size, num_points, _ = points.shape
    device = points.device

    if random_start:
        current = torch.randint(
            0, num_points, (batch_size,), device=device, dtype=torch.long
        )
    else:
        current = torch.zeros(batch_size, device=device, dtype=torch.long)

    selected = torch.empty(batch_size, num_samples, device=device, dtype=torch.long)
    min_dist = torch.full(
        (batch_size, num_points), float("inf"), device=device, dtype=points.dtype
    )
    batch = torch.arange(batch_size, device=device)

    for i in range(num_samples):
        selected[:, i] = current
        if i + 1 == num_samples:
            break  # last index recorded; skip the unused final update
        centroid = points[batch, current].unsqueeze(1)  # (B, 1, D)
        dist = torch.sum((points - centroid) ** 2, dim=-1)  # (B, N)
        min_dist = torch.minimum(min_dist, dist)
        current = torch.argmax(min_dist, dim=1)  # (B,), stays on device

    if was_unbatched:
        selected = selected.squeeze(0)
    return selected
