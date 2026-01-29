# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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


def knn_impl(
    points: torch.Tensor,
    queries: torch.Tensor,
    k: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform kNN search with torch.

    Args:
        points (torch.Tensor): Query points, shape (M, D)
        queries (torch.Tensor): Reference points, shape (N, D)
        k (int): Number of neighbors

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - indices (torch.Tensor): Indices of the top-k nearest neighbors, shape (N, k)
            - distances (torch.Tensor): Distances to the top-k nearest neighbors, shape (N, k)
    """
    validate_inputs(points, queries)
    # M, D = p1.shape
    # N, D_feat = p2_features.shape

    # Compute pairwise distances: (M, N)
    dists = torch.norm(points[:, None, :] - queries[None, :, :], dim=-1)

    # Find top-k nearest neighbors
    topk_dists, topk_idx = torch.topk(dists, k=k, dim=0, largest=False, sorted=True)

    return topk_idx.T, topk_dists.T
