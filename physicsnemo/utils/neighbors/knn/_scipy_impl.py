# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

from physicsnemo.utils.version_check import check_min_version

SCIPY_AVAILABLE = check_min_version("scipy", "1.7.0")

if SCIPY_AVAILABLE:
    from scipy.spatial import KDTree

    @torch.library.custom_op("physicsnemo::knn_scipy", mutates_args=())
    def knn_impl(
        points: torch.Tensor, queries: torch.Tensor, k: int = 3
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Use dlpack to move the data without copying between pytorch and cuml:
        points = points.detach().numpy()
        queries = queries.detach().numpy()

        interp_func = KDTree(points)
        distance, indices = interp_func.query(queries, k=k)

        # Ensure dtype compatibility: cast distances to the dtype of queries:
        distance = distance.astype(queries.dtype)

        indices = torch.from_numpy(indices)
        distance = torch.from_numpy(distance)

        # This reshape is to prevent scipy from eating the second dimension whten k ==1
        return indices.reshape(queries.shape[0], k), distance.reshape(
            queries.shape[0], k
        )

    @knn_impl.register_fake
    def _(
        points: torch.Tensor, queries: torch.Tensor, k: int = 3
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if points.device != queries.device:
            raise RuntimeError("points and queries must be on the same device")

        dist_output = torch.empty(
            queries.shape[0], k, device=queries.device, dtype=queries.dtype
        )
        idx_output = torch.empty(
            queries.shape[0], k, device=queries.device, dtype=torch.int64
        )

        return idx_output, dist_output
else:

    def knn_impl(
        points: torch.Tensor,
        queries: torch.Tensor,
        k: int = 3,
    ) -> None:
        """
        Dummy implementation for when scipy is not available.

        Args:
            points (torch.Tensor): The points to search in.
            queries (torch.Tensor): The queries to search for.
            k (int): The number of neighbors to search for.

        Raises:
            ImportError: If scipy is not installed.
        """

        raise ImportError(
            "scipy is not installed, can not be used as a backend for a knn search"
        )
