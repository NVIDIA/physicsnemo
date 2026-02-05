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

from physicsnemo.core.version_check import OptionalImport

from .utils import validate_inputs

scipy_spatial = OptionalImport("scipy.spatial")


@torch.library.custom_op("physicsnemo::knn_scipy", mutates_args=())
def knn_impl(
    points: torch.Tensor, queries: torch.Tensor, k: int = 3
) -> tuple[torch.Tensor, torch.Tensor]:
    if points.device.type != "cpu":
        raise ValueError(
            f"`knn` scipy implementation does not support CUDA, got {points.device=}"
        )
    validate_inputs(points, queries)
    restore_dtype = points.dtype
    if restore_dtype == torch.bfloat16:
        points = points.to(torch.float32)
        queries = queries.to(torch.float32)
    # Use dlpack to move the data without copying between pytorch and cuml:
    points = points.detach().numpy()
    queries = queries.detach().numpy()

    KDTree = scipy_spatial.KDTree
    interp_func = KDTree(points)
    distance, indices = interp_func.query(queries, k=k)

    # Ensure dtype compatibility: cast distances to the dtype of queries:
    distance = distance.astype(queries.dtype)

    indices = torch.from_numpy(indices)
    distance = torch.from_numpy(distance)

    # This reshape is to prevent scipy from eating the second dimension whten k ==1
    indices = indices.reshape(queries.shape[0], k)
    distance = distance.reshape(queries.shape[0], k)
    if restore_dtype == torch.bfloat16:
        distance = distance.to(restore_dtype)
    return indices, distance


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
