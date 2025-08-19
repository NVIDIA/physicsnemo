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

from typing import Literal

import torch

from ._torch_impl import knn_impl as knn_torch
from ._cuml_impl import knn_impl as knn_cuml


def knn(
    points: torch.Tensor,
    queries: torch.Tensor,
    k: int,
    backend: Literal["cuml", "torch"] = "cuml",
) -> tuple[torch.Tensor]:
    """
    Perform a k-nearest neighbor search on torch tensors.  Can be done with
    torch directly, or leverage RAPIDS cuML algorithm.

    Args:
        points: Tensor of shape (N, 3) containing the points to search from.
        queries: Tensor of shape (M, 3) containing the points to search for.
        k: Number of nearest neighbors to return for each query point.
        backend: Backend to use for the search.

    """

    if backend not in ["cuml", "torch"]:
        raise ValueError(
            f"`knn` backend must be either 'cuml' or 'torch', got {backend=}"
        )

    # Num neighbors is returned, because in the warp version
    # it's essential to get the backwards pass right.

    # We never actually return it from here.
    # (If you update to use it in the future, check it carefully!)

    if backend == "cuml":
        indices, distances = knn_cuml(
            points, queries, k
        )
    elif backend == "torch":
        indices, distances = knn_torch(
            points, queries, k
        )

    return indices, distances
