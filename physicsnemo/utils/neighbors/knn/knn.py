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

from ._cuml_impl import knn_impl as knn_cuml
from ._scipy_impl import knn_impl as knn_scipy
from ._torch_impl import knn_impl as knn_torch


def knn(
    points: torch.Tensor,
    queries: torch.Tensor,
    k: int,
    backend: Literal["cuml", "torch", "scipy", "auto"] = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a k-nearest neighbor search on torch tensors.  Can be done with
    torch directly, or leverage RAPIDS cuML algorithm.

    The "auto" backend will dispatch to the optimal version for the data
    device of the input tensors.

    Args:
        points: Tensor of shape (N, 3) containing the points to search from.
        queries: Tensor of shape (M, 3) containing the points to search for.
        k: Number of nearest neighbors to return for each query point.
        backend: Backend to use for the search.

    Returns:
        indices: Tensor of shape (M, k) containing the indices of the k nearest neighbors for each query point.
        distances: Tensor of shape (M, k) containing the distances to the k nearest neighbors for each query point.

    """

    if backend not in ["cuml", "torch", "scipy", "auto"]:
        raise ValueError(
            f"`knn` backend must be in ['cuml', 'torch', 'scipy', 'auto'], got {backend=}"
        )

    if points.device != queries.device:
        raise ValueError(
            f"`knn` points and queries must be on the same device, got {points.device=} and {queries.device=}"
        )

    if points.dtype != queries.dtype:
        raise ValueError(
            f"`knn` points and queries must have the same dtype, got {points.dtype=} and {queries.dtype=}"
        )

    # cuml is GPU only
    # scip is CPU only.

    # Compute follows data:
    # auto will dispatch to scipy if points.device==cpu and cuml if points.device==cuda
    # the brute-force torch backend will never be reached unless requested explicitly.

    if backend == "auto":
        if points.is_cuda:
            backend = "cuml"
        else:
            backend = "scipy"

    # Cuml foes not support bfloat16:
    # Autocast to float32:
    original_dtype = points.dtype

    if points.dtype == torch.bfloat16 and (backend == "cuml" or backend == "scipy"):
        points = points.to(torch.float32)
        queries = queries.to(torch.float32)

    match backend:
        case "scipy":
            if points.device.type != "cpu":
                raise ValueError(
                    f"`knn` scipy backend does not support CUDA, got {points.device=}"
                )
            method = knn_scipy
        case "cuml":
            if points.device.type != "cuda":
                raise ValueError(
                    f"`knn` cuml backend does not support CPU, got {points.device=}"
                )
            method = knn_cuml
        case "torch":
            method = knn_torch
        case _:
            raise NotImplementedError(f"Unknown backend: {backend}")

    indices, distances = method(points, queries, k)

    # Return the distances in the original dtype:
    distances = distances.to(original_dtype)

    return indices, distances
