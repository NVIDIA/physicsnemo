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


import importlib

import torch

from physicsnemo.core.version_check import check_version_spec

from .utils import validate_inputs

CUML_AVAILABLE = check_version_spec("cuml", "24.0.0", hard_fail=False)
CUPY_AVAILABLE = check_version_spec("cupy", "13.0.0", hard_fail=False)

if CUML_AVAILABLE and CUPY_AVAILABLE:
    cuml = importlib.import_module("cuml")
    cp = importlib.import_module("cupy")

    @torch.library.custom_op("physicsnemo::knn_cuml", mutates_args=())
    def knn_impl(
        points: torch.Tensor, queries: torch.Tensor, k: int = 3
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if points.device.type != "cuda":
            raise ValueError(
                f"`knn` cuml implementation does not support CPU, got {points.device=}"
            )
        validate_inputs(points, queries)
        restore_dtype = points.dtype
        if restore_dtype == torch.bfloat16:
            points = points.to(torch.float32)
            queries = queries.to(torch.float32)
        # Create a cuml handle to ensure we use the right stream:
        torch_stream = torch.cuda.current_stream()

        # Get the raw CUDA stream pointer (as an integer)
        ptr = torch_stream.cuda_stream

        # Build a cuML handle with that stream
        handle = cuml.Handle(stream=ptr)

        # Use dlpack to move the data without copying between pytorch and cuml:
        points = cp.from_dlpack(points)
        queries = cp.from_dlpack(queries)

        # Construct the knn:
        knn = cuml.neighbors.NearestNeighbors(n_neighbors=k, handle=handle)
        # First pass partitions everything in points to make lookups fast
        knn.fit(points)

        # Second pass uses that partition to quickly find neighbors of points in points
        distance, indices = knn.kneighbors(queries)

        # convert back to pytorch:
        distance = torch.from_dlpack(distance)
        indices = torch.from_dlpack(indices)

        # Return torch objects.
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
else:

    def knn_impl(*args, **kwargs) -> None:
        """
        Dummy implementation for when cuml is not available.
        """

        raise ImportError(
            "physics nemo kNN: cuml or cupy is not installed, can not be used as an implementation for a knn search"
            "Please install cuml and cupy, for installation instructions see: https://docs.rapids.ai/install"
        )
