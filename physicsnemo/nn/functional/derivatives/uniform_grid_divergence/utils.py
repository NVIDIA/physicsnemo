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


def validate_vector_field(vector_field: torch.Tensor) -> int:
    """Validate a channel-first vector field and return grid dimensionality."""
    if not torch.is_floating_point(vector_field):
        raise TypeError("vector_field must be a floating-point tensor")
    if vector_field.ndim < 2 or vector_field.ndim > 4:
        raise ValueError(
            "uniform_grid_divergence expects a channel-first 1D-3D vector field with "
            "shape (dim, *grid_shape)"
        )

    grid_ndim = vector_field.ndim - 1
    if vector_field.shape[0] != grid_ndim:
        raise ValueError(
            "vector_field.shape[0] must match grid dimensionality "
            f"({grid_ndim}), got {vector_field.shape[0]}"
        )
    return grid_ndim
