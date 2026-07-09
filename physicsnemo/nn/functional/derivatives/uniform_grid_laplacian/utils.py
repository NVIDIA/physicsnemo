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


def validate_scalar_field(field: torch.Tensor) -> None:
    """Validate a scalar 1D/2D/3D grid field."""
    if not torch.is_floating_point(field):
        raise TypeError("field must be a floating-point tensor")
    if field.ndim < 1 or field.ndim > 3:
        raise ValueError(
            "uniform_grid_laplacian expects a scalar 1D-3D grid field with shape "
            "(n0,), (n0,n1), or (n0,n1,n2)"
        )
