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

"""
Utility functions: padding (generic) and CO2-specific normalization/visualization.
"""

from utils.co2_normalization import (
    denormalize_inputs,
    dnorm_dP,
    dnorm_inj,
    dnorm_lam,
    dnorm_P,
    dnorm_Swi,
    dnorm_temp,
    extract_reservoir_mask,
)
from utils.co2_visualization import (
    create_pcolor_func,
    get_time_labels,
    plot_4x3_comparison,
    setup_plotting_grid,
)
from utils.padding import (
    compute_right_pad_to_multiple,
    compute_right_pad_to_multiple_per_dim,
    pad_right_nd,
    pad_spatial_right,
)

__all__ = [
    # CO2-specific normalization
    "dnorm_dP",
    "dnorm_inj",
    "dnorm_temp",
    "dnorm_P",
    "dnorm_lam",
    "dnorm_Swi",
    "extract_reservoir_mask",
    "denormalize_inputs",
    # CO2-specific visualization
    "setup_plotting_grid",
    "get_time_labels",
    "create_pcolor_func",
    "plot_4x3_comparison",
    # Padding (generic)
    "compute_right_pad_to_multiple",
    "compute_right_pad_to_multiple_per_dim",
    "pad_right_nd",
    "pad_spatial_right",
]
