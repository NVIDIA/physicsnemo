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

"""Sampling operations for meshes.

This module provides functions for sampling points on meshes, including:
- Random uniform point sampling on cells using Dirichlet distributions
- Spatial data sampling at query points with interpolation
- BVH-accelerated sampling for large meshes (via the ``bvh`` parameter)
"""

from physicsnemo.mesh.sampling.random_point_sampling import (
    sample_random_points_on_cells,
)
from physicsnemo.mesh.sampling.sample_data import (
    compute_barycentric_coordinates,
    find_all_containing_cells,
    find_containing_cells,
    find_nearest_cells,
    sample_data_at_points,
)
