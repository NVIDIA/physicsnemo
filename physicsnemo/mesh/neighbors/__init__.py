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

"""Neighbor and adjacency computation for simplicial meshes.

This module provides GPU-compatible functions for computing various adjacency
relationships in simplicial meshes, including point-to-cells, point-to-points,
and cell-to-cells adjacency.

All adjacency relationships are returned as Adjacency tensorclass objects using
offset-indices encoding for efficient representation of ragged arrays.
"""

from physicsnemo.mesh.neighbors._adjacency import Adjacency, build_adjacency_from_pairs
from physicsnemo.mesh.neighbors._cell_neighbors import (
    get_cell_to_cells_adjacency,
    get_cell_to_points_adjacency,
)
from physicsnemo.mesh.neighbors._point_neighbors import (
    get_point_to_cells_adjacency,
    get_point_to_points_adjacency,
)
