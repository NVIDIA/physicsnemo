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

"""Mesh repair and cleanup utilities.

Tools for fixing common mesh problems including duplicates, degenerates,
holes, and orientation issues.
"""

from physicsnemo.mesh.repair.degenerate_removal import remove_degenerate_cells
from physicsnemo.mesh.repair.duplicate_removal import remove_duplicate_vertices
from physicsnemo.mesh.repair.hole_filling import fill_holes
from physicsnemo.mesh.repair.isolated_removal import remove_isolated_vertices
from physicsnemo.mesh.repair.orientation import fix_orientation
from physicsnemo.mesh.repair.pipeline import repair_mesh

__all__ = [
    "remove_duplicate_vertices",
    "remove_degenerate_cells",
    "remove_isolated_vertices",
    "fix_orientation",
    "fill_holes",
    "repair_mesh",
]
