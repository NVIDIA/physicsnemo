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

from .._moved_to_mesh import MOVED_TO_MESH, moved_to_mesh
from .farthest_point_sampling import FarthestPointSampling, farthest_point_sampling
from .sdf import SignedDistanceField, signed_distance_field

__all__ = [
    "FarthestPointSampling",
    "SignedDistanceField",
    "farthest_point_sampling",
    "signed_distance_field",
]
# Deprecated aliases of the mesh functionals, also resolved by __getattr__.
__all__ += MOVED_TO_MESH[__name__]


def __getattr__(name):
    # Lazy legacy re-exports of the functionals moved to physicsnemo.mesh.
    return moved_to_mesh(__name__, name)
