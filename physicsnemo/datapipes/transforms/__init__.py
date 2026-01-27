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

"""
Transforms module - Operations on Samples.

Transforms are composable operations that take a Sample and return a modified Sample.
They are designed for GPU preprocessing
"""

from physicsnemo.datapipes.transforms.base import Transform
from physicsnemo.datapipes.transforms.compose import Compose

# NOTE: Downsample and ToDevice transforms are not yet implemented
from physicsnemo.datapipes.transforms.concat_fields import (
    ConcatFields,
    NormalizeVectors,
)
from physicsnemo.datapipes.transforms.field_processing import (
    BroadcastGlobalFeatures,
)
from physicsnemo.datapipes.transforms.field_slice import FieldSlice
from physicsnemo.datapipes.transforms.geometric import (
    ComputeNormals,
    ComputeSDF,
    Scale,
    Translate,
)
from physicsnemo.datapipes.transforms.normalize import Normalize
from physicsnemo.datapipes.transforms.spatial import (
    BoundingBoxFilter,
    CenterOfMass,
    CreateGrid,
    KNearestNeighbors,
)
from physicsnemo.datapipes.transforms.subsample import (
    SubsamplePoints,
    poisson_sample_indices_fixed,
    shuffle_array,
)
from physicsnemo.datapipes.transforms.utility import (
    ConstantField,
    Purge,
    Rename,
)

__all__ = [
    # Base
    "Transform",
    "Compose",
    # Existing transforms
    "Normalize",
    # Subsampling
    "SubsamplePoints",
    "poisson_sample_indices_fixed",
    "shuffle_array",
    # Geometric
    "ComputeSDF",
    "ComputeNormals",
    "Translate",
    "Scale",
    # Field processing
    "FieldSlice",
    "BroadcastGlobalFeatures",
    # Concat / feature building
    "ConcatFields",
    "NormalizeVectors",
    # Spatial
    "BoundingBoxFilter",
    "CreateGrid",
    "KNearestNeighbors",
    "CenterOfMass",
    # Utility
    "Rename",
    "Purge",
    "ConstantField",
]
