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
Mesh transforms and augmentations.

Transforms operate on Mesh (single-mesh) or, via ``td.apply(transform,
call_on_nested=True)``, on a ``TensorDict[str, Mesh]``. Type-based
only; no key-based filtering.
"""

from physicsnemo.datapipes.transforms.mesh.augmentations import (
    RandomRotateMesh,
    RandomScaleMesh,
    RandomTranslateMesh,
)
from physicsnemo.datapipes.transforms.mesh.base import MeshTransform
from physicsnemo.datapipes.transforms.mesh.transforms import (
    CenterMesh,
    ComputeCellCentroids,
    ComputeSurfaceNormals,
    DropMeshFields,
    MeshToDomainMesh,
    MeshToTensorDict,
    NormalizeMeshFields,
    RenameMeshFields,
    RestructureTensorDict,
    RotateMesh,
    ScaleMesh,
    SetGlobalField,
    SubsampleMesh,
    TranslateMesh,
)

__all__ = [
    "MeshTransform",
    "ComputeCellCentroids",
    "ComputeSurfaceNormals",
    "ScaleMesh",
    "TranslateMesh",
    "RotateMesh",
    "CenterMesh",
    "SubsampleMesh",
    "DropMeshFields",
    "RenameMeshFields",
    "SetGlobalField",
    "NormalizeMeshFields",
    "MeshToDomainMesh",
    "MeshToTensorDict",
    "RestructureTensorDict",
    "RandomScaleMesh",
    "RandomTranslateMesh",
    "RandomRotateMesh",
]
