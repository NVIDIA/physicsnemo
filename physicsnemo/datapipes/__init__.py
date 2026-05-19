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
datapipe - High-performance GPU-centric data loading for Scientific ML

A modular, composable data pipeline for physics and scientific machine learning.
Designed for clean separation of concerns:

- **Readers**: Load data from sources → TensorDict tuples with CPU tensors
- **Transforms**: Process TensorDict data
- **Dataset**: Reader + transforms pipeline with optional auto device transfer
- **DataLoader**: Batched iteration with optional prefetching

"""

from tensordict import TensorDict

from physicsnemo.datapipes.collate import (
    Collator,
    ConcatCollator,
    DefaultCollator,
    FunctionCollator,
    concat_collate,
    default_collate,
    get_collator,
)
from physicsnemo.datapipes.dataloader import DataLoader
from physicsnemo.datapipes.dataset import Dataset
from physicsnemo.datapipes.mesh_dataset import MeshDataset
from physicsnemo.datapipes.multi_dataset import MultiDataset
from physicsnemo.datapipes.protocols import DatasetBase
from physicsnemo.datapipes.readers import (
    DomainMeshReader,
    HDF5Reader,
    MeshReader,
    NumpyReader,
    Reader,
    TensorStoreZarrReader,
    VTKReader,
    ZarrReader,
)
from physicsnemo.datapipes.registry import (
    COMPONENT_REGISTRY,
    ComponentRegistry,
    register,
    register_resolvers,
)
from physicsnemo.datapipes.transforms import (
    BoundingBoxFilter,
    BroadcastGlobalFeatures,
    CenterMesh,
    CenterOfMass,
    Compose,
    ComputeCellCentroids,
    ComputeNormals,
    ComputeSDF,
    ComputeSurfaceNormals,
    ConcatFields,
    ConstantField,
    CreateGrid,
    DropMeshFields,
    FieldSlice,
    KNearestNeighbors,
    MeshToTensorDict,
    MeshTransform,
    Normalize,
    NormalizeMeshFields,
    NormalizeVectors,
    Purge,
    RandomRotateMesh,
    RandomScaleMesh,
    RandomTranslateMesh,
    Rename,
    RenameMeshFields,
    RestructureTensorDict,
    RotateMesh,
    Scale,
    ScaleMesh,
    SetGlobalField,
    SubsampleMesh,
    SubsamplePoints,
    Transform,
    Translate,
    TranslateMesh,
)

# Auto-register OmegaConf resolvers so ${dp:ComponentName} works in Hydra configs
register_resolvers()

__all__ = [
    #
    "TensorDict",  # Re-export from tensordict
    "DatasetBase",
    "Dataset",
    "MeshDataset",
    "DataLoader",
    "MultiDataset",
    # Transforms - Base
    "Transform",
    "Compose",
    # Transforms - Normalization
    "Normalize",
    # Transforms - Subsampling
    "SubsamplePoints",
    # Transforms - Geometric
    "ComputeSDF",
    "ComputeNormals",
    "Translate",
    "Scale",
    # Transforms - Field processing
    "FieldSlice",
    "BroadcastGlobalFeatures",
    # Transforms - Concat / feature building
    "ConcatFields",
    "NormalizeVectors",
    # Transforms - Spatial
    "BoundingBoxFilter",
    "CreateGrid",
    "KNearestNeighbors",
    "CenterOfMass",
    # Transforms - Utility
    "Rename",
    "Purge",
    "ConstantField",
    # Transforms - Mesh
    "MeshTransform",
    "ComputeCellCentroids",
    "ComputeSurfaceNormals",
    "MeshToTensorDict",
    "ScaleMesh",
    "TranslateMesh",
    "RotateMesh",
    "CenterMesh",
    "SubsampleMesh",
    "DropMeshFields",
    "RenameMeshFields",
    "NormalizeMeshFields",
    "SetGlobalField",
    "RestructureTensorDict",
    "RandomScaleMesh",
    "RandomTranslateMesh",
    "RandomRotateMesh",
    # Readers
    "Reader",
    "HDF5Reader",
    "ZarrReader",
    "NumpyReader",
    "VTKReader",
    "TensorStoreZarrReader",
    "MeshReader",
    "DomainMeshReader",
    # Collation
    "Collator",
    "DefaultCollator",
    "ConcatCollator",
    "FunctionCollator",
    "default_collate",
    "concat_collate",
    "get_collator",
    # Registry
    "ComponentRegistry",
    "COMPONENT_REGISTRY",
    "register",
    "register_resolvers",
]
