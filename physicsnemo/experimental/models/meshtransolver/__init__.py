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

r"""Hybrid mesh-attention models for irregular simulation meshes.

The models in this package combine local MeshGraphNet processing with one of
PhysicsNeMo's global physics-attention processors.
"""

from .contact import (
    CONTACT_FEATURE_DIM,
    KINEMATIC_CONTACT_FEATURE_DIM,
    ContactGraph,
    SparseContactBlock,
    SparseContactGraphBuilder,
    contact_edge_features,
    merge_contact_graphs,
    smooth_contact_cutoff,
)
from .functional_contact import ContactSearchTopology, FunctionalContactGraphBuilder
from .meshtransolver import (
    MeshAttentionHybrid,
    MeshContextFiLM,
    MeshGeoFLARE,
    MeshGeometryContextEncoder,
    MeshGeoTransolver,
    MeshTransolver,
)
from .surface_contact import SurfaceContactGraphBuilder

__all__ = [
    "MeshAttentionHybrid",
    "MeshGeometryContextEncoder",
    "MeshContextFiLM",
    "MeshTransolver",
    "MeshGeoTransolver",
    "MeshGeoFLARE",
    "CONTACT_FEATURE_DIM",
    "KINEMATIC_CONTACT_FEATURE_DIM",
    "ContactSearchTopology",
    "FunctionalContactGraphBuilder",
    "SurfaceContactGraphBuilder",
    "smooth_contact_cutoff",
    "ContactGraph",
    "SparseContactBlock",
    "SparseContactGraphBuilder",
    "contact_edge_features",
    "merge_contact_graphs",
]
