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

r"""DeFormer: the network, its contact-aware corrector and their building blocks.

The network is a GeoTransolver/FLARE backbone with static kNN-attention
stages before and after it, FiLM-conditioned on a mesh
context. The corrector is a MeshGraphNet-style message-passing model over the
structural edges and the contact edges discovered on a predicted shape.
"""

from .batching import LocalNeighborhoodData
from .corrector import DeFormerCorrector, mesh_edge_features
from .deformer import (
    DeFormer,
    DeFormerMetaData,
    MeshContextFiLM,
    MeshGeometryContextEncoder,
)
from .local_attention import (
    KNNAttentionBlock,
    KNNAttentionProcessor,
    LocalNeighborhood,
)

__all__ = [
    "MeshGeometryContextEncoder",
    "MeshContextFiLM",
    "DeFormer",
    "DeFormerCorrector",
    "DeFormerMetaData",
    "LocalNeighborhood",
    "LocalNeighborhoodData",
    "KNNAttentionBlock",
    "KNNAttentionProcessor",
    "mesh_edge_features",
]
