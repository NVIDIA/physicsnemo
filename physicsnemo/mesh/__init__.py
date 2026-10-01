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

from typing import TYPE_CHECKING

from physicsnemo.mesh.domain_mesh import DomainMesh
from physicsnemo.mesh.fields import (
    FieldSchema,
    FieldSchemaLike,
    RankSpec,
    RankSpecLike,
    _missing_attribute,
)
from physicsnemo.mesh.mesh import MESH_FIELD_ASSOCIATIONS, Mesh, MeshFieldAssociation

### The rank-spec helpers removed in 2.3 were re-exported here; importing one
### names its replacement (see physicsnemo.mesh.fields). Hidden from type
### checkers so that they keep reporting unknown names as missing.
if not TYPE_CHECKING:

    def __getattr__(name: str):
        """Point imports of removed names at their replacements (PEP 562)."""
        _missing_attribute(__name__, name)


__all__ = [
    "DomainMesh",
    "MESH_FIELD_ASSOCIATIONS",
    "Mesh",
    "MeshFieldAssociation",
    "FieldSchema",
    "FieldSchemaLike",
    "RankSpec",
    "RankSpecLike",
]
