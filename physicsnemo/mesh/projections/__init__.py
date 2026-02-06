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

"""Projection operations for mesh extrusion, embedding, and spatial dimension manipulation.

This module provides functionality for:
- Embedding meshes in higher-dimensional spaces (non-destructive)
- Projecting meshes to lower-dimensional spaces (lossy)
- Extruding manifolds to higher dimensions
"""

from physicsnemo.mesh.projections._embed import embed
from physicsnemo.mesh.projections._extrude import extrude
from physicsnemo.mesh.projections._project import project
