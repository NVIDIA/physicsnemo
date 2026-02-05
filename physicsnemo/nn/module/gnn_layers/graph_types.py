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
This file creates a uniform interface for the graph type, usable in typing contexts.
"""

from typing import TYPE_CHECKING, TypeAlias, Union

from physicsnemo.core.version_check import OptionalImport

# Lazy imports for optional dependencies - no import happens until accessed
_pyg = OptionalImport("torch_geometric")
_torch_scatter = OptionalImport("torch_scatter")

# Type alias that works regardless of whether PyG is installed
if TYPE_CHECKING:
    from torch_geometric.data import Data as PyGData
    from torch_geometric.data import HeteroData as PyGHeteroData

    GraphType: TypeAlias = Union[PyGData, PyGHeteroData]
else:
    # At runtime, we use None as a placeholder to avoid triggering imports.
    # Actual type checks should use isinstance with OptionalImport accessors.
    GraphType: TypeAlias = None
