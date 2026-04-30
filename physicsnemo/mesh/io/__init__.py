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

"""I/O utilities for PhysicsNeMo Mesh.

This module provides functions to convert between PhysicsNeMo Mesh and other
mesh formats, particularly PyVista.

Submodules that depend on optional packages are loaded lazily via the
module-level ``__getattr__`` (PEP 562) so that ``import physicsnemo.mesh.io``
succeeds without those packages installed; the import only fails when the
attribute is actually accessed.
"""

from typing import TYPE_CHECKING

__all__ = ["from_pyvista", "to_pyvista"]

if TYPE_CHECKING:
    from physicsnemo.mesh.io.io_pyvista import from_pyvista, to_pyvista


def __getattr__(name: str):  # PEP 562
    if name in {"from_pyvista", "to_pyvista"}:
        from physicsnemo.mesh.io import io_pyvista

        return getattr(io_pyvista, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
