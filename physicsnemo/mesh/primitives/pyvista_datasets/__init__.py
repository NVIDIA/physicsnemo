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

"""PyVista example dataset wrappers.

These functions wrap PyVista's built-in example datasets, converting them
to physicsnemo.mesh format. PyVista handles caching of downloaded datasets
automatically.

This module requires pyvista to be installed.
"""

from physicsnemo.mesh.primitives.pyvista_datasets import (
    airplane,
    ant,
    bunny,
    cow,
    globe,
    hexbeam,
    tetbeam,
)
