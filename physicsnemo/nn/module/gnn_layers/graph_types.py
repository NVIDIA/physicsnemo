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
This file creates a uniform interface for the graph type, usable in typing contexts.
"""

import importlib
from typing import TypeAlias

from physicsnemo.core.version_check import check_version_spec

PYG_AVAILABLE = check_version_spec(
    "torch_geometric", hard_fail=False
) and check_version_spec("torch_scatter", hard_fail=False)

if PYG_AVAILABLE:
    PyGData = importlib.import_module("torch_geometric.data").Data
    PyGHeteroData = importlib.import_module("torch_geometric.data").HeteroData

    GraphType: TypeAlias = PyGData | PyGHeteroData

else:
    GraphType: TypeAlias = None


def raise_missing_pyg_error():
    msg = "MeshGraphNet requires PyTorch Geometric and torch_scatter.\n"
    "Install it from here:\n"
    "  https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html\n"

    raise ImportError(msg)
