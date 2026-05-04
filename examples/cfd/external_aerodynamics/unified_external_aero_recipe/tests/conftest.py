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

"""Shared pytest fixtures and path setup for the unified external aero recipe tests.

The recipe's `src/` modules are imported by their bare names (e.g.
``from collate import build_collate_fn``); the production entry point
(`src/train.py`) gets this for free because `src/datasets.py` runs
``sys.path.insert(0, str(Path(__file__).resolve().parent))`` at import
time. For tests, we make the same insertion explicit here so each test
file can simply ``from collate import ...``.

Importing :mod:`physicsnemo.datapipes` also registers the ``${dp:...}``
OmegaConf resolver, and the recipe-local :mod:`nondim` and :mod:`sdf`
modules register their custom transforms into the global datapipe
registry. Without those side-effect imports the dataset YAMLs cannot be
instantiated.
"""

from __future__ import annotations

import sys
from pathlib import Path

_RECIPE_ROOT = Path(__file__).resolve().parent.parent
_SRC = _RECIPE_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

### Side-effect imports: register the ${dp:...} resolver plus the
### recipe-local NonDimensionalizeByMetadata / ComputeSDFFromBoundary /
### DropBoundary transforms.
import physicsnemo.datapipes  # noqa: E402, F401
import nondim  # noqa: E402, F401
import sdf  # noqa: E402, F401
