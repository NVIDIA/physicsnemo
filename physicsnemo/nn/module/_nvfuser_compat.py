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

"""Compatibility shim for the legacy ``nvfuser`` package and the newer
``nvfuser_direct`` package.

The nvFuser Python frontend is split into two distributions: the legacy
``nvfuser`` package (older PyTorch containers) and ``nvfuser_direct`` (newer
containers). This module hides the difference behind a single import surface
and reimplements the two helpers that exist only in the legacy package
(``compute_contiguity`` and ``define_constant``) so the rest of PhysicsNeMo
can target either backend without conditionals.

Importing is also defended against orphan ``.dist-info`` metadata: some
container images ship pip metadata for ``nvfuser`` without the actual package
files, in which case ``importlib.metadata`` reports it as installed but
``import nvfuser`` raises ``ModuleNotFoundError``. We treat that as
"unavailable" and fall back gracefully.
"""

import importlib
import importlib.util
import logging
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


def _try_import():
    """Return ``(module, backend_name)`` for the first usable nvfuser backend.

    Tries the legacy ``nvfuser`` package first for backward compatibility,
    then ``nvfuser_direct``. Uses ``find_spec`` before ``import_module`` so
    that orphan distribution metadata does not turn a soft "missing optional
    dep" into a hard import failure.
    """
    for name in ("nvfuser", "nvfuser_direct"):
        if importlib.util.find_spec(name) is None:
            continue
        try:
            return importlib.import_module(name), name
        except ImportError as e:
            logger.warning(
                "Found %s on sys.path but failed to import (%s); trying next backend.",
                name,
                e,
            )
    return None, None


nvfuser, _BACKEND = _try_import()
NV_FUSER_AVAILABLE: bool = nvfuser is not None

if NV_FUSER_AVAILABLE:
    FusionDefinition = nvfuser.FusionDefinition
    DataType = nvfuser.DataType
else:
    FusionDefinition = None  # type: ignore[assignment]
    DataType = None  # type: ignore[assignment]


def compute_contiguity(
    sizes: Sequence[int], strides: Sequence[int]
) -> List[Optional[bool]]:
    """Per-dim contiguity flags expected by ``FusionDefinition.define_tensor``.

    Mirrors the legacy ``nvfuser.compute_contiguity`` helper, which was
    removed in ``nvfuser_direct``. The convention (preserved here) is one
    entry per dimension, where each entry is ``True``/``False`` indicating
    whether that dim is contiguous w.r.t. the inner dims, or ``None`` for
    broadcast/size-1 dims.
    """
    n = len(sizes)
    out: List[Optional[bool]] = [None] * n
    expected = 1
    for i in range(n - 1, -1, -1):
        if sizes[i] == 1:
            out[i] = None
        else:
            out[i] = strides[i] == expected
            expected = sizes[i] * strides[i]
    return out
