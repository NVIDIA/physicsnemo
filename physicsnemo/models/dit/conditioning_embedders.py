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

"""Backward-compatible shim for DiT conditioning embedders.

.. deprecated::
    ``physicsnemo.models.dit.conditioning_embedders`` has been moved to
    ``physicsnemo.nn.module.conditioning_embedders``. Import from :mod:`physicsnemo.nn`
    instead. This shim will be removed in a future release.
"""

import warnings

from physicsnemo.core.warnings import LegacyFeatureWarning

warnings.warn(
    "physicsnemo.models.dit.conditioning_embedders is deprecated. "
    "Import conditioning embedder components from physicsnemo.nn instead "
    "(e.g. `from physicsnemo.nn import ConditioningEmbedder`). "
    "This backward-compatibility shim will be removed in a future release.",
    LegacyFeatureWarning,
    stacklevel=2,
)

from physicsnemo.nn.module.conditioning_embedders import (  # noqa: E402, F401
    ConditioningEmbedder,
    ConditioningEmbedderType,
    DiTConditionEmbedder,
    EDMConditionEmbedder,
    ZeroConditioningEmbedder,
    get_conditioning_embedder,
)

__all__ = [
    "ConditioningEmbedder",
    "ConditioningEmbedderType",
    "DiTConditionEmbedder",
    "EDMConditionEmbedder",
    "ZeroConditioningEmbedder",
    "get_conditioning_embedder",
]
