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

r"""Legacy import shims for the FLARE model.

The model now lives in :mod:`physicsnemo.models.flare`. Import it from there
instead:

.. code-block:: python

    from physicsnemo.models.flare import FLARE

Importing from this legacy namespace emits a
:class:`~physicsnemo.core.warnings.LegacyFeatureWarning`; PhysicsNeMo will
remove these shims in a future release.
"""

import warnings

from physicsnemo.core.warnings import LegacyFeatureWarning

from .flare import FLARE

warnings.warn(
    "Importing from 'physicsnemo.experimental.models.flare' is deprecated. "
    "Use 'from physicsnemo.models.flare import FLARE' instead. "
    "This backward-compatibility shim will be removed in a future release.",
    LegacyFeatureWarning,
    stacklevel=2,
)

__all__ = ["FLARE"]
