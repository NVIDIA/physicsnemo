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

r"""Legacy import shims for the GeoTransolver model and its components.

The model, its metadata, and the context projector components now live in
:mod:`physicsnemo.models.geotransolver`; the GALE attention layers now live in
:mod:`physicsnemo.nn`. Import them from there instead:

.. code-block:: python

    from physicsnemo.models.geotransolver import ContextProjector, GeoTransolver
    from physicsnemo.nn import GALE, GALEBlock

Importing from this legacy namespace emits a
:class:`~physicsnemo.core.warnings.LegacyFeatureWarning`; PhysicsNeMo will
remove these shims in a future release.
"""

import warnings

from physicsnemo.core.warnings import LegacyFeatureWarning
from physicsnemo.nn import (
    ConcreteDropout,
    collect_concrete_dropout_losses,
    get_concrete_dropout_rates,
)

from .context_projector import (
    ContextProjector,
    GeometricFeatureProcessor,
    GlobalContextBuilder,
    MultiScaleFeatureExtractor,
    StructuredContextProjector,
)
from .gale import (
    GALE,
    GALE_FA,
    GALE_block,
    GALEStructuredMesh2D,
    GALEStructuredMesh3D,
)
from .geotransolver import GeoTransolver, GeoTransolverMetaData

warnings.warn(
    "Importing from 'physicsnemo.experimental.models.geotransolver' is deprecated. "
    "Import GeoTransolver, its metadata, and the context projector components "
    "from 'physicsnemo.models.geotransolver', and the GALE attention layers "
    "(GALE_block is now named GALEBlock) from 'physicsnemo.nn' instead. "
    "This backward-compatibility shim will be removed in a future release.",
    LegacyFeatureWarning,
    stacklevel=2,
)

__all__ = [
    "GeoTransolver",
    "GeoTransolverMetaData",
    "GALE",
    "GALE_FA",
    "GALE_block",
    "GALEStructuredMesh2D",
    "GALEStructuredMesh3D",
    "ContextProjector",
    "GeometricFeatureProcessor",
    "GlobalContextBuilder",
    "MultiScaleFeatureExtractor",
    "StructuredContextProjector",
    "ConcreteDropout",
    "collect_concrete_dropout_losses",
    "get_concrete_dropout_rates",
]
