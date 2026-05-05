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

r"""GeoTransolver: Geometry-Aware Physics Attention Transformer.

This module provides the GeoTransolver model and its components for learning
physics-based representations with geometry and global context awareness.

Classes
-------
GeoTransolver
    Main model class combining GALE attention with geometry and global context.
GeoTransolverMetaData
    Data class for storing essential meta data needed for the GeoTransolver model.
GALE
    Geometry-Aware Latent Embeddings attention layer.
GALE_FA
    GALE with FLARE self-attention backend.
GALE_block
    Transformer block using GALE or GALE_FA attention.
GALEStructuredMesh2D
    GALE with Conv2d slice projection for 2D structured grids.
GALEStructuredMesh3D
    GALE with Conv3d slice projection for 3D structured grids.
ContextProjector
    Projects context features onto physical state slices.
StructuredContextProjector
    Context projector with Conv2d/Conv3d geometry encoding on structured grids.
GeometricFeatureProcessor
    Processes geometric features at a single spatial scale using BQWarp.
MultiScaleFeatureExtractor
    Multi-scale geometric feature extraction over multiple radii.
GlobalContextBuilder
    Orchestrates context construction for the model.

Functions
---------
collect_concrete_dropout_losses
    Collect concrete dropout regularization losses from a model.
get_concrete_dropout_rates
    Get concrete dropout rates from a model.

Examples
--------
Basic usage:

>>> import torch
>>> from physicsnemo.experimental.models.geotransolver import GeoTransolver
>>> model = GeoTransolver(
...     functional_dim=64,
...     out_dim=3,
...     n_hidden=256,
...     n_layers=4,
...     use_te=False,
... )
>>> x = torch.randn(2, 1000, 64)
>>> output = model(x)
>>> output.shape
torch.Size([2, 1000, 3])
"""

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