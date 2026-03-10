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
GALE
    Geometry-Aware Latent Embeddings attention layer.
GALE_block
    Transformer block using GALE attention.
ContextProjector
    Projects context features onto physical state slices.
GlobalContextBuilder
    Orchestrates context construction for the model.

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

from .context_projector import ContextProjector, GlobalContextBuilder
from .gale import GALE, GALE_block
from .geotransolver import GeoTransolver, GeoTransolverMetaData

__all__ = [
    "GeoTransolver",
    "GeoTransolverMetaData",
    "GALE",
    "GALE_block",
    "ContextProjector",
    "GlobalContextBuilder",
]