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

r"""HealDA data-assimilation models and building blocks.

:class:`HealDAv2` (current) uses :class:`PixelCrossAttention` and
:class:`ObsTokenizerFiLM`. :class:`HealDA` (v1) uses the
:class:`MultiSensorObsEmbedder` stack (:class:`SensorEmbedder`,
:class:`UniformFusion`, :class:`ObsTokenizer`).
"""

from .healda import HealDA, HealDAMetaData
from .healda_v2 import HealDAv2, HealDAv2MetaData
from .obs_context import ObsContext, PixelGroupMap
from .obs_tokenizer import ObsTokenizerFiLM
from .pixel_cross_attention import PixelCrossAttention
from .point_embed import (
    MultiSensorObsEmbedder,
    ObsTokenizer,
    SensorEmbedder,
    UniformFusion,
)
from .scatter_aggregator import ScatterAggregator, scatter_mean

__all__ = [
    "HealDAv2",
    "HealDAv2MetaData",
    "HealDA",
    "HealDAMetaData",
    "PixelCrossAttention",
    "ObsTokenizerFiLM",
    "ObsContext",
    "PixelGroupMap",
    "MultiSensorObsEmbedder",
    "ObsTokenizer",
    "SensorEmbedder",
    "UniformFusion",
    "ScatterAggregator",
    "scatter_mean",
]
