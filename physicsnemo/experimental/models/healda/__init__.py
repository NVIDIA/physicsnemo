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
Experimental HealDA models and layers.

Warning: This module is experimental and APIs may change without notice.
Per MOD-002a, new models start here before promotion to physicsnemo.models.
"""
from .dit_layers import (
    HealDA,
    HealDAMetaData,
    convert_healda_state_dict,
    map_healda_to_pnm_block_keys,
)

# Config types (model constructor params only)
# Training configs (ModelConfigV1, ObsConfig) are in examples/weather/healda/config/
from .config import (
    ModelSensorConfig,
    SensorEmbedderConfig,
)

# Data types
from .types import (
    Batch,
    UnifiedObservation,
    split_by_sensor,
)

# Domain
from .domain import Domain, HealPixDomain

# Embedding layers (FrequencyEmbedding, CalendarEmbedding are HealDA-specific)
# For PositionalEmbedding and FourierEmbedding, use physicsnemo.nn directly
from .embedding import (
    CalendarEmbedding,
    EmbedNoiseLabels,
    FrequencyEmbedding,
)

# HEALPix tokenizer/detokenizer
from .healpix_layers import (
    HPXPatchDetokenizer,
    HPXPatchTokenizer,
)

# Obs embedding
from .point_embed import MultiSensorObsEmbedding
from .scatter_aggregator import ScatterAggregator, scatter_mean
