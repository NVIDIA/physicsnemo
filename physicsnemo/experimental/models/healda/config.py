# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Model configuration dataclasses for HealDA.

These are constructor parameter types used by the model.
Training-specific configs (ModelConfigV1, ObsConfig) are in examples/weather/healda/.
"""
import dataclasses


@dataclasses.dataclass(frozen=True)
class ModelSensorConfig:
    """Configuration for a single sensor type.
    
    Parameters
    ----------
    sensor_id : int
        Unique identifier for the sensor type.
    nchannel : int
        Number of channels for this sensor.
    platform_ids : tuple[int, ...]
        Tuple of platform IDs associated with this sensor.
    """
    sensor_id: int
    nchannel: int
    platform_ids: tuple[int, ...]  # Use tuple since frozen=True


@dataclasses.dataclass(frozen=True)
class SensorEmbedderConfig:
    """Configuration for sensor embedding module.
    
    Parameters
    ----------
    embed_dim : int, optional, default=32
        Initial tokenization dimension for observations.
    meta_dim : int, optional, default=28
        Dimension of static metadata features.
    fusion_dim : int, optional, default=512
        Dimension after sensor fusion.
    use_channel_platform_embedding_table : bool, optional, default=False
        Whether to use embedding tables for channel and platform IDs.
    """
    embed_dim: int = 32
    meta_dim: int = 28
    fusion_dim: int = 512
    use_channel_platform_embedding_table: bool = False
