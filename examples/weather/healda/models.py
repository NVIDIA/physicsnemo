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
import torch

from physicsnemo.experimental.models.healda import HealDA
from config.model_config import ModelConfigV1


def _get_condition_dim(config: ModelConfigV1, hidden_size: int) -> int | None:
    """Determine condition_dim from config.

    Returns None for VIT mode (no diffusion conditioning),
    or the embedding dimension for diffusion mode.
    """
    if config.as_vit:
        return None
    # Default to 4 * hidden_size if not specified
    return config.emb_channels or 4 * hidden_size


def get_model(config: ModelConfigV1) -> torch.nn.Module:
    """Instantiate HealDA model from config."""
    if config.architecture == "dit-test":
        hidden_size = 128  # 2 heads * 64 dim
        return HealDA(
            in_channels=config.condition_channels,
            out_channels=config.out_channels,
            sensor_embedder_config=config.sensor_embedder_config,
            sensors=config.sensors,
            hidden_size=hidden_size,
            num_layers=1,
            num_heads=2,
            level_in=6,
            level_model=5,
            time_length=config.time_length,
            drop_path=config.drop_path,
            dropout=config.p_dropout,
            qk_norm_type="rmsnorm" if config.qk_rms_norm else None,
            condition_dim=_get_condition_dim(config, hidden_size),
            noise_channels=config.noise_channels or 1024,
            label_dim=config.label_dim,
            label_dropout=config.label_dropout if config.label_dropout > 0 else None,
        )
    elif config.architecture == "dit-l_reg_hpx6_per_sensor":
        hidden_size = 1024  # 16 heads * 64 dim
        return HealDA(
            in_channels=config.condition_channels,
            out_channels=config.out_channels,
            sensor_embedder_config=config.sensor_embedder_config,
            sensors=config.sensors,
            hidden_size=hidden_size,
            num_layers=24,
            num_heads=16,
            level_in=6,
            level_model=5,
            time_length=config.time_length,
            drop_path=config.drop_path,
            dropout=config.p_dropout,
            qk_norm_type="rmsnorm" if config.qk_rms_norm else None,
            condition_dim=_get_condition_dim(config, hidden_size),
            noise_channels=config.noise_channels or 1024,
            label_dim=config.label_dim,
            label_dropout=config.label_dropout if config.label_dropout > 0 else None,
        )
    else:
        raise NotImplementedError(config.architecture)
