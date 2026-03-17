# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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


def get_model(config: ModelConfigV1) -> torch.nn.Module:
    """Instantiate HealDA model from config."""
    if config.architecture == "dit-test":
        hidden_size = 128  # 2 heads * 64 dim
        return _build_healda(config, hidden_size=hidden_size, num_layers=1, num_heads=2)
    elif config.architecture == "dit-l_reg_hpx6_per_sensor":
        hidden_size = 1024  # 16 heads * 64 dim
        return _build_healda(
            config, hidden_size=hidden_size, num_layers=24, num_heads=16
        )
    else:
        raise NotImplementedError(config.architecture)


def _build_healda(
    config: ModelConfigV1,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
) -> HealDA:
    diffusion_conditioning = not config.as_vit
    condition_embed_dim = config.emb_channels if diffusion_conditioning else None

    return HealDA(
        in_channels=config.condition_channels,
        out_channels=config.out_channels,
        nchannel_per_sensor=config.nchannel_per_sensor,
        nplatform_per_sensor=config.nplatform_per_sensor,
        sensor_names=config.sensor_names,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        level_in=6,
        level_model=5,
        time_length=config.time_length,
        embed_dim=config.embed_dim,
        meta_dim=config.meta_dim,
        fusion_dim=config.fusion_dim,
        drop_path=config.drop_path,
        dropout=config.p_dropout,
        qk_norm_type="RMSNorm" if config.qk_rms_norm else None,
        diffusion_conditioning=diffusion_conditioning,
        condition_dim=config.label_dim,
        condition_embed_dim=condition_embed_dim,
        noise_channels=config.noise_channels,
        condition_dropout=config.label_dropout,
        compile_obs_embedder=config.compile_obs_embedder,
    )
