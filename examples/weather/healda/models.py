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

from physicsnemo.models.healda import DiT, ModelConfigV1


def get_model(config: ModelConfigV1) -> torch.nn.Module:
    """Instantiate DiT model from config. Supports 'dit-test' (small) and 'dit' (full) architectures."""
    if config.architecture == "dit-test":
        return DiT(
            in_channels=config.condition_channels,
            out_channels=config.out_channels,
            num_attention_heads=2,
            num_layers=1,
            attention_head_dim=64,
            level_in=6,
            level_model=5,
            obs_config=config.obs_config,
            drop_path=config.drop_path,
            dropout=config.p_dropout,
            time_length=config.time_length,
            label_dim=config.label_dim,
            label_dropout=config.label_dropout,
            group_norm_eps=config.group_norm_eps,
            use_gains=config.pos_emb_gains,
            temporal_attention=config.dit_temporal_attention,
            embed_v2=config.embed_v2,
            compile_dit=config.compile_dit,
            qk_rms_norm=config.qk_rms_norm,
            allow_nans_condition=config.allow_nans_condition,
            embed_v2_meta_dim=28,
            sensor_embedder_config=config.sensor_embedder_config,
            sensors=config.sensors,
            as_vit=config.as_vit,
        )
    elif config.architecture == "dit-l_reg_hpx6_per_sensor":
        return DiT(
            in_channels=config.condition_channels,
            out_channels=config.out_channels,
            num_attention_heads=16,
            num_layers=24,
            attention_head_dim=64,
            level_in=6,
            level_model=5,
            label_dim=config.label_dim,
            label_dropout=config.label_dropout,
            legacy_label_bias=config.legacy_label_bias,
            obs_config=config.obs_config,
            drop_path=config.drop_path,
            dropout=config.p_dropout,
            group_norm_eps=config.group_norm_eps,
            use_gains=config.pos_emb_gains,
            time_length=config.time_length,
            temporal_attention=config.dit_temporal_attention,
            embed_v2=True,
            embed_v2_meta_dim=28,
            sensor_embedder_config=config.sensor_embedder_config,
            sensors=config.sensors,
            compile_dit=config.compile_dit,
            qk_rms_norm=config.qk_rms_norm,
            allow_nans_condition=config.allow_nans_condition,
            as_vit=config.as_vit,
            emb_channels=config.emb_channels,
            noise_channels=config.noise_channels,
        )
    else:
        raise NotImplementedError(config.architecture)
