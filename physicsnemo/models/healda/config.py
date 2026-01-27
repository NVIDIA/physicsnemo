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
import dataclasses
import json


def _filter_to_dataclass_fields(d: dict, cls) -> dict:
    """Filter dict to only include fields defined in the dataclass."""
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in d.items() if k in valid_fields}


@dataclasses.dataclass(frozen=True)
class ObsConfig:
    """Observation dataset configuration."""

    use_obs: bool = False
    innovation_type: str = "none"
    context_start: int = -21  # start/end in hours
    context_end: int = 3
    randomize_interval_times: bool = False
    out_channels: int = -1
    embed_dim: int = 0
    use_infrared: bool = False
    use_conv: bool = False
    use_density: bool = False
    conv_uv_in_situ_only: bool = False
    conv_gps_level1_only: bool = False
    dropout: float = 0.0
    # Optional list of global observation channel IDs to drop
    # (these correspond to GLOBAL_CHANNEL_ID in the unified obs schema).
    # When None, no channels are explicitly dropped by ID.
    drop_obs_channel_ids: list[int] | None = None


@dataclasses.dataclass(frozen=True)
class ModelSensorConfig:
    sensor_id: int
    nchannel: int
    platform_ids: tuple[int, ...]  # Use tuple since frozen=True


@dataclasses.dataclass(frozen=True)
class SensorEmbedderConfig:
    """Sensor embedding configuration. Used per sensor."""

    embed_dim: int = 32  # initial tokenization dimension
    meta_dim: int = 28  # dimension of static metadata features
    fusion_dim: int = 512  # sensor fusion dimension
    use_channel_platform_embedding_table: bool = False


@dataclasses.dataclass
class ModelConfigV1:
    architecture: str = "dit-l_reg_hpx6_per_sensor"
    label_dim: int = 0
    out_channels: int = 1
    condition_channels: int = 0
    time_length: int = 1
    label_dropout: float = 0.0
    legacy_label_bias: bool = (
        False  # For loading old checkpoints with trained label bias
    )

    obs_config: ObsConfig = ObsConfig()

    p_dropout: float = 0.0
    drop_path: float = 0.0
    group_norm_eps: float = 1e-6
    pos_emb_gains: bool = False

    # dit settings
    dit_temporal_attention: bool = False
    compile_dit: bool = False
    qk_rms_norm: bool = False
    embed_v2: bool = False
    allow_nans_condition: bool = False
    emb_channels: int | None = None
    noise_channels: int | None = None
    """
    the number of channels to use for the noise and label embedding, defaults to 4 * inner_dim.
    """
    as_vit: bool = False  # run DiT without noise/label conditioning

    # obs encoder settings
    sensor_embedder_config: SensorEmbedderConfig | None = dataclasses.field(
        default_factory=SensorEmbedderConfig
    )
    sensors: dict[str, ModelSensorConfig] | None = None

    def dumps(self):
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def loads(cls, s):
        d = json.loads(s)

        # Filter out fields that aren't in the current model config definition
        d = _filter_to_dataclass_fields(d, cls)

        if isinstance(d.get("obs_config"), dict):
            d["obs_config"] = ObsConfig(
                **_filter_to_dataclass_fields(d["obs_config"], ObsConfig)
            )

        if isinstance(d.get("sensor_embedder_config"), dict):
            embed_cfg = d["sensor_embedder_config"]
            # Backwards compat: old checkpoints had sensors/sensor_config nested inside
            # Move it to top-level ModelConfigV1.sensors
            nested_sensors = embed_cfg.pop("sensors", None) or embed_cfg.pop(
                "sensor_config", None
            )
            if nested_sensors and "sensors" not in d:
                d["sensors"] = nested_sensors
            # Filter to only known fields
            d["sensor_embedder_config"] = SensorEmbedderConfig(
                **_filter_to_dataclass_fields(embed_cfg, SensorEmbedderConfig)
            )

        # Handle sensors dict at top level
        if isinstance(d.get("sensors"), dict):
            d["sensors"] = {
                k: ModelSensorConfig(**v) if isinstance(v, dict) else v
                for k, v in d["sensors"].items()
            }

        return cls(**d)
