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
"""
Training configuration dataclasses for HealDA.

These are training-specific configs with serialization for checkpointing.
"""

import dataclasses
import json


def _filter_to_dataclass_fields(d: dict, cls) -> dict:
    """Filter dict to only include fields defined in the dataclass."""
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in d.items() if k in valid_fields}


@dataclasses.dataclass(frozen=True)
class ObsConfig:
    """Observation dataset configuration for training."""

    use_obs: bool = False
    context_start: int = -21  # start/end in hours
    context_end: int = 3
    use_infrared: bool = False
    use_conv: bool = False
    conv_uv_in_situ_only: bool = False
    conv_gps_level1_only: bool = False
    drop_obs_channel_ids: list[int] | None = None


@dataclasses.dataclass
class ModelConfigV1:
    """Training configuration for HealDA."""

    architecture: str = "dit-l_reg_hpx6_per_sensor"
    label_dim: int = 0
    out_channels: int = 1
    condition_channels: int = 0
    time_length: int = 1
    label_dropout: float = 0.0

    obs_config: ObsConfig = dataclasses.field(default_factory=ObsConfig)

    p_dropout: float = 0.0
    drop_path: float = 0.0

    qk_rms_norm: bool = False
    as_vit: bool = False  # run DiT without noise/label conditioning
    emb_channels: int | None = None
    noise_channels: int | None = None

    # Sensor config
    nchannel_per_sensor: list[int] = dataclasses.field(default_factory=list)
    nplatform_per_sensor: list[int] = dataclasses.field(default_factory=list)
    sensor_names: list[str] = dataclasses.field(default_factory=list)

    # Obs embedder settings
    embed_dim: int = 32
    meta_dim: int = 28
    fusion_dim: int = 512
    compile_obs_embedder: bool = True

    def dumps(self) -> str:
        """Serialize config to JSON string for checkpointing."""
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def loads(cls, s: str) -> "ModelConfigV1":
        """Deserialize config from JSON string."""
        d = json.loads(s)

        # Filter out fields that aren't in the current model config definition
        d = _filter_to_dataclass_fields(d, cls)

        if isinstance(d.get("obs_config"), dict):
            d["obs_config"] = ObsConfig(
                **_filter_to_dataclass_fields(d["obs_config"], ObsConfig)
            )

        return cls(**d)
