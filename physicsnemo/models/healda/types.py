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
from typing import Optional, TypedDict

import torch


@dataclasses.dataclass
class UnifiedObservation:
    """Unified observation structure for both satellite and conventional observations."""

    # Core observation data
    obs: torch.Tensor  # Observation values
    time: torch.Tensor  # Observation times

    # Pre-computed metadata
    float_metadata: (
        torch.Tensor
    )  # Pre-computed float features (e.g., angles, local solar time)

    # Integer metadata for spatial aggregation and embedding lookups
    # Shape: (n_obs, n_metadata_fields=6) - transposed for better torch.compile performance
    # Typical fields: sensor_id, pix, channel_id, platform_id, obs_type, global_channel_id
    int_metadata: torch.Tensor  # dtype=torch.long

    class bucket_index:
        sensor = 0
        pix = 1
        local_channel = 2
        platform = 3
        obs_type = 4
        global_channel = 5

    offsets: torch.Tensor | None = (
        None  # 3D: (n_active_sensors, batch, time) cumulative start indices
    )
    batch_idx: torch.Tensor | None = (
        None  # (n_obs,) batch index - computed from offsets in post_init (batch context, not intrinsic property)
    )
    sensor_id_to_local: torch.Tensor | None = (
        None  # (max_sensor_id active + 1,) map: sensor_id -> local_idx (-1 if inactive)
    )
    hpx_level: int | None = (
        None  # the hpx level (pix is at int_metadata[:, bucket_index.pix])
    )

    def __post_init__(self):
        """Automatically compute batch_idx from offsets after construction."""
        if self.batch_idx is None:
            if self.offsets is not None:
                self.batch_idx = offsets_to_batch_idx(self.offsets)
            else:
                # No offsets = single batch, all observations belong to batch 0
                self.batch_idx = torch.zeros(
                    self.obs.shape[0], dtype=torch.long, device=self.obs.device
                )

    @property
    def batch_dims(self):
        """Return (batch, time) shape from 3D offsets (S, B, T)."""
        if self.offsets is not None:
            return self.offsets.shape[-2:]
        else:
            return ()

    def __repr__(self):
        nobs = self.obs.shape[0]
        return f"UnifiedObservation({nobs=}, batch_dims={self.batch_dims})"

    def to(self, device=None, dtype=None, non_blocking=True):
        """Move all tensors to device and/or convert dtype."""

        def _move_tensor(x):
            if x is None:
                return
            return x.to(device=device, dtype=dtype, non_blocking=non_blocking)

        return UnifiedObservation(
            obs=_move_tensor(self.obs),
            time=_move_tensor(self.time),
            float_metadata=_move_tensor(self.float_metadata),
            int_metadata=_move_tensor(self.int_metadata),
            offsets=_move_tensor(self.offsets),
            batch_idx=_move_tensor(self.batch_idx),
            sensor_id_to_local=_move_tensor(self.sensor_id_to_local),
            hpx_level=self.hpx_level,
        )

    def record_stream(self, stream):
        """Mark"""
        self.obs.record_stream(stream)
        self.time.record_stream(stream)
        self.float_metadata.record_stream(stream)
        self.int_metadata.record_stream(stream)
        self.batch_idx.record_stream(stream)
        if self.offsets is not None:
            self.offsets.record_stream(stream)
        if self.sensor_id_to_local is not None:
            self.sensor_id_to_local.record_stream(stream)


class Batch(TypedDict):
    """Input of DA model on which Obs Encoder operates"""

    target: torch.Tensor  # (b, c, t, x) - main atmospheric variables
    condition: torch.Tensor  # (b, c_cond, t, x) - conditioning variables
    second_of_day: torch.Tensor  # (b, t) - seconds of day
    day_of_year: torch.Tensor  # (b, t) - day of year
    labels: torch.Tensor  # (b, num_classes) - one-hot encoded labels
    timestamp: torch.Tensor  # (b,) - timestamps as seconds since epoch
    unified_obs: Optional[UnifiedObservation]  # Unified observation data (v2)
    # Residual training fields (optional)
    background: Optional[
        torch.Tensor
    ]  # (b, c, t, x) - background data for residual training
    residual_target: Optional[torch.Tensor]  # (b, c, t, x) - residual target
    residual_denormalized: Optional[
        torch.Tensor
    ]  # (b, c, t, x) - denormalized residual
    background_label: Optional[torch.Tensor]  # (b, num_classes) - background label
    lag_steps: Optional[torch.Tensor]  # (b,) - lag steps for residual training


def offsets_to_batch_idx(offsets):
    """Convert 3D cumulative-end offsets to (batch, time) indices.

    offsets is (S, B, T) with cumulative ends.
    Returns index in [0, B*T) for each observation, ignoring sensor dimension.
    """
    S, B, T = offsets.shape
    bt_size = B * T

    offsets_flat = offsets.flatten()
    offsets_with_zero = torch.cat(
        [torch.tensor([0], device=offsets.device, dtype=offsets.dtype), offsets_flat]
    )
    sizes = offsets_with_zero.diff()  # num obs per group of sensor obs

    # Assign each group an index in [0, S*B*T), then map to [0, B*T) with mod
    window_indices = torch.arange(
        sizes.shape[0], dtype=torch.long, device=offsets.device
    )
    bt_indices = window_indices % bt_size

    return bt_indices.repeat_interleave(sizes)


@torch.compiler.disable
def split_by_sensor(
    obs: UnifiedObservation, target_sensor_ids: list[int]
) -> dict[int, UnifiedObservation]:
    """
    Slice a UnifiedObservation into per-sensor sub-objects using its precomputed offsets.

    Args:
        obs: UnifiedObservation
        target_sensor_ids: list of int sensor IDs to extract

    Returns:
        dict[int, UnifiedObservation]: mapping from sensor_id -> sliced UnifiedObservation.
                                       If a sensor_id has no data, returns an empty slice
                                       (same structure, 0 rows).
    """
    if obs.offsets is None or obs.sensor_id_to_local is None:
        raise ValueError("offsets is required for split_by_sensor")

    out = {}
    offsets = obs.offsets  # [S,B,T]
    sensor_id_to_local = obs.sensor_id_to_local  # [max_sensor_id+1]

    device = obs.obs.device
    B, T = obs.batch_dims
    total_obs = obs.obs.shape[0]

    obs_count = 0
    for sensor_id in target_sensor_ids:
        if sensor_id < len(sensor_id_to_local):
            s_local = sensor_id_to_local[sensor_id].item()
        else:
            s_local = -1

        if s_local < 0:
            # Not active -> return zero-length slice
            start = end = 0
            sensor_offsets = torch.zeros((1, B, T), dtype=offsets.dtype, device=device)
        else:
            # Each sensor's last cumulative offset is total rows for that sensor
            end = offsets[s_local, -1, -1].item()
            # Adjust offsets to be relative to this sensor's start
            start = 0 if s_local == 0 else offsets[s_local - 1, -1, -1].item()
            sensor_offsets = offsets[s_local : s_local + 1] - start

        if not (0 <= start <= total_obs and start <= end <= total_obs):
            raise ValueError(
                f"Invalid offsets for sensor {sensor_id}: start={start}, end={end}, "
                f"total_obs={total_obs}."
            )
        length = end - start

        def _narrow(x):
            return (
                torch.narrow(x, 0, start, length) if length > 0 else x.narrow(0, 0, 0)
            )

        # single-sensor map: only this sensor maps to local idx 0
        single_sensor_map = torch.full(
            (sensor_id + 1,), -1, dtype=torch.int32, device=device
        )
        single_sensor_map[sensor_id] = 0

        out[sensor_id] = UnifiedObservation(
            obs=_narrow(obs.obs),
            time=_narrow(obs.time),
            float_metadata=_narrow(obs.float_metadata),
            int_metadata=_narrow(obs.int_metadata),
            hpx_level=obs.hpx_level,
            offsets=sensor_offsets,  # (1, B, T) relative to sliced data
            batch_idx=_narrow(obs.batch_idx),  # Also narrow batch_idx
            sensor_id_to_local=single_sensor_map,
        )
        obs_count += length

    return out
