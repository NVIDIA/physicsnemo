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
from datetime import timedelta
from enum import Enum
from typing import Any, Protocol

import numpy as np
import torch

from physicsnemo.experimental.models.healda import Domain


class TimeUnit(Enum):
    """Time units supported by the dataset.
    Values are the pandas frequency strings (offset aliases)"""

    HOUR = "h"
    DAY = "D"
    MINUTE = "min"
    SECOND = "s"

    def to_timedelta(self, steps: float) -> timedelta:
        return {
            TimeUnit.HOUR: timedelta(hours=steps),
            TimeUnit.DAY: timedelta(days=steps),
            TimeUnit.MINUTE: timedelta(minutes=steps),
            TimeUnit.SECOND: timedelta(seconds=steps),
        }[self]


@dataclasses.dataclass
class BatchInfo:
    """Metadata describing model output"""

    channels: list[str]
    time_step: int = 1  # Time (in units `time_unit`) between consecutive frames
    time_unit: TimeUnit = TimeUnit.HOUR
    scales: Any | None = None
    center: Any | None = None

    def __post_init__(self):
        if isinstance(self.time_unit, str):
            raise ValueError("Time unit is an str. Should be a TimeUnit.")

    @staticmethod
    def loads(s):
        kw = json.loads(s)

        if "time_unit" in kw:
            kw["time_unit"] = TimeUnit(kw["time_unit"])

        # Ignore deprecated residual normalization field if present
        kw.pop("residual_normalization", None)

        return BatchInfo(**kw)

    def asdict(self):
        """Return a dictionary representation of the BatchInfo, suitable for JSON serialization."""
        out = {}
        out["channels"] = self.channels
        out["time_step"] = self.time_step
        out["time_unit"] = self.time_unit.value  # TimeUnit is always a TimeUnit enum
        # Convert numpy arrays to lists for JSON compatibility
        if self.scales is not None:
            out["scales"] = np.asarray(self.scales).tolist()
        else:
            out["scales"] = None
        if self.center is not None:
            out["center"] = np.asarray(self.center).tolist()
        else:
            out["center"] = None
        return out

    def sel_channels(self, channels: list[str]):
        channels = list(channels)
        index = np.array([self.channels.index(ch) for ch in channels])
        scales = None
        if self.scales is not None:
            scales = np.asarray(self.scales)[index]

        center = None
        if self.center is not None:
            center = np.asarray(self.center)[index]

        return BatchInfo(
            time_step=self.time_step,
            time_unit=self.time_unit,
            channels=channels,
            scales=scales,
            center=center,
        )

    def denormalize(self, x):
        scales = torch.as_tensor(self.scales).to(x)
        scales = scales.view(-1, 1, 1)

        center = torch.as_tensor(self.center).to(x)
        center = center.view(-1, 1, 1)
        return x * scales + center

    def get_time_delta(self, t: int) -> timedelta:
        """Gets time offset of the t-th frame in a frame sequence."""
        total_steps = t * self.time_step
        return self.time_unit.to_timedelta(total_steps)


@dataclasses.dataclass
class DatasetMetadata:
    name: str
    start: str
    end: str
    time_step: int  # time between successive data points in `time_unit`
    time_unit: TimeUnit

    @property
    def freq(self) -> str:
        return f"{self.time_step}{self.time_unit.value}"


@dataclasses.dataclass(frozen=True)
class VariableConfig:
    """Input variable set"""

    name: str
    variables_2d: list[str]
    variables_3d: list[str]
    levels: list[int]
    variables_static: list[str] = dataclasses.field(default_factory=list)


class SpatioTemporalDataset(Protocol):
    """Protocol for time-indexed gridded datasets."""

    @property
    def domain(self) -> Domain:
        pass

    def __len__(self) -> int:
        pass

    @property
    def num_channels(self) -> int:
        pass

    @property
    def condition_channels(self) -> int:
        pass

    @property
    def augment_channels(self) -> int:
        return 0

    @property
    def label_dim(self) -> int:
        return 0

    @property
    def time_length(self) -> int:
        pass

    @property
    def batch_info(self) -> BatchInfo:
        return BatchInfo(
            channels=[str(i) for i in range(self.num_channels)],
        )

    def metadata(self) -> Any:
        """Unstructured metadata about the dataset and the values it yields

        Can be used to save normalization constants, timestamps, channel names,
        config values, etc. The training code will avoid looking into this, but
        could be useful for inference.

        """
        return {}

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """

        Returns:
            image: shaped (num_channels, time_length, x)
            labels: shaped (label_dim,)
            condition: shaped (condition_channels, time_length, x)
        """
        pass
