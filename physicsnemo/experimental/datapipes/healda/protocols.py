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
"""Protocols defining the extension points for HealDA data loading.

Custom data sources and transforms can be plugged into the HealDA pipeline by
implementing these protocols. No inheritance is required -- any class with the
right method signatures will satisfy the protocol.

See ``UFSUnifiedLoader`` for a reference ``ObsLoader`` implementation, and
``ERA5ObsTransform`` for a reference ``Transform`` / ``DeviceTransform``
implementation.
"""

from typing import Any, Protocol, runtime_checkable

import cftime
import pandas as pd
import torch


@runtime_checkable
class ObsLoader(Protocol):
    """Load observations for a set of timestamps.

    Implementations fetch observation data (satellite, conventional, or other)
    for the requested times and return it as a dictionary.  The method is async
    to allow concurrent I/O when composed with other loaders.

    The canonical return key is ``"obs"`` mapping to a ``list[pa.Table]``
    with one table per requested timestamp.  Custom loaders may use different
    keys as long as the downstream ``Transform`` expects them.

    Example::

        class MyObsLoader:
            async def sel_time(self, times):
                tables = [load_obs_for(t) for t in times]
                return {"obs": tables}
    """

    async def sel_time(self, times: pd.DatetimeIndex) -> dict[str, list[Any]]:
        """Load observation data for the given timestamps.

        Args:
            times: Timestamps to load observations for.

        Returns:
            Dictionary mapping field names to per-timestep data lists.
        """
        ...


class Transform(Protocol):
    """CPU-side batch transform, called inside DataLoader worker processes.

    Converts raw loaded frames (state arrays + observation tables) into an
    intermediate batch dictionary suitable for ``pin_memory`` and collation.
    Must NOT use CUDA.

    Args:
        times: ``list[list[cftime.datetime]]`` shaped ``(batch, time_per_sample)``.
        frames: ``list[list[dict]]`` shaped ``(batch, time_per_sample)``.
            Each inner dict has keys from the loaders (e.g. ``"state"``, ``"obs"``).

    Returns:
        Batch dict with tensors (``target``, ``condition``, time encodings, etc.).
    """

    def transform(
        self,
        times: list[list[cftime.datetime]],
        frames: list[list[dict[str, Any]]],
    ) -> dict[str, Any]: ...


class DeviceTransform(Protocol):
    """GPU-side transform, called in the ``prefetch_map`` background thread.

    Moves the CPU batch to the target device and performs GPU-accelerated
    featurization (e.g. observation metadata computation, HEALPix pixel lookup).

    Args:
        batch: Output of ``Transform.transform()``.
        device: Target ``torch.device``.

    Returns:
        Batch dict with all tensors on the target device.
    """

    def device_transform(
        self, batch: dict[str, Any], device: torch.device
    ) -> dict[str, Any]: ...
