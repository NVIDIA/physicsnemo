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
"""HealDA data loading pipeline.

Provides the complete data pipeline for HealDA training: observation loading,
ERA5 state loading, two-stage transforms (CPU + GPU), distributed sampling,
and background CUDA prefetching.

Key entry points:

- :class:`ObsERA5Dataset` — map-style dataset combining ERA5 state + observations
- :class:`UFSUnifiedLoader` — parquet-based observation loader
- :class:`ERA5ObsTransform` — two-stage transform with Triton feature kernels
- :func:`prefetch_map` — background CUDA stream prefetching
- :class:`RestartableDistributedSampler` — stateful distributed sampler with checkpoint support

Protocols for custom loaders/transforms:

- :class:`ObsLoader` — async observation loading interface
- :class:`Transform` — CPU-side batch transform
- :class:`DeviceTransform` — GPU-side batch transform
"""

from physicsnemo.experimental.datapipes.healda.dataset import (
    ObsERA5Dataset,
    identity_collate,
)
from physicsnemo.experimental.datapipes.healda.prefetch import prefetch_map
from physicsnemo.experimental.datapipes.healda.protocols import (
    DeviceTransform,
    ObsLoader,
    Transform,
)
from physicsnemo.experimental.datapipes.healda.samplers import (
    RestartableDistributedSampler,
)
from physicsnemo.experimental.datapipes.healda.types import (
    Batch,
    BatchInfo,
    TimeUnit,
    UnifiedObservation,
    VariableConfig,
    empty_batch,
    split_by_sensor,
)

__all__ = [
    # Dataset
    "ObsERA5Dataset",
    # Protocols
    "ObsLoader",
    "Transform",
    "DeviceTransform",
    # Types
    "UnifiedObservation",
    "Batch",
    "BatchInfo",
    "VariableConfig",
    "TimeUnit",
    "empty_batch",
    "split_by_sensor",
    # Infrastructure
    "prefetch_map",
    "RestartableDistributedSampler",
    "identity_collate",
]
