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

r"""Alpha diffusion distillation APIs with optional training backends."""

import importlib
import warnings

from physicsnemo.core.warnings import ExperimentalFeatureWarning

from .base import BaseDistillationLoss, LossMap
from .checkpoint import CheckpointStateBundle
from .losses import DistillationLoss, WeightedDistillationLoss

warnings.warn(
    "physicsnemo.diffusion.distillation is alpha. Its API may change without backward compatibility.",
    ExperimentalFeatureWarning,
    stacklevel=2,
)

__all__ = [
    "BaseDistillationLoss",
    "CheckpointStateBundle",
    "LossMap",
    "DistillationLoss",
    "WeightedDistillationLoss",
    "DistillationStrategy",
]


def __getattr__(name: str):
    r"""Load the optional training strategy only when requested."""
    if name == "DistillationStrategy":
        return importlib.import_module(f"{__name__}.strategy").DistillationStrategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
