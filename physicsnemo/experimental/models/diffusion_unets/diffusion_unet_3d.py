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

"""Backward-compatible import shim for :class:`DiffusionUNet3D`."""

import warnings

from physicsnemo.models.diffusion_unets.diffusion_unet_3d import DiffusionUNet3D

warnings.warn(
    "`physicsnemo.experimental.models.diffusion_unets.DiffusionUNet3D` has "
    "moved to `physicsnemo.models.diffusion_unets.DiffusionUNet3D`.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DiffusionUNet3D"]
