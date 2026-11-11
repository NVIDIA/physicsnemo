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

"""Backward-compatible import shims for the 3D diffusion U-Net layers."""

import warnings

from physicsnemo.nn import Conv3D, GroupNorm3D, UNetAttention3D, UNetBlock3D

warnings.warn(
    "The 3D diffusion U-Net layers have moved from `physicsnemo.experimental.nn` "
    "to `physicsnemo.nn`.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Conv3D", "GroupNorm3D", "UNetAttention3D", "UNetBlock3D"]
