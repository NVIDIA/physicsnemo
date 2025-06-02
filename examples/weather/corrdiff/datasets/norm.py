# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import numpy as np


def normalize(x, center, scale):
    """Normalize input data 'x' using center and scale values."""
    center = np.asarray(center)
    scale = np.asarray(scale)
    if not (center.ndim == 1 and scale.ndim == 1):
        raise ValueError("center and scale must be 1D arrays")
    return (x - center[np.newaxis, :, np.newaxis, np.newaxis]) / scale[
        np.newaxis, :, np.newaxis, np.newaxis
    ]


def denormalize(x, center, scale):
    """Denormalize input data 'x' using center and scale values."""
    center = np.asarray(center)
    scale = np.asarray(scale)
    if not (center.ndim == 1 and scale.ndim == 1):
        raise ValueError("center and scale must be 1D arrays")
    return (
        x * scale[np.newaxis, :, np.newaxis, np.newaxis]
        + center[np.newaxis, :, np.newaxis, np.newaxis]
    )
