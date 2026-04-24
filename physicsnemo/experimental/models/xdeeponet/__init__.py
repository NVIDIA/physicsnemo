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

"""xDeepONet — the extended DeepONet family.

Config-driven assembly of eight DeepONet-based architectures sharing a
common branch/trunk/decoder pattern:

- ``deeponet``, ``u_deeponet``, ``fourier_deeponet``, ``conv_deeponet``,
  ``hybrid_deeponet`` — single-branch variants.
- ``mionet``, ``fourier_mionet`` — two-branch multi-input variants.
- ``tno`` — Temporal Neural Operator (branch2 = previous solution).

Both 2D and 3D spatial versions are provided.  :class:`DeepONetWrapper` and
:class:`DeepONet3DWrapper` are the recommended entry points; see their class
docstrings for usage examples and the branch/trunk configuration schema.
"""

from .branches import MLPBranch, SpatialBranch, SpatialBranch3D, TrunkNet
from .deeponet import DeepONet, DeepONet3D
from .wrappers import DeepONet3DWrapper, DeepONetWrapper

__all__ = [
    # Core architectures
    "DeepONet",
    "DeepONet3D",
    # Convenience wrappers (recommended entry points)
    "DeepONetWrapper",
    "DeepONet3DWrapper",
    # Building blocks
    "TrunkNet",
    "MLPBranch",
    "SpatialBranch",
    "SpatialBranch3D",
]
