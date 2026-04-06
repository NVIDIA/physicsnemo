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

"""
Neural operator model architectures.

Available models:
- FNO variants: UFNO, UFNONet, FNO4D, FNO4DNet
- DeepONet variants: DeepONet, DeepONet3D, DeepONetWrapper, DeepONet3DWrapper
- U-Net: UNet2D, UNet3D, PhysicsNemoUNet2D, PhysicsNemoUNet3D, StandaloneUNet
"""

from models.physicsnemo_unet import (
    PhysicsNemoUNet2D,
    PhysicsNemoUNet3D,
    StandaloneUNet,
)
from models.unet import UNet2D, UNet3D
from models.xdeeponet import (
    DeepONet,
    DeepONet3D,
    DeepONet3DWrapper,
    DeepONetWrapper,
    MLPBranch,
    SpatialBranch,
    SpatialBranch3D,
    TrunkNet,
)
from models.xfno import FNO4D, UFNO, FNO4DNet, UFNONet

__all__ = [
    # FNO
    "UFNO",
    "UFNONet",
    "FNO4D",
    "FNO4DNet",
    # DeepONet
    "TrunkNet",
    "MLPBranch",
    "SpatialBranch",
    "SpatialBranch3D",
    "DeepONet",
    "DeepONet3D",
    "DeepONetWrapper",
    "DeepONet3DWrapper",
    # U-Net
    "UNet2D",
    "UNet3D",
    "PhysicsNemoUNet2D",
    "PhysicsNemoUNet3D",
    "StandaloneUNet",
]
