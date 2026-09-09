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
This package contains the implementation of the Deep Learning Weather Prediction (DLWP) recurrent UNet on the HEALPix mesh.
It handles the forward pass of the model, the backward pass, and the initialization of the hidden states.

The main classes are:
- HEALPixRecUNet: The main class for the DLWP recurrent UNet on the HEALPix mesh.
- HEALPixUNet: The main class for the DLWP UNet on the HEALPix mesh.
"""

from .HEALPixRecUNet import HEALPixRecUNet
from .HEALPixUNet import HEALPixUNet

__all__ = ["HEALPixRecUNet", "HEALPixUNet"]
