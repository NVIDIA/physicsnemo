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

import logging
from typing import Union

import torch

Tensor = torch.Tensor
logger = logging.getLogger("__name__")


def check_datapipe_device(sample: Tensor, device: Union[str, torch.device]) -> bool:
    """Checks if datapipe loads sample to correct device

    Parameters
    ----------
    sample : Tensor
        Torch tensor to check device on.
    device : str
        expected device to load too

    Returns
    -------
    bool
        Test passed
    """
    if isinstance(device, str):
        device = torch.device(device)
    # Need a index id if cuda
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    # Check if sample is on correct device
    if sample.device != device:
        logger.warning("Datapipe loading sample on incorrect device")
        logger.warning(f"Expected Device: {type(device)}")
        logger.warning(f"Device: {type(sample.device)}")
        return False
    return True
