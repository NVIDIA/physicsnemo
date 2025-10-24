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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class ChannelMetadata:
    """Metadata describing a data channel."""

    name: str
    level: str = ""
    auxiliary: bool = False


class DownscalingDataset(torch.utils.data.Dataset, ABC):
    """Abstract class for dataset with downscaling paired data

    A DownscalingDataset has high-resolution output data (target) paired with
    low-resolution input data.
    """

    @abstractmethod
    def longitude(self) -> np.ndarray:
        """Get longitude values from the dataset."""
        pass

    @abstractmethod
    def latitude(self) -> np.ndarray:
        """Get latitude values from the dataset."""
        pass

    @abstractmethod
    def input_channels(self) -> List[ChannelMetadata]:
        """Metadata for the input channels. A list of ChannelMetadata, one for each channel"""
        pass

    @abstractmethod
    def output_channels(self) -> List[ChannelMetadata]:
        """Metadata for the output channels. A list of ChannelMetadata, one for each channel"""
        pass

    @abstractmethod
    def time(self) -> List:
        """Get time values from the dataset."""
        pass

    @abstractmethod
    def image_shape(self) -> Tuple[int, int]:
        """Get the (height, width) of the data (same for input and output)."""
        pass

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from physical units to normalized data."""
        return x

    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from normalized data to physical units."""
        return x

    def normalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from physical units to normalized data."""
        return x

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from normalized data to physical units."""
        return x

    def info(self) -> dict:
        """Get information about the dataset."""
        return {}
