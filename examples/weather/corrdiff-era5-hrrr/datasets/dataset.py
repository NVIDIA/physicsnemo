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

from abc import ABC, abstractmethod

import numpy as np
import torch


class DownscalingDataset(torch.utils.data.Dataset, ABC):
    """An abstract class that defines the interface for downscaling datasets.

    All datasets must inherit from this class and define the methods marked as
    @abstractmethod. The other methods have default implementations, which a
    dataset can override, for example to provide a normalization scheme.

    Besides the methods below, all datasets must also define the following:
    - `__init__`, which should accept a `train` argument indicating whether the
        dataset is for training or validation, and the dataset parameters from
        the dataset config as keyword arguments
    - `__len__`, which should return the number of samples in the dataset
    - `__getitem__`, which should return a dictionary containing the following keys:
        - `background`: a numpy.ndarray of shape `(num_channels_background, height, width)`
            containing the low-resolution conditioning interpolated to the
            high-resolution grid
        - `state`: a numpy.ndarray of shape `(num_channels_state, height, width)`
            containing the high-resolution target
        - `scalar_conditions` (optional): a numpy.ndarray of shape
            `(num_channels_scalar_conditions,)` containing conditions that apply to
            the entire sample (for example, the time of year), named by
            `scalar_condition_channels()`
        `__getitem__` should return normalized outputs. The training loop does not
        normalize them, for performance reasons.

    `inference.py` also needs a `times` attribute, a numpy.datetime64 array with
    the time of each sample, and uses `latitude()` and `longitude()` for the
    coordinates of the saved samples.

    See `hrrr_era5.py` for an example dataset.
    """

    @abstractmethod
    def background_channels(self) -> list[str]:
        """Metadata for the background channels. A list of channel names, one for each channel"""
        pass

    @abstractmethod
    def state_channels(self) -> list[str]:
        """Metadata for the state channels. A list of channel names, one for each channel"""
        pass

    def scalar_condition_channels(self) -> list[str]:
        """Metadata for the scalar condition channels. A list of channel names, one for each channel"""
        return []

    @abstractmethod
    def image_shape(self) -> tuple[int, int]:
        """Get the (height, width) of the data."""
        pass

    def latitude(self) -> np.ndarray:
        """Latitude of each pixel, shape `image_shape()`."""
        return np.full(self.image_shape(), np.nan)

    def longitude(self) -> np.ndarray:
        """Longitude of each pixel, shape `image_shape()`."""
        return np.full(self.image_shape(), np.nan)

    def denormalize_background(self, x: np.ndarray) -> np.ndarray:
        """Convert background from normalized data to physical units."""
        return x

    def denormalize_state(self, x: np.ndarray) -> np.ndarray:
        """Convert state from normalized data to physical units."""
        return x


def worker_init(wrk_id):
    """Give each dataloader worker its own NumPy random state."""
    np.random.seed(torch.utils.data.get_worker_info().seed % (2**32 - 1))
