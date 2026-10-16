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

from typing import Any

import numpy as np

from .dataset import DownscalingDataset


class MockDataset(DownscalingDataset):
    """A minimal mock dataset implementation for testing without real data.

    Args:
        num_state_channels: Number of channels in the state (target) data
        num_background_channels: Number of channels in the background data
        num_scalar_cond_channels: Number of scalar conditions
        image_size: Tuple of (height, width) for the images
        num_samples: Number of samples in the dataset
    """

    def __init__(
        self,
        num_state_channels: int = 3,
        num_background_channels: int = 4,
        num_scalar_cond_channels: int = 2,
        image_size: tuple[int, int] = (64, 32),
        num_samples: int = 20,
        train: bool = True,
    ):
        self._num_state_channels = num_state_channels
        self._num_background_channels = num_background_channels
        self._num_scalar_cond_channels = num_scalar_cond_channels
        self._image_size = tuple(image_size)
        self._num_samples = num_samples

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a sample with random data."""
        rng = np.random.default_rng(seed=idx)  # Use idx as seed for reproducibility
        (height, width) = self._image_size

        item = {
            "background": rng.normal(
                size=(self._num_background_channels, height, width)
            ).astype(np.float32),
            "state": rng.normal(size=(self._num_state_channels, height, width)).astype(
                np.float32
            ),
        }
        if self._num_scalar_cond_channels:
            item["scalar_conditions"] = rng.normal(
                size=(self._num_scalar_cond_channels,)
            ).astype(np.float32)
        return item

    def background_channels(self) -> list[str]:
        """Return metadata for background channels."""
        return [f"background_{i}" for i in range(self._num_background_channels)]

    def state_channels(self) -> list[str]:
        """Return metadata for state channels."""
        return [f"state_{i}" for i in range(self._num_state_channels)]

    def scalar_condition_channels(self) -> list[str]:
        """Return metadata for scalar condition channels."""
        return [f"scalar_cond_{i}" for i in range(self._num_scalar_cond_channels)]

    def image_shape(self) -> tuple[int, int]:
        """Return the (height, width) of the data."""
        return self._image_size
