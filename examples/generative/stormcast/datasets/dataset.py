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

import numpy as np
import torch


class StormCastDataset(torch.utils.data.Dataset, ABC):
    """An abstract class that defines the interface for StormCast datasets."""

    @abstractmethod
    def background_channels(self) -> list[str]:
        """Metadata for the background channels. A list of channel names, one for each channel"""
        pass

    @abstractmethod
    def state_channels(self) -> list[str]:
        """Metadata for the state channels. A list of channel names, one for each channel"""
        pass

    @abstractmethod
    def image_shape(self) -> tuple[int, int]:
        """Get the (height, width) of the data."""
        pass

    def latitude(self) -> np.ndarray:
        return np.full(self.image_shape(), np.nan)

    def longitude(self) -> np.ndarray:
        return np.full(self.image_shape(), np.nan)

    def normalize_background(self, x: np.ndarray) -> np.ndarray:
        """Convert background from physical units to normalized data."""
        return x

    def denormalize_background(self, x: np.ndarray) -> np.ndarray:
        """Convert background from normalized data to physical units."""
        return x

    def normalize_state(self, x: np.ndarray) -> np.ndarray:
        """Convert state from physical units to normalized data."""
        return x

    def denormalize_state(self, x: np.ndarray) -> np.ndarray:
        """Convert state from normalized data to physical units."""
        return x

    def get_invariants(self) -> np.ndarray | None:
        """Return invariants used for training, or None if no invariants are used."""
        return None


def worker_init(wrk_id):
    np.random.seed(torch.utils.data.get_worker_info().seed % (2**32 - 1))
