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


class StormCastDataset(torch.utils.data.Dataset, ABC):
    """An abstract class that defines the interface for StormCast datasets.

    All datasets must inherit from this class and implement the methods marked as @abstractmethod.
    The other methods have default implementations and can be overridden by the dataset if needed,
    for example to provide a normalization scheme.

    In addition to the methods defined below, all datasets must also implement the following:
    - `__init__`, which should accept a `params` argument containing the dataset parameters
        and a `train` argument indicating whether the dataset is for training or validation
    - `__len__`, which should return the number of samples in the dataset
    - `__getitem__`, which should return a dictionary containing the following keys:
        - `background`: a numpy.ndarray or torch.Tensor of shape
            `(num_channels_background, height, width)` containing the background data
        - `state`: a 2-tuple of numpy.ndarray or torch.Tensor of shape
            `(num_channels_state, height, width)` with index 0 containing the input state data
            and index 1 containing the target state data
        - `lead_time_label` (optional): this must be returned if lead_time_steps > 0. A single
            integer indicating which lead time embedding should be used
        The outputs of __getitem__ should be already normalized (this is not done in the training
        loop for performance reasons).

    An example implementation of a dataset is given in `data_loader_hrrr_era5.py`.
    """

    lead_time_steps: int = 0  # number of lead time embeddings

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
        return np.full(self.image_shape(), np.nan)

    def longitude(self) -> np.ndarray:
        return np.full(self.image_shape(), np.nan)

    def normalize_background(
        self, x: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Convert background from physical units to normalized data."""
        return x

    def denormalize_background(
        self, x: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Convert background from normalized data to physical units."""
        return x

    def normalize_state(
        self, x: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Convert state from physical units to normalized data."""
        return x

    def denormalize_state(
        self, x: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Convert state from normalized data to physical units."""
        return x

    def get_invariants(self) -> np.ndarray | None:
        """Return invariants used for training, or None if no invariants are used."""
        return None


def worker_init(wrk_id):
    np.random.seed(torch.utils.data.get_worker_info().seed % (2**32 - 1))
