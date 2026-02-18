# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Module that defines the classic two-moon classification dataset to
use as a demonstration dataset for a minimal active learning workflow.
"""

import torch
from torch.utils.data import Dataset

__all__ = ["MoonsDataset"]


def make_moons(n_samples: int = 2000, sigma: float = 0.25) -> torch.Tensor:
    """
    Make the classic two-moon dataset. Code was adapted from
    ``sklearn``, but modified slightly for the purposes of
    this particular example.

    Parameters
    ----------
    n_samples: int
        The number of samples to generate.

    Returns
    -------
    X_values: torch.Tensor
        The input features.
    y_values: torch.Tensor
        The target labels.
    """
    outer_grid = torch.linspace(0.0, torch.pi, int(n_samples * 0.7))
    inner_grid = torch.linspace(0.0, torch.pi, int(n_samples * 0.3))
    outer_x = torch.cos(outer_grid)
    outer_y = torch.sin(outer_grid)
    inner_x = 1 - torch.cos(inner_grid)
    inner_y = 1 - torch.sin(inner_grid) - 0.5
    outer = torch.stack([outer_x, outer_y], dim=-1)
    inner = torch.stack([inner_x, inner_y], dim=-1)
    X_values = torch.cat([outer, inner], dim=0)
    # add some noise to the coordinates
    X_values += torch.randn_like(X_values) * sigma
    y_values = torch.zeros(n_samples)
    y_values[outer_grid.shape[0] :] = 1
    return X_values, y_values


class MoonsDataset(Dataset):
    """
    Generate the classic two-moon dataset, repurposed for a minimal
    active learning example.

    This class implements the `DataPool` protocol by subclassing
    ``Dataset``, which provides all the methods except for ``append``,
    which we implement here.

    The intuition is to have one of the moons be data poor, and a quasi-
    intelligent query strategy will help overcome class imbalance to
    some extent, as it will hopefully have higher uncertainty in its
    classifier output to reflect this.

    Attributes
    ----------
    initial_samples: float
        The initial number of samples to hold out for training.
    total_samples: int
        The total number of samples to generate.
    train_indices: torch.LongTensor | None
        The indices of the samples to use for training.
    X_values: torch.Tensor
        The full set of input features; i.e. the coordinates
        of a point in 2D space.
    y_values: torch.Tensor
        The target labels; 0 for the outer moon, 1 for the inner moon.
    sigma: float
        The standard deviation of the noise to add to the coordinates.
    """

    def __init__(
        self,
        initial_samples: float = 0.05,
        total_samples: int = 1000,
        train_indices: torch.LongTensor | None = None,
        sigma: float = 0.25,
    ):
        super().__init__()
        self.initial_samples = initial_samples
        self.total_samples = total_samples
        # this holds the full dataset for training
        self.X_values, self.y_values = make_moons(total_samples, sigma)
        # this corresponds to the subset that is actually exposed
        # during training; it grows as we 'label' more samples
        if train_indices is None:
            # initial hold out for training
            train_indices = torch.randperm(total_samples)[
                : int(total_samples * initial_samples)
            ]
        self.train_indices = train_indices

    def __len__(self) -> int:
        """Return the length of the training subset."""
        return len(self.train_indices)

    def _sample_indices(self) -> torch.LongTensor:
        """Return the indices that are not currently in training."""
        all_indices = torch.arange(self.total_samples)
        mask = ~torch.isin(all_indices, self.train_indices)
        return all_indices[mask]

    def append(self, item: int) -> None:
        """Append a single index to the training set; needed for 'labeling'."""
        self.train_indices = torch.cat(
            [self.train_indices, torch.tensor([item])], dim=0
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve a single coordinate-label pair from the dataset."""
        actual_index = self.train_indices[index]
        x_val = self.X_values[actual_index, :]
        y_val = self.y_values[actual_index]
        return x_val, y_val
