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
This code defines a custom dataset class H5Dataset for loading and
normalizing data stored in .h5 files. The dataset is initialized with
a list of file paths and global statistics (mean and standard deviation)
for normalizing the data. The data is normalized using z-score normalization,
and NaN values can be replaced with zeros. The code also provides a function
create_dataloader to create a PyTorch DataLoader for efficient batch loading
with configurable parameters such as batch size, number of workers, and
prefetching. This setup is ideal for handling large datasets stored in .h5
files while leveraging parallel data loading for efficiency.
"""

import os
import sys
import json
import h5py
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader


# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from utils import find_h5_files


class H5Dataset(Dataset):  # TODO: Use a Dali datapipe for better performance
    """
    Custom dataset class for loading

    Parameters:
    ----------
        file_list (list of str): List of paths to .h5 files.
        mean (np.ndarray): Global mean for normalization.
        std (np.ndarray): Global standard deviation for normalization.
    """

    def __init__(self, file_list, mean, std, nan_to_0=True):
        self.file_list = file_list
        self.mean = mean
        self.std = std
        self.nan_to_0 = nan_to_0

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        with h5py.File(file_path, "r") as hf:
            data = hf["data"][:]

        # Normalize data using z-score
        data = (data - self.mean[:, None, None, None]) / self.std[:, None, None, None]

        # Convert to PyTorch tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Replace nan with zeros
        if self.nan_to_0:
            data_tensor = torch.nan_to_num(data_tensor, nan=0.0)

        return data_tensor


def create_dataloader(
    file_list,
    mean,
    std,
    sampler,
    nan_to_0=True,
    batch_size=1,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
):
    """
    Creates a DataLoader for the H5Dataset with prefetching.

    Args:
        file_list (list of str): List of paths to .h5 files.
        mean (np.ndarray): Global mean for normalization.
        std (np.ndarray): Global standard deviation for normalization.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of worker processes for data loading.
        pin_memory (bool): If True, the data loader will copy tensors into CUDA pinned memory.
        prefetch_factor (int): Number of samples to prefetch.

    Returns:
        DataLoader: Configured DataLoader for the dataset.
    """
    dataset = H5Dataset(file_list, mean, std, nan_to_0)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        sampler=sampler,
    )
    return dataloader


if __name__ == "__main__":
    data_path = "drivaer_aws_h5"
    stats_file = "global_stats.json"

    # Load global statistics
    with open(stats_file, "r") as f:
        stats = json.load(f)
    mean = np.array(stats["mean"])
    std = np.array(stats["std_dev"])

    # Find all .h5 files in the directory
    file_list = find_h5_files(data_path)

    # Create DataLoader
    dataloader = create_dataloader(
        file_list, mean, std, nan_to_0=True, batch_size=2, num_workers=1
    )

    # Example usage
    for batch in dataloader:
        print(batch.shape)  # Print batch shape
