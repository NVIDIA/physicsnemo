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
This code processes voxel data stored in .h5 files to compute global
mean and standard deviation, for various data fields. It identifies
all .h5 files in a directory, processes each file to accumulate statistics for
specific fields (like coordinates and pressure), and then aggregates the results
across all files. The code supports parallel processing to handle multiple files
simultaneously, speeding up the computation. Finally, the global statistics are
saved to a JSON file.
"""

import os
import sys
import h5py
import numpy as np
import json
import hydra

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from hydra import to_absolute_path
from omegaconf import DictConfig

# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from utils import find_h5_files


def process_file(h5_file):
    """
    Processes a single .h5 file to compute the sum, sum of squares, and count for each variable.
    """
    print(h5_file)
    with h5py.File(h5_file, "r") as hf:
        data = hf["data"][:]
        nan_mask = np.isnan(data)
        sum_data = np.mean(data, axis=(1, 2, 3), where=~nan_mask)
        sum_squares = np.mean(data**2, axis=(1, 2, 3), where=~nan_mask)

    return sum_data, sum_squares


def aggregate_results(results):
    """
    Aggregates the results from all files to compute global mean and standard deviation.
    """
    total_sum = None
    total_sum_squares = None
    total_count = 0

    for sum_data, sum_squares in results:
        if total_sum is None:
            total_sum = np.zeros(sum_data.shape)
            total_sum_squares = np.zeros(sum_squares.shape)

        total_sum += sum_data
        total_sum_squares += sum_squares
        total_count += 1

    global_mean = total_sum / total_count
    global_variance = (total_sum_squares / total_count) - (global_mean**2)
    global_std = np.sqrt(global_variance)

    return global_mean, global_std


def compute_global_stats(h5_files, num_workers=4):
    """
    Computes the global mean and standard deviation for each variable across all .h5 files
    using parallel processing.
    """
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_file, h5_files),
                total=len(h5_files),
                desc="Processing H5 Files",
                unit="file",
            )
        )

    # Aggregate the results from all files
    global_mean, global_std = aggregate_results(results)

    return global_mean, global_std


def save_stats_to_json(mean, std_dev, output_file):
    """
    Saves the global mean and standard deviation to a JSON file.
    """
    stats = {
        "mean": mean.tolist(),  # Convert numpy arrays to lists
        "std_dev": std_dev.tolist(),  # Convert numpy arrays to lists
    }

    with open(output_file, "w") as f:
        json.dump(stats, f, indent=4)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    data_path = to_absolute_path(
        cfg.partitions_path
    )  # Directory containing the .bin graph files with partitions
    output_file = to_absolute_path(cfg.stats_file)  # File to save the global statistics

    # Find all .h5 files in the directory
    h5_files = find_h5_files(data_path)

    # Compute global statistics with parallel processing
    global_mean, global_std = compute_global_stats(
        h5_files, num_workers=cfg.num_preprocess_workers
    )

    # Save statistics to a JSON file
    save_stats_to_json(global_mean, global_std, output_file)

    # Print the results
    print("Global Mean:", global_mean)
    print("Global Standard Deviation:", global_std)
    print(f"Statistics saved to {output_file}")


if __name__ == "__main__":
    main()
