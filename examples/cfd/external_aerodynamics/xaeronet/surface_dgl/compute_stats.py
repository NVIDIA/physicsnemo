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

"""
This code processes partitioned graph data stored in .bin files to compute global
mean and standard deviation,for various node and edge data fields. It identifies
all .bin files in a directory, processes each file to accumulate statistics for
specific fields (like coordinates and pressure), and then aggregates the results
across all files. The code supports parallel processing to handle multiple files
simultaneously, speeding up the computation. Finally, the global statistics are
saved to a JSON file.
"""

import os
import json
import numpy as np
import dgl
import hydra

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


def find_bin_files(data_path):
    """
    Finds all .bin files in the specified directory.
    """
    return [
        os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".bin")
    ]


def process_file(bin_file):
    """
    Processes a single .bin file containing graph partitions to compute the mean, mean of squares, and count for each variable.
    """
    graphs, _ = dgl.load_graphs(bin_file)

    # Initialize dictionaries to accumulate stats
    node_fields = ["coordinates", "normals", "area", "pressure", "shear_stress"]
    edge_fields = ["x"]

    field_means = {}
    field_square_means = {}
    counts = {}

    # Initialize stats accumulation for each partitioned graph
    for field in node_fields + edge_fields:
        field_means[field] = 0
        field_square_means[field] = 0
        counts[field] = 0

    # Loop through each partition in the file
    for graph in graphs:
        # Process node data
        for field in node_fields:
            if field in graph.ndata:
                data = graph.ndata[field].numpy()

                if data.ndim == 1:
                    data = np.expand_dims(data, axis=-1)

                # Compute mean, mean of squares, and count for each partition
                field_mean = np.mean(data, axis=0)
                field_square_mean = np.mean(data**2, axis=0)
                count = data.shape[0]

                # Accumulate stats across partitions
                field_means[field] += field_mean * count
                field_square_means[field] += field_square_mean * count
                counts[field] += count
            else:
                print(f"Warning: Node field '{field}' not found in {bin_file}")

        # Process edge data
        for field in edge_fields:
            if field in graph.edata:
                data = graph.edata[field].numpy()

                field_mean = np.mean(data, axis=0)
                field_square_mean = np.mean(data**2, axis=0)
                count = data.shape[0]

                field_means[field] += field_mean * count
                field_square_means[field] += field_square_mean * count
                counts[field] += count
            else:
                print(f"Warning: Edge field '{field}' not found in {bin_file}")

    return field_means, field_square_means, counts


def aggregate_results(results):
    """
    Aggregates the results from all files to compute global mean and standard deviation.
    """
    total_mean = {}
    total_square_mean = {}
    total_count = {}

    # Initialize totals with zeros for each field
    for field in results[0][0].keys():
        total_mean[field] = 0
        total_square_mean[field] = 0
        total_count[field] = 0

    # Accumulate weighted sums and counts
    for field_means, field_square_means, counts in results:
        for field in field_means:
            total_mean[field] += field_means[field]
            total_square_mean[field] += field_square_means[field]
            total_count[field] += counts[field]

    # Compute global mean and standard deviation
    global_mean = {}
    global_std = {}

    for field in total_mean:
        global_mean[field] = total_mean[field] / total_count[field]
        variance = (total_square_mean[field] / total_count[field]) - (
            global_mean[field] ** 2
        )
        global_std[field] = np.sqrt(
            np.maximum(variance, 0)
        )  # Ensure no negative variance due to rounding errors

    return global_mean, global_std


def compute_global_stats(bin_files, num_workers=4):
    """
    Computes the global mean and standard deviation for each field across all .bin files
    using parallel processing.
    """
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_file, bin_files),
                total=len(bin_files),
                desc="Processing BIN Files",
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
        "mean": {
            k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in mean.items()
        },
        "std_dev": {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in std_dev.items()
        },
    }

    with open(output_file, "w") as f:
        json.dump(stats, f, indent=4)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    data_path = to_absolute_path(
        cfg.partitions_path
    )  # Directory containing the .bin graph files with partitions
    output_file = to_absolute_path(cfg.stats_file)  # File to save the global statistics
    # Find all .bin files in the directory
    bin_files = find_bin_files(data_path)

    # Compute global statistics with parallel processing
    global_mean, global_std = compute_global_stats(
        bin_files, num_workers=cfg.num_preprocess_workers
    )

    # Save statistics to a JSON file
    save_stats_to_json(global_mean, global_std, output_file)

    # Print the results
    print("Global Mean:", global_mean)
    print("Global Standard Deviation:", global_std)
    print(f"Statistics saved to {output_file}")


if __name__ == "__main__":
    main()
