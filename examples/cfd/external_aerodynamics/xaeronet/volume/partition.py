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
This code partitions large data batches into smaller sub-batches, while
managing boundary regions (halos) and applying filters for physics-based
computations. It provides a function to partition a single batch and handle
inner and halo regions, and uses parallel processing to efficiently partition
multiple batches from a data loader simultaneously. This setup is particularly
useful for distributed or large-scale computations where handling boundaries
between partitions is critical for accuracy and performance.
"""

import numpy as np
import concurrent.futures


def partition_batch(batch, num_partitions, partition_width, halo_width):
    # Preallocate data list and filter list
    data = [None] * num_partitions
    filter = np.zeros((num_partitions, partition_width + 2 * halo_width), dtype=bool)
    phys_filter = np.zeros(
        (num_partitions, partition_width + 2 * halo_width), dtype=bool
    )

    # Handle first partition
    data[0] = batch[:, :, 0 : partition_width + 2 * halo_width, :, :]

    # Handle middle partitions
    for i in range(1, num_partitions - 1):
        start_idx = i * partition_width - halo_width
        end_idx = (i + 1) * partition_width + halo_width
        data[i] = batch[:, :, start_idx:end_idx, :, :]

    # Handle last partition
    data[num_partitions - 1] = batch[
        :, :, (num_partitions - 1) * partition_width - 2 * halo_width :, :, :
    ]

    # Create filter for inner nodes
    filter[0, 0:partition_width] = True
    filter[1 : num_partitions - 1, halo_width : partition_width + halo_width] = True
    filter[num_partitions - 1, 2 * halo_width :] = True

    # Create padded filters for physics loss
    phys_filter[0, 0 : partition_width + 1] = True
    phys_filter[
        1 : num_partitions - 1, halo_width - 1 : partition_width + halo_width + 1
    ] = True
    phys_filter[num_partitions - 1, 2 * halo_width - 1 :] = True

    return data, filter, phys_filter


# Function to process each batch (partitioning and filtering)
def process_batch(batch, num_partitions, partition_width, halo_width):
    data_i, filter_i, phys_filter_i = partition_batch(
        batch, num_partitions, partition_width, halo_width
    )
    return data_i, filter_i, phys_filter_i, batch


# Efficient processing of valid_dataloader batches using parallelism
def parallel_partitioning(
    dataloader, num_partitions=7, partition_width=100, halo_width=40
):
    data, filter, phys_filter, batch = [], [], [], []

    # Use ThreadPoolExecutor for CPU-bound tasks
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        # Submit tasks in parallel
        for batch_i in dataloader:
            futures.append(
                executor.submit(
                    process_batch, batch_i, num_partitions, partition_width, halo_width
                )
            )

        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            data_i, filter_i, phys_filter_i, batch_i = future.result()
            data.append(data_i)
            filter.append(filter_i)
            phys_filter.append(phys_filter_i)
            batch.append(batch_i)

    print("Partitioning completed")
    return data, filter, phys_filter, batch
