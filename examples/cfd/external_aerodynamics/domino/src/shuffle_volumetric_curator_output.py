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

import os
import multiprocessing as mp
from functools import partial

import numpy as np
import shutil

import zarr
from numcodecs import Blosc

"""
This script reads each zarr file from a specified directory, and copies the
data to the output directory.  For the keys "volume_fields" and "volume_mesh_centers",
the script will apply a permutation (aka shuffle) of those fields in tandem.

Since the datasets used are often very large, this script also applies
sharding to the output files which is a Zarr3 feature.

Therefore, zarr >= 3.0 is required.
"""


def check_file_completeness(input_file: str, output_file: str) -> bool:
    """
    Check if the output file exists and contains all required data from input file.
    """
    if not os.path.exists(output_file):
        return False

    in_file = zarr.open(input_file, mode="r")
    try:
        out_file = zarr.open(output_file, mode="r")
    except zarr.errors.PathNotFoundError:
        print(f"No output, returning False")
        return False

    # Check if all keys except 'filename' exist and have same shapes
    for key in in_file.keys():
        if key == "filename":
            continue
        if key not in out_file and key not in out_file.attrs:
            print(f"Key {key} not in output, returning False")
            return False
        if isinstance(in_file[key], zarr.Array):
            if key in out_file.attrs:
                continue
            if in_file[key].shape != out_file[key].shape:
                print(f"Key {key} shape mismatch, returning False")
                return False
    return True


def store_array(store, name: str, data: np.ndarray):
    # By default, chunk size is 10k points:
    chunk_size = (10_000,) + data.shape[1:]
    # By default, shard size is 2 million points:
    shard_size = (2_000_000,) + data.shape[1:]

    zarr.create_array(
        store=store,
        name=name,
        data=data,
        chunks=chunk_size,
        shards=shard_size,
        compressors="auto",
    )


def copy_file_with_shuffled_volume_data(
    input_file: str, output_file: str, random_seed: int | None = None
):
    """
    Copy a file with shuffled volume data, using Zarr v3 sharding for efficient storage.
    Only processes if the output file doesn't exist or is incomplete.
    """
    file_is_complete = check_file_completeness(input_file, output_file)
    if file_is_complete:
        print(f"Skipping {output_file} - already complete")
        return True

    print(f"Processing {input_file} -> {output_file}")

    # return False

    # if the output folder exists but isn't complete, purge it.
    # It's probably an interrupted conversion.
    if os.path.exists(output_file):
        shutil.rmtree(output_file)

    # return file_is_complete
    volume_keys = ["volume_fields", "volume_mesh_centers"]

    in_file = zarr.open(input_file, mode="r")

    # Create store with sharding configuration
    store = zarr.storage.LocalStore(output_file)
    root = zarr.group(store=store)

    # First copy all non-volume data
    for key in in_file.keys():
        if key not in volume_keys:
            if key == "filename":
                continue
            in_data = in_file[key]
            if in_data.shape != ():
                # For array data, use the same chunks as input but with sharding
                store_array(store, key, in_data[:])
            else:
                # Store scalar values as attributes
                root.attrs[key] = in_data[()]

    # Open and shuffle the volume data
    volume_fields = in_file["volume_fields"][:]
    volume_mesh_centers = in_file["volume_mesh_centers"][:]

    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate a permutation
    permutation = np.random.permutation(volume_fields.shape[0])

    # Shuffle the volume data
    shuffled_volume_fields = volume_fields[permutation]
    shuffled_volume_mesh_centers = volume_mesh_centers[permutation]

    store_array(store, "volume_fields", shuffled_volume_fields)
    store_array(store, "volume_mesh_centers", shuffled_volume_mesh_centers)

    print(f"Processed {output_file} - COMPLETE")
    return True


def process_file(file: str, top_dir: str, out_dir: str):
    """
    Process a single file, creating output directory if needed.
    """
    os.makedirs(out_dir, exist_ok=True)
    input_path = os.path.join(top_dir, file)
    output_path = os.path.join(out_dir, file)
    return copy_file_with_shuffled_volume_data(input_path, output_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Shuffle volumetric curator output")
    parser.add_argument("--input-dir", required=True, help="Input directory path")
    parser.add_argument("--output-dir", required=True, help="Output directory path")
    parser.add_argument(
        "--num-cores", type=int, default=64, help="Number of cores to use"
    )
    args = parser.parse_args()

    # Get list of files to process
    files = os.listdir(args.input_dir)

    # Create a partial function with fixed directories
    process_func = partial(
        process_file, top_dir=args.input_dir, out_dir=args.output_dir
    )

    # Use multiprocessing to process files in parallel
    num_cores = max(1, args.num_cores)  # Leave one core free
    print(f"Processing {len(files)} files using {num_cores} cores")

    with mp.Pool(num_cores) as pool:
        results = pool.map(process_func, files)
        print(f"Results: {results}")
        print(f"Total conversions: {sum(results)}")


if __name__ == "__main__":
    main()
