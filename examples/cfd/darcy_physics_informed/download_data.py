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

import h5py
import numpy as np

from utils import download_FNO_dataset

download_FNO_dataset("Darcy_241", outdir="datasets/")

filename = "datasets/Darcy_241/piececonst_r241_N1024_smooth1.hdf5"

output_filenames = [
    "datasets/Darcy_241/train.hdf5",
    "datasets/Darcy_241/validation.hdf5",
]

# split_percentage = [80, 20]
split_percentage = [10, 10]

with (
    h5py.File(output_filenames[0], "w") as f_part1,
    h5py.File(output_filenames[1], "w") as f_part2,
):
    with h5py.File(filename, "r") as f:
        # Loop through all the datasets in the input file
        for key in f.keys():
            data = f[key][:]  # Load current dataset to memory

            # Calculate the split index based on the percentage
            split_idx = []
            for split in split_percentage:
                split_idx.append(int(len(data) * split / 100))

            # Split the data into two parts based on the calculated index
            data_part1, data_part2 = (
                data[: split_idx[0]],
                data[split_idx[0] : split_idx[0] + split_idx[1]],
            )

            # Save the data parts to the new files
            f_part1.create_dataset(key, data=data_part1)
            f_part2.create_dataset(key, data=data_part2)
