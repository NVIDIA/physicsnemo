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
import glob
import numpy as np
import xarray as xr
import h5py
from scipy.sparse import coo_matrix
from collections import defaultdict


def processor(files, dest):
    # group files by year
    files_by_year = defaultdict(list)
    for file in files:
        basename = os.path.basename(file)
        year = basename.rsplit("_", 1)[0]
        files_by_year[year].append(file)

    input_map_wts = xr.open_dataset("./map_LL721x1440_CS64.nc")
    i = input_map_wts.row.values - 1
    j = input_map_wts.col.values - 1
    data = input_map_wts.S.values
    M = coo_matrix((data, (i, j)))

    results = {}
    # process files year by year
    for year, filenames in files_by_year.items():
        result_arrays = []
        filenames = sorted(filenames, key=lambda x: x[-4])
        for filename in filenames:
            with xr.open_dataset(filename) as ds:
                data_var_name = list(ds.data_vars)[0]
                # read the data variable and multiply by the matrix
                data = ds[data_var_name].values
                num_time = data.shape[0]
                result = np.reshape(
                    np.reshape(data, (num_time, -1)) * M.T, (num_time, 6, 64, 64)
                )
                result_arrays.append(result.astype(np.float32))

        # concatenate the arrays
        result_stack = np.stack(result_arrays, axis=1)
        results[year] = result_stack

    for year, result in results.items():
        print(year, result.shape)
        if not os.path.exists(dest):
            os.makedirs(dest)
        output_filename = dest + f"{year}.h5"
        print(output_filename)
        # store result in a HDF5 file
        with h5py.File(output_filename, "w") as hf:
            hf.create_dataset("fields", data=result)


processor(glob.glob("./data/train_temp/*.nc"), "./data/train/")
processor(glob.glob("./data/test_temp/*.nc"), "./data/test/")
processor(glob.glob("./data/out_of_sample_temp/*.nc"), "./data/out_of_sample/")
