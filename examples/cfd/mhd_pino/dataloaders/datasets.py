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

import torch
import numpy as np
import yaml
from torch.utils import data
import os
import sys
import glob
import time
import h5py
import sys


class Dedalus2DDataset(data.Dataset):
    "Dataset for MHD 2D Dataset"

    def __init__(
        self,
        data_path,
        output_names="output-????",
        field_names=["magnetic field", "velocity"],
        num_train=None,
        num_test=None,
        use_train=True,
    ):
        self.data_path = data_path
        self.output_names = output_names
        raw_path = os.path.join(data_path, output_names, "*.h5")
        files_raw = sorted(glob.glob(raw_path))
        self.files_raw = files_raw
        self.num_files_raw = num_files_raw = len(files_raw)
        self.field_names = field_names
        self.use_train = use_train

        if (num_train is None) or (num_train > num_files_raw):
            num_train = num_files_raw
        self.num_train = num_train
        self.train_files = self.files_raw[:num_train]
        if (num_test is None) or (num_test > (num_files_raw - num_train)):
            num_test = num_files_raw - num_train
        self.num_test = num_test
        self.test_end = test_end = num_train + num_test
        self.test_files = self.files_raw[num_train:test_end]
        if (self.use_train) or (self.test_files is None):
            files = self.train_files
        else:
            files = self.test_files
        self.files = files
        self.num_files = num_files = len(files)

    def __len__(self):
        length = len(self.files)
        return length

    def __getitem__(self, index):
        "Gets item for dataloader"
        file = self.files[index]

        field_names = self.field_names
        fields = {}
        coords = []
        with h5py.File(file, mode="r") as h5file:
            data_file = h5file["tasks"]
            keys = list(data_file.keys())
            if field_names is None:
                field_names = keys
            for field_name in field_names:
                if field_name in data_file:
                    field = data_file[field_name][:]
                    fields[field_name] = field
                else:
                    print(f"field name {field_name} not found")
        dataset = fields
        return dataset

    def get_coords(self, index):
        "Gets coordinates of t, x, y for dataloader"
        file = self.files[index]
        with h5py.File(file, mode="r") as h5file:
            data_file = h5file["tasks"]
            keys = list(data_file.keys())
            dims = data_file[keys[0]].dims

            ndims = len(dims)
            t = dims[0]["sim_time"][:]
            x = dims[ndims - 2][0][:]
            y = dims[ndims - 1][0][:]
        return t, x, y
