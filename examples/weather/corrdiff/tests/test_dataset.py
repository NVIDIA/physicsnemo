# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# from training.dataset import _ZarrDataset, FilterTime
# import torch
# import datetime
# import numpy as np
# import os


# import pytest

# path = "/lustre/fsw/sw_climate_fno/nbrenowitz/2023-01-24-cwb-4years.zarr"


# @pytest.fixture
# def dataset():
#     if not os.path.isdir(path):
#         pytest.skip()
#     else:
#         return _ZarrDataset(path)


# def test_zarr_dataset(dataset):
#     dataset[0]
#     assert len(dataset) > 0
#     assert dataset[len(dataset) - 1]


# def test_zarr_dataset_channels(dataset):
#     assert dataset.input_channels()
#     assert dataset.output_channels()


# def test_zarr_dataset_time(dataset):
#     isinstance(dataset.time()[0].year, int)


# def test_zarr_dataset_get_valid_time_index(dataset):
#     ans = dataset._get_valid_time_index(0)
#     assert isinstance(ans, np.int64)

# def test_filter_time():
#     class MockData(torch.utils.data.Dataset):

#         def __getitem__(self, idx):
#             return self.time()[idx]

#         def time(self):
#             return [datetime.datetime(2018, 1, 1), datetime.datetime(1970, 1, 1)]

#     data = MockData()
#     filtered = FilterTime(data, lambda time: time.year > 1990)
#     assert filtered.time() == [datetime.datetime(2018, 1, 1)]
#     assert filtered[0]
