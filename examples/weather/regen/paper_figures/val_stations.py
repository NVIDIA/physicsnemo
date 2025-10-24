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
from utils import find_takeout, find_takeout_random
import xarray as xr
import numpy as np

stat_loc = xr.open_dataarray("figure_data/station_locations_on_grid.nc")
num_leave = 50
valid = []
# valid = np.load("more_random_val_stations.npy").tolist()
for _ in range(num_leave):
    bool_array = stat_loc.values.astype(bool)
    for indices in valid:
        bool_array[indices[0], indices[1]] = False
    valid.append(find_takeout_random(bool_array))
    print(len(valid))
np.save("figure_data/evenmore_random_val_stations", np.array(valid))
