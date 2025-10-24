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


import netCDF4 as nc

# Open the NetCDF file
file_path = "image_outdir_0_score.nc"  # Replace with the path to your NetCDF file
dataset = nc.Dataset(file_path, "r")  # 'r' stands for read mode

# Access variables and attributes
print("Variables:")
for var_name, var in dataset.variables.items():
    print(f"{var_name}: {var[:]}")  # Access the data for each variable

print("\nGlobal attributes:")
for attr_name in dataset.ncattrs():
    print(f"{attr_name}: {getattr(dataset, attr_name)}")  # Access global attributes

# Close the NetCDF file when done
dataset.close()
