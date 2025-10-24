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

import os
import random
import shutil

# Define the directory that contains the original files
data_dir = "results"

# Define the directories for your train, validation, and test datasets
train_dir = "./dataset/train"
valid_dir = "./dataset/validation"
test_dir = "./dataset/test"

# Create directories if they do not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get all the files in the original directory
all_files = [
    f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))
]

# Shuffle the files
random.shuffle(all_files)

# Get the count of all files
all_files_count = len(all_files)

# Calculate the size of each dataset
train_size = int(all_files_count * 0.8)
valid_size = int(all_files_count * 0.1)
test_size = all_files_count - train_size - valid_size  # Ensure all files are used

# Split the files
train_files = all_files[:train_size]
valid_files = all_files[train_size : train_size + valid_size]
test_files = all_files[train_size + valid_size :]


# Function to copy files
def copy_files(files, dest_dir):
    for f in files:
        shutil.copy(os.path.join(data_dir, f), os.path.join(dest_dir, f))


# Copy the files
copy_files(train_files, train_dir)
copy_files(valid_files, valid_dir)
copy_files(test_files, test_dir)
