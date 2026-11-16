#!/bin/bash
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

# This is a Bash script designed to identify and remove corrupted files after downloading the AWS DrivAer dataset.
# The script defines two functions: check_and_remove_corrupted_extension and check_all_runs.
# The check_and_remove_corrupted_extension function checks for files in a given directory that have extra characters after their extension.
# If such a file is found, it is considered corrupted, and the function removes it.
# The check_all_runs function iterates over all directories in a specified local directory (LOCAL_DIR), checking for corrupted files with the extensions ".vtu", ".stl", and ".vtp".
# The script begins the cleanup process by calling the check_all_runs function. The target directory for this operation is set as "./drivaer_data_full".

# Set the local directory to check the files
LOCAL_DIR="./drivaer_data_full"  # <--- This is the directory where the files are downloaded.

# Function to check if a file has extra characters after the extension and remove it
check_and_remove_corrupted_extension() {
    local dir=$1
    local base_filename=$2
    local extension=$3

    # Find any files with extra characters after the extension
    for file in "$dir/$base_filename"$extension*; do
        if [[ -f "$file" && "$file" != "$dir/$base_filename$extension" ]]; then
            echo "Corrupted file detected: $file (extra characters after extension), removing it."
            rm "$file"
        fi
    done
}

# Function to go over all the run directories and check files
check_all_runs() {
    for RUN_DIR in "$LOCAL_DIR"/run_*; do
        echo "Checking folder: $RUN_DIR"

        # Check for corrupted .vtu files
        base_vtu="volume_${RUN_DIR##*_}"
        check_and_remove_corrupted_extension "$RUN_DIR" "$base_vtu" ".vtu"

        # Check for corrupted .stl files
        base_stl="drivaer_${RUN_DIR##*_}"
        check_and_remove_corrupted_extension "$RUN_DIR" "$base_stl" ".stl"

        # Check for corrupted .vtp files
        base_stl="drivaer_${RUN_DIR##*_}"
        check_and_remove_corrupted_extension "$RUN_DIR" "$base_stl" ".vtp"
    done
}

# Start checking
check_all_runs
