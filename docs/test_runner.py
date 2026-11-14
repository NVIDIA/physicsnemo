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
import subprocess

import pytest

# Collecting all the Python files in the scripts directory
script_files = [f for f in os.listdir("test_scripts/") if f.endswith(".py")]

# print(script_files)
@pytest.mark.parametrize("script_file", script_files)
def test_script_execution(script_file):
    """Test if a script runs without error."""
    filepath = os.path.join("test_scripts/", script_file)
    print(filepath)
    result = subprocess.run(["python", filepath], capture_output=True, text=True)

    # Check that the script executed successfully
    assert (
        result.returncode == 0
    ), f"Script {script_file} failed with error:\n{result.stderr}"
