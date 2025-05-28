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
from pathlib import Path

import zarr
from pytest_utils import import_or_fail


# Verifies that a Zarr file in a publicly accessible S3 bucket can be read using MSC (Multi-Storage Client).
# See the [Multi-Storage Client README](/examples/multi_storage_client/README.md) for further information.
@import_or_fail(["multistorageclient"])
def test_msc_read(pytestconfig):

    # Point at the MSC config file which specifies access information for the S3 bucket
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    os.environ["MSC_CONFIG"] = f"{current_dir}/msc_config_public_read.yaml"

    # Open a publicly accessible zarr file in an S3 bucket
    zarr_group = zarr.open(
        "msc://cmip6-pds/CMIP6/ScenarioMIP/NOAA-GFDL/GFDL-ESM4/ssp119/r1i1p1f1/day/tas/gr1/v20180701",
        mode="r",
    )

    # Verify the group has content
    assert len(zarr_group) > 0
