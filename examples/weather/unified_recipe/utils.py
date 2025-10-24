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

# Utils for unified training recipe

import os
import fsspec
import zarr
import numpy as np


def get_filesystem(
    type: str,  # "file" or "s3"
    key: str = None,
    endpoint_url: str = None,
    region_name: str = None,
):
    """
    Get filesystem object based on the type

    Parameters
    ----------
    type : str
        Type of filesystem. Currently supports "file" and "s3"
    key : str
        Key for s3
    endpoint_url : str
        Endpoint url for s3
    region_name : str
        Region name for s3

    Returns
    -------
    fs : fsspec.filesystem
        Filesystem object
    """

    if type == "file":
        fs = fsspec.filesystem("file")
    elif type == "s3":
        fs = fsspec.filesystem(
            "s3",
            key=key,
            secret=os.environ["AWS_SECRET_ACCESS_KEY"],
            client_kwargs={
                "endpoint_url": endpoint_url,
                "region_name": region_name,
            },
        )
    else:
        raise ValueError(f"Unknown type: {type}")

    return fs
