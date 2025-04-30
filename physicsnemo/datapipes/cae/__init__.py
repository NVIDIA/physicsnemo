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

from physicsnemo.utils.version_check import check_package_installed

# This could and should be reoragnized into meta-data level requirements on pipelines.
domino_datapipe_requirements = ["warp", "scipy"]
mesh_datapipe_requirements = ["vtk"]


def __getattr__(name):
    """
    This file is meant to provide information
    """
    if name == "DoMINODataPipe":
        missing = [
            p for p in domino_datapipe_requirements if not check_package_installed(p)
        ]

        if missing:
            raise ImportError(
                f"Cannot import DoMINODataPipe: Missing required packages: {', '.join(missing)}"
            )
        else:
            from .domino_datapipe import DoMINODataPipe

            return DoMINODataPipe

    if name == "MeshDatapipe":
        missing = [
            p for p in mesh_datapipe_requirements if not check_package_installed(p)
        ]
        if missing:
            raise ImportError(
                f"Cannot import MeshDatapipe: Missing required packages: {', '.join(missing)}"
            )
        else:
            from .mesh_datapipe import MeshDatapipe

            return MeshDatapipe

    raise AttributeError(
        f"module 'physicsnemo.datapipes.cae' has no attribute '{name}'"
    )
