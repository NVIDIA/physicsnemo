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

"""Configuration utilities for AirfRANS dataset paths."""

import platform
from pathlib import Path


def default_airfrans_data_dir() -> Path | None:
    """
    Get the AirfRANS dataset directory based on the current hostname.

    Returns:
        Path to the AirfRANS Dataset directory, or None if the hostname is not recognized.
    """
    hostname = platform.node()

    if hostname == "NV-pds":  # local
        return Path("/home/psharpe/gh/aerodynamics_datasets/airfrans/Dataset")
    elif hostname.endswith("eos.clusters.nvidia.com"):  # EOS
        return Path(
            "/lustre/fsw/coreai_modulus_cae/datasets/airfrans/Dataset"
        )
    elif hostname.startswith("nvl72"):  # OCI-HSG
        return Path(
            "/lustre/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/datasets/airfrans/Dataset"
        )
    else:
        return None
