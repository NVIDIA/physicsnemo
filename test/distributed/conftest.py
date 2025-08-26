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


import pytest

from physicsnemo.distributed import DistributedManager

"""
PhysicsNeMo Distributed Testing Utilities

This folder is meant to enable testing of distributed utilities, with some
restrictions.  This is a place to put unit tests that require a relatively 
simple distributed environment: every test uses the same environment set up,
including process groups and domain parallel sizes, etc.

By default, this is a 1D mesh with name "domain" to test domain parallel tools.

This isn't the place to test 2D parallelism, if that's needed do it in 
tests/spawn_distributed where the distributed environment is created fresh 
every time.
"""


@pytest.fixture(scope="session", autouse=False)
def distributed_mesh():
    """Initialize the domain-parallel mesh once per test session"""
    # Setup
    mesh = DistributedManager().initialize_mesh([-1], ["domain"])
    yield mesh


@pytest.fixture(scope="session", autouse=True)
def distributed_mesh_2d():
    """Initialize the 2D mesh once per test session"""

    # Divide the number of visible GPUs in 2 for the mesh calculation.
    # raise an exception if the number of GPUs isn't divisible
    dm = DistributedManager()
    num_gpus = dm.world_size

    if num_gpus % 2 != 0:
        raise ValueError(
            f"Number of GPUs ({num_gpus}) must be divisible by 2 for 2D mesh testing"
        )

    num_gpus_per_dim = num_gpus // 2

    # Create a mesh with the same number of GPUs per dimension
    mesh = dm.initialize_mesh([-1, num_gpus_per_dim], ["axis1", "axis2"])

    yield mesh


def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished, right before returning exit status"""

    if DistributedManager.is_initialized():
        DistributedManager.cleanup()
