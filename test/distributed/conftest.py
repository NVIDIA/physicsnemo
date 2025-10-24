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

There are two modes of distributed testing in PhysicsNeMo.

@pytest.mark.multigpu_dynamic
-----------------------------
The first, and conceptually simplest/most flexible, is to use a dynamic process
pool in a multi gpu environment.  This workflow will spawn one process
per GPU, initialize the distributed utilities, run whatever tests are 
needed, and then tear down the distributed environment and destroy the processes.
Though it is simple conceptually, it is also very slow: initializing
processes and setting up the enviroment takes time.  So, use this
mode when necessary but use the other mode when possible.

These tests will run with `py.test --multigpu-dynamic [...]`

@pytest.mark.multigpu_static
----------------------------
A much faster mode of testing is to spawn one pytest process
per GPU, set up the distributed environment, and then run tests
as usual.  In this mode, there is nearly no overhead from pytest itself
and process spawning/startup.  However, take care when developing
tests in this mode because they are ALL collective tests.  For example,
if rank == 0:
  assert False
will hang your entire test suite...

These test run like `torchrun  --nproc-per-node 8 -m pytest --multigpu-static [...]`
"""


@pytest.fixture(scope="session", autouse=False)
def distributed_mesh(request):
    """Initialize the domain-parallel mesh once per test session when distributed_static marker is used"""

    # Setup
    mesh = DistributedManager().initialize_mesh([-1], ["domain"])
    yield mesh


@pytest.fixture(scope="session", autouse=False)
def distributed_mesh_2d(request):
    """Initialize the 2D mesh once per test session when distributed_static marker is used"""

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
