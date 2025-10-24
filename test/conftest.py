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
import pathlib
from collections import defaultdict

import pytest

NFS_DATA_PATH = "/data/nfs/modulus-data"


# Total time per file
file_timings = defaultdict(float)


def pytest_runtest_logreport(report):
    if report.when == "call":
        # report.nodeid format: path::TestClass::test_name
        filename = report.nodeid.split("::")[0]
        file_timings[filename] += report.duration


def pytest_sessionfinish(session, exitstatus):
    print("\n=== Test durations by file ===")
    for filename, duration in sorted(
        file_timings.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"{filename}: {duration:.2f} seconds")


def pytest_addoption(parser):
    parser.addoption(
        "--multigpu-dynamic",
        action="store_true",
        default=False,
        help="run multigpu tests that require dynamic initialization",
    )
    parser.addoption(
        "--multigpu-static",
        action="store_true",
        default=False,
        help="run multigpu tests that can use static initialization",
    )
    parser.addoption(
        "--fail-on-missing-modules",
        action="store_true",
        default=False,
        help="fail tests if required modules are missing",
    )
    parser.addoption(
        "--nfs-data-dir", action="store", default=None, help="path to test data"
    )


@pytest.fixture(scope="session")
def nfs_data_dir(request):
    data_dir = pathlib.Path(
        request.config.getoption("--nfs-data-dir")
        or os.environ.get("TEST_DATA_DIR", NFS_DATA_PATH)
    )
    if not data_dir.exists():
        pytest.skip(
            "NFS volumes not set up with CI data repo. Run `make get-data` from the root directory of the repo"
        )
    print(f"Using {data_dir} as data directory")
    return data_dir


def pytest_configure(config):
    config.addinivalue_line("markers", "multigpu_dynamic: mark test as multigpu to run")
    config.addinivalue_line(
        "markers", "multigpu_static: mark test to run only with --multigpu-static flag"
    )

    # Conditionally register the distributed_print plugin for multigpu tests
    static_flag = config.getoption("--multigpu-static")

    if static_flag:
        # Initialize the distributed manager for static tests
        from physicsnemo.distributed import DistributedManager

        DistributedManager.initialize()
        # Only load the plugin when running distributed tests
        config.pluginmanager.register(
            __import__("plugins.distributed_print", fromlist=[""]),
            name="distributed_print",
        )


def pytest_collection_modifyitems(config, items):
    dynamic_flag = config.getoption("--multigpu-dynamic")
    static_flag = config.getoption("--multigpu-static")

    # Ensure options are mutually exclusive
    if dynamic_flag and static_flag:
        raise pytest.UsageError(
            "Cannot specify both --multigpu-dynamic and --multigpu-static flags"
        )

    # Skip tests based on which flag is provided
    if dynamic_flag:
        # Running dynamic tests, skip static tests
        skip_static = pytest.mark.skip(
            reason="skipping static and single-gpu tests when --multigpu-dynamic is specified"
        )
        for item in items:
            if "multigpu_dynamic" not in item.keywords:
                item.add_marker(skip_static)
    elif static_flag:
        # Running static tests, skip dynamic tests
        skip_dynamic = pytest.mark.skip(
            reason="skipping dynamic and single-gpu tests when --multigpu-static is specified"
        )
        for item in items:
            if "multigpu_static" not in item.keywords:
                item.add_marker(skip_dynamic)
    else:
        # No flags provided, skip all multigpu tests
        skip_all = pytest.mark.skip(
            reason="need either --multigpu-dynamic or --multigpu-static option to run"
        )
        for item in items:
            if (
                "multigpu_dynamic" in item.keywords
                or "multigpu_static" in item.keywords
            ):
                item.add_marker(skip_all)
