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
        "--multigpu", action="store_true", default=False, help="run multigpu tests"
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
    config.addinivalue_line("markers", "multigpu: mark test as multigpu to run")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--multigpu") and not config.getoption("-m"):
        skip_multigpu = pytest.mark.skip(reason="need --multigpu option to run")
        for item in items:
            if "multigpu" in item.keywords:
                item.add_marker(skip_multigpu)
