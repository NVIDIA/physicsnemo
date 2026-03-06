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

# Set before any import that might load HDF5 to avoid [Errno -101] NetCDF: HDF
# error when opening .nc on bind-mounted /workspace.
if "HDF5_USE_FILE_LOCKING" not in os.environ:
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Import netCDF4 before h5py (or other HDF5-using packages) so they share the
# same HDF5 linkage and avoid library version conflict -101 errors.
try:
    import netCDF4  # noqa: F401
except ImportError:
    pass

import importlib
import importlib.util
import pathlib
import random
from collections import defaultdict
from importlib import metadata

import numpy as np
import pytest
import torch
from packaging.requirements import Requirement
from packaging.version import Version

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
    nfs_data_dir_opt = request.config.getoption("--nfs-data-dir")
    test_data_dir_env = os.environ.get("TEST_DATA_DIR")
    if nfs_data_dir_opt:
        data_dir = pathlib.Path(nfs_data_dir_opt)
    elif test_data_dir_env:
        # get-data clones into $(TEST_DATA_DIR)/modulus-data
        data_dir = pathlib.Path(test_data_dir_env) / "modulus-data"
    else:
        data_dir = pathlib.Path(NFS_DATA_PATH)
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
            __import__("test.plugins.distributed_print", fromlist=[""]),
            name="distributed_print",
        )

        # And this one sets up distributed fixtures for static parallel tests.
        config.pluginmanager.register(
            __import__("test.plugins.distributed_fixtures", fromlist=[""]),
            name="distributed_fixtures",
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


def _check_requirement(spec):
    """
    Return True if the requirement is satisfied, False otherwise.

    Spec may be a plain module name (e.g. "zarr") or a name with version
    specifier (e.g. "zarr>=3.0.0"). Uses packaging.requirements.Requirement
    for parsing and importlib.metadata for the installed version.
    """
    req = Requirement(spec)
    module_name = req.name
    if importlib.util.find_spec(module_name) is None:
        return False
    if req.specifier:
        try:
            installed = metadata.version(module_name)
        except Exception:
            return False
        if Version(installed) not in req.specifier:
            return False
    return True


def requires_module(names):
    """
    Decorator to skip a test if *any* of the given modules are missing
    or do not satisfy the requested version.

    Accepts a single spec or a list/tuple of specs. Each spec may be a
    module name (e.g. ``"zarr"``) or a name with version specifier
    (e.g. ``"zarr>=3.0.0"``).
    """
    if isinstance(names, str):
        names = [names]

    skip = not all(_check_requirement(spec) for spec in names)
    return pytest.mark.skipif(skip, reason="")


@pytest.fixture(params=["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []))
def device(request):
    """Device fixture that automatically skips CUDA tests when not available."""
    return request.param


@pytest.fixture(autouse=True, scope="function")
def seed_random_state():
    """Reset all random number generators to a fixed seed before each test.

    This ensures test reproducibility and isolation - each test starts with
    identical RNG state regardless of test execution order or subset.

    Tests that need a specific seed can still call torch.manual_seed() etc.
    explicitly, which will override this fixture's seeding.
    """
    SEED = 95051

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # CUDA seeding (no-op if CUDA unavailable)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    yield


@pytest.fixture(autouse=True, scope="function")
def reset_dynamo_state():
    """Reset torch._dynamo state after each test.

    This ensures test isolation by cleaning up dynamo's compiled function cache
    and resetting configuration options like error_on_recompile. Without this,
    tests that set error_on_recompile=True can cause subsequent tests to fail
    when they trigger recompilation with different tensor shapes.
    """
    yield
    # Reset after test completes
    torch._dynamo.reset()
    torch._dynamo.config.error_on_recompile = False
