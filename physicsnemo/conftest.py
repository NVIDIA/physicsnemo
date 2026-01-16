# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""
Conftest for physicsnemo package doctests.

This file provides hooks to gracefully handle doctest failures due to:
1. Missing optional dependencies at module import time (e.g., DALI, pyvista, vtk)
2. Missing optional dependencies within doctest examples (e.g., torch_geometric)
3. CUDA not being available when doctests require GPU

Usage:
    # Run doctests on all source files (skips files/tests with missing deps)
    uv run pytest --doctest-modules physicsnemo/

    # Run doctests on a specific directory
    uv run pytest --doctest-modules physicsnemo/datapipes/
"""

import importlib
import logging
import os

import pytest


def pytest_ignore_collect(collection_path, config):
    """
    Skip collection for source files that fail to import due to missing
    optional dependencies.

    When running doctests (--doctest-modules), pytest tries to import each module.
    If a module raises ImportError at import time (common for optional deps),
    this hook catches that and skips the file instead of failing collection.

    This allows running `pytest --doctest-modules` without having all optional
    dependencies installed.  It's not really meant for permanent use, instead
    this is a shim until we have full CI container support / env support.
    """
    # Only check .py files
    if collection_path.suffix != ".py":
        return None

    # Skip conftest files
    if collection_path.name == "conftest.py":
        return None

    # Only apply import checking when collecting doctests
    if not config.option.doctestmodules:
        return None

    # Get path relative to project root
    try:
        rel_path = collection_path.relative_to(config.rootpath)
        rel_str = str(rel_path)

        # Skip files in test directories
        if rel_str.startswith("test" + os.sep) or (os.sep + "test" + os.sep) in rel_str:
            return None

        # Skip experimental directory - it's excluded from linting and has known import issues
        if (os.sep + "experimental" + os.sep) in rel_str or rel_str.startswith(
            "physicsnemo" + os.sep + "experimental" + os.sep
        ):
            return True
    except ValueError:
        # Path is not relative to rootpath
        return None

    # Try to import the module to see if it has missing dependencies
    try:
        # Convert path to module name
        # e.g., physicsnemo/datapipes/cae/mesh_datapipe.py -> physicsnemo.datapipes.cae.mesh_datapipe
        if rel_path.name == "__init__.py":
            module_name = str(rel_path.parent).replace(os.sep, ".")
        else:
            module_name = str(rel_path.with_suffix("")).replace(os.sep, ".")

        # Skip if module name is empty or starts with a dot
        if not module_name or module_name.startswith("."):
            return None

        # Try importing the module
        importlib.import_module(module_name)
    except ImportError:
        # Module has missing dependencies - skip doctest collection for this file
        return True
    except Exception as e:
        # Other errors (syntax, etc.) - let pytest handle normally
        # Log at debug level for troubleshooting
        logging.getLogger(__name__).debug(
            f"Unexpected exception while checking imports for {collection_path}: {e}"
        )

    # Return None to let pytest's default collection handle this file
    return None


# Exception types that indicate missing optional dependencies or unavailable hardware
SKIPPABLE_EXCEPTIONS = (
    ImportError,
    ModuleNotFoundError,
)

# Error messages that indicate CUDA/GPU is not available
CUDA_UNAVAILABLE_MESSAGES = (
    "Torch not compiled with CUDA enabled",
    "CUDA is not available",
    "No CUDA GPUs are available",
    "cuda runtime error",
    "CUDA out of memory",
)


def _should_skip_doctest_exception(exc_value):
    """
    Check if an exception from a doctest should result in a skip.

    Returns a skip reason string if should skip, None otherwise.
    """
    # Handle UnexpectedException from doctest (wraps the actual exception)
    if hasattr(exc_value, "exc_info"):
        _, actual_exc, _ = exc_value.exc_info
        if actual_exc is not None:
            exc_value = actual_exc

    # Check for import/module errors (missing optional dependencies)
    if isinstance(exc_value, SKIPPABLE_EXCEPTIONS):
        module_name = getattr(exc_value, "name", str(exc_value))
        return f"missing optional dependency: {module_name}"

    # Check for CUDA unavailability errors
    exc_str = str(exc_value)
    if isinstance(exc_value, (AssertionError, RuntimeError)):
        for msg in CUDA_UNAVAILABLE_MESSAGES:
            if msg in exc_str:
                return f"CUDA not available: {msg}"

    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Convert doctest failures due to missing optional dependencies or
    unavailable CUDA into skipped tests instead of failures.

    This allows doctests to include examples that require optional dependencies
    (like torch_geometric) or CUDA without causing test failures when those
    dependencies aren't installed or hardware isn't available.
    """
    outcome = yield
    report = outcome.get_result()

    # Only process doctest items that failed during the call phase
    if report.when != "call" or report.outcome != "failed":
        return

    # Check if this is a doctest item
    if not hasattr(item, "dtest"):
        return

    # Check if the failure was due to a skippable exception
    if call.excinfo is not None:
        skip_reason = _should_skip_doctest_exception(call.excinfo.value)
        if skip_reason:
            report.outcome = "skipped"
            # longrepr for skipped tests must be a tuple of (filename, lineno, reason)
            # The reason should start with "Skipped: " for pytest to parse it correctly
            report.longrepr = (
                str(item.fspath),
                item.dtest.lineno,
                f"Skipped: {skip_reason}",
            )
