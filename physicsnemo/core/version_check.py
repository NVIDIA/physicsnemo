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


"""
Utilities for version compatibility checking.

This is used to provide a uniform and consistent way to check for missing
packages, when not all packages are required for the base physicsnemo
install.  Additionally, for some packages (it's not mandatory to do this),
we have a registry of packages -> install tip that is used
to provide a helpful error message.
"""

import functools
from importlib import metadata
from typing import Optional

from packaging.version import parse


@functools.lru_cache(maxsize=None)
def get_installed_version(distribution_name: str) -> Optional[str]:
    """
    Return the installed version for a given distribution without importing it.
    Uses importlib.metadata to avoid heavy import-time side effects.
    Cached for repeated lookups.
    """

    # First, try exact match:
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        pass

    # Some packages have only partial matches, like `cupy`
    for dist in metadata.distributions():
        name = dist.metadata["Name"].lower()
        if name.startswith(distribution_name):
            return dist.version

    return None


def check_version_spec(
    distribution_name: str,
    spec: str = "0.0.0",
    *,
    error_msg: Optional[str] = None,
    hard_fail: bool = False,
) -> bool:
    """
    Check whether the installed distribution satisfies a PEP 440 version specifier.

    Args:
        distribution_name: Distribution (package) name as installed by pip
        spec: version specifier (e.g., '2.4') (Not PEP 440 to allow dev versions, etc.)
        error_msg: Optional custom error message
        hard_fail: Whether to raise an ImportError if the version requirement is not met
    Returns:
        True if version requirement is met; False if not and hard_fail=False

    Raises:
        ImportError: If package is not installed or requirement not satisfied (and hard_fail=True)
    """
    installed = get_installed_version(distribution_name)
    if installed is None:
        if hard_fail:
            raise ImportError(
                f"Package '{distribution_name}' is required but not installed."
            )
        else:
            return False

    ok = parse(installed) >= parse(spec)
    if not ok:
        msg = (
            error_msg
            or f"{distribution_name} {spec} is required, but found {installed}"
        )
        if hard_fail:
            raise ImportError(msg)
        return False

    return True


def require_version_spec(package_name: str, spec: str = "0.0.0"):
    """
    Decorator that checks a package version requirement before function execution.

    Args:
        package_name: Name of the package to check
        spec: Minimum version required (e.g., '2.4')

    Returns:
        Decorator function that checks version requirement before execution

    Example:
        @require_version("torch", "2.3")
        def my_function():
            # This function will only execute if torch >= 2.3
            pass
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            check_version_spec(package_name, spec, hard_fail=True)
            return func(*args, **kwargs)

        return wrapper

    return decorator
