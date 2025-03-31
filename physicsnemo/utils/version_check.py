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


"""
Utilities for version compatibility checking.

Specifically in use to prevent some newer physicsnemo modules from being used with
and older version of pytorch.

"""

import importlib
from typing import Optional

from packaging import version

# Dictionary mapping module paths to their version requirements
# This can be expanded as needed for different modules
VERSION_REQUIREMENTS = {
    "physicsnemo.distributed.shard_tensor": {"torch": "2.5.9"},
    "device_mesh": {"torch": "2.4.0"},
}


def check_min_version(
    package_name: str, min_version: str, error_msg: Optional[str] = None
) -> bool:
    """
    Check if an installed package meets the minimum version requirement.

    Args:
        package_name: Name of the package to check
        min_version: Minimum required version string (e.g. '2.6.0')
        error_msg: Optional custom error message

    Returns:
        True if version requirement is met

    Raises:
        ImportError: If package is not installed or version is too low
    """
    try:
        package = importlib.import_module(package_name)
        package_version = getattr(package, "__version__", "0.0.0")
    except ImportError:
        raise ImportError(f"Package {package_name} is required but not installed.")

    if version.parse(package_version) < version.parse(min_version):
        msg = (
            error_msg
            or f"{package_name} version {min_version} or higher is required, but found {package_version}"
        )
        raise ImportError(msg)

    return True


def check_module_requirements(module_path: str) -> None:
    """
    Check all version requirements for a specific module.

    Args:
        module_path: The import path of the module to check requirements for

    Raises:
        ImportError: If any requirement is not met
    """
    if module_path not in VERSION_REQUIREMENTS:
        return

    for package, min_version in VERSION_REQUIREMENTS[module_path].items():
        check_min_version(package, min_version)


def require_version(package_name: str, min_version: str):
    """
    Decorator that prevents a function from being called unless the
    specified package meets the minimum version requirement.

    Args:
        package_name: Name of the package to check
        min_version: Minimum required version string (e.g. '2.3')

    Returns:
        Decorator function that checks version requirement before execution

    Example:
        @require_version("torch", "2.3")
        def my_function():
            # This function will only execute if torch >= 2.3
            pass
    """

    def decorator(func):
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Verify the package meets minimum version before executing
            check_min_version(package_name, min_version)

            # If we get here, version check passed
            return func(*args, **kwargs)

        return wrapper

    return decorator
