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

import importlib
from functools import wraps

import pytest
from packaging.version import Version


def import_or_fail(
    module_names: str | list[str] | tuple,
    min_versions: str | list[str] | tuple | None = None,
):
    """
    Try to import a module and skip the test if the module is not available
    or if the version is below the minimum required version.

    Args:
        module_names (str): Name of the modules to import.
        min_versions (str, optional): Minimum required versions of the modules.
    """

    def decorator(test_func):
        @pytest.mark.usefixtures("pytestconfig")
        @wraps(test_func)
        def wrapper(*args, **kwargs):
            pytestconfig = kwargs.get("pytestconfig")
            if pytestconfig is None:
                raise ValueError(
                    "pytestconfig must be passed as an argument when using the import_or_fail_decorator."
                )
            _import_or_fail(module_names, pytestconfig, min_versions)

            return test_func(*args, **kwargs)

        return wrapper

    return decorator


def _import_or_fail(module_names, config, min_versions=None):
    if not isinstance(module_names, (list, tuple)):
        module_names = [module_names]  # allow single names
    if min_versions is not None and not isinstance(min_versions, (list, tuple)):
        min_versions = [min_versions]  # allow single value for min_versions

    if min_versions is None:
        min_versions = [None] * len(module_names)
    elif len(min_versions) != len(module_names):
        raise ValueError(
            "The length of module_names and min_versions must be the same."
        )

    for module_name, min_version in zip(module_names, min_versions):
        if config.getoption("--fail-on-missing-modules"):
            __import__(module_name)
        else:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, "__version__"):
                    if (
                        isinstance(module.__version__, str)
                        or module.__version__ is None
                    ):
                        pytest.importorskip(module_name, min_version)
                    elif isinstance(module.__version__, Version):
                        # pytest importorskip only works for modulues that return the version as str.
                        version_check = Version(min_version)
                        if module.__version__ < version_check:
                            pytest.skip(
                                f"{module_name} {module.__version__} is less than the required version {version_check}"
                            )
            except ModuleNotFoundError:
                pytest.importorskip(module_name, min_version)
