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

from importlib import metadata
from unittest.mock import patch

import pytest

from physicsnemo.core.version_check import (
    check_version_spec,
    get_installed_version,
    require_version_spec,
)


def test_get_installed_version_found():
    """get_installed_version returns version string when package is installed"""
    # Clear the LRU cache for testing:
    get_installed_version.cache_clear()

    with patch(
        "physicsnemo.core.version_check.metadata.version", return_value="2.6.0"
    ) as mock_version:
        assert get_installed_version("torch") == "2.6.0"
        mock_version.assert_called_once_with("torch")


def test_get_installed_version_not_found():
    """get_installed_version returns None when package is not installed"""
    with patch(
        "physicsnemo.core.version_check.metadata.version",
        side_effect=metadata.PackageNotFoundError,
    ):
        assert get_installed_version("nonexistent_package") is None


def test_check_version_spec_failure_hard():
    """check_version_spec raises ImportError when requirement is not met and hard_fail=True"""
    with patch(
        "physicsnemo.core.version_check.get_installed_version", return_value="2.5.0"
    ):
        with pytest.raises(ImportError) as excinfo:
            check_version_spec("torch", "2.6.0", hard_fail=True)
    msg = str(excinfo.value)
    assert "torch 2.6.0 is required" in msg
    assert "found 2.5.0" in msg


def test_check_version_spec_failure_soft():
    """check_version_spec returns False when requirement not met and hard_fail=False"""
    with patch(
        "physicsnemo.core.version_check.get_installed_version", return_value="2.5.0"
    ):
        assert check_version_spec("torch", "2.6.0", hard_fail=False) is False


def test_check_version_spec_custom_error_message():
    """check_version_spec uses provided custom error message"""
    with patch(
        "physicsnemo.core.version_check.get_installed_version", return_value="2.5.0"
    ):
        with pytest.raises(ImportError) as excinfo:
            check_version_spec(
                "torch", "2.6.0", error_msg="Custom error", hard_fail=True
            )
    assert "Custom error" in str(excinfo.value)


def test_check_version_spec_package_not_found_hard():
    """Raises with clear message when package is not installed and hard_fail=True"""
    with patch(
        "physicsnemo.core.version_check.get_installed_version", return_value=None
    ):
        with pytest.raises(ImportError) as excinfo:
            check_version_spec("torch", "2.0.0", hard_fail=True)
    assert "Package 'torch' is required but not installed." in str(excinfo.value)


def test_check_version_spec_package_not_found_soft():
    """Returns False when package is not installed and hard_fail=False"""
    with patch(
        "physicsnemo.core.version_check.get_installed_version", return_value=None
    ):
        assert check_version_spec("torch", "2.0.0", hard_fail=False) is False


def test_require_version_spec_success():
    """Decorator allows execution when requirement is met"""
    with patch("physicsnemo.core.version_check.check_version_spec", return_value=True):

        @require_version_spec("torch", "2.5.0")
        def fn():
            return "ok"

        assert fn() == "ok"


def test_require_version_spec_failure():
    """Decorator prevents execution when requirement is not met"""
    with patch(
        "physicsnemo.core.version_check.check_version_spec",
        side_effect=ImportError("not satisfied"),
    ):

        @require_version_spec("torch", "2.6.0")
        def fn():
            return "ok"

        with pytest.raises(ImportError) as excinfo:
            fn()
    assert "not satisfied" in str(excinfo.value)
