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

from unittest.mock import MagicMock, patch

import pytest

from physicsnemo.utils.version_check import (
    VERSION_REQUIREMENTS,
    check_min_version,
    check_module_requirements,
)


def test_check_min_version_success():
    """Test that check_min_version succeeds when version requirement is met"""
    with patch("importlib.import_module") as mock_import:
        # Create a mock module with version 2.6.0
        mock_module = MagicMock()
        mock_module.__version__ = "2.6.0"
        mock_import.return_value = mock_module

        # Should pass with same version
        assert check_min_version("torch", "2.6.0") is True

        # Should pass with lower required version
        assert check_min_version("torch", "2.5.0") is True


def test_check_min_version_failure():
    """Test that check_min_version raises ImportError when version requirement is not met"""
    with patch("importlib.import_module") as mock_import:
        # Create a mock module with version 2.5.0
        mock_module = MagicMock()
        mock_module.__version__ = "2.5.0"
        mock_import.return_value = mock_module

        # Should fail with higher required version
        with pytest.raises(ImportError) as excinfo:
            check_min_version("torch", "2.6.0")

        assert "torch version 2.6.0 or higher is required" in str(excinfo.value)


def test_check_min_version_custom_error():
    """Test that check_min_version uses custom error message if provided"""
    with patch("importlib.import_module") as mock_import:
        # Create a mock module with version 2.5.0
        mock_module = MagicMock()
        mock_module.__version__ = "2.5.0"
        mock_import.return_value = mock_module

        custom_msg = "Custom error message"
        with pytest.raises(ImportError) as excinfo:
            check_min_version("torch", "2.6.0", error_msg=custom_msg)

        assert custom_msg in str(excinfo.value)


def test_check_min_version_package_not_found():
    """Test that check_min_version raises ImportError when package is not installed"""
    with patch("importlib.import_module", side_effect=ImportError("Package not found")):
        with pytest.raises(ImportError) as excinfo:
            check_min_version("nonexistent_package", "1.0.0")

        assert "Package nonexistent_package is required but not installed" in str(
            excinfo.value
        )


def test_check_module_requirements_success():
    """Test that check_module_requirements succeeds when all requirements are met"""
    with patch(
        "physicsnemo.utils.version_check.check_min_version"
    ) as mock_check_min_version:
        mock_check_min_version.return_value = True

        # Should run check_min_version for known module
        check_module_requirements("physicsnemo.distributed.shard_tensor")
        mock_check_min_version.assert_called_once_with("torch", "2.5.9")


def test_check_module_requirements_unknown_module():
    """Test that check_module_requirements does nothing for unknown modules"""
    with patch(
        "physicsnemo.utils.version_check.check_min_version"
    ) as mock_check_min_version:
        # Should not call check_min_version for unknown module
        check_module_requirements("unknown.module.path")
        mock_check_min_version.assert_not_called()


def test_version_requirements_structure():
    """Test that VERSION_REQUIREMENTS dictionary has the expected structure"""
    assert "physicsnemo.distributed.shard_tensor" in VERSION_REQUIREMENTS
    assert "torch" in VERSION_REQUIREMENTS["physicsnemo.distributed.shard_tensor"]
    assert (
        VERSION_REQUIREMENTS["physicsnemo.distributed.shard_tensor"]["torch"] == "2.5.9"
    )


def test_require_version_success():
    """Test that require_version decorator allows function to run when version requirement is met"""
    with patch("importlib.import_module") as mock_import:
        # Create a mock module with version 2.6.0
        mock_module = MagicMock()
        mock_module.__version__ = "2.6.0"
        mock_import.return_value = mock_module

        # Create a decorated function
        from physicsnemo.utils.version_check import require_version

        @require_version("torch", "2.5.0")
        def test_function():
            return "Function executed"

        # Function should execute normally when version requirement is met
        assert test_function() == "Function executed"


def test_require_version_failure():
    """Test that require_version decorator prevents function from running when version requirement is not met"""
    with patch("importlib.import_module") as mock_import:
        # Create a mock module with version 2.5.0
        mock_module = MagicMock()
        mock_module.__version__ = "2.5.0"
        mock_import.return_value = mock_module

        # Create a decorated function
        from physicsnemo.utils.version_check import require_version

        @require_version("torch", "2.6.0")
        def test_function():
            return "Function executed"

        # Function should raise ImportError when version requirement is not met
        with pytest.raises(ImportError) as excinfo:
            test_function()

        assert "torch version 2.6.0 or higher is required" in str(excinfo.value)


def test_require_version_package_not_found():
    """Test that require_version decorator raises ImportError when package is not installed"""
    with patch("importlib.import_module", side_effect=ImportError("Package not found")):
        # Create a decorated function
        from physicsnemo.utils.version_check import require_version

        @require_version("nonexistent_package", "1.0.0")
        def test_function():
            return "Function executed"

        # Function should raise ImportError when package is not installed
        with pytest.raises(ImportError) as excinfo:
            test_function()

        assert "Package nonexistent_package is required but not installed" in str(
            excinfo.value
        )
