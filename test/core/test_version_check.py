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
from unittest.mock import MagicMock, patch

import pytest

from physicsnemo.core.version_check import (
    OptionalImport,
    _format_install_hint,
    _optional_import_registry,
    check_version_spec,
    get_installed_version,
    get_package_hint,
    is_package_available,
    register_package_hint,
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
    """Decorator prevents execution when requirement is not met."""
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


# =============================================================================
# Tests for get_installed_version - normalized name matching
# =============================================================================


class TestGetInstalledVersionNormalization:
    """Tests for PEP 503 name normalization in get_installed_version."""

    def setup_method(self):
        """Clear cache before each test."""
        get_installed_version.cache_clear()

    def test_normalized_name_underscore_to_hyphen(self):
        """Finds package when searching with underscores but installed with hyphens."""

        def mock_version(name):
            if name == "torch-geometric":
                return "2.5.0"
            raise metadata.PackageNotFoundError(name)

        with patch(
            "physicsnemo.core.version_check.metadata.version", side_effect=mock_version
        ):
            with patch("physicsnemo.core.version_check.metadata.distributions"):
                # Search with underscores, should find hyphenated version
                result = get_installed_version("torch_geometric")
                assert result == "2.5.0"

    def test_normalized_name_hyphen_to_underscore(self):
        """Finds package when searching with hyphens but installed with underscores."""

        def mock_version(name):
            if name == "some-package":
                return "1.0.0"
            raise metadata.PackageNotFoundError(name)

        with patch(
            "physicsnemo.core.version_check.metadata.version", side_effect=mock_version
        ):
            with patch("physicsnemo.core.version_check.metadata.distributions"):
                result = get_installed_version("some-package")
                assert result == "1.0.0"

    def test_prefix_match_with_hyphen_delimiter(self):
        """Finds variant packages like cupy-cuda12x when searching for cupy."""
        # Create mock distribution
        mock_dist = MagicMock()
        mock_dist.metadata = {"Name": "cupy-cuda12x"}
        mock_dist.version = "13.0.0"

        with patch(
            "physicsnemo.core.version_check.metadata.version",
            side_effect=metadata.PackageNotFoundError,
        ):
            with patch(
                "physicsnemo.core.version_check.metadata.distributions",
                return_value=[mock_dist],
            ):
                result = get_installed_version("cupy")
                assert result == "13.0.0"

    def test_no_false_positive_without_hyphen(self):
        """Ensures 'torch' doesn't match 'torchvision' (no hyphen delimiter)."""
        # Create mock distribution for torchvision
        mock_dist = MagicMock()
        mock_dist.metadata = {"Name": "torchvision"}
        mock_dist.version = "0.18.0"

        with patch(
            "physicsnemo.core.version_check.metadata.version",
            side_effect=metadata.PackageNotFoundError,
        ):
            with patch(
                "physicsnemo.core.version_check.metadata.distributions",
                return_value=[mock_dist],
            ):
                result = get_installed_version("torch")
                # Should NOT find torchvision when searching for torch
                assert result is None

    def test_no_false_positive_partial_name(self):
        """Ensures 'numpy' doesn't match 'numpy-stl' (different package)."""
        mock_dist = MagicMock()
        mock_dist.metadata = {"Name": "numpy-stl"}
        mock_dist.version = "3.0.0"

        with patch(
            "physicsnemo.core.version_check.metadata.version",
            side_effect=metadata.PackageNotFoundError,
        ):
            with patch(
                "physicsnemo.core.version_check.metadata.distributions",
                return_value=[mock_dist],
            ):
                # numpy should NOT match numpy-stl — they are unrelated packages.
                # Prefix matching is restricted to _VARIANT_BASE_PACKAGES.
                result = get_installed_version("numpy")
                assert result is None

    def test_exact_match_takes_precedence(self):
        """Exact match should be returned even if prefix match exists."""
        mock_dist = MagicMock()
        mock_dist.metadata = {"Name": "foo-bar"}
        mock_dist.version = "2.0.0"

        def mock_version(name):
            if name == "foo":
                return "1.0.0"
            raise metadata.PackageNotFoundError(name)

        with patch(
            "physicsnemo.core.version_check.metadata.version", side_effect=mock_version
        ):
            with patch(
                "physicsnemo.core.version_check.metadata.distributions",
                return_value=[mock_dist],
            ):
                result = get_installed_version("foo")
                # Should return exact match, not prefix match
                assert result == "1.0.0"


# =============================================================================
# Tests for is_package_available
# =============================================================================


class TestIsPackageAvailable:
    """Tests for is_package_available function."""

    def setup_method(self):
        """Clear caches before each test."""
        is_package_available.cache_clear()
        get_installed_version.cache_clear()

    def test_available_package(self):
        """Returns True when package is installed."""
        with patch(
            "physicsnemo.core.version_check.get_installed_version", return_value="1.0.0"
        ):
            assert is_package_available("some_package") is True

    def test_unavailable_package(self):
        """Returns False when package is not installed."""
        with patch(
            "physicsnemo.core.version_check.get_installed_version", return_value=None
        ):
            assert is_package_available("nonexistent_package") is False


# =============================================================================
# Tests for check_version_spec - additional edge cases
# =============================================================================


class TestCheckVersionSpecEdgeCases:
    """Additional edge case tests for check_version_spec."""

    def test_version_exactly_met(self):
        """Returns True when installed version exactly matches requirement."""
        with patch(
            "physicsnemo.core.version_check.get_installed_version", return_value="2.6.0"
        ):
            assert check_version_spec("torch", "2.6.0") is True

    def test_version_exceeded(self):
        """Returns True when installed version exceeds requirement."""
        with patch(
            "physicsnemo.core.version_check.get_installed_version", return_value="3.0.0"
        ):
            assert check_version_spec("torch", "2.6.0") is True

    def test_dev_version_comparison(self):
        """Handles development version strings correctly."""
        with patch(
            "physicsnemo.core.version_check.get_installed_version",
            return_value="2.7.0.dev20240101",
        ):
            assert check_version_spec("torch", "2.6.0") is True

    def test_default_spec_any_version(self):
        """Default spec '0.0.0' accepts any installed version."""
        with patch(
            "physicsnemo.core.version_check.get_installed_version", return_value="0.0.1"
        ):
            assert check_version_spec("torch") is True


# =============================================================================
# Tests for register_package_hint and get_package_hint
# =============================================================================


class TestPackageHints:
    """Tests for package hint registration and retrieval."""

    def test_register_and_get_custom_hint(self):
        """Can register and retrieve a custom package hint."""
        custom_hint = "Install with: pip install my-special-package"
        register_package_hint("my_special_package", custom_hint)

        result = get_package_hint("my_special_package")
        assert result == custom_hint

    def test_get_registered_hint(self):
        """Returns registered hint for known packages."""
        hint = get_package_hint("torch_geometric")
        assert "torch_geometric" in hint
        assert "gnns" in hint  # Should mention the optional group

    def test_get_fallback_hint_for_unknown_package(self):
        """Returns generic hint for unknown packages."""
        hint = get_package_hint("completely_unknown_package_xyz")
        assert "completely_unknown_package_xyz" in hint
        assert "pip install" in hint


# =============================================================================
# Tests for _format_install_hint
# =============================================================================


class TestFormatInstallHint:
    """Tests for _format_install_hint helper function."""

    def test_group_based_hint(self):
        """Formats hint with optional dependency group."""
        hint = _format_install_hint("mypackage", group="extras")
        assert "mypackage" in hint
        assert "[extras]" in hint
        assert "physicsnemo[extras]" in hint

    def test_direct_install_hint(self):
        """Formats hint with direct pip install."""
        hint = _format_install_hint("mypackage", direct_install="my-package")
        assert "mypackage" in hint
        assert "pip install my-package" in hint

    def test_direct_hint_custom_text(self):
        """Formats hint with custom installation text."""
        hint = _format_install_hint(
            "mypackage", direct_hint="Build from source at github.com/example"
        )
        assert "mypackage" in hint
        assert "Build from source" in hint

    def test_docs_url_included(self):
        """Includes documentation URL when provided."""
        hint = _format_install_hint(
            "mypackage",
            direct_install="mypackage",
            docs_url="https://docs.example.com",
        )
        assert "https://docs.example.com" in hint


# =============================================================================
# Tests for OptionalImport
# =============================================================================


class TestOptionalImport:
    """Tests for OptionalImport lazy import wrapper."""

    def setup_method(self):
        """Clear the OptionalImport registry before each test."""
        _optional_import_registry.clear()
        is_package_available.cache_clear()
        get_installed_version.cache_clear()

    def test_no_import_on_instantiation(self):
        """Module is not imported when OptionalImport is created."""
        # Use a real pip-installed package (packaging is a dependency)
        opt = OptionalImport("packaging.version")
        # Module should be None (not yet imported)
        assert object.__getattribute__(opt, "_module") is None

    def test_import_on_attribute_access(self):
        """Module is imported when an attribute is accessed."""
        # Use a real pip-installed package
        opt = OptionalImport("packaging.version")
        # Access an attribute - this should trigger the import
        result = opt.parse
        # Module should now be loaded
        assert object.__getattribute__(opt, "_module") is not None
        # The attribute should be the real parse function
        assert callable(result)

    def test_module_cached_after_first_access(self):
        """Module is cached after first successful import."""
        opt = OptionalImport("packaging.utils")
        # Access an attribute to trigger import
        _ = opt.canonicalize_name
        # Module should be cached
        module = object.__getattribute__(opt, "_module")
        assert module is not None
        # Access another attribute and verify module is the same
        _ = opt.canonicalize_version
        assert object.__getattribute__(opt, "_module") is module

    def test_raises_import_error_when_unavailable(self):
        """Raises ImportError with helpful message when package is missing."""
        opt = OptionalImport("totally_fake_missing_package_abc123")

        with pytest.raises(ImportError) as excinfo:
            _ = opt.some_attribute

        error_msg = str(excinfo.value)
        assert "Missing optional dependency" in error_msg
        assert "totally_fake_missing_package_abc123" in error_msg

    def test_custom_package_hint(self):
        """Uses custom package hint when provided."""
        custom_hint = "Special install instructions here"

        opt = OptionalImport("another_fake_pkg_xyz789", package_hint=custom_hint)

        with pytest.raises(ImportError) as excinfo:
            _ = opt.some_attribute

        assert custom_hint in str(excinfo.value)

    def test_dunder_probe_raises_attribute_error_when_unavailable(self):
        """Dunder attribute probes raise AttributeError (not ImportError) when missing.

        Python's inspect, hasattr, and doctest machinery probe for dunders like
        __wrapped__.  These must raise AttributeError so hasattr() returns False
        and introspection doesn't crash.
        """
        opt = OptionalImport("fake_dunder_test_pkg_000")

        # Dunder probe should raise AttributeError, not ImportError
        with pytest.raises(AttributeError):
            _ = opt.__wrapped__

        # hasattr should return False without raising
        assert not hasattr(opt, "__wrapped__")

        # Non-dunder access should still raise ImportError with install hint
        with pytest.raises(ImportError):
            _ = opt.some_function

    def test_dunder_works_when_available(self):
        """Dunder attributes are accessible when the module is available."""
        opt = OptionalImport("packaging")

        # Should not raise - packaging is installed
        result = opt.__name__
        assert result == "packaging"

    def test_available_property_true(self):
        """available property returns True when package is installed."""
        opt = OptionalImport("pytest")  # pytest is installed for running tests
        assert opt.available is True

    def test_available_property_false(self):
        """available property returns False when package is not installed."""
        opt = OptionalImport("nonexistent_pkg_xyz_99999")
        assert opt.available is False

    def test_repr_not_loaded(self):
        """__repr__ shows correct state before module is loaded."""
        opt = OptionalImport("packaging.specifiers")
        repr_str = repr(opt)

        assert "OptionalImport" in repr_str
        assert "packaging.specifiers" in repr_str
        assert "not loaded" in repr_str

    def test_repr_loaded(self):
        """__repr__ shows correct state after module is loaded."""
        opt = OptionalImport("packaging.markers")
        _ = opt.Marker  # Trigger load

        repr_str = repr(opt)
        assert "loaded" in repr_str
        assert "not loaded" not in repr_str

    def test_same_instance_returned_for_same_module(self):
        """Same OptionalImport instance is returned for the same module name."""
        opt1 = OptionalImport("packaging.requirements")
        opt2 = OptionalImport("packaging.requirements")

        assert opt1 is opt2

    def test_different_instances_for_different_modules(self):
        """Different OptionalImport instances for different module names."""
        opt1 = OptionalImport("packaging.tags")
        opt2 = OptionalImport("packaging.metadata")

        assert opt1 is not opt2

    def test_submodule_import(self):
        """Can import submodules like packaging.version."""
        opt = OptionalImport("packaging.version")
        # Access an attribute from the submodule
        result = opt.Version
        # Should be the real Version class
        assert result is not None
        # Verify it works correctly
        v = result("1.0.0")
        assert str(v) == "1.0.0"

    def test_root_package_checked_for_availability(self):
        """Availability check uses root package name, not full module path."""
        # Test with a known missing package
        opt = OptionalImport("nonexistent_test_pkg_12345.sub.module")

        with pytest.raises(ImportError) as excinfo:
            _ = opt.attr

        # Error message should reference root package
        assert "nonexistent_test_pkg_12345" in str(excinfo.value)


# =============================================================================
# Tests for require_version_spec - additional cases
# =============================================================================


class TestRequireVersionSpecAdditional:
    """Additional tests for require_version_spec decorator."""

    def setup_method(self):
        """Clear caches and registry."""
        _optional_import_registry.clear()
        is_package_available.cache_clear()
        get_installed_version.cache_clear()

    def test_default_spec_accepts_any_version(self):
        """Decorator with default spec='0.0.0' accepts any installed version."""

        # Use a real pip-installed package
        @require_version_spec("packaging")
        def fn():
            return "executed"

        assert fn() == "executed"

    def test_raises_import_error_when_package_missing(self):
        """Raises ImportError when package is missing."""

        @require_version_spec("nonexistent_test_pkg_67890", "1.0.0")
        def fn():
            return "executed"

        with pytest.raises(ImportError) as excinfo:
            fn()

        assert "Missing optional dependency" in str(excinfo.value)

    def test_preserves_function_metadata(self):
        """Decorator preserves wrapped function's metadata."""

        # Use a real pip-installed package
        @require_version_spec("pytest")
        def my_documented_function():
            """This is the docstring."""
            return "ok"

        assert my_documented_function.__name__ == "my_documented_function"
        assert "docstring" in my_documented_function.__doc__
