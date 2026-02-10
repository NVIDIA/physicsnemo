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
Utilities for version compatibility checking and optional dependency handling.

This module provides:

1. Version checking without importing (``check_version_spec``, ``get_installed_version``)
2. Version requirement decorator (``require_version_spec``)
3. Lazy optional imports with clear error messages (``OptionalImport``)

Example usage::

    from physicsnemo.core.version_check import OptionalImport

    # Lazy import - no import happens until first attribute access
    _pyg = OptionalImport("torch_geometric.data")

    def my_gnn_function(data):
        # Import happens here; raises ImportError with install hint if missing
        Data = _pyg.Data
        ...

"""

import functools
import importlib
import os
import sys
from importlib import metadata
from types import ModuleType
from typing import Dict, Optional

from packaging.utils import canonicalize_name
from packaging.version import parse

# =============================================================================
# Terminal color support (ANSI escape codes, standard library only)
# =============================================================================


def _supports_color() -> bool:
    """Check if the terminal supports ANSI color codes."""
    # Respect NO_COLOR environment variable (https://no-color.org/)
    if os.environ.get("NO_COLOR"):
        return False
    # Force color if requested
    if os.environ.get("FORCE_COLOR"):
        return True
    # Check if stdout is a tty
    if not hasattr(sys.stderr, "isatty") or not sys.stderr.isatty():
        return False
    # Windows terminal detection
    if sys.platform == "win32":
        # Windows 10+ supports ANSI codes in modern terminals
        return os.environ.get("TERM") or os.environ.get("WT_SESSION")
    return True


class _Colors:
    """ANSI color codes for terminal output."""

    # Only use colors if terminal supports them
    _enabled = _supports_color()

    # Colors
    RED = "\033[91m" if _enabled else ""
    GREEN = "\033[92m" if _enabled else ""
    YELLOW = "\033[93m" if _enabled else ""
    BLUE = "\033[94m" if _enabled else ""
    MAGENTA = "\033[95m" if _enabled else ""
    CYAN = "\033[96m" if _enabled else ""
    WHITE = "\033[97m" if _enabled else ""

    # Styles
    BOLD = "\033[1m" if _enabled else ""
    DIM = "\033[2m" if _enabled else ""
    UNDERLINE = "\033[4m" if _enabled else ""

    # Reset
    RESET = "\033[0m" if _enabled else ""


def _format_install_hint(
    package_name: str,
    group: Optional[str] = None,
    direct_install: Optional[str] = None,
    direct_hint: Optional[str] = None,
    docs_url: Optional[str] = None,
) -> str:
    """
    Format a colored install hint message.

    Parameters
    ----------
    package_name : str
        Display name of the package.
    group : str, optional
        physicsnemo optional dependency group (e.g., "graph", "transformer").
        If provided, shows `pip install physicsnemo[group]` instructions.
    direct_install : str, optional
        Package name for direct pip install (e.g., "warp-lang").
        If provided, shows `pip install <direct_install>` instructions.
    direct_hint : str, optional
        Custom installation instructions string. Can be a command, URL, or
        any freeform text. Displayed as-is after the header.
    docs_url : str, optional
        URL to documentation. Shown at the end as a dim link.
    """
    c = _Colors
    lines = []

    # Header with package name
    lines.append(f"{c.CYAN}{package_name}{c.RESET} is required for this feature.")

    # Group-based install (physicsnemo optional dep)
    if group:
        lines[0] = (
            f"{c.CYAN}{package_name}{c.RESET} is part of the "
            f"{c.YELLOW}[{group}]{c.RESET} optional dependency group."
        )
        lines.append(f"\n{c.BOLD}Install with:{c.RESET}")
        lines.append(f"  {c.GREEN}uv pip install physicsnemo[{group}]{c.RESET}")
        lines.append(f"  {c.GREEN}pip install physicsnemo[{group}]{c.RESET}")

    # Direct pip install
    elif direct_install:
        lines.append(f"\n{c.BOLD}Install with:{c.RESET}")
        lines.append(f"  {c.GREEN}uv pip install {direct_install}{c.RESET}")
        lines.append(f"  {c.GREEN}pip install {direct_install}{c.RESET}")

    # Custom hint (command, URL, or any text)
    if direct_hint:
        lines.append(f"\n{c.BOLD}Installation:{c.RESET}")
        lines.append(f"  {c.GREEN}{direct_hint}{c.RESET}")

    # Documentation URL
    if docs_url:
        lines.append(f"\n{c.DIM}Documentation: {docs_url}{c.RESET}")

    return "\n".join(lines)


# Registry for package install hints (maps package name to install instructions)
# Each entry should include:
#   - The optional dependency group (if applicable)
#   - Install commands for both uv and pip
#   - Link to documentation if needed
_PACKAGE_HINTS: Dict[str, str] = {
    # Graph neural network packages
    "torch_geometric": _format_install_hint(
        "torch_geometric",
        group="gnns",
        docs_url="https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html",
    ),
    "torch_scatter": _format_install_hint(
        "torch_scatter",
        group="gnns",
        docs_url="https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html",
    ),
    "torch_sparse": _format_install_hint(
        "torch_sparse",
        group="gnns",
        docs_url="https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html",
    ),
    "torch_cluster": _format_install_hint(
        "torch_cluster",
        group="gnns",
        docs_url="https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html",
    ),
    # Core scientific packages
    "scipy": _format_install_hint(
        "scipy",
        direct_install="scipy",
    ),
    "scikit-learn": _format_install_hint(
        "scikit-learn",
        direct_install="scikit-learn",
    ),
    "scikit-image": _format_install_hint(  # To be removed with utils/mesh/
        "scikit-image",
        direct_install="scikit-image",
    ),
    # Data format packages
    "xarray": _format_install_hint(
        "xarray",
        group="datapipes-extras",
    ),
    "zarr": _format_install_hint(
        "zarr",
        group="datapipes-extras",
    ),
    "h5py": _format_install_hint(
        "h5py",
        direct_install="h5py",
    ),
    "netCDF4": _format_install_hint(
        "netCDF4",
        group="model-extras",
    ),
    "tfrecord": _format_install_hint(
        "tfrecord",
        direct_install="tfrecord",
    ),
    "tensorstore": _format_install_hint(
        "tensorstore",
        direct_install="tensorstore",
    ),
    # Visualization packages
    "pyvista": _format_install_hint(
        "pyvista",
        group="mesh-extras",
    ),
    "vtk": _format_install_hint(
        "vtk",
        group="model-extras",
    ),
    # NVIDIA packages
    "warp": _format_install_hint(
        "warp-lang",
        direct_install="warp-lang",
    ),
    "transformer_engine": _format_install_hint(
        "transformer_engine",
        group="perf",
    ),
    "nvidia.dali": _format_install_hint(
        "nvidia-dali",
        group="perf",
    ),
    "cuml": _format_install_hint(
        "cuml",
        group="perf",
    ),
    "cupy": _format_install_hint(
        "cupy",
        group="perf",
    ),
    "rmm": _format_install_hint(
        "rmm",
        group="perf",
    ),
    "nvfuser": _format_install_hint(
        "nvfuser",
        direct_hint="Available in PyTorch container >= 23.10",
        docs_url="https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/",
    ),
    "apex": _format_install_hint(
        "apex",
        direct_hint="See https://github.com/NVIDIA/apex#installation",
    ),
    "onnxruntime": _format_install_hint(
        "onnxruntime",
        direct_install="onnxruntime-gpu",
        docs_url="https://onnxruntime.ai/docs/install/",
    ),
    # Neural network extras
    "natten": _format_install_hint(
        "natten",
        group="nn-extras",
    ),
    "earth2grid": _format_install_hint(
        "earth2grid",
        direct_install="earth2grid",
    ),
    # Logging and profiling
    "wandb": _format_install_hint(
        "wandb",
        group="utils-extras",
    ),
    "mlflow": _format_install_hint(
        "mlflow",
        group="utils-extras",
    ),
    "line_profiler": _format_install_hint(
        "line_profiler",
        group="utils-extras",
    ),
    # Mesh utilities
    "numpy-stl": _format_install_hint(
        "numpy-stl",
        group="utils-extras",
    ),
    "stl": _format_install_hint(
        "numpy-stl",
        group="utils-extras",
    ),
    "shapely": _format_install_hint(
        "shapely",
        direct_install="shapely",
    ),
    # Miscellaneous
    "sparse_dot_mkl": _format_install_hint(
        "sparse_dot_mkl",
        direct_install="sparse_dot_mkl",
    ),
    "wrapt": _format_install_hint(
        "wrapt",
        direct_install="wrapt",
    ),
}


def register_package_hint(package_name: str, hint: str) -> None:
    """
    Register a custom install hint for a package.

    Parameters
    ----------
    package_name : str
        The package distribution name.
    hint : str
        Install instructions to show when the package is missing.
    """
    _PACKAGE_HINTS[package_name] = hint


def get_package_hint(package_name: str) -> str:
    """
    Get the install hint for a package.

    Parameters
    ----------
    package_name : str
        The package distribution name.

    Returns
    -------
    str
        Install hint, or a generic message if not registered.
    """
    if package_name in _PACKAGE_HINTS:
        return _PACKAGE_HINTS[package_name]

    # Generic fallback with colors
    c = _Colors
    return (
        f"{c.CYAN}{package_name}{c.RESET} is required for this feature.\n"
        f"\n{c.BOLD}Install with:{c.RESET}\n"
        f"  {c.GREEN}uv pip install {package_name}{c.RESET}\n"
        f"  {c.GREEN}pip install {package_name}{c.RESET}"
    )


# Packages known to ship under build-variant distribution names.
# For example, ``cupy`` is distributed as ``cupy-cuda11x``, ``cupy-cuda12x``,
# etc. depending on the CUDA toolkit version.  Only these base names will
# trigger the slower prefix-match fallback in :func:`get_installed_version`;
# all other packages require an exact (or PEP 503-normalized) name match.
# This prevents false positives such as ``"numpy"`` matching ``"numpy-stl"``,
# which is an entirely unrelated package.
_VARIANT_BASE_PACKAGES: frozenset = frozenset(
    {
        "cupy",  # cupy-cuda11x, cupy-cuda12x, …
        "onnxruntime",  # onnxruntime-gpu, onnxruntime-openvino, …
        "warp",  # pip installs as `warp-lang`
    }
)


@functools.lru_cache(maxsize=None)
def get_installed_version(distribution_name: str) -> Optional[str]:
    """
    Return the installed version for a given distribution without importing it.

    Uses importlib.metadata to avoid heavy import-time side effects.
    Cached for repeated lookups.

    Parameters
    ----------
    distribution_name : str
        The package distribution name (as installed by pip).

    Returns
    -------
    Optional[str]
        The installed version string, or None if not installed.

    Notes
    -----
    This function handles variant package names like ``cupy-cuda12x`` when
    searching for ``cupy``, but **only** for packages listed in
    ``_VARIANT_BASE_PACKAGES``.  It uses PEP 503 name normalization and
    requires an exact match or (for known variant packages) a
    hyphen-delimited prefix match.  Unregistered packages must match
    exactly — e.g. searching for ``"numpy"`` will **not** match
    ``"numpy-stl"``.
    """
    # First, try exact match (handles most cases)
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        pass

    # Normalize the name per PEP 503 (lowercase, replace ._- with -)
    normalized_name = canonicalize_name(distribution_name)

    # Try normalized name directly (handles torch_geometric vs torch-geometric)
    try:
        return metadata.version(normalized_name)
    except metadata.PackageNotFoundError:
        pass

    # Handle variant packages like cupy-cuda12x when searching for cupy.
    # Only apply prefix matching for packages registered in
    # _VARIANT_BASE_PACKAGES to avoid false positives (e.g., "numpy"
    # should not match "numpy-stl").
    if normalized_name in _VARIANT_BASE_PACKAGES:
        normalized_prefix = normalized_name + "-"
        for dist in metadata.distributions():
            dist_normalized = canonicalize_name(dist.metadata["Name"])
            if dist_normalized.startswith(normalized_prefix):
                return dist.version

    return None


@functools.lru_cache(maxsize=None)
def is_package_available(distribution_name: str) -> bool:
    """
    Check if a package is installed (any version).

    Parameters
    ----------
    distribution_name : str
        The package distribution name.

    Returns
    -------
    bool
        True if the package is installed, False otherwise.
    """
    return get_installed_version(distribution_name) is not None


def check_version_spec(
    distribution_name: str,
    spec: str = "0.0.0",
    *,
    error_msg: Optional[str] = None,
    hard_fail: bool = False,
) -> bool:
    """
    Check whether the installed distribution satisfies a version specifier.

    Parameters
    ----------
    distribution_name : str
        Distribution (package) name as installed by pip.
    spec : str, default="0.0.0"
        Minimum version specifier (e.g., '2.4'). Uses simple >= comparison,
        not full PEP 440, to allow dev versions etc.
    error_msg : str, optional
        Custom error message to use on failure.
    hard_fail : bool, default=False
        Whether to raise an ImportError if the version requirement is not met.

    Returns
    -------
    bool
        True if version requirement is met; False if not and hard_fail=False.

    Raises
    ------
    ImportError
        If package is not installed or requirement not satisfied (and hard_fail=True).
    """
    installed = get_installed_version(distribution_name)
    if installed is None:
        if hard_fail:
            hint = get_package_hint(distribution_name)
            raise ImportError(
                f"Package '{distribution_name}' is required but not installed.\n{hint}"
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

    Uses OptionalImport internally to provide helpful error messages with
    installation hints when the package is missing.

    Parameters
    ----------
    package_name : str
        Name of the package to check.
    spec : str, default="0.0.0"
        Minimum version required (e.g., '2.4').

    Returns
    -------
    Callable
        Decorator function that checks version requirement before execution.

    Raises
    ------
    ImportError
        If the package is missing or does not satisfy the version requirement.

    Example
    -------
    >>> @require_version_spec("pyvista")
    ... def load_mesh():
    ...     import pyvista as pv
    ...     return pv.examples.download_bunny()

    If pyvista is not installed or the version is too low, calling the function
    will raise ``ImportError`` with installation instructions.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Use OptionalImport to get the nice error message with install hints
            opt = OptionalImport(package_name)
            if not opt.available:
                opt._get_module()  # This will raise ImportError with hint
            # If version specified, also check version.
            # check_version_spec raises ImportError with hard_fail=True,
            # which we let propagate directly.
            if spec != "0.0.0":
                check_version_spec(package_name, spec, hard_fail=True)
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Private registry for OptionalImport instances - ensures same instance
# is returned for the same module name across the codebase
_optional_import_registry: dict[str, "OptionalImport"] = {}


class OptionalImport:
    """
    Lazy import wrapper for optional dependencies.

    Delays the import of an optional module until it is actually accessed.
    If the module is not available, raises ``ImportError`` with a helpful
    message on first access. This is safe because the import is lazy —
    it only raises on attribute access, not at import time, so
    ``import physicsnemo`` won't crash for users who don't need specific
    optional dependencies.

    Instances are cached by module name - calling ``OptionalImport("foo")``
    multiple times returns the same instance.

    Parameters
    ----------
    module_name : str
        The fully qualified module name to import (e.g., "torch_geometric.data").
    package_hint : str, optional
        Custom install hint. If not provided, uses the registered hint for the
        root package, or a generic message. Only used on first instantiation
        for a given module name.

    Example
    -------
    >>> # At module level (no import happens here)
    >>> torch_geometric = OptionalImport("torch_geometric")
    >>> torch_scatter = OptionalImport("torch_scatter")
    >>>
    >>> def my_function(data):
    ...     # Import happens here on first access
    ...     return torch_scatter.scatter(data, ...)

    If torch_scatter is not installed, accessing ``torch_scatter.scatter``
    will raise ``ImportError`` with install instructions::

        ImportError: Missing optional dependency: torch_scatter

        torch_scatter is part of the [graph] optional dependency group.
        Install with:
          uv pip install physicsnemo[graph]
          pip install physicsnemo[graph]

    Notes
    -----
    This class uses ``__getattr__`` to lazily import the module on first attribute
    access. The actual module is cached after first successful import.

    Instances are deduplicated: if multiple files create ``OptionalImport("torch_geometric")``,
    they all receive the same instance. This ensures consistent state and avoids
    redundant import attempts.
    """

    def __new__(cls, module_name: str, package_hint: Optional[str] = None):
        # Return cached instance if it exists
        if module_name in _optional_import_registry:
            return _optional_import_registry[module_name]

        # Create new instance
        instance = object.__new__(cls)
        _optional_import_registry[module_name] = instance
        return instance

    def __init__(self, module_name: str, package_hint: Optional[str] = None):
        # Skip re-initialization if already initialized (cached instance)
        # Use try/except instead of hasattr to avoid triggering __getattr__
        try:
            object.__getattribute__(self, "_module_name")
            return  # Already initialized
        except AttributeError:
            pass

        # Use object.__setattr__ to avoid triggering our __setattr__
        object.__setattr__(self, "_module_name", module_name)
        object.__setattr__(self, "_package_hint", package_hint)
        object.__setattr__(self, "_module", None)

    def _get_module(self) -> ModuleType:
        """Import the module, raising ImportError with helpful message if unavailable."""
        # Use object.__getattribute__ to avoid triggering __getattr__ recursion
        module = object.__getattribute__(self, "_module")
        if module is not None:
            return module

        module_name = object.__getattribute__(self, "_module_name")
        root_pkg = module_name.split(".")[0]

        # Check availability before attempting import
        if not is_package_available(root_pkg):
            package_hint = object.__getattribute__(self, "_package_hint")
            hint = package_hint or get_package_hint(root_pkg)
            # ImportError is the standard exception for missing modules.
            # This is safe because OptionalImport is lazy — it only raises
            # on attribute access, not at import time, so
            # 'import physicsnemo' won't crash for users who don't need
            # this optional dependency.  Using ImportError also lets
            # pytest conftest hooks (SKIPPABLE_EXCEPTIONS) auto-skip
            # doctests when optional deps are missing.
            c = _Colors
            raise ImportError(
                f"\n{c.RED}{c.BOLD}Missing optional dependency: {root_pkg}{c.RESET}\n\n{hint}"
            )

        # Package is available, import it
        module = importlib.import_module(module_name)
        object.__setattr__(self, "_module", module)
        return module

    def __getattr__(self, name: str):
        """Proxy attribute access to the lazily imported module."""
        # For dunder (protocol) attribute probes, don't trigger a full
        # import error.  Python's inspect, hasattr, pickle, copy, and
        # doctest machinery probe for dunders like __wrapped__, __reduce__,
        # __class__, etc.  If the module isn't available, these must raise
        # AttributeError so that hasattr() returns False and introspection
        # works correctly.
        if name.startswith("__") and name.endswith("__"):
            # If the module is already loaded, look up the dunder on it
            module = object.__getattribute__(self, "_module")
            if module is not None:
                return getattr(module, name)
            # Module not loaded yet — check availability without importing
            module_name = object.__getattribute__(self, "_module_name")
            root_pkg = module_name.split(".")[0]
            if not is_package_available(root_pkg):
                raise AttributeError(name)
            # Package is available but not yet loaded; import and look up
            module = self._get_module()
            return getattr(module, name)

        module = self._get_module()
        return getattr(module, name)

    def __repr__(self) -> str:
        try:
            module_name = object.__getattribute__(self, "_module_name")
            module = object.__getattribute__(self, "_module")
            status = "loaded" if module is not None else "not loaded"
            return f"<OptionalImport({module_name!r}) [{status}]>"
        except AttributeError:
            return "<OptionalImport [uninitialized]>"

    @property
    def available(self) -> bool:
        """Check if the module is available without importing it."""
        module_name = object.__getattribute__(self, "_module_name")
        root_pkg = module_name.split(".")[0]
        return is_package_available(root_pkg)
