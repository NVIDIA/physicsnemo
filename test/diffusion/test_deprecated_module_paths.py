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

"""Tests for the deprecated diffusion module paths kept as import shims."""

import importlib
import re
import subprocess
import sys
import warnings

import pytest

# =============================================================================
# Constants and Configurations
# =============================================================================

# Deprecated module path -> (supported package path, names it must re-export)
DEPRECATED_MODULES = {
    "physicsnemo.diffusion.samplers.solvers": (
        "physicsnemo.diffusion.samplers",
        (
            "Solver",
            "EulerSolver",
            "HeunSolver",
            "EDMStochasticEulerSolver",
            "EDMStochasticHeunSolver",
        ),
    ),
    "physicsnemo.diffusion.noise_schedulers.noise_schedulers": (
        "physicsnemo.diffusion.noise_schedulers",
        (
            "NoiseScheduler",
            "LinearGaussianNoiseScheduler",
            "EDMNoiseScheduler",
            "EDMLogUniformNoiseScheduler",
            "VENoiseScheduler",
            "IDDPMNoiseScheduler",
            "VPNoiseScheduler",
            "StudentTEDMNoiseScheduler",
        ),
    ),
}

DEPRECATED_PATHS = list(DEPRECATED_MODULES)

# Phrase shared by both shim warnings, used to tell them apart from the
# pre-existing LegacyFeatureWarning raised by the legacy sampler modules.
SHIM_WARNING_MARKER = "lives in its own module"


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.parametrize("old_path", DEPRECATED_PATHS, ids=DEPRECATED_PATHS)
def test_deprecated_module_warns_on_import(old_path):
    """Importing a deprecated path warns and names the supported package."""
    new_path = DEPRECATED_MODULES[old_path][0]
    # Drop the cached module so the shim body, and its warning, run again.
    sys.modules.pop(old_path, None)
    with pytest.warns(DeprecationWarning, match=re.escape(new_path)):
        importlib.import_module(old_path)


@pytest.mark.parametrize("old_path", DEPRECATED_PATHS, ids=DEPRECATED_PATHS)
def test_deprecated_module_reexports_the_same_objects(old_path):
    """A shim re-exports the package's classes themselves, not copies."""
    new_path, names = DEPRECATED_MODULES[old_path]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old_module = importlib.import_module(old_path)
    new_module = importlib.import_module(new_path)
    for name in names:
        assert getattr(old_module, name) is getattr(new_module, name)


@pytest.mark.parametrize("old_path", DEPRECATED_PATHS, ids=DEPRECATED_PATHS)
def test_supported_package_does_not_import_the_shim(old_path):
    """Importing the supported package must not pull in the shim or its warning.

    Runs in a subprocess because a module body executes once per process: by the
    time this test runs, the shim is already cached, so an in-process check
    would pass regardless of what the package imports.
    """
    new_path = DEPRECATED_MODULES[old_path][0]
    snippet = (
        "import sys, warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        f"    import {new_path}\n"
        f"assert {old_path!r} not in sys.modules, 'package imports the deprecated shim'\n"
        f"leaked = [str(w.message) for w in caught if {SHIM_WARNING_MARKER!r} in str(w.message)]\n"
        "assert not leaked, leaked\n"
    )
    subprocess.run(  # noqa: S603 - interpreter and snippet are test constants
        [sys.executable, "-c", snippet],
        check=True,
        capture_output=True,
        text=True,
    )
