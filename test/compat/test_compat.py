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

import importlib
import re
import sys
import warnings

import pytest

from physicsnemo.compat import COMPAT_MAP, install

# Old paths whose new target may be missing in minimal installs (optional deps).
# If install() skips these, resolution/parent tests skip; warning tests allow them.
KNOWN_OPTIONAL_OLD_NAMES = frozenset(
    {
        "physicsnemo.utils.graphcast",
        "physicsnemo.utils.diffusion",
        "physicsnemo.utils.domino",
        "physicsnemo.launch.utils.checkpoint",
        "physicsnemo.launch.logging",
    }
)


@pytest.fixture(scope="module")
def compat_installed():
    """Run compat.install() once per test module; suppress deprecation output."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module="physicsnemo.compat",
        )
        install()


@pytest.mark.parametrize(
    "old_name,new_name",
    list(COMPAT_MAP.items()),
    ids=list(COMPAT_MAP.keys()),
)
def test_compat_install_old_path_resolves_to_new_module(
    compat_installed, old_name, new_name
):
    """For each COMPAT_MAP entry, old-name import returns the same module as new-name."""
    if old_name not in sys.modules:
        pytest.skip(
            f"Compat alias {old_name!r} not registered (new module may have optional deps)"
        )
    old_mod = importlib.import_module(old_name)
    new_mod = importlib.import_module(new_name)
    assert old_mod is new_mod


@pytest.mark.parametrize(
    "old_name,new_name",
    list(COMPAT_MAP.items()),
    ids=list(COMPAT_MAP.keys()),
)
def test_compat_parent_package_has_old_submodule_attribute(
    compat_installed, old_name, new_name
):
    """Old paths are reachable via parent package attribute (from pkg import sub)."""
    if old_name not in sys.modules:
        pytest.skip(
            f"Compat alias {old_name!r} not registered (new module may have optional deps)"
        )
    parent_name, child = old_name.rsplit(".", 1)
    try:
        parent_mod = importlib.import_module(parent_name)
    except ModuleNotFoundError:
        pytest.skip(
            f"Parent package {parent_name!r} does not exist (e.g. removed package)"
        )
    new_mod = importlib.import_module(new_name)
    assert getattr(parent_mod, child) is new_mod


def test_compat_install_no_failed_import_warnings():
    """install() does not emit 'Failed to import new module' unless from known optional targets."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", UserWarning)
        install()
    failed = [
        x
        for x in w
        if x.category is UserWarning and "Failed to import new module" in str(x.message)
    ]
    # Message format: "Failed to import new module '...' for compat alias 'old_name'"
    alias_re = re.compile(r"for compat alias '([^']+)'")
    for rec in failed:
        match = alias_re.search(str(rec.message))
        assert match, f"Could not parse alias from: {rec.message}"
        old_name = match.group(1)
        assert old_name in KNOWN_OPTIONAL_OLD_NAMES, (
            f"Unexpected failed import for {old_name!r}; add to KNOWN_OPTIONAL_OLD_NAMES if optional"
        )


def test_compat_map_structure():
    """COMPAT_MAP has valid keys/values and is non-empty."""
    assert len(COMPAT_MAP) > 0
    for key, value in COMPAT_MAP.items():
        assert isinstance(key, str) and len(key) > 0
        assert isinstance(value, str) and len(value) > 0
        assert "." in key, f"Key should be a module path: {key!r}"
    assert len(COMPAT_MAP) == len(set(COMPAT_MAP.keys())), (
        "COMPAT_MAP keys must be unique"
    )
