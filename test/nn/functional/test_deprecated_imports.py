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

"""Tests for the deprecated nn.functional names of functionals moved to utils."""

import re
import subprocess
import sys

import pytest

import physicsnemo.utils
from physicsnemo.core.warnings import LegacyFeatureWarning
from physicsnemo.nn import functional

MOVED_TO_UTILS = ["weighted_multinomial", "WeightedMultinomial"]


@pytest.mark.parametrize("name", MOVED_TO_UTILS)
def test_moved_name_warns_and_returns_utils_object(name):
    """The old name warns, names the new import, and returns the same object."""
    with pytest.warns(
        LegacyFeatureWarning,
        match=re.escape(f"from physicsnemo.utils import {name}"),
    ):
        legacy = getattr(functional, name)

    assert legacy is getattr(physicsnemo.utils, name)


def test_from_import_of_moved_name_warns():
    """``from physicsnemo.nn.functional import ...`` still works, with a warning."""
    with pytest.warns(LegacyFeatureWarning):
        from physicsnemo.nn.functional import weighted_multinomial

    assert weighted_multinomial is physicsnemo.utils.weighted_multinomial


def test_wildcard_import_keeps_moved_names():
    """``import *`` still provides the moved names, with the warning."""
    namespace = {}
    with pytest.warns(LegacyFeatureWarning):
        exec("from physicsnemo.nn.functional import *", namespace)  # noqa: S102

    for name in MOVED_TO_UTILS:
        assert namespace[name] is getattr(physicsnemo.utils, name)


def test_unknown_name_raises_attribute_error():
    """Other names raise AttributeError; safe_normalize never shipped here."""
    with pytest.raises(AttributeError, match="no attribute 'safe_normalize'"):
        functional.safe_normalize  # noqa: B018


def test_library_imports_do_not_use_deprecated_names():
    """No physicsnemo module reaches the moved functionals through nn.functional.

    Runs in a subprocess because a module body executes once per process: by the
    time this test runs, the modules are already cached, so an in-process check
    would pass regardless of what they import.
    """
    snippet = (
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import physicsnemo.datapipes.cae.transolver_datapipe\n"
        "    import physicsnemo.datapipes.transforms\n"
        "    import physicsnemo.mesh\n"
        "    import physicsnemo.models.domino.utils\n"
        "    import physicsnemo.nn.functional\n"
        "leaked = [\n"
        "    str(w.message) for w in caught\n"
        "    if \"from 'physicsnemo.nn.functional' is deprecated\" in str(w.message)\n"
        "]\n"
        "assert not leaked, leaked\n"
    )
    subprocess.run(  # noqa: S603 - interpreter and snippet are test constants
        [sys.executable, "-c", snippet], check=True
    )
