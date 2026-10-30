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

"""Tests for the deprecated nn.functional names of functionals moved to utils or mesh."""

import importlib
import re
import subprocess
import sys

import pytest

import physicsnemo.utils
from physicsnemo.core.warnings import LegacyFeatureWarning
from physicsnemo.mesh.calculus import _lsq_gradient
from physicsnemo.nn import functional
from physicsnemo.nn.functional._moved_to_mesh import MOVED_TO_MESH

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


@pytest.mark.parametrize(
    ("module", "name"),
    [(module, name) for module, names in MOVED_TO_MESH.items() for name in names],
)
def test_moved_to_mesh_name_warns_and_returns_mesh_object(module, name):
    """The old name warns at the caller and returns the object mesh defines."""
    with pytest.warns(
        LegacyFeatureWarning,
        match=re.escape(f"Importing {name!r} from {module!r} is deprecated"),
    ) as record:
        legacy = getattr(importlib.import_module(module), name)

    caller = next(w for w in record if issubclass(w.category, LegacyFeatureWarning))
    assert caller.filename == __file__
    assert legacy.__module__.startswith("physicsnemo.mesh.")
    assert getattr(importlib.import_module(legacy.__module__), name) is legacy


@pytest.mark.parametrize(
    ("name", "use_instead"),
    [
        (
            "mesh_lsq_gradient",
            "physicsnemo.mesh.calculus.compute_gradient_points_lsq on a Mesh",
        ),
        ("morph_points", "physicsnemo.mesh.transformations.deform.morph on a Mesh"),
        ("remeshing", "physicsnemo.mesh.remeshing.remesh on a Mesh"),
    ],
)
def test_moved_to_mesh_warning_names_the_replacement(name, use_instead):
    """Names with a public replacement say what to use instead."""
    with pytest.warns(LegacyFeatureWarning, match=re.escape(f"Use {use_instead}")):
        getattr(functional, name)


def test_from_import_of_moved_mesh_name_warns():
    """The import that the DoMINO examples used still works, with a warning."""
    with pytest.warns(LegacyFeatureWarning, match="compute_gradient_points_lsq"):
        from physicsnemo.nn.functional.derivatives import mesh_lsq_gradient

    assert mesh_lsq_gradient is _lsq_gradient.mesh_lsq_gradient


def test_wildcard_import_keeps_moved_names():
    """``import *`` still provides the moved names, with the warning."""
    namespace = {}
    with pytest.warns(LegacyFeatureWarning):
        exec("from physicsnemo.nn.functional import *", namespace)  # noqa: S102

    for name in MOVED_TO_UTILS:
        assert namespace[name] is getattr(physicsnemo.utils, name)
    for name in MOVED_TO_MESH["physicsnemo.nn.functional"]:
        assert namespace[name].__module__.startswith("physicsnemo.mesh.")


@pytest.mark.parametrize("module", list(MOVED_TO_MESH))
def test_unknown_name_raises_attribute_error(module):
    """Other names raise AttributeError; safe_normalize never shipped here."""
    with pytest.raises(AttributeError, match="no attribute 'safe_normalize'"):
        getattr(importlib.import_module(module), "safe_normalize")


def test_library_imports_do_not_use_deprecated_names():
    """No physicsnemo module reaches the moved functionals through nn.functional,
    and importing nn.functional does not import mesh, whose aliases resolve
    lazily.

    Runs in a subprocess because a module body executes once per process: by the
    time this test runs, the modules are already cached, so an in-process check
    would pass regardless of what they import.
    """
    snippet = (
        "import sys\n"
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import physicsnemo.nn.functional\n"
        "    mesh = [m for m in sys.modules if m.startswith('physicsnemo.mesh')]\n"
        "    assert not mesh, mesh\n"
        "    import physicsnemo.datapipes.cae.transolver_datapipe\n"
        "    import physicsnemo.datapipes.transforms\n"
        "    import physicsnemo.mesh\n"
        "    import physicsnemo.models.domino.utils\n"
        "leaked = [\n"
        "    str(w.message) for w in caught\n"
        '    if "from \'physicsnemo.nn.functional" in str(w.message)\n'
        "]\n"
        "assert not leaked, leaked\n"
    )
    subprocess.run(  # noqa: S603 - interpreter and snippet are test constants
        [sys.executable, "-c", snippet], check=True
    )
