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

"""The exact geometry's seams that the patch-then-compare tests force still reach the
code.

Each test here forces a seam as another test forces it (a spy put in a function's
place, a module constant patched) and asserts, through :mod:`._forced`, that the
forced code ran and the other did not, or that the patched name was looked up. If a
function moves into the physicsnemo library and the caller stops looking it up in
ISLA's module, the seam's test here fails, where the patch-then-compare test would
pass vacuously: retarget that test's patch, then this one. A seam whose forcing
test already asserts that the forced code ran (the tree read's resolvers and limits,
the least-norm point, the closest weights) has no test here.
"""

import types

import pytest
import torch

### The modules that own each seam. A move retargets these imports (and the
### patch-then-compare tests' patches); nothing else in this file names a module.
from physicsnemo.experimental.models.isla import _geometry as geometry
from physicsnemo.experimental.models.isla import _geometry_fast as geometry_fast
from physicsnemo.experimental.models.isla._geometry import exact_boundary_geometry
from test.experimental.models.isla._forced import ran, reaches
from test.experimental.models.isla._geometry_cases import (
    _capped_cube,
    _circle,
    _near_wall,
    _sphere,
)

D = torch.float64


# --------------------------------------------------------------------------- the helper
def _watched(x):
    return x + 1


def test_the_helper_sees_a_call_under_any_name_and_a_patch_only_where_looked_up(
    monkeypatch,
):
    """:class:`ran` counts a function's runs whatever name it was called by, on other
    threads too, with their arguments and return values; :func:`reaches` records only
    the calls that look the patched name up (a copy of the name elsewhere, as an
    import into another module makes, bypasses it): the vacuous case it exists for."""
    import threading

    alias = _watched
    owner = types.SimpleNamespace(f=_watched)
    calls = reaches(monkeypatch, owner, "f")
    with ran(_watched) as r:
        alias(1)
        owner.f(2)
        t = threading.Thread(target=_watched, args=(3,))
        t.start()
        t.join()
    assert r.count(_watched) == 3 and r.returns(_watched) == [2, 3, 4]
    assert [c["x"] for c in r.calls(_watched)] == [1, 2, 3]
    assert len(calls) == 1
    alias(4)
    assert r.count(_watched) == 3


# --------------------------------------------------------------------------- the exact geometry
def _geometry_case(dim):
    mesh = _circle(32) if dim == 2 else _sphere(2)
    return _near_wall(dim, 200, 3, band=0.5), mesh.points, mesh.cells


@pytest.mark.parametrize("kernel", ["reference", "fast"])
@pytest.mark.parametrize("dim", [2, 3])
def test_a_named_search_kernel_runs_that_search(dim, kernel, monkeypatch):
    """``exact_boundary_geometry(kernel=...)`` with the two search ops spied
    (test_isla_geometry_fast's ``_run``, test_isla_geometry_thin's ``recorded``): the
    named op's spy is reached, and the fast search's own code (the welded canonical
    cells) runs only under ``"fast"``."""
    spies = {
        n: reaches(monkeypatch, geometry, n)
        for n in ("_closest_candidates", "_closest_candidates_fast")
    }
    fast_only = [geometry_fast.canonical_cells]
    with ran(*fast_only) as r:
        exact_boundary_geometry(*_geometry_case(dim), kernel=kernel)
    named = "_closest_candidates_fast" if kernel == "fast" else "_closest_candidates"
    assert [len(v) for k, v in spies.items() if k == named] == [1]
    assert all(not v for k, v in spies.items() if k != named)
    assert all((r.count(f) > 0) == (kernel == "fast") for f in fast_only)


@pytest.mark.parametrize("kernel", ["reference", "fast"])
def test_the_winding_spies_reach_the_three_dimensional_search(kernel, monkeypatch):
    """``geometry._winding_sign`` and ``geometry._packed_winding_sum`` spied
    (test_isla_geometry_fast's ``_run`` reads the winding number through them): both
    spies are reached under either kernel."""
    sign = reaches(monkeypatch, geometry, "_winding_sign")
    tree_w = reaches(monkeypatch, geometry, "_packed_winding_sum")
    exact_boundary_geometry(*_geometry_case(3), kernel=kernel)
    assert len(sign) == len(tree_w) == 1


def test_the_switch_and_the_accurate_point_reach_the_closest_weights(monkeypatch):
    """``geometry._SWITCH_UNITS`` patched and ``geometry._accurate_weights`` replaced
    (test_isla_geometry_fast's guard tests, test_isla_geometry_thin's switch test):
    on a capped cube's thin cells the reference weights call the accurate point, and
    the switch's patched value changes the weights."""
    p, c, _ = _capped_cube(1e-6, "cap")
    g = torch.Generator().manual_seed(0)
    b = torch.rand(200, 3, dtype=D, generator=g)
    s = p[c[0]].expand(200, 3, 3).contiguous()
    q = ((b / b.sum(1, keepdim=True))[:, :, None] * s).sum(1)
    accurate = reaches(monkeypatch, geometry, "_accurate_weights")
    w1 = geometry._closest_weights(q, s)
    assert accurate
    monkeypatch.setattr(geometry, "_SWITCH_UNITS", float("inf"))
    assert bool((geometry._closest_weights(q, s) != w1).any())
