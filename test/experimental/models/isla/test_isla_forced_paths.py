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

"""The seams that ISLA's patch-then-compare tests force still reach the code.

Each test here forces one seam as an existing test forces it (a resolver patched to
name a path, a module constant patched to disqualify one, a spy put in a function's
place) and asserts, through :mod:`._forced`, that the forced code ran and the other
did not, or that the patched name was looked up. When a function moves into
the physicsnemo library and the caller stops looking it up in ISLA's module, the test
of its seam fails here, instead of the patch-then-compare test passing vacuously:
retarget that test's patch (and the owner named below), then this one. The forced
paths' outputs are compared with the reference's in the tests that force them; here
only that they ran.
"""

import types

import pytest
import torch

### The modules that own each seam. A move retargets these imports (and the
### patch-then-compare tests' patches); nothing else in this file names a module.
from physicsnemo.experimental.models.isla import ISLA
from physicsnemo.experimental.models.isla import _fused_basis as fused_basis
from physicsnemo.experimental.models.isla import _geometry as geometry
from physicsnemo.experimental.models.isla import _geometry_fast as geometry_fast
from physicsnemo.experimental.models.isla import _tree_build as tree_build
from physicsnemo.experimental.models.isla import _tree_read as tree_read
from physicsnemo.experimental.models.isla import _tree_walk as tree_walk
from physicsnemo.experimental.models.isla import model as isla_model
from physicsnemo.experimental.models.isla._geometry import exact_boundary_geometry
from physicsnemo.experimental.models.isla._tree_read import (
    BoundaryForest,
    _KernelBasis,
    kernel_geo_width,
)
from physicsnemo.mesh.spatial.cluster_tree import ClusterTree
from test.experimental.models.isla._forced import ran, reaches
from test.experimental.models.isla._geometry_cases import (
    _box_ties,
    _capped_cube,
    _circle,
    _near_wall,
    _sphere,
)
from test.experimental.models.isla._meshes import cloud, live, points_mesh, soup

D = torch.float64
#: The reference walk and build: GLOBE's cluster tree, tree by tree.
REFERENCE_WALK = ClusterTree.find_point_interaction_pairs
REFERENCE_BUILD = ClusterTree.from_points


def _forest(n=300, seed=1, **kw):
    pts, nrm, w, _ = cloud(n, d=3, seed=seed)
    q, _, _, _ = cloud(40, d=3, seed=seed + 1, scale=(3.0,) * 3)
    return BoundaryForest.build(pts, w, nrm, None, theta=1.0, **kw), q, (pts, w, nrm)


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


# --------------------------------------------------------------------------- the walk
@pytest.mark.parametrize("walk", ["reference", "fast"])
def test_a_forced_walk_runs_that_walk(walk, monkeypatch):
    """``tree_read.resolve_walk`` patched to name a walk (test_isla_tree_read's
    ``test_tree_walk_resolution_and_the_models_walks_agree`` and
    ``..._falls_back_to_the_reference_past_its_limits``): the forest's lists come from
    that walk alone (the fast walk on Warp's CPU backend)."""
    f, q, _ = _forest()
    calls = []
    monkeypatch.setattr(
        tree_read, "resolve_walk", lambda *a, w=walk: calls.append(a) or w
    )
    with ran(tree_walk.walk_pairs, REFERENCE_WALK) as r:
        f.point_pairs(q)
    assert calls, "the patched resolver was not consulted"
    fast, ref = r.count(tree_walk.walk_pairs), r.count(REFERENCE_WALK)
    assert fast == (1 if walk == "fast" else 0)
    assert (ref > 0) == (walk == "reference")
    if walk == "fast":
        assert r.returns(tree_walk.walk_pairs)[0] is not None


def test_a_walk_stack_below_the_forests_depth_disqualifies_the_fast_walk(monkeypatch):
    """``tree_walk.WALK_STACK`` patched below the forest's depth (the past-its-limits
    test): the function that decides reads the patched constant."""
    f, q, _ = _forest()
    on_cuda = types.SimpleNamespace(is_cuda=True, device=torch.device("cuda"))
    with ran(tree_walk.walk_fits) as r:
        assert tree_read.resolve_walk("auto", on_cuda, 1.0, f) == "fast"
        monkeypatch.setattr(tree_walk, "WALK_STACK", len(f.levels) + 1)
        assert tree_read.resolve_walk("auto", on_cuda, 1.0, f) == "reference"
    assert r.returns(tree_walk.walk_fits) == [True, False]


def test_lists_past_warps_limit_fall_back_to_the_reference_walk(monkeypatch):
    """``tree_walk.WARP_MAX_ELEMENTS`` patched to the list length, the fast walk forced
    (the past-its-limits test): the fast walk runs, finds its lists too long and
    returns None, and the reference walks the points."""
    f, q, _ = _forest()
    n = f.point_pairs(q, walk="reference")[0].numel()
    monkeypatch.setattr(tree_walk, "WARP_MAX_ELEMENTS", n)
    monkeypatch.setattr(tree_read, "resolve_walk", lambda *a: "fast")
    with ran(tree_walk.walk_pairs, REFERENCE_WALK) as r:
        f.point_pairs(q)
    assert r.returns(tree_walk.walk_pairs) == [None]
    assert r.count(REFERENCE_WALK) > 0


# --------------------------------------------------------------------------- the build
@pytest.mark.parametrize("build", ["reference", "fast"])
def test_a_forced_build_runs_that_build(build, monkeypatch):
    """``tree_read.resolve_tree_build`` patched to name a build
    (``test_tree_build_auto_resolves_to_the_reference_off_cuda``,
    ``test_tree_read_kernel_is_the_tree_reads_one_execution_attribute``): the forest
    comes from that build alone (the fast one from ``forest_tables``)."""
    calls = []
    monkeypatch.setattr(
        tree_read, "resolve_tree_build", lambda *a, b=build: calls.append(a) or b
    )
    with ran(tree_build.forest_tables, REFERENCE_BUILD) as r:
        _forest()
    assert calls, "the patched resolver was not consulted"
    tables = r.returns(tree_build.forest_tables)
    if build == "fast":
        assert (
            len(tables) == 1 and tables[0] is not None and r.count(REFERENCE_BUILD) == 0
        )
    else:
        assert not tables and r.count(REFERENCE_BUILD) > 0


def test_a_cell_limit_at_the_cell_count_leaves_the_forest_to_the_reference(monkeypatch):
    """``tree_walk.WARP_MAX_ELEMENTS`` patched to the cell count and ``tree_read.forest_tables``
    spied (``test_fast_forest_build_refuses_what_it_cannot_build``): the spy is
    reached, the tables decline and the reference build makes the forest."""
    calls = reaches(monkeypatch, tree_read, "forest_tables")
    monkeypatch.setattr(tree_walk, "WARP_MAX_ELEMENTS", 300)
    with ran(tree_build.forest_tables, REFERENCE_BUILD) as r:
        _forest(build="fast")
    assert len(calls) == 1
    assert (
        r.returns(tree_build.forest_tables) == [None] and r.count(REFERENCE_BUILD) > 0
    )


# --------------------------------------------------------------------------- the kernel basis
def test_the_fused_basis_seam_and_the_pair_chunk_reach_the_chunked_basis(monkeypatch):
    """``tree_read.fused_basis.fused_kernel_basis`` replaced by a CPU stand-in and
    ``tree_read.TREE_READ_PAIR_CHUNK`` patched
    (``test_fused_basis_chunks_sum_their_weight_gradients_as_the_reference``, and the
    spy of ``test_tree_read_pairs_fused_on_cuda_builds_the_basis_in_the_fused_kernel``):
    the basis from the plan calls the stand-in once per chunk of the patched size."""
    calls = []

    def stand_in(weights, qi, si, q_off, s_off, geometry):
        calls.append(int(qi.shape[0]))
        return qi.new_zeros(qi.shape[0], 16, dtype=D)

    monkeypatch.setattr(tree_read, "TREE_READ_PAIR_CHUNK", 300)
    monkeypatch.setattr(tree_read.fused_basis, "fused_kernel_basis", stand_in)
    assert tree_read.fused_basis is fused_basis
    plan = types.SimpleNamespace(
        qi=torch.zeros(1000, dtype=torch.int32), bj=torch.zeros(1000, dtype=torch.int32)
    )
    basis = _KernelBasis(kernel_geo_width(1, 0)).double()
    geometry_ = {"query_r": torch.zeros(1, 3, dtype=D)}
    with ran(tree_read._chunked_basis_from_plan) as r:
        tree_read._chunked_basis_from_plan(
            basis, plan, 0, 1000, 0, 0, geometry_, weights=()
        )
    assert r.count(tree_read._chunked_basis_from_plan) == 1
    assert calls == [300, 300, 300, 100]


# --------------------------------------------------------------------------- the model's seams
def _dtt_forward():
    """One evaluation forward of a small float64 ISLA-DTT model (its read live) on
    CPU, on a 3-D cloud's soup with 40 prediction points."""
    torch.manual_seed(0)
    m = ISLA(
        output_schema={"p": {"rank": 0}, "v": {"rank": 1}},
        boundary_schemas={"wall": {}},
        global_schema={"g": {"rank": 1}},
        interior=True,
        hidden=32,
        n_layers=2,
        n_slices=8,
    )
    m = live(m).double().eval()
    pts, nrm, w, drv = cloud(200, d=3, seed=1)
    q, _, _, _ = cloud(40, d=3, seed=2, scale=(3.0,) * 3)
    with torch.no_grad():
        m(
            boundary_meshes={"wall": soup(pts, nrm, w)},
            prediction_points=points_mesh(q),
            global_data={"g": drv},
        )


def test_the_models_pair_assembly_seam_reaches_the_read(monkeypatch):
    """``model.walk_range_pairs`` spied
    (test_isla_tree_read's ``test_tree_read_cuda_matches_cpu_float64`` captures the
    plans through it): an ISLA-DTT forward looks the name up in the model's module,
    once, and the tree read's pair assembly runs behind it."""
    calls = reaches(monkeypatch, isla_model, "walk_range_pairs")
    with ran(tree_read.walk_range_pairs) as r:
        _dtt_forward()
    assert len(calls) == 1 and r.count(tree_read.walk_range_pairs) == 1


@pytest.mark.parametrize("kernel", ["reference", "fused"])
def test_a_forced_tree_read_kernel_reaches_the_read(kernel, monkeypatch):
    """``ISLA._resolve_tree_read_kernel`` patched to name the read
    (``test_tree_read_kernel_resolves_to_the_reference_on_cpu`` forces ``"fused"`` on
    CPU): the model consults the patched resolver and hands its name to the walk
    (``tree_read_walk``'s ``kernel``), whose pairs' basis comes from the plan
    (``_chunked_basis_from_plan``) either way."""
    calls = []
    monkeypatch.setattr(
        ISLA,
        "_resolve_tree_read_kernel",
        lambda self, p, k=kernel: calls.append(p) or k,
    )
    with ran(tree_read.tree_read_walk, tree_read._chunked_basis_from_plan) as r:
        _dtt_forward()
    assert calls, "the patched resolver was not consulted"
    assert [c["kernel"] for c in r.calls(tree_read.tree_read_walk)] == [kernel]
    assert r.count(tree_read._chunked_basis_from_plan) > 0


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


def test_the_least_norm_point_reaches_the_tie_directions(monkeypatch):
    """``geometry._min_norm_points`` spied
    (test_isla_interior_geometry's ``test_tie_directions_take_the_exact_least_norm_point``):
    the tie directions of a square's diagonals call it."""
    calls = reaches(monkeypatch, geometry, "_min_norm_points")
    mesh, q, _ = _box_ties(2)
    exact_boundary_geometry(q, mesh.points, mesh.cells)
    assert calls


def test_the_fast_weights_fall_back_through_the_closest_weights_seam(monkeypatch):
    """``geometry._closest_weights`` spied (test_isla_geometry_fast's
    ``test_guard_kernel_falls_back_to_the_torch_code_off_cuda``): off CUDA the fast
    weights call the torch code through the module's name."""
    closest = geometry._closest_weights
    calls = reaches(monkeypatch, geometry, "_closest_weights")
    g = torch.Generator().manual_seed(1)
    s = torch.randn(50, 3, 3, dtype=D, generator=g)
    q = torch.randn(50, 3, dtype=D, generator=g)
    closest(q, s, None, fast=True)
    assert len(calls) == 1
