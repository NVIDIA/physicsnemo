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
import math

import numpy as np
import pytest
import torch

from physicsnemo.core.version_check import check_version_spec
from physicsnemo.nn.functional.neighbors.reference_geodesic import (
    reference_geodesic_exclusions,
)

requires_scipy = pytest.mark.skipif(
    not check_version_spec("scipy", hard_fail=False),
    reason="Reference-geodesic algorithm tests require the optional SciPy dependency",
)


def test_missing_scipy_is_delayed_until_geodesic_preprocessing(monkeypatch):
    """Missing SciPy leaves imports usable and gives an actionable call error."""
    from physicsnemo.core import version_check
    from physicsnemo.nn.functional.neighbors.surface_contact import (
        closest_point_triangle,
    )

    module = importlib.import_module(
        "physicsnemo.nn.functional.neighbors.reference_geodesic"
    )
    available = version_check.is_package_available
    findable = version_check._is_module_findable
    monkeypatch.setattr(
        version_check,
        "is_package_available",
        lambda name: False if name == "scipy" else available(name),
    )
    monkeypatch.setattr(
        version_check,
        "_is_module_findable",
        lambda name: False if name.startswith("scipy") else findable(name),
    )
    monkeypatch.setattr(module._scipy_sparse, "_module", None)

    # Reload exercises the module's import path, not just an already-loaded proxy.
    importlib.reload(module)
    assert module._scipy_sparse._module is None
    points = torch.tensor([[0.25, 0.25, 1.0]])
    triangles = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    torch.testing.assert_close(
        closest_point_triangle(points, triangles).distance, torch.ones(1)
    )
    with pytest.raises(
        ImportError, match="reference_geodesic_exclusions requires SciPy"
    ):
        module.reference_geodesic_exclusions(
            torch.empty(0, 3),
            torch.empty(0, 3, dtype=torch.long),
            torch.empty(0),
            torch.empty(0),
        )


def strip():
    positions = torch.tensor(
        [[float(i), float(j), 0.0] for i in range(5) for j in range(2)]
        + [[0.0, 0.0, 0.01], [1.0, 0.0, 0.01], [0.0, 1.0, 0.01]],
        dtype=torch.float64,
    )
    faces = torch.tensor(
        [[2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1] for i in range(4)]
        + [[10, 11, 12, 12]]
    )
    return positions, faces


def dense_oracle(positions, faces, ng, fg, factor, scale=1.0, minimum=0.0):
    """Independent Floyd-Warshall oracle on tiny test meshes only."""
    n = len(positions)
    d = np.full((n, n), np.inf)
    np.fill_diagonal(d, 0.0)
    xyz = positions.numpy()
    for face in faces.tolist():
        corners = face[:3] if len(face) == 4 and face[2] == face[3] else face
        for a, b in zip(corners, corners[1:] + corners[:1]):
            d[a, b] = d[b, a] = np.linalg.norm(xyz[a] - xyz[b])
    for k in range(n):
        d = np.minimum(d, d[:, k, None] + d[None, k, :])
    return {
        (q, f)
        for q in range(n)
        for f, face in enumerate(faces.tolist())
        if q in face
        or d[q, face].min()
        < factor * max(minimum, scale * (float(ng[q]) + float(fg[f])))
    }


@pytest.mark.parametrize("factor", [0.0, 1.0, math.sqrt(2), 3.0])
@pytest.mark.parametrize("scale,minimum", [(1.0, 0.0), (0.7, 0.2), (0.0, 1.1)])
@requires_scipy
def test_matches_dense_oracle_with_pair_specific_gaps(factor, scale, minimum):
    p, f = strip()
    ng = torch.linspace(0.0, 1.4, len(p), dtype=torch.float64)
    fg = torch.tensor([0.1, 0.9, 0.3, 1.2, 0.5], dtype=torch.float64)
    actual = reference_geodesic_exclusions(
        p, f, ng, fg, distance_scale=factor, gap_scale=scale, gap_min=minimum
    )
    expected = dense_oracle(p, f, ng, fg, factor, scale, minimum)
    assert list(map(tuple, actual.T.tolist())) == sorted(expected)
    assert (0, 4) not in expected  # Nearly coincident but disconnected sheet.


@requires_scipy
def test_strict_boundary_incidence_and_optional_disable():
    p, f = strip()
    g = torch.ones(len(p), dtype=torch.float64) / 2
    fg = torch.ones(len(f), dtype=torch.float64) / 2

    def pairs(factor):
        return set(
            map(
                tuple,
                reference_geodesic_exclusions(
                    p, f, g, fg, distance_scale=factor
                ).T.tolist(),
            )
        )

    assert (0, 0) in pairs(0.0)
    assert (0, 1) not in pairs(1.0)  # Nearest target vertex exactly 1 mm away.
    assert (0, 1) in pairs(1.0 + 1e-10)
    assert (0, 2) in pairs(2.1)  # Can span more than one ring if gap warrants it.
    zeros = reference_geodesic_exclusions(p, f, g * 0, fg * 0)
    assert set(map(tuple, zeros.T.tolist())) == pairs(0.0)


@requires_scipy
def test_graph_distance_not_euclidean_and_no_quad_diagonal_shortcut():
    p, f = strip()
    # Far material end folds spatially next to node 0, without shortening the
    # original edge path. The input is the reference, not that folded state.
    folded = p.clone()
    folded[8] = p[0]
    result = reference_geodesic_exclusions(
        p, f, torch.ones(len(p)), torch.ones(len(f)), distance_scale=0.75
    )
    pairs = set(map(tuple, result.T.tolist()))
    assert torch.linalg.norm(folded[0] - folded[8]) == 0
    assert (0, 3) not in pairs
    p = torch.tensor(
        [[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 1, 0], [1, 2, 0]],
        dtype=torch.float64,
    )
    f = torch.tensor([[0, 1, 2, 3], [2, 4, 5, 5]])
    pairs = reference_geodesic_exclusions(
        p, f, torch.ones(6) / 2, torch.ones(2) / 2, distance_scale=1.5
    )
    assert (0, 1) not in set(map(tuple, pairs.T.tolist()))  # Path 2, not sqrt(2).


@requires_scipy
def test_zero_length_material_edges_and_no_normalization_or_gradient_graph():
    p, f = strip()
    p[2] = p[0]
    p.requires_grad_()
    ng = torch.ones(len(p), requires_grad=True) * 0.2
    fg = torch.ones(len(f), requires_grad=True) * 0.2
    actual = reference_geodesic_exclusions(p, f, ng, fg)
    expected = dense_oracle(p.detach(), f, ng.detach(), fg.detach(), math.sqrt(2))
    assert set(map(tuple, actual.T.tolist())) == expected
    assert actual.grad_fn is None
    # Rigid transforms and consistent physical-unit scaling preserve eligibility.
    transformed = p.detach()[:, [2, 0, 1]] + torch.tensor([100.0, -20.0, 7.0])
    torch.testing.assert_close(
        actual, reference_geodesic_exclusions(transformed, f, ng, fg)
    )
    torch.testing.assert_close(
        actual, reference_geodesic_exclusions(p * 1000, f, ng * 1000, fg * 1000)
    )


@requires_scipy
def test_triangle_padding_and_empty_mesh():
    p = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    f = torch.tensor([[0, 1, 2], [1, 2, 3]])
    g = torch.ones(4)
    fg = torch.ones(2)
    torch.testing.assert_close(
        reference_geodesic_exclusions(p, f, g, fg),
        reference_geodesic_exclusions(p, torch.cat((f, f[:, -1:]), 1), g, fg),
    )
    assert reference_geodesic_exclusions(
        p, torch.empty((0, 4), dtype=torch.long), g, torch.empty(0)
    ).shape == (2, 0)


@pytest.mark.parametrize(
    "option,value",
    [
        ("distance_scale", -1.0),
        ("distance_scale", float("nan")),
        ("gap_scale", float("inf")),
        ("gap_min", -1.0),
        ("max_pairs", 0),
        ("max_pairs", True),
    ],
)
@requires_scipy
def test_invalid_options(option, value):
    p, f = strip()
    with pytest.raises(ValueError):
        reference_geodesic_exclusions(
            p, f, torch.ones(len(p)), torch.ones(len(f)), **{option: value}
        )


@requires_scipy
def test_budget_never_silently_truncates():
    p, f = strip()
    with pytest.raises(RuntimeError, match="no pairs truncated"):
        reference_geodesic_exclusions(
            p, f, torch.ones(len(p)), torch.ones(len(f)), max_pairs=1
        )
    with pytest.raises(RuntimeError, match="no pairs truncated"):
        reference_geodesic_exclusions(
            p, f, torch.ones(len(p)) * 10, torch.ones(len(f)) * 10, max_pairs=24
        )


@pytest.mark.parametrize(
    "which", ["position", "node_gap", "face_gap", "index", "repeated"]
)
@requires_scipy
def test_invalid_geometry(which):
    p, f = strip()
    ng = torch.ones(len(p))
    fg = torch.ones(len(f))
    if which == "position":
        p[0, 0] = float("nan")
    elif which == "node_gap":
        ng[0] = -1
    elif which == "face_gap":
        fg[0] = float("inf")
    elif which == "index":
        f[0, 0] = len(p)
    else:
        f[0] = torch.tensor([0, 0, 1, 2])
    with pytest.raises(ValueError):
        reference_geodesic_exclusions(p, f, ng, fg)
