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

"""Geometric transforms decide on the host whatever the host already knows.

Python-number arguments, string axes, rotations, and scalar scales need no
device synchronization: their validity, invertibility, and similarity are known
without reading device memory. ``assume_valid_axis`` and ``assume_similarity``
let callers promise the same for device tensors and general matrices.
"""

import pytest
import torch

from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.calculus.measure import point_measures, set_point_measures
from physicsnemo.mesh.primitives.planar import structured_grid
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.mesh.transformations.geometric import (
    _is_similarity_transform,
    _scale_assumptions,
    rotation_matrix,
    scale_matrix,
    transform,
)
from test.mesh.mesh.test_slicing_sync import _cuda_sync_budget

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required to detect synchronizations"
)

_CACHED = (
    ("cell", "areas"),
    ("cell", "centroids"),
    ("cell", "normals"),
    ("point", "normals"),
)


def _surface(device) -> Mesh:
    """A closed surface in 3D with every propagated cache populated."""
    mesh = sphere_icosahedral.load(subdivisions=2, device=device)
    _ = mesh.cell_areas, mesh.cell_centroids, mesh.cell_normals, mesh.point_normals
    return mesh


def _cache_keys(mesh: Mesh) -> set:
    return {key for key in _CACHED if mesh._cache.get(key, None) is not None}


def _assert_same_mesh(actual: Mesh, expected: Mesh) -> None:
    torch.testing.assert_close(actual.points, expected.points, rtol=0, atol=0)
    assert _cache_keys(actual) == _cache_keys(expected)
    for key in _cache_keys(expected):
        torch.testing.assert_close(actual._cache[key], expected._cache[key])


### Host-side decisions


@pytest.mark.parametrize(
    "factor, expected",
    [
        (2.0, (True, True)),
        (0.0, (False, True)),
        (1e-4, (False, True)),  # |det| = 1e-12 is below the threshold in 3D
        ([1.0, 2.0, 3.0], (True, False)),
        ([2.0, -2.0, 2.0], (True, True)),  # a reflection is still a similarity
        ([1.0, 0.0, 1.0], (False, False)),
        (1e160, (True, True)),  # |det| overflows to inf
        ([1e160, 1e-160, 1.0], (True, False)),  # 1e160 ** 2 overflows to inf
        (torch.tensor(2.0), (None, True)),
        (torch.tensor([1.0, 2.0, 3.0]), (None, None)),
    ],
)
def test_scale_assumptions(factor, expected):
    assert _scale_assumptions(factor, 3) == expected


@pytest.mark.parametrize(
    "factor", [2.0, 0.0, [1.0, 2.0, 3.0], [2.0, -2.0, 2.0], [1.0, 0.0, 1.0]]
)
def test_scale_matches_runtime_tests(factor, device):
    """Deciding on the host gives exactly what the runtime tests decide."""
    mesh = _surface(device)
    matrix = scale_matrix(factor, 3, mesh.points.device, mesh.points.dtype)
    _assert_same_mesh(mesh.scale(factor), transform(mesh, matrix))


@pytest.mark.parametrize(
    "factor", [[1e160, 1e-160, 1.0], [1e200, 1e200, 1e-200], [1e-200, 1e-200, 1e-200]]
)
def test_extreme_scale_assumptions_match_runtime_tests(factor):
    """Factors whose products overflow or underflow in float64 decide alike."""
    matrix = scale_matrix(factor, 3, torch.device("cpu"), torch.float64)
    runtime = (
        bool(torch.linalg.det(matrix).abs() > 1e-10),
        _is_similarity_transform(matrix),
    )
    assert _scale_assumptions(factor, 3) == runtime


@pytest.mark.parametrize("axis", [[0.0, 0.0, 2.0], "z", (1.0, 1.0, 0.0)])
@pytest.mark.parametrize("center", [None, [0.5, -1.0, 2.0]])
def test_rotate_matches_runtime_tests(axis, center, device):
    mesh = _surface(device)
    matrix = rotation_matrix(0.4, axis, 3, mesh.points.device, mesh.points.dtype)
    if center is None:
        expected = transform(mesh, matrix)
    else:
        expected = transform(mesh.translate([-0.5, 1.0, -2.0]), matrix)
        expected = expected.translate(center)
    _assert_same_mesh(mesh.rotate(0.4, axis=axis, center=center), expected)


@pytest.mark.parametrize("matrix_kind", ["pivoting rotation", "shear"])
def test_propagated_caches_match_recomputation(matrix_kind, device):
    """Propagated normals and areas equal those recomputed from the moved points."""
    mesh = _surface(device)
    if matrix_kind == "pivoting rotation":  # its LU factorization needs row swaps
        matrix = rotation_matrix(2.0, "z", 3, mesh.points.device, mesh.points.dtype)
    else:
        matrix = torch.tensor(
            [[1.0, 0.5, 0.0], [0.0, 1.0, 0.2], [0.0, 0.0, 1.0]], device=device
        )
    moved = transform(mesh, matrix, assume_invertible=True)
    fresh = Mesh(points=moved.points, cells=moved.cells)

    assert ("cell", "normals") in _cache_keys(moved)
    torch.testing.assert_close(moved.cell_normals, fresh.cell_normals)
    torch.testing.assert_close(moved.cell_areas, fresh.cell_areas)
    if matrix_kind == "pivoting rotation":
        assert ("point", "normals") in _cache_keys(moved)
        torch.testing.assert_close(moved.point_normals, fresh.point_normals)


### Promises


def test_assume_similarity_controls_point_normal_propagation(device):
    mesh = _surface(device)
    rotation = mesh.rotate(0.3, axis="x")
    shear = torch.tensor(
        [[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device
    )

    # Default: decided at runtime
    assert ("point", "normals") in _cache_keys(rotation)
    assert ("point", "normals") not in _cache_keys(transform(mesh, shear))
    # Promises override the runtime test either way
    kept = transform(mesh, shear, assume_invertible=True, assume_similarity=True)
    assert ("point", "normals") in _cache_keys(kept)
    quarter = rotation.points.new_tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.6, -0.8], [0.0, 0.8, 0.6]]
    )
    dropped = transform(mesh, quarter, assume_similarity=False)
    assert ("point", "normals") not in _cache_keys(dropped)


def test_assume_similarity_scales_point_measures(device):
    mesh = structured_grid.load(n_x=4, n_y=3, device=device)
    surface = Mesh(points=torch.nn.functional.pad(mesh.points, (0, 1)))
    set_point_measures(
        surface, torch.ones(surface.n_points, device=device), dimension=2
    )
    stretch = 2.0 * torch.eye(3, device=device)

    with pytest.raises(ValueError, match="support geometry"):
        surface.scale([2.0, 3.0, 4.0])
    torch.testing.assert_close(
        point_measures(transform(surface, stretch, assume_similarity=True)),
        torch.full((surface.n_points,), 4.0, device=device),
    )
    with pytest.raises(ValueError, match="support geometry"):
        transform(surface, stretch, assume_similarity=False)


def test_rotation_axis_validation(device):
    mesh = _surface(device)
    for axis in ([0.0, 0.0, 0.0], torch.zeros(3, device=device)):
        with pytest.raises(ValueError, match="near-zero length"):
            mesh.rotate(0.3, axis=axis)
    with pytest.raises(NotImplementedError, match="axis shape \\(3,\\)"):
        mesh.rotate(0.3, axis=[1.0, 0.0])
    with pytest.raises(ValueError, match="implies 3D rotation"):
        structured_grid.load(device=device).rotate(0.3, axis=[0.0, 0.0, 1.0])

    # A broken promise is not checked: a zero axis gives a NaN rotation
    promised = mesh.rotate(
        0.3, axis=torch.zeros(3, device=device), assume_valid_axis=True
    )
    assert promised.points.isnan().all()


def test_domain_mesh_threads_assumptions(device):
    interior = structured_grid.load(n_x=4, n_y=3, device=device)
    interior = Mesh(
        points=torch.nn.functional.pad(interior.points, (0, 1)), cells=interior.cells
    )
    domain = DomainMesh(interior=interior, boundaries={"wall": _surface(device)})
    shear = torch.tensor(
        [[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device
    )

    kept = domain.transform(shear, assume_invertible=True, assume_similarity=True)
    assert ("point", "normals") in _cache_keys(kept.boundaries["wall"])
    assert ("point", "normals") in _cache_keys(
        domain.rotate(0.3, axis="z").boundaries["wall"]
    )
    assert ("point", "normals") not in _cache_keys(
        domain.scale([1.0, 2.0, 3.0]).boundaries["wall"]
    )
    axis = torch.tensor([0.0, 0.0, 1.0], device=device)
    torch.testing.assert_close(
        domain.rotate(0.3, axis=axis, assume_valid_axis=True).interior.points,
        domain.rotate(0.3, axis="z").interior.points,
    )


### Synchronizations


@requires_cuda
@pytest.mark.parametrize(
    "operation",
    [
        lambda m: m.translate([1.0, 2.0, 3.0]),
        lambda m: m.rotate(0.3, axis=[0.0, 0.0, 1.0]),
        lambda m: m.rotate(0.3, axis="y", center=[1.0, 0.0, 0.0]),
        lambda m: m.rotate(
            0.3, axis=torch.ones(3, device="cuda"), assume_valid_axis=True
        ),
        lambda m: m.scale(2.0),
        lambda m: m.scale([1.0, 2.0, 3.0], center=(0.0, 1.0, 0.0)),
        lambda m: m.scale(torch.full((), 2.0, device="cuda"), assume_invertible=True),
        lambda m: m.transform(
            torch.eye(3, device="cuda"), assume_invertible=True, assume_similarity=True
        ),
    ],
)
def test_transforms_are_sync_free(operation):
    mesh = _surface("cuda")
    operation(mesh)  # warm up lazy CUDA initialization
    torch.cuda.synchronize()

    with _cuda_sync_budget(0):
        operation(mesh)
    torch.cuda.synchronize()


@requires_cuda
def test_python_arguments_stay_on_the_host_under_a_cuda_default_device():
    """Arguments built from Python values do not follow the default device."""
    mesh = _surface("cuda")
    operations = [
        lambda: mesh.translate([1.0, 2.0, 3.0]),
        lambda: mesh.rotate(0.3, axis=[0.0, 0.0, 1.0]),
        lambda: mesh.rotate(0.3, axis="y", center=[1.0, 0.0, 0.0]),
        lambda: mesh.scale([1.0, 2.0, 3.0]),
    ]
    with torch.device("cuda"):
        for operation in operations:
            operation()
        torch.cuda.synchronize()

        with _cuda_sync_budget(0):
            for operation in operations:
                operation()
    torch.cuda.synchronize()


@requires_cuda
def test_planar_rotation_and_domain_transforms_are_sync_free():
    grid = structured_grid.load(n_x=4, n_y=3, device="cuda")
    _ = grid.cell_areas
    interior = Mesh(
        points=torch.nn.functional.pad(grid.points, (0, 1)), cells=grid.cells
    )
    domain = DomainMesh(interior=interior, boundaries={"wall": _surface("cuda")})
    operations = [
        lambda: grid.rotate(0.3),
        lambda: domain.translate([1.0, 0.0, 0.0]),
        lambda: domain.rotate(0.3, axis="x", center=[0.0, 1.0, 0.0]),
        lambda: domain.scale(2.0),
        lambda: domain.scale([1.0, 2.0, 3.0]),
    ]
    for operation in operations:
        operation()
    torch.cuda.synchronize()

    with _cuda_sync_budget(0):
        for operation in operations:
            operation()
    torch.cuda.synchronize()


@requires_cuda
def test_random_augmentations_are_sync_free():
    from physicsnemo.datapipes.transforms.mesh.augmentations import (
        RandomRotateMesh,
        RandomScaleMesh,
    )

    mesh = _surface("cuda")
    transforms = [
        RandomScaleMesh().to("cuda"),
        RandomRotateMesh(mode="uniform").to("cuda"),
    ]
    for t in transforms:
        t(mesh)
    torch.cuda.synchronize()

    with _cuda_sync_budget(0):
        for t in transforms:
            t(mesh)
    torch.cuda.synchronize()
