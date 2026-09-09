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

"""Regression tests for synchronization-free mesh linear algebra.

``torch.linalg.inv`` and ``torch.linalg.solve`` read a LAPACK/cuSOLVER status code
back to the host purely so they can raise on singular input, which costs a
device-to-host synchronization on every call. The mesh hot paths use the ``_ex``
variants with ``check_errors=False`` instead. These tests pin both halves of that
trade: that the synchronization is really gone, and what the lost error check now
means for callers who feed in a singular matrix anyway.
"""

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.geometry.dual_meshes import compute_cotan_weights_fem
from physicsnemo.mesh.transformations.geometric import transform

_CUDA = torch.cuda.is_available()


def _curve_mesh(device: torch.device | str) -> Mesh:
    """Return a non-axis-aligned 1-manifold with all normal caches populated."""
    mesh = Mesh(
        points=torch.tensor(
            [[0.0, 0.0], [1.0, 0.25], [1.5, 1.5]],
            dtype=torch.float64,
            device=device,
        ),
        cells=torch.tensor([[0, 1], [1, 2]], dtype=torch.long, device=device),
    )
    _ = mesh.cell_areas
    _ = mesh.cell_normals
    _ = mesh.point_normals
    return mesh


def _triangle_mesh(device: torch.device | str) -> Mesh:
    return Mesh(
        points=torch.tensor(
            [[0.0, 0.0], [1.25, 0.1], [0.2, 1.1]],
            dtype=torch.float64,
            device=device,
        ),
        cells=torch.tensor([[0, 1, 2]], dtype=torch.long, device=device),
    )


def test_singular_matrix_with_assume_invertible_poisons_caches(device) -> None:
    """``assume_invertible=True`` is a promise, and breaking it is now silent.

    Without the error check there is nothing left to detect a singular matrix, so
    the inverse-transpose step yields non-finite values and they reach the caller
    through the ordinary accessors rather than through an exception. This is the
    documented contract for ``assume_invertible=True``; the point of pinning it is
    that ``assume_invertible=None`` must remain the safe default.
    """
    mesh = _curve_mesh(device)
    # Rank-1 projection onto the x-axis: singular, and a plausible user input.
    singular = torch.tensor(
        [[1.0, 0.0], [0.0, 0.0]], dtype=mesh.points.dtype, device=device
    )

    promised = transform(mesh, singular, assume_invertible=True)
    assert not bool(torch.isfinite(promised.cell_normals).any())
    assert not bool(torch.isfinite(promised.cell_areas).any())
    assert not bool(torch.isfinite(promised.point_normals).any())

    # The escape hatches must still be safe: both drop the caches and recompute.
    for kwargs in ({}, {"assume_invertible": False}):
        checked = transform(mesh, singular, **kwargs)
        assert ("cell", "normals") not in checked._cache.keys(include_nested=True)
        assert bool(torch.isfinite(checked.cell_areas).all())


@pytest.mark.skipif(not _CUDA, reason="CUDA required to detect host synchronizations")
def test_cotan_weight_inverse_is_sync_free(monkeypatch) -> None:
    """FEM Gram-matrix inversion must not synchronize CUDA for error checks."""
    device = torch.device("cuda")
    mesh = _triangle_mesh(device)
    _ = mesh.cell_areas

    # Topology extraction has independently data-dependent output. Replace it
    # with the known topology so this test isolates the fixed-size inversion.
    edges = torch.tensor([[0, 1], [0, 2], [1, 2]], device=device)
    inverse = torch.arange(3, device=device)
    from physicsnemo.mesh.utilities import _topology

    monkeypatch.setattr(_topology, "extract_unique_edges", lambda _: (edges, inverse))

    # Warm lazy CUDA-library initialization and allocator growth outside the guard.
    compute_cotan_weights_fem(mesh)
    torch.cuda.synchronize()

    # The surrounding topology path contains separate data-dependent indexing,
    # so guard the inversion call itself. The called assertion also makes this a
    # regression test against reverting to the synchronizing ``linalg.inv``.
    original_inv_ex = torch.linalg.inv_ex
    called = False

    def checked_inv_ex(*args, **kwargs):
        nonlocal called
        called = True
        previous_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            return original_inv_ex(*args, **kwargs)
        finally:
            torch.cuda.set_sync_debug_mode(previous_mode)

    monkeypatch.setattr(torch.linalg, "inv_ex", checked_inv_ex)
    compute_cotan_weights_fem(mesh)
    torch.cuda.synchronize()
    assert called


@pytest.mark.skipif(not _CUDA, reason="CUDA required to detect host synchronizations")
def test_cached_normal_transformation_is_sync_free() -> None:
    """Inverse-transpose cache propagation must not synchronize CUDA.

    A 1-manifold with ``assume_invertible=True`` is the configuration in which
    ``transform`` has no other reason to synchronize: it skips both the runtime
    determinant test and the similarity test, which are host reads of a device
    scalar. Keep it that way when editing this test -- ``set_sync_debug_mode`` is a
    prototype feature, and a ``bool(cuda_tensor)`` under it has been observed to
    abort the interpreter rather than raise, which would take the whole session
    down instead of failing one test.
    """
    device = torch.device("cuda")
    mesh = _curve_mesh(device)
    matrix = torch.tensor(
        [[1.75, 0.3], [-0.2, 1.25]], dtype=mesh.points.dtype, device=device
    )

    transform(mesh, matrix, assume_invertible=True)
    torch.cuda.synchronize()

    previous_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        transform(mesh, matrix, assume_invertible=True)
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)
    torch.cuda.synchronize()
