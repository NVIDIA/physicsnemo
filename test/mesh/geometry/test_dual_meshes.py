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

"""Robustness of the FEM cotangent weights against degenerate cells.

``compute_cotan_weights_fem`` inverts a per-cell Gram matrix. Degenerate cells make
that matrix singular, so it substitutes an isotropic one for them. These tests pin
the two ways that substitution has to stay scale-free: it must handle cells with no
extent at all, and cells that are flat but live at large coordinate values.
"""

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.geometry.dual_meshes import compute_cotan_weights_fem
from physicsnemo.mesh.primitives import surfaces


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_padding_leaves_cotan_weights_unchanged(device, dtype) -> None:
    """``Mesh.pad`` promises null cells that don't affect computations.

    Padding cells put every vertex on the same point, so their edge matrix is
    exactly zero and their Gram matrix is exactly singular. Regularizing by adding
    a bare identity used to leave that untouched in float32, because the degeneracy
    threshold underflowed, so the inverse raised and ``laplacian`` died on any
    padded float32 mesh.
    """
    base = surfaces.sphere_uv.load(theta_resolution=8, phi_resolution=8)
    mesh = Mesh(
        points=base.points.to(dtype=dtype, device=device),
        cells=base.cells.to(device=device),
    )
    padded = mesh.pad_to_next_power()
    assert padded.n_cells > mesh.n_cells, "test needs padding to actually happen"

    weights, edges = compute_cotan_weights_fem(mesh)
    padded_weights, padded_edges = compute_cotan_weights_fem(padded)

    n_real = len(edges)
    torch.testing.assert_close(padded_edges[:n_real], edges)
    torch.testing.assert_close(padded_weights[:n_real], weights)
    # Edges that exist only because of the padding must carry no weight at all.
    assert bool((padded_weights[n_real:] == 0).all())

    # The public entry point that the singular inverse used to break.
    values = torch.ones(padded.n_points, dtype=dtype, device=device)
    assert bool(torch.isfinite(padded.laplacian(values)).all())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_degenerate_cell_weights_are_finite_and_scale_invariant(device, dtype) -> None:
    """A flat cell must be neutralised at any coordinate scale.

    Cotangent weights on a 2-manifold are dimensionless, so scaling the mesh must
    not move them. Adding a bare identity to the Gram matrix broke that: it is a
    no-op once the entries of G exceed the dtype's unit-in-last-place, which for
    float32 happens around coordinates of 1e4 -- a car body measured in
    millimetres. Such a cell stayed exactly singular and the inverse raised.
    """
    # Cell 0 is collinear (degenerate); cell 1 is a healthy right triangle. They
    # share an edge, so a mishandled cell 0 would also corrupt cell 1's weights.
    unit_points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=dtype,
        device=device,
    )
    cells = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.long, device=device)

    reference = None
    for scale in (1.0, 1e2, 1e4, 1e6):
        mesh = Mesh(points=unit_points * scale, cells=cells)
        weights, _ = compute_cotan_weights_fem(mesh)

        assert bool(torch.isfinite(weights).all()), f"non-finite weights at {scale=}"
        if reference is None:
            reference = weights
        else:
            torch.testing.assert_close(
                weights, reference, msg=f"weights changed at {scale=}"
            )
