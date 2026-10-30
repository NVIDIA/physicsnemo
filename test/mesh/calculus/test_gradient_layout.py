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

"""Regression tests for the public gradient tensor layout."""

import pytest
import torch
import torch.nn.functional as F

from physicsnemo.mesh import Mesh


def _single_tetrahedron() -> Mesh:
    return Mesh(
        points=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        ),
        cells=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
    )


def _tetrahedral_star() -> Mesh:
    return Mesh(
        points=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.5],
            ],
            dtype=torch.float64,
        ),
        cells=torch.tensor(
            [[0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]],
            dtype=torch.long,
        ),
    )


def _square_with_center() -> Mesh:
    return Mesh(
        points=torch.tensor(
            [
                [-1.0, -1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=torch.float64,
        ),
        cells=torch.tensor(
            [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=torch.long
        ),
    )


@pytest.mark.parametrize("data_source", ["points", "cells"])
def test_affine_vector_calculus_uses_derivative_first_jacobian(
    data_source: str,
) -> None:
    mesh = _single_tetrahedron() if data_source == "points" else _tetrahedral_star()
    coordinates = mesh.points if data_source == "points" else mesh.cell_centroids
    # derivatives[k, j] = partial v_j / partial x_k. This matrix is deliberately
    # nonsymmetric so transposing the derivative and value axes cannot pass.
    derivatives = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]],
        dtype=mesh.points.dtype,
    )
    vector_field = coordinates @ derivatives

    gradient = mesh.gradient(
        vector_field, gradient_type="extrinsic", data_source=data_source
    )

    torch.testing.assert_close(gradient, derivatives.expand_as(gradient))
    torch.testing.assert_close(
        mesh.divergence(vector_field, data_source=data_source),
        coordinates.new_full((len(coordinates),), 16.0),
    )
    torch.testing.assert_close(
        mesh.curl(vector_field, data_source=data_source),
        coordinates.new_tensor([-2.0, 4.0, -2.0]).expand_as(coordinates),
    )

    if data_source == "points":
        mesh.point_data["affine_vector"] = vector_field
        derived = mesh.compute_point_derivatives(
            keys="affine_vector", method="lsq", gradient_type="extrinsic"
        )
        stored_gradient = derived.point_data["affine_vector_gradient"]
    else:
        mesh.cell_data["affine_vector"] = vector_field
        derived = mesh.compute_cell_derivatives(
            keys="affine_vector", method="lsq", gradient_type="extrinsic"
        )
        stored_gradient = derived.cell_data["affine_vector_gradient"]
    torch.testing.assert_close(stored_gradient, gradient)


def test_intrinsic_gradient_projects_the_derivative_axis() -> None:
    planar_mesh = _square_with_center()
    mesh = Mesh(F.pad(planar_mesh.points, (0, 1)), planar_mesh.cells)
    vector_field = torch.zeros_like(mesh.points)
    vector_field[:, 2] = mesh.points[:, 0]

    gradient = mesh.gradient(vector_field, gradient_type="intrinsic")

    expected = torch.zeros_like(gradient)
    expected[:, 0, 2] = 1.0  # partial v_z / partial x
    torch.testing.assert_close(gradient, expected, atol=1.0e-14, rtol=1.0e-14)


def test_intrinsic_gradient_preserves_trailing_value_dimensions() -> None:
    planar_mesh = _square_with_center()
    mesh = Mesh(F.pad(planar_mesh.points, (0, 1)), planar_mesh.cells)
    derivatives = torch.arange(1, 19, dtype=mesh.points.dtype).reshape(3, 2, 3)
    values = torch.einsum("nd,dab->nab", mesh.points, derivatives)

    gradient = mesh.gradient(values, gradient_type="intrinsic")

    expected = derivatives.clone()
    expected[2] = 0.0
    assert gradient.shape == (mesh.n_points, mesh.n_spatial_dims, 2, 3)
    torch.testing.assert_close(gradient, expected.expand_as(gradient))


def test_dec_and_lsq_preserve_trailing_value_dimensions() -> None:
    mesh = _square_with_center()
    derivatives = torch.arange(1, 13, dtype=mesh.points.dtype).reshape(2, 2, 3)
    values = torch.einsum("nd,dab->nab", mesh.points, derivatives)
    expected = derivatives.expand(mesh.n_points, -1, -1, -1)

    dec_gradient = mesh.gradient(values, method="dec", gradient_type="extrinsic")
    lsq_gradient = mesh.gradient(values, method="lsq", gradient_type="extrinsic")

    assert dec_gradient.shape == (mesh.n_points, 2, 2, 3)
    torch.testing.assert_close(dec_gradient, expected)
    torch.testing.assert_close(lsq_gradient, expected)


@pytest.mark.parametrize("location", ["points", "cells"])
@pytest.mark.parametrize("gradient_type", ["extrinsic", "intrinsic", "both"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mesh_name", ["sphere", "cube"])
def test_lsq_derivatives_of_several_fields_match_one_field_at_a_time(
    location, gradient_type, dtype, mesh_name
) -> None:
    """The LSQ derivatives of several fields share their LSQ systems.

    Each field must still get the gradient it gets alone, bit for bit, with
    nested keys, mixed ranks and two dtypes in one call.
    """
    from physicsnemo.mesh.calculus.derivatives import (
        compute_cell_derivatives,
        compute_point_derivatives,
    )
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
    from physicsnemo.mesh.primitives.volumes import cube_volume

    if mesh_name == "sphere":
        mesh = sphere_icosahedral.load(subdivisions=2)
    else:
        mesh = cube_volume.load(subdivisions=3)
    mesh = Mesh(points=mesh.points.to(dtype), cells=mesh.cells)
    n = mesh.n_points if location == "points" else mesh.n_cells
    g = torch.Generator().manual_seed(0)
    fields = {
        "p": torch.randn(n, generator=g, dtype=dtype),
        ("flow", "u"): torch.randn(n, 3, generator=g, dtype=dtype),
        ("flow", "k"): torch.randn(n, generator=g, dtype=dtype),
        "S": torch.randn(n, 2, 2, generator=g, dtype=dtype),
        # Strided fields: a transposed tensor and every other column
        "T": torch.randn(n, 2, 2, generator=g, dtype=dtype).transpose(1, 2),
        "w": torch.randn(n, 6, generator=g, dtype=dtype)[:, ::2],
    }
    ### A field of the other dtype: its own group. (The intrinsic point
    ### gradient of a field whose dtype differs from the surface mesh's raises.)
    if not (
        gradient_type == "intrinsic" and location == "points" and mesh_name == "sphere"
    ):
        other = torch.float64 if dtype == torch.float32 else torch.float32
        fields["q"] = torch.randn(n, generator=g, dtype=other)
    data = mesh.point_data if location == "points" else mesh.cell_data
    for key, value in fields.items():
        data[key] = value
    compute = (
        compute_point_derivatives if location == "points" else compute_cell_derivatives
    )

    together = compute(mesh, keys=list(fields), gradient_type=gradient_type)
    out = together.point_data if location == "points" else together.cell_data
    suffixes = (
        ["_gradient"]
        if gradient_type != "both"
        else ["_gradient_extrinsic", "_gradient_intrinsic"]
    )
    for key in fields:
        alone = compute(mesh, keys=key, gradient_type=gradient_type)
        expected = alone.point_data if location == "points" else alone.cell_data
        for suffix in suffixes:
            out_key = (
                key + suffix if isinstance(key, str) else key[:-1] + (key[-1] + suffix,)
            )
            assert torch.equal(out[out_key], expected[out_key]), out_key
