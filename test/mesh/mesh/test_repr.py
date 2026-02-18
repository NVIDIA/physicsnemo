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

"""Tests for Mesh __repr__ method."""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.mesh.mesh import Mesh


def test_repr_simple_case():
    """Test simple case with ≤3 fields total, no nesting, empty dicts."""
    points = torch.randn(4842, 3)
    cells = torch.randint(0, 4842, (19147, 3))
    mesh = Mesh(
        points=points,
        cells=cells,
        point_data={},
        cell_data={"noise": torch.randn(19147)},
        global_data={},
    )

    result = repr(mesh)

    expected = r"""Mesh(manifold_dim=2, spatial_dim=3, n_points=4842, n_cells=19147)
    point_data : {}
    cell_data  : {noise: ()}
    global_data: {}"""

    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"


def test_repr_many_fields():
    """Test many fields (>3) triggers multiline formatting with point_data and cell_data."""
    points = torch.randn(100, 3)
    cells = torch.randint(0, 100, (50, 3))
    mesh = Mesh(
        points=points,
        cells=cells,
        point_data={
            "temperature": torch.randn(100),
            "pressure": torch.randn(100),
            "velocity": torch.randn(100, 3),
            "stress": torch.randn(100, 3, 3),
        },
        cell_data={},
        global_data={},
    )

    result = repr(mesh)

    # Keys are alphabetically sorted: pressure, stress, temperature, velocity
    expected = r"""Mesh(manifold_dim=2, spatial_dim=3, n_points=100, n_cells=50)
    point_data : {
        pressure   : (),
        stress     : (3, 3),
        temperature: (),
        velocity   : (3,)}
    cell_data  : {}
    global_data: {}"""

    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"


def test_repr_deeply_nested():
    """Test deeply nested TensorDicts with many fields across point_data and cell_data."""
    points = torch.randn(100, 3)
    cells = torch.randint(0, 100, (50, 3))
    mesh = Mesh(
        points=points,
        cells=cells,
        point_data={
            "temperature": torch.randn(100),
            "flow": TensorDict(
                {
                    "pressure": torch.randn(100),
                    "velocity": torch.randn(100, 3),
                    "turbulence": torch.randn(100, 3, 3),
                },
                batch_size=[100],
            ),
        },
        cell_data={
            "material": TensorDict(
                {
                    "density": torch.randn(50),
                    "elasticity": torch.randn(50, 6),
                },
                batch_size=[50],
            )
        },
        global_data={"timestep": torch.tensor(0.01)},
    )

    result = repr(mesh)

    # Keys are alphabetically sorted at all levels
    # Top level point_data: flow, temperature
    # Nested flow: pressure, turbulence, velocity
    expected = r"""Mesh(manifold_dim=2, spatial_dim=3, n_points=100, n_cells=50)
    point_data : {
        flow       : {pressure: (), turbulence: (3, 3), velocity: (3,)},
        temperature: ()}
    cell_data  : {material: {density: (), elasticity: (6,)}}
    global_data: {timestep: ()}"""

    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"


def test_repr_complex_nested():
    """Test multiple nested levels with a lower-dimensional mesh (manifold_dim=1)."""
    points = torch.randn(100, 2)
    cells = torch.randint(0, 100, (50, 2))
    mesh = Mesh(
        points=points,
        cells=cells,
        point_data={"position": torch.randn(100, 2)},
        cell_data={
            "state": TensorDict(
                {
                    "thermal": TensorDict(
                        {
                            "temperature": torch.randn(50),
                            "heat_flux": torch.randn(50, 2),
                        },
                        batch_size=[50],
                    ),
                    "mechanical": TensorDict(
                        {
                            "stress": torch.randn(50),
                            "strain": torch.randn(50),
                        },
                        batch_size=[50],
                    ),
                },
                batch_size=[50],
            )
        },
        global_data={},
    )

    result = repr(mesh)

    # Keys are alphabetically sorted at all levels
    # state: mechanical, thermal (alphabetically)
    # mechanical: strain, stress
    # thermal: heat_flux, temperature
    expected = r"""Mesh(manifold_dim=1, spatial_dim=2, n_points=100, n_cells=50)
    point_data : {position: (2,)}
    cell_data  : {
        state: {
            mechanical: {strain: (), stress: ()},
            thermal   : {heat_flux: (2,), temperature: ()}}}
    global_data: {}"""

    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"


def test_repr_empty_mesh():
    """Test repr for a mesh with no point_data, cell_data, or global_data."""
    points = torch.randn(10, 3)
    cells = torch.randint(0, 10, (5, 3))
    mesh = Mesh(
        points=points,
        cells=cells,
        point_data={},
        cell_data={},
        global_data={},
    )

    result = repr(mesh)

    expected = r"""Mesh(manifold_dim=2, spatial_dim=3, n_points=10, n_cells=5)
    point_data : {}
    cell_data  : {}
    global_data: {}"""

    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"


def test_repr_with_device():
    """Test that device info is shown when explicitly set."""
    points = torch.randn(100, 3)
    cells = torch.randint(0, 100, (50, 3))
    mesh = Mesh(
        points=points,
        cells=cells,
        point_data={},
        cell_data={
            "pressure": torch.randn(50),
            "velocity": torch.randn(50, 3),
        },
        global_data={},
    )

    # Explicitly set device using .to()
    mesh = mesh.to("cpu")

    result = repr(mesh)

    # Device should be shown since it was explicitly set
    expected = r"""Mesh(manifold_dim=2, spatial_dim=3, n_points=100, n_cells=50, device=cpu)
    point_data : {}
    cell_data  : {pressure: (), velocity: (3,)}
    global_data: {}"""

    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"


@pytest.mark.cuda
def test_repr_with_cuda_device():
    """Test that CUDA device displays correctly when explicitly set."""
    points = torch.randn(100, 3)
    cells = torch.randint(0, 100, (50, 3))
    mesh = Mesh(
        points=points,
        cells=cells,
        point_data={},
        cell_data={
            "pressure": torch.randn(50),
            "velocity": torch.randn(50, 3),
        },
        global_data={},
    )

    # Explicitly set device to cuda:0
    mesh = mesh.to("cuda:0")

    result = repr(mesh)

    # Device should show cuda:0
    expected = r"""Mesh(manifold_dim=2, spatial_dim=3, n_points=100, n_cells=50, device=cuda:0)
    point_data : {}
    cell_data  : {pressure: (), velocity: (3,)}
    global_data: {}"""

    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"


def test_repr_with_cached_data():
    """Test that cached data is included in repr output."""
    points = torch.randn(10, 3)
    cells = torch.randint(0, 10, (5, 3))
    mesh = Mesh(
        points=points,
        cells=cells,
        point_data={},
        cell_data={},
        global_data={},
    )

    # Access a cached property to populate cache
    _ = mesh.cell_centroids

    result = repr(mesh)

    # Should include _cache in the output
    assert "_cache" in result, f"Expected _cache in output but got:\n{result}"
    assert "centroids" in result, f"Expected centroids in output but got:\n{result}"


def test_repr_cache_always_last():
    """Test that _cache appears last in repr, after user-defined fields."""
    points = torch.randn(10, 3)
    cells = torch.randint(0, 10, (5, 3))

    mesh = Mesh(
        points=points,
        cells=cells,
        point_data={},
        cell_data={
            "zebra": torch.randn(5),
            "alpha": torch.randn(5),
            "beta": torch.randn(5),
        },
        global_data={},
    )

    # Trigger cache population
    _ = mesh.cell_centroids
    _ = mesh.cell_areas

    result = repr(mesh)

    # _cache should appear AFTER alpha, beta, zebra (not before due to underscore)
    expected = r"""Mesh(manifold_dim=2, spatial_dim=3, n_points=10, n_cells=5)
    point_data : {}
    cell_data  : {
        alpha : (),
        beta  : (),
        zebra : (),
        _cache: {areas: (), centroids: (3,)}}
    global_data: {}"""

    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"

    # Verify by checking order in the string
    alpha_pos = result.index("alpha")
    beta_pos = result.index("beta")
    zebra_pos = result.index("zebra")
    cache_pos = result.index("_cache")

    assert alpha_pos < beta_pos < zebra_pos < cache_pos, (
        f"Keys not in correct order: alpha={alpha_pos}, beta={beta_pos}, zebra={zebra_pos}, _cache={cache_pos}"
    )
