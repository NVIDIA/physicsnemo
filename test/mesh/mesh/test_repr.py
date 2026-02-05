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
    """Test many fields (>3) without nesting."""
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


def test_repr_nested_few_fields():
    """Test nested TensorDict with ≤3 total fields."""
    points = torch.randn(100, 3)
    cells = torch.randint(0, 100, (50, 3))
    mesh = Mesh(
        points=points,
        cells=cells,
        point_data={},
        cell_data={
            "flow": TensorDict(
                {
                    "pressure": torch.randn(50),
                    "velocity": torch.randn(50, 3),
                },
                batch_size=[50],
            )
        },
        global_data={},
    )

    result = repr(mesh)

    expected = r"""Mesh(manifold_dim=2, spatial_dim=3, n_points=100, n_cells=50)
    point_data : {}
    cell_data  : {flow: {pressure: (), velocity: (3,)}}
    global_data: {}"""

    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"


def test_repr_nested_many_fields():
    """Test nested TensorDict with >3 total fields."""
    points = torch.randn(100, 3)
    cells = torch.randint(0, 100, (50, 3))
    mesh = Mesh(
        points=points,
        cells=cells,
        point_data={},
        cell_data={
            "temperature": torch.randn(50),
            "flow": TensorDict(
                {
                    "pressure": torch.randn(50),
                    "velocity": torch.randn(50, 3),
                },
                batch_size=[50],
            ),
        },
        global_data={},
    )

    result = repr(mesh)

    # Keys are alphabetically sorted: flow, temperature
    expected = r"""Mesh(manifold_dim=2, spatial_dim=3, n_points=100, n_cells=50)
    point_data : {}
    cell_data  : {
        flow       : {pressure: (), velocity: (3,)},
        temperature: ()}
    global_data: {}"""

    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"


def test_repr_deeply_nested():
    """Test deeply nested with many fields."""
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


def test_repr_with_device():
    """Test with explicitly set device."""
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


def test_repr_complex_nested():
    """Test multiple nested levels with various field counts."""
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
    """Test edge case with empty mesh."""
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


def test_repr_single_field():
    """Test edge case with single field."""
    points = torch.randn(10, 3)
    cells = torch.randint(0, 10, (5, 3))
    mesh = Mesh(
        points=points,
        cells=cells,
        point_data={"temperature": torch.randn(10)},
        cell_data={},
        global_data={},
    )

    result = repr(mesh)

    expected = r"""Mesh(manifold_dim=2, spatial_dim=3, n_points=10, n_cells=5)
    point_data : {temperature: ()}
    cell_data  : {}
    global_data: {}"""

    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"


def test_repr_exactly_three_fields():
    """Test edge case with exactly 3 fields."""
    points = torch.randn(10, 3)
    cells = torch.randint(0, 10, (5, 3))
    mesh = Mesh(
        points=points,
        cells=cells,
        point_data={
            "temperature": torch.randn(10),
            "pressure": torch.randn(10),
            "velocity": torch.randn(10, 3),
        },
        cell_data={},
        global_data={},
    )

    result = repr(mesh)

    # Keys are alphabetically sorted: pressure, temperature, velocity
    expected = r"""Mesh(manifold_dim=2, spatial_dim=3, n_points=10, n_cells=5)
    point_data : {pressure: (), temperature: (), velocity: (3,)}
    cell_data  : {}
    global_data: {}"""

    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"


def test_repr_with_cached_data():
    """Test that cached data is included by default."""
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


def test_repr_with_multiple_cached_fields():
    """Test that multiple cached fields are formatted correctly."""
    points = torch.randn(10, 3)
    cells = torch.randint(0, 10, (5, 3))
    mesh = Mesh(
        points=points,
        cells=cells,
        point_data={},
        cell_data={"pressure": torch.randn(5)},
        global_data={},
    )

    # Access multiple cached properties to populate cache
    _ = mesh.cell_centroids
    _ = mesh.cell_areas

    result = repr(mesh)

    # Should show multiline format for cell_data since it has >3 total fields
    # (pressure + _cache.centroids + _cache.areas = 3 fields in _cache, but total >3)
    assert "_cache" in result
    assert "centroids" in result
    assert "areas" in result

    # Verify structure is correct
    lines = result.split("\n")
    # Find the cell_data line
    cell_data_line_idx = None
    for i, line in enumerate(lines):
        if "cell_data" in line:
            cell_data_line_idx = i
            break

    # cell_data should be multiline since it has pressure + _cache with multiple fields
    assert "{" in lines[cell_data_line_idx], "cell_data should start multiline format"


def test_repr_with_subclass():
    """Test that subclasses show their own class name."""

    class CustomMesh(Mesh):
        pass

    points = torch.randn(10, 3)
    cells = torch.randint(0, 10, (5, 3))
    mesh = CustomMesh(
        points=points,
        cells=cells,
        point_data={},
        cell_data={},
        global_data={},
    )

    result = repr(mesh)

    # Should start with CustomMesh, not Mesh
    assert result.startswith("CustomMesh("), (
        f"Expected 'CustomMesh(' at start but got:\n{result}"
    )


def test_repr_colon_alignment():
    """Test that colons are properly aligned at each level."""
    points = torch.randn(10, 3)
    cells = torch.randint(0, 10, (5, 3))
    mesh = Mesh(
        points=points,
        cells=cells,
        point_data={"a": torch.randn(10)},
        cell_data={"temperature": torch.randn(5), "p": torch.randn(5)},
        global_data={},
    )

    result = repr(mesh)
    lines = result.split("\n")

    # Find colon positions in main data fields
    colon_positions = [line.index(":") for line in lines[1:4] if ":" in line]

    # All colons should be at the same position
    assert len(set(colon_positions)) == 1, (
        f"Colons not aligned: {colon_positions}\nOutput:\n{result}"
    )


def test_repr_indentation():
    """Test that 4-space indentation is used at each level."""
    points = torch.randn(10, 3)
    cells = torch.randint(0, 10, (5, 3))
    mesh = Mesh(
        points=points,
        cells=cells,
        point_data={
            "a": torch.randn(10),
            "b": torch.randn(10),
            "c": torch.randn(10),
            "d": torch.randn(10),
        },
        cell_data={},
        global_data={},
    )

    result = repr(mesh)
    lines = result.split("\n")

    # Check that data field lines have 4 spaces
    assert lines[1].startswith("    "), f"Expected 4 spaces but got: {repr(lines[1])}"

    # Check that nested field lines have 8 spaces (4 * 2)
    # Find a line that's a field inside point_data
    for line in lines[2:]:
        if ":" in line and line.strip().startswith(("a:", "b:", "c:", "d:")):
            assert line.startswith("        "), (
                f"Expected 8 spaces but got: {repr(line)}"
            )
            break


def test_repr_alphabetical_sorting():
    """Test that keys are displayed in alphabetical order."""
    points = torch.randn(10, 3)
    cells = torch.randint(0, 10, (5, 3))

    # Create mesh with keys in non-alphabetical order
    mesh = Mesh(
        points=points,
        cells=cells,
        point_data={
            "zebra": torch.randn(10),
            "alpha": torch.randn(10),
            "delta": torch.randn(10),
            "beta": torch.randn(10),
        },
        cell_data={},
        global_data={},
    )

    result = repr(mesh)

    # Keys should be sorted: alpha, beta, delta, zebra
    expected = r"""Mesh(manifold_dim=2, spatial_dim=3, n_points=10, n_cells=5)
    point_data : {
        alpha: (),
        beta : (),
        delta: (),
        zebra: ()}
    cell_data  : {}
    global_data: {}"""

    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"


def test_repr_cache_always_last():
    """Test that _cache appears last, even though '_' comes before letters alphabetically."""
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


def test_repr_edge_case_empty_cells():
    """Test that __repr__ handles meshes with zero cells gracefully."""
    # Create a mesh with zero cells (edge case)
    points = torch.randn(10, 3)
    cells = torch.randint(0, 10, (0, 3))  # Zero cells
    mesh = Mesh(
        points=points,
        cells=cells,
        point_data={},
        cell_data={},
        global_data={},
    )

    result = repr(mesh)

    # Should handle zero cells gracefully
    assert "n_cells=0" in result
    assert "n_points=10" in result

    expected = r"""Mesh(manifold_dim=2, spatial_dim=3, n_points=10, n_cells=0)
    point_data : {}
    cell_data  : {}
    global_data: {}"""

    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"
