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

"""Tests for Mesh.pad() and Mesh.pad_to_next_power() methods.

Tests validate padding functionality for torch.compile compatibility, including:
- Padding points and cells to target sizes
- Padding with data fields (point_data, cell_data)
- Custom padding values
- Power-based padding for compile cache efficiency
- Error handling for invalid inputs
- Device preservation
"""

import math

import pytest
import torch

from physicsnemo.mesh import Mesh

### Helper Functions ###


def create_simple_mesh(
    n_points: int = 10,
    n_cells: int = 5,
    n_spatial_dims: int = 3,
    n_manifold_dims: int = 2,
    device: str = "cpu",
    add_point_data: bool = False,
    add_cell_data: bool = False,
) -> Mesh:
    """Create a simple mesh for testing."""
    torch.manual_seed(42)
    points = torch.randn(n_points, n_spatial_dims, device=device)
    cells = torch.randint(0, n_points, (n_cells, n_manifold_dims + 1), device=device)

    point_data = {}
    cell_data = {}

    if add_point_data:
        point_data["temperature"] = torch.randn(n_points, device=device)
        point_data["velocity"] = torch.randn(n_points, 3, device=device)

    if add_cell_data:
        cell_data["pressure"] = torch.randn(n_cells, device=device)

    return Mesh(
        points=points,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,
    )


### Test Classes ###


class TestPadBasic:
    """Tests for basic pad() functionality."""

    def test_pad_points_only(self):
        """Test padding only points."""
        mesh = create_simple_mesh(n_points=10, n_cells=5)

        padded = mesh.pad(target_n_points=20)

        assert padded.n_points == 20
        assert padded.n_cells == 5  # Unchanged

    def test_pad_cells_only(self):
        """Test padding only cells."""
        mesh = create_simple_mesh(n_points=10, n_cells=5)

        padded = mesh.pad(target_n_cells=15)

        assert padded.n_points == 10  # Unchanged
        assert padded.n_cells == 15

    def test_pad_both_points_and_cells(self):
        """Test padding both points and cells."""
        mesh = create_simple_mesh(n_points=10, n_cells=5)

        padded = mesh.pad(target_n_points=20, target_n_cells=15)

        assert padded.n_points == 20
        assert padded.n_cells == 15

    def test_pad_no_targets_returns_self(self):
        """Test that padding with no targets returns self."""
        mesh = create_simple_mesh(n_points=10, n_cells=5)

        padded = mesh.pad()

        assert padded is mesh

    def test_pad_same_size_no_change(self):
        """Test that padding to same size works correctly."""
        mesh = create_simple_mesh(n_points=10, n_cells=5)

        padded = mesh.pad(target_n_points=10, target_n_cells=5)

        assert padded.n_points == 10
        assert padded.n_cells == 5

    def test_pad_preserves_original_points(self):
        """Test that original points are preserved after padding."""
        mesh = create_simple_mesh(n_points=10, n_cells=5)
        original_points = mesh.points.clone()

        padded = mesh.pad(target_n_points=20)

        assert torch.allclose(padded.points[:10], original_points)

    def test_pad_preserves_original_cells(self):
        """Test that original cells are preserved after padding."""
        mesh = create_simple_mesh(n_points=10, n_cells=5)
        original_cells = mesh.cells.clone()

        padded = mesh.pad(target_n_cells=15)

        assert torch.equal(padded.cells[:5], original_cells)

    def test_pad_points_use_last_point(self):
        """Test that padded points replicate the last point."""
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        padded = mesh.pad(target_n_points=5)

        # Last point should be replicated
        assert torch.allclose(padded.points[3], points[-1])
        assert torch.allclose(padded.points[4], points[-1])

    def test_pad_cells_use_last_point_index(self):
        """Test that padded cells reference the last point index."""
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        padded = mesh.pad(target_n_cells=3)

        # Padded cells should reference last point index (2)
        assert torch.all(padded.cells[1] == 2)
        assert torch.all(padded.cells[2] == 2)


class TestPadWithData:
    """Tests for padding with point_data and cell_data."""

    def test_pad_with_point_data(self):
        """Test that point_data is correctly padded."""
        mesh = create_simple_mesh(n_points=10, n_cells=5, add_point_data=True)

        padded = mesh.pad(target_n_points=20)

        assert padded.point_data["temperature"].shape[0] == 20
        assert padded.point_data["velocity"].shape[0] == 20

    def test_pad_with_cell_data(self):
        """Test that cell_data is correctly padded."""
        mesh = create_simple_mesh(n_points=10, n_cells=5, add_cell_data=True)

        padded = mesh.pad(target_n_cells=15)

        assert padded.cell_data["pressure"].shape[0] == 15

    def test_pad_default_value_is_nan(self):
        """Test that default padding value is NaN."""
        mesh = create_simple_mesh(n_points=10, n_cells=5, add_point_data=True)

        padded = mesh.pad(target_n_points=20)

        # Padded values should be NaN
        assert torch.isnan(padded.point_data["temperature"][10:]).all()

    def test_pad_custom_value(self):
        """Test padding with custom value."""
        mesh = create_simple_mesh(n_points=10, n_cells=5, add_point_data=True)

        padded = mesh.pad(target_n_points=20, data_padding_value=0.0)

        # Padded values should be 0.0
        assert torch.allclose(padded.point_data["temperature"][10:], torch.zeros(10))

    def test_pad_negative_value(self):
        """Test padding with negative value."""
        mesh = create_simple_mesh(n_points=10, n_cells=5, add_cell_data=True)

        padded = mesh.pad(target_n_cells=15, data_padding_value=-999.0)

        assert torch.allclose(
            padded.cell_data["pressure"][5:], torch.full((10,), -999.0)
        )

    def test_pad_preserves_original_data(self):
        """Test that original data values are preserved."""
        mesh = create_simple_mesh(n_points=10, n_cells=5, add_point_data=True)
        original_temp = mesh.point_data["temperature"].clone()

        padded = mesh.pad(target_n_points=20, data_padding_value=0.0)

        assert torch.allclose(padded.point_data["temperature"][:10], original_temp)


class TestPadErrors:
    """Tests for error handling in pad()."""

    def test_pad_target_points_less_than_current_raises(self):
        """Test that target_n_points < n_points raises ValueError."""
        mesh = create_simple_mesh(n_points=10, n_cells=5)

        with pytest.raises(ValueError, match="target_n_points=.* must be >= "):
            mesh.pad(target_n_points=5)

    def test_pad_target_cells_less_than_current_raises(self):
        """Test that target_n_cells < n_cells raises ValueError."""
        mesh = create_simple_mesh(n_points=10, n_cells=5)

        with pytest.raises(ValueError, match="target_n_cells=.* must be >= "):
            mesh.pad(target_n_cells=3)


class TestPadToNextPowerBasic:
    """Tests for basic pad_to_next_power() functionality."""

    def test_pad_to_next_power_basic(self):
        """Test basic pad_to_next_power functionality."""
        mesh = create_simple_mesh(n_points=100, n_cells=50)

        padded = mesh.pad_to_next_power(power=2.0)

        # For base 2: 100 points -> 128, 50 cells -> 64
        assert padded.n_points >= 100
        assert padded.n_cells >= 50
        # Should be a power of 2
        assert (padded.n_points & (padded.n_points - 1)) == 0 or padded.n_points == 1
        assert (padded.n_cells & (padded.n_cells - 1)) == 0 or padded.n_cells == 1

    def test_pad_to_next_power_1_5(self):
        """Test pad_to_next_power with power=1.5."""
        mesh = create_simple_mesh(n_points=100, n_cells=50)

        padded = mesh.pad_to_next_power(power=1.5)

        # Result should be >= original sizes
        assert padded.n_points >= 100
        assert padded.n_cells >= 50

    def test_pad_to_next_power_small_mesh(self):
        """Test pad_to_next_power with small mesh sizes."""
        mesh = create_simple_mesh(n_points=3, n_cells=2)

        padded = mesh.pad_to_next_power(power=2.0)

        # 3 points -> next power of 2 >= 3 is 4
        # 2 cells -> next power of 2 >= 2 is 2
        assert padded.n_points >= 3
        assert padded.n_cells >= 2

    def test_pad_to_next_power_exact_power(self):
        """Test pad_to_next_power when size is already a power."""
        mesh = create_simple_mesh(n_points=128, n_cells=64)

        padded = mesh.pad_to_next_power(power=2.0)

        # Already powers of 2, should stay same
        assert padded.n_points == 128
        assert padded.n_cells == 64

    def test_pad_to_next_power_with_data(self):
        """Test pad_to_next_power with data fields."""
        mesh = create_simple_mesh(
            n_points=100, n_cells=50, add_point_data=True, add_cell_data=True
        )

        padded = mesh.pad_to_next_power(power=2.0)

        assert padded.point_data["temperature"].shape[0] == padded.n_points
        assert padded.cell_data["pressure"].shape[0] == padded.n_cells

    def test_pad_to_next_power_custom_value(self):
        """Test pad_to_next_power with custom padding value."""
        mesh = create_simple_mesh(n_points=100, n_cells=50, add_point_data=True)

        padded = mesh.pad_to_next_power(power=2.0, data_padding_value=-1.0)

        # Padded values should be -1.0
        assert torch.allclose(
            padded.point_data["temperature"][100:],
            torch.full((padded.n_points - 100,), -1.0),
        )


class TestPadToNextPowerErrors:
    """Tests for error handling in pad_to_next_power()."""

    def test_pad_to_next_power_invalid_power_raises(self):
        """Test that power <= 1 raises ValueError."""
        mesh = create_simple_mesh(n_points=100, n_cells=50)

        with pytest.raises(ValueError, match="power must be > 1"):
            mesh.pad_to_next_power(power=1.0)

    def test_pad_to_next_power_negative_power_raises(self):
        """Test that negative power raises ValueError."""
        mesh = create_simple_mesh(n_points=100, n_cells=50)

        with pytest.raises(ValueError, match="power must be > 1"):
            mesh.pad_to_next_power(power=0.5)


class TestPadDeviceHandling:
    """Tests for device handling in pad methods."""

    def test_pad_preserves_cpu_device(self):
        """Test that padding CPU mesh produces CPU result."""
        mesh = create_simple_mesh(n_points=10, n_cells=5, device="cpu")

        padded = mesh.pad(target_n_points=20, target_n_cells=15)

        assert padded.points.device.type == "cpu"
        assert padded.cells.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_pad_preserves_cuda_device(self):
        """Test that padding CUDA mesh produces CUDA result."""
        mesh = create_simple_mesh(n_points=10, n_cells=5, device="cuda")

        padded = mesh.pad(target_n_points=20, target_n_cells=15)

        assert padded.points.device.type == "cuda"
        assert padded.cells.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_pad_to_next_power_preserves_cuda_device(self):
        """Test that pad_to_next_power preserves CUDA device."""
        mesh = create_simple_mesh(n_points=100, n_cells=50, device="cuda")

        padded = mesh.pad_to_next_power(power=2.0)

        assert padded.points.device.type == "cuda"
        assert padded.cells.device.type == "cuda"


class TestPadParametrized:
    """Parametrized tests for padding across dimensions."""

    @pytest.mark.parametrize(
        "n_spatial_dims,n_manifold_dims",
        [
            (2, 1),
            (2, 2),
            (3, 1),
            (3, 2),
            (3, 3),
        ],
    )
    def test_pad_various_dimensions(self, n_spatial_dims, n_manifold_dims, device):
        """Test padding across various dimension combinations."""
        mesh = create_simple_mesh(
            n_points=10,
            n_cells=5,
            n_spatial_dims=n_spatial_dims,
            n_manifold_dims=n_manifold_dims,
            device=device,
        )

        padded = mesh.pad(target_n_points=20, target_n_cells=15)

        assert padded.n_points == 20
        assert padded.n_cells == 15
        assert padded.n_spatial_dims == n_spatial_dims
        assert padded.n_manifold_dims == n_manifold_dims
        assert padded.points.device.type == device

    @pytest.mark.parametrize("power", [1.5, 2.0, 3.0])
    def test_pad_to_next_power_various_bases(self, power):
        """Test pad_to_next_power with various power bases."""
        mesh = create_simple_mesh(n_points=100, n_cells=50)

        padded = mesh.pad_to_next_power(power=power)

        # Result should be >= original sizes
        assert padded.n_points >= 100
        assert padded.n_cells >= 50

        # Result should be floor(power^n) for some integer n
        # The implementation uses floor after computing the power, so we need to check
        # that the result is achievable as floor(power^n) for some n
        def is_valid_padded_size(n, base):
            if n <= 0:
                return False
            # Find n such that floor(base^n) = result
            log_val = math.log(n) / math.log(base)
            n_ceil = math.ceil(log_val)
            expected = int(base**n_ceil)
            # Allow small tolerance for floating point computation
            return n == expected

        assert is_valid_padded_size(padded.n_points, power)
        assert is_valid_padded_size(padded.n_cells, power)


class TestPadEdgeCases:
    """Tests for edge cases in padding."""

    def test_pad_single_point(self):
        """Test padding mesh with single point."""
        points = torch.tensor([[0.0, 0.0, 0.0]])
        cells = torch.tensor([[0]])
        mesh = Mesh(points=points, cells=cells)

        padded = mesh.pad(target_n_points=5)

        assert padded.n_points == 5
        # All padded points should be at the original point position
        assert torch.allclose(padded.points, points.expand(5, -1))

    def test_pad_empty_cells(self):
        """Test padding mesh with empty cells."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        cells = torch.zeros((0, 3), dtype=torch.long)
        mesh = Mesh(points=points, cells=cells)

        padded = mesh.pad(target_n_cells=5)

        assert padded.n_cells == 5
        # All cells should reference the last point index (2)
        assert torch.all(padded.cells == 2)

    def test_pad_preserves_global_data(self):
        """Test that global_data is preserved unchanged."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells, global_data={"time": torch.tensor(1.5)})

        padded = mesh.pad(target_n_points=10)

        assert torch.allclose(padded.global_data["time"], torch.tensor(1.5))
