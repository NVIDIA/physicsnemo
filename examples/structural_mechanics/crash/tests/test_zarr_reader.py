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

import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

# Import functions from zarr_reader
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import zarr_reader


def create_mock_zarr_store(
    store_path: Path,
    num_timesteps: int = 3,
    num_nodes: int = 4,
    thickness_value: float = 1.0,
):
    """
    Helper function to create a mock Zarr store with crash simulation data.

    Args:
        store_path: Path where the Zarr store should be created
        num_timesteps: Number of timesteps
        num_nodes: Number of nodes
        thickness_value: Constant thickness value for all nodes

    Returns:
        Tuple of (mesh_pos, node_thickness, edges) arrays that were written
    """
    store_path.mkdir(exist_ok=True)

    # Create mock data
    mesh_pos = np.random.randn(num_timesteps, num_nodes, 3).astype(np.float32)
    node_thickness = np.ones(num_nodes, dtype=np.float32) * thickness_value
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)

    # Write to Zarr store
    store = zarr.open(str(store_path), mode="w")
    store.create_dataset("mesh_pos", data=mesh_pos, dtype=np.float32)
    store.create_dataset("thickness", data=node_thickness, dtype=np.float32)
    store.create_dataset("edges", data=edges, dtype=np.int64)

    return mesh_pos, node_thickness, edges


@pytest.fixture
def mock_zarr_store():
    """Create a temporary Zarr store with mock crash simulation data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "Run001.zarr"
        mesh_pos, node_thickness, edges = create_mock_zarr_store(
            store_path, thickness_value=2.0
        )
        yield temp_dir, mesh_pos, node_thickness, edges


@pytest.fixture
def mock_zarr_directory():
    """Create a directory with multiple Zarr stores."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create multiple zarr stores
        for i in range(3):
            store_path = temp_path / f"Run{i:03d}.zarr"
            create_mock_zarr_store(store_path)

        # Create a non-zarr directory (should be ignored)
        (temp_path / "NotAZarr").mkdir()

        # Create a regular file (should be ignored)
        (temp_path / "some_file.txt").touch()

        yield temp_dir


def test_find_zarr_stores(mock_zarr_directory):
    """Test that find_zarr_stores correctly identifies Zarr directories."""
    zarr_stores = zarr_reader.find_zarr_stores(mock_zarr_directory)

    assert len(zarr_stores) == 3, f"Expected 3 zarr stores, got {len(zarr_stores)}"
    assert all(path.endswith(".zarr") for path in zarr_stores)
    assert all("Run" in Path(path).name for path in zarr_stores)


def test_find_zarr_stores_empty_directory():
    """Test find_zarr_stores with empty directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        zarr_stores = zarr_reader.find_zarr_stores(temp_dir)
        assert len(zarr_stores) == 0, (
            "Should return empty list for directory with no zarr stores"
        )


def test_find_zarr_stores_nonexistent_directory():
    """Test find_zarr_stores with nonexistent directory."""
    zarr_stores = zarr_reader.find_zarr_stores("/nonexistent/path")
    assert len(zarr_stores) == 0, "Should return empty list for nonexistent directory"


def test_load_zarr_store(mock_zarr_store):
    """Test loading data from a Zarr store."""
    temp_dir, expected_mesh_pos, expected_thickness, expected_edges = mock_zarr_store
    store_path = Path(temp_dir) / "Run001.zarr"

    mesh_pos, edges, point_data_dict = zarr_reader.load_zarr_store(str(store_path))

    # Check shapes
    assert mesh_pos.shape == expected_mesh_pos.shape
    assert edges.shape == expected_edges.shape
    assert "thickness" in point_data_dict, "Should have thickness in point_data"
    assert point_data_dict["thickness"].shape == expected_thickness.shape

    # Check data types
    assert mesh_pos.dtype == np.float64, "mesh_pos should be float64"
    assert point_data_dict["thickness"].dtype == np.float32, (
        "thickness should be float32"
    )
    assert edges.dtype == np.int64, "edges should be int64"

    # Check values
    np.testing.assert_array_almost_equal(mesh_pos, expected_mesh_pos)
    np.testing.assert_array_almost_equal(
        point_data_dict["thickness"], expected_thickness
    )
    np.testing.assert_array_equal(edges, expected_edges)


def test_load_zarr_store_missing_fields():
    """Test that loading a Zarr store with missing required fields raises KeyError."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "incomplete.zarr"
        store_path.mkdir()

        # Create store with only thickness (missing mesh_pos and edges)
        store = zarr.open(str(store_path), mode="w")
        store.create_dataset("thickness", data=np.ones(4, dtype=np.float32))

        # Should raise KeyError for missing mesh_pos
        with pytest.raises(KeyError, match="mesh_pos"):
            zarr_reader.load_zarr_store(str(store_path))

        # Test missing edges
        store_path2 = Path(temp_dir) / "incomplete2.zarr"
        store_path2.mkdir()
        store2 = zarr.open(str(store_path2), mode="w")
        store2.create_dataset(
            "mesh_pos", data=np.random.randn(3, 4, 3).astype(np.float32)
        )

        # Should raise KeyError for missing edges
        with pytest.raises(KeyError, match="edges"):
            zarr_reader.load_zarr_store(str(store_path2))


def test_load_zarr_store_multiple_point_data_fields():
    """Test that load_zarr_store dynamically reads all point data fields."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "multi_fields.zarr"
        store_path.mkdir()

        # Create store with multiple point data fields
        num_nodes = 10
        store = zarr.open(str(store_path), mode="w")
        store.create_dataset(
            "mesh_pos", data=np.random.randn(3, num_nodes, 3).astype(np.float32)
        )
        store.create_dataset("edges", data=np.array([[0, 1]], dtype=np.int64))
        # Add multiple point data fields
        store.create_dataset("thickness", data=np.ones(num_nodes, dtype=np.float32))
        store.create_dataset(
            "stress", data=np.random.randn(num_nodes).astype(np.float32)
        )
        store.create_dataset(
            "temperature", data=np.random.randn(num_nodes).astype(np.float32)
        )
        # This should be skipped (mesh connectivity, not point data)
        store.create_dataset(
            "mesh_connectivity_flat", data=np.array([0, 1, 2], dtype=np.int64)
        )

        mesh_pos, edges, point_data_dict = zarr_reader.load_zarr_store(str(store_path))

        # Should have all three point data fields
        assert "thickness" in point_data_dict
        assert "stress" in point_data_dict
        assert "temperature" in point_data_dict
        # Should NOT include mesh connectivity
        assert "mesh_connectivity_flat" not in point_data_dict
        # Should NOT include mesh_pos or edges
        assert "mesh_pos" not in point_data_dict
        assert "edges" not in point_data_dict

        # Check that all point data fields have correct shape
        for name, data in point_data_dict.items():
            assert data.shape == (num_nodes,), (
                f"{name} should have shape ({num_nodes},)"
            )
            assert data.dtype == np.float32, f"{name} should be float32"


def test_load_zarr_store_2d_feature_arrays():
    """Test that load_zarr_store correctly handles 2D feature arrays [N, K]."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "2d_features.zarr"
        store_path.mkdir()

        # Create store with 2D feature array
        num_nodes = 8
        feature_dim = 3
        store = zarr.open(str(store_path), mode="w")
        store.create_dataset(
            "mesh_pos", data=np.random.randn(3, num_nodes, 3).astype(np.float32)
        )
        store.create_dataset("edges", data=np.array([[0, 1]], dtype=np.int64))
        # Add 1D feature (thickness)
        store.create_dataset("thickness", data=np.ones(num_nodes, dtype=np.float32))
        # Add 2D feature array [N, K] (e.g., stress tensor components)
        stress_tensor = np.random.randn(num_nodes, feature_dim).astype(np.float32)
        store.create_dataset("stress_tensor", data=stress_tensor)

        mesh_pos, edges, point_data_dict = zarr_reader.load_zarr_store(str(store_path))

        # Should have both 1D and 2D features
        assert "thickness" in point_data_dict
        assert "stress_tensor" in point_data_dict

        # Check 1D feature shape
        assert point_data_dict["thickness"].shape == (num_nodes,)
        assert point_data_dict["thickness"].ndim == 1

        # Check 2D feature shape
        assert point_data_dict["stress_tensor"].shape == (num_nodes, feature_dim)
        assert point_data_dict["stress_tensor"].ndim == 2

        # Verify values match
        np.testing.assert_array_almost_equal(
            point_data_dict["stress_tensor"], stress_tensor
        )


def test_process_zarr_data(mock_zarr_directory):
    """Test processing multiple Zarr stores."""
    srcs, dsts, point_data = zarr_reader.process_zarr_data(
        data_dir=mock_zarr_directory,
        num_samples=2,
    )

    # Check we got 2 samples
    assert len(srcs) == 2, f"Expected 2 samples, got {len(srcs)}"
    assert len(dsts) == 2
    assert len(point_data) == 2

    # Check each sample has correct structure
    for i in range(2):
        assert srcs[i].ndim == 1, "srcs should be 1D array"
        assert dsts[i].ndim == 1, "dsts should be 1D array"
        assert len(srcs[i]) == len(dsts[i]), "srcs and dsts should have same length"

        # Check point_data structure
        assert "coords" in point_data[i], "point_data should have 'coords' key"
        assert "thickness" in point_data[i], "point_data should have 'thickness' key"

        coords = point_data[i]["coords"]
        thickness = point_data[i]["thickness"]

        assert coords.ndim == 3, "coords should be [T,N,3]"
        assert coords.shape[-1] == 3, "coords last dimension should be 3"
        assert thickness.ndim == 1, "thickness should be 1D"
        assert len(thickness) == coords.shape[1], (
            "thickness length should match num_nodes"
        )


def test_process_zarr_data_no_stores():
    """Test that processing directory with no Zarr stores raises error."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError, match="No .zarr stores found"):
            zarr_reader.process_zarr_data(
                data_dir=temp_dir,
                num_samples=1,
            )


def test_process_zarr_data_validation():
    """Test that process_zarr_data validates data shapes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "bad_store.zarr"
        store_path.mkdir()

        # Create store with invalid mesh_pos shape (should be [T,N,3])
        store = zarr.open(str(store_path), mode="w")
        store.create_dataset(
            "mesh_pos", data=np.random.randn(3, 4, 2).astype(np.float32)
        )  # Wrong last dim
        store.create_dataset("thickness", data=np.ones(4, dtype=np.float32))
        store.create_dataset("edges", data=np.array([[0, 1]], dtype=np.int64))

        with pytest.raises(ValueError, match="mesh_pos must be"):
            zarr_reader.process_zarr_data(
                data_dir=temp_dir,
                num_samples=1,
            )


def test_process_zarr_data_edge_bounds():
    """Test that process_zarr_data validates edge indices are within bounds."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "bad_edges.zarr"
        store_path.mkdir()

        num_nodes = 4
        store = zarr.open(str(store_path), mode="w")
        store.create_dataset(
            "mesh_pos", data=np.random.randn(3, num_nodes, 3).astype(np.float32)
        )
        store.create_dataset("thickness", data=np.ones(num_nodes, dtype=np.float32))
        # Edge references node 10 which is out of bounds
        store.create_dataset("edges", data=np.array([[0, 10]], dtype=np.int64))

        with pytest.raises(ValueError, match="Edge indices out of bounds"):
            zarr_reader.process_zarr_data(
                data_dir=temp_dir,
                num_samples=1,
            )


def test_reader_class(mock_zarr_directory):
    """Test the Reader class callable interface."""
    reader = zarr_reader.Reader()

    srcs, dsts, point_data = reader(
        data_dir=mock_zarr_directory,
        num_samples=2,
        split="train",
    )

    assert len(srcs) == 2
    assert len(dsts) == 2
    assert len(point_data) == 2


def test_natural_sorting(mock_zarr_directory):
    """Test that Zarr stores are sorted naturally (Run1, Run2, ..., Run10)."""
    temp_path = Path(mock_zarr_directory)

    # Add more stores with different numbering
    for i in [10, 5, 20]:
        store_path = temp_path / f"Run{i}.zarr"
        create_mock_zarr_store(store_path)

    zarr_stores = zarr_reader.find_zarr_stores(mock_zarr_directory)
    store_names = [Path(p).name for p in zarr_stores]

    # Should be sorted: Run000, Run001, Run002, Run5, Run10, Run20
    assert store_names[0] == "Run000.zarr"
    assert store_names[1] == "Run001.zarr"
    assert store_names[2] == "Run002.zarr"
    assert "Run5.zarr" in store_names
    assert "Run10.zarr" in store_names
    assert "Run20.zarr" in store_names
