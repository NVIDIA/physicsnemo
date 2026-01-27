# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Tests for specialized readers (VTKReader, TensorStoreZarrReader)."""

import json

import numpy as np
import pytest
import torch
from tensordict import TensorDict

# ============================================================================
# TensorStoreZarrReader Tests
# ============================================================================


class TestTensorStoreZarrReader:
    """Tests for TensorStoreZarrReader."""

    @pytest.fixture
    def tensorstore_available(self):
        """Check if tensorstore is available."""
        try:
            import tensorstore  # noqa: F401

            return True
        except ImportError:
            pytest.skip("TensorStore not installed")

    @pytest.fixture
    def zarr_v2_data_dir(self, tmp_path):
        """Create Zarr v2 format test data."""
        zarr = pytest.importorskip("zarr")

        for i in range(5):
            group_path = tmp_path / f"sample_{i:03d}.zarr"
            root = zarr.open(group_path, mode="w")
            root.create_array(
                "positions", data=np.random.randn(100, 3).astype(np.float32)
            )
            root.create_array(
                "features", data=np.random.randn(100, 8).astype(np.float32)
            )
            root.attrs["sample_id"] = i
            root.attrs["scale_factor"] = float(i + 1)

        return tmp_path

    @pytest.fixture
    def zarr_v3_data_dir(self, tmp_path):
        """Create Zarr v3 format test data (manual structure)."""
        for i in range(3):
            group_path = tmp_path / f"sample_{i:03d}.zarr"
            group_path.mkdir(parents=True)

            # Create zarr.json for group (v3 format)
            group_meta = {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {"sample_id": i, "scale": float(i + 1)},
            }
            with open(group_path / "zarr.json", "w") as f:
                json.dump(group_meta, f)

            # Create an array subdirectory
            array_path = group_path / "data"
            array_path.mkdir()

            # Create zarr.json for array (v3 format)
            array_meta = {
                "zarr_format": 3,
                "node_type": "array",
                "shape": [50, 4],
                "data_type": "float32",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [50, 4]},
                },
            }
            with open(array_path / "zarr.json", "w") as f:
                json.dump(array_meta, f)

        return tmp_path

    def test_import_error_without_tensorstore(self, tmp_path):
        """Test that ImportError is raised when tensorstore not installed."""
        # This test checks the error handling
        from physicsnemo.datapipes.readers.tensorstore_zarr import (
            TENSORSTORE_AVAILABLE,
            TensorStoreZarrReader,
        )

        if TENSORSTORE_AVAILABLE:
            pytest.skip("TensorStore is installed, cannot test ImportError")

        with pytest.raises(ImportError, match="TensorStore is required"):
            TensorStoreZarrReader(tmp_path)

    def test_basic_loading(self, tensorstore_available, zarr_v2_data_dir):
        """Test basic data loading."""
        from physicsnemo.datapipes.readers.tensorstore_zarr import (
            TensorStoreZarrReader,
        )

        reader = TensorStoreZarrReader(zarr_v2_data_dir, group_pattern="sample_*.zarr")

        assert len(reader) == 5
        assert "positions" in reader.fields
        assert "features" in reader.fields

        data, metadata = reader[0]

        assert isinstance(data, TensorDict)
        assert data["positions"].shape == (100, 3)
        assert data["features"].shape == (100, 8)

    def test_field_selection(self, tensorstore_available, zarr_v2_data_dir):
        """Test loading specific fields."""
        from physicsnemo.datapipes.readers.tensorstore_zarr import (
            TensorStoreZarrReader,
        )

        reader = TensorStoreZarrReader(
            zarr_v2_data_dir,
            fields=["positions"],
            group_pattern="sample_*.zarr",
        )

        assert reader.fields == ["positions"]

        data, metadata = reader[0]

        assert "positions" in data
        assert "features" not in data

    def test_default_values(self, tensorstore_available, zarr_v2_data_dir):
        """Test default values for missing fields."""
        from physicsnemo.datapipes.readers.tensorstore_zarr import (
            TensorStoreZarrReader,
        )

        default_tensor = torch.zeros(10)
        reader = TensorStoreZarrReader(
            zarr_v2_data_dir,
            fields=["positions", "missing_field"],
            default_values={"missing_field": default_tensor},
            group_pattern="sample_*.zarr",
        )

        data, metadata = reader[0]

        assert "missing_field" in data
        torch.testing.assert_close(data["missing_field"], default_tensor)

    def test_missing_required_field_raises(
        self, tensorstore_available, zarr_v2_data_dir
    ):
        """Test that missing required field raises KeyError."""
        from physicsnemo.datapipes.readers.tensorstore_zarr import (
            TensorStoreZarrReader,
        )

        reader = TensorStoreZarrReader(
            zarr_v2_data_dir,
            fields=["positions", "nonexistent"],
            group_pattern="sample_*.zarr",
        )

        with pytest.raises(KeyError, match="nonexistent"):
            reader[0]

    def test_path_not_found_raises(self, tensorstore_available, tmp_path):
        """Test that nonexistent path raises FileNotFoundError."""
        from physicsnemo.datapipes.readers.tensorstore_zarr import (
            TensorStoreZarrReader,
        )

        with pytest.raises(FileNotFoundError):
            TensorStoreZarrReader(tmp_path / "nonexistent")

    def test_path_not_directory_raises(self, tensorstore_available, tmp_path):
        """Test that file path raises ValueError."""
        from physicsnemo.datapipes.readers.tensorstore_zarr import (
            TensorStoreZarrReader,
        )

        file_path = tmp_path / "file.txt"
        file_path.write_text("test")

        with pytest.raises(ValueError, match="directory"):
            TensorStoreZarrReader(file_path)

    def test_no_zarr_groups_raises(self, tensorstore_available, tmp_path):
        """Test that empty directory raises ValueError."""
        from physicsnemo.datapipes.readers.tensorstore_zarr import (
            TensorStoreZarrReader,
        )

        with pytest.raises(ValueError, match="No Zarr groups"):
            TensorStoreZarrReader(tmp_path, group_pattern="*.zarr")

    def test_coordinated_subsampling(self, tensorstore_available, zarr_v2_data_dir):
        """Test coordinated subsampling."""
        from physicsnemo.datapipes.readers.tensorstore_zarr import (
            TensorStoreZarrReader,
        )

        reader = TensorStoreZarrReader(
            zarr_v2_data_dir,
            group_pattern="sample_*.zarr",
            coordinated_subsampling={
                "n_points": 50,
                "target_keys": ["positions", "features"],
            },
        )

        data, metadata = reader[0]

        # Subsampled to 50 points
        assert data["positions"].shape[0] == 50
        assert data["features"].shape[0] == 50

    def test_coordinated_subsampling_too_few_points_raises(
        self, tensorstore_available, zarr_v2_data_dir
    ):
        """Test that requesting more points than available raises."""
        from physicsnemo.datapipes.readers.tensorstore_zarr import (
            TensorStoreZarrReader,
        )

        reader = TensorStoreZarrReader(
            zarr_v2_data_dir,
            group_pattern="sample_*.zarr",
            coordinated_subsampling={
                "n_points": 1000,  # More than 100 available
                "target_keys": ["positions"],
            },
        )

        with pytest.raises(ValueError, match="less than"):
            reader[0]

    def test_sample_metadata(self, tensorstore_available, zarr_v2_data_dir):
        """Test metadata includes source info."""
        from physicsnemo.datapipes.readers.tensorstore_zarr import (
            TensorStoreZarrReader,
        )

        reader = TensorStoreZarrReader(zarr_v2_data_dir, group_pattern="sample_*.zarr")

        data, metadata = reader[0]

        assert "source_file" in metadata
        assert "source_filename" in metadata
        assert "index" in metadata
        assert metadata["index"] == 0

    def test_negative_indexing(self, tensorstore_available, zarr_v2_data_dir):
        """Test negative indexing."""
        from physicsnemo.datapipes.readers.tensorstore_zarr import (
            TensorStoreZarrReader,
        )

        reader = TensorStoreZarrReader(zarr_v2_data_dir, group_pattern="sample_*.zarr")

        last_data, _ = reader[-1]
        also_last, _ = reader[4]

        torch.testing.assert_close(last_data["positions"], also_last["positions"])

    def test_repr(self, tensorstore_available, zarr_v2_data_dir):
        """Test string representation."""
        from physicsnemo.datapipes.readers.tensorstore_zarr import (
            TensorStoreZarrReader,
        )

        reader = TensorStoreZarrReader(zarr_v2_data_dir, group_pattern="sample_*.zarr")

        repr_str = repr(reader)

        assert "TensorStoreZarrReader" in repr_str
        assert "len=5" in repr_str

    def test_repr_with_subsampling(self, tensorstore_available, zarr_v2_data_dir):
        """Test repr with subsampling info."""
        from physicsnemo.datapipes.readers.tensorstore_zarr import (
            TensorStoreZarrReader,
        )

        reader = TensorStoreZarrReader(
            zarr_v2_data_dir,
            group_pattern="sample_*.zarr",
            coordinated_subsampling={"n_points": 50, "target_keys": ["positions"]},
        )

        repr_str = repr(reader)

        assert "subsampling=50" in repr_str

    def test_supports_coordinated_subsampling_property(
        self, tensorstore_available, zarr_v2_data_dir
    ):
        """Test _supports_coordinated_subsampling property."""
        from physicsnemo.datapipes.readers.tensorstore_zarr import (
            TensorStoreZarrReader,
        )

        reader = TensorStoreZarrReader(zarr_v2_data_dir, group_pattern="sample_*.zarr")

        assert reader._supports_coordinated_subsampling is True

    def test_pin_memory(self, tensorstore_available, zarr_v2_data_dir):
        """Test pin_memory option."""
        pytest.importorskip("torch")

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for pin_memory test")

        from physicsnemo.datapipes.readers.tensorstore_zarr import (
            TensorStoreZarrReader,
        )

        reader = TensorStoreZarrReader(
            zarr_v2_data_dir,
            group_pattern="sample_*.zarr",
            pin_memory=True,
        )

        data, _ = reader[0]

        assert data["positions"].is_pinned()


# ============================================================================
# VTKReader Tests
# ============================================================================


class TestVTKReader:
    """Tests for VTKReader."""

    @pytest.fixture
    def pyvista_available(self):
        """Check if pyvista is available."""
        try:
            import pyvista  # noqa: F401

            return True
        except ImportError:
            pytest.skip("PyVista not installed")

    @pytest.fixture
    def stl_data_dir(self, tmp_path, pyvista_available):
        """Create STL test data using pyvista."""
        import pyvista as pv

        for i in range(3):
            sample_dir = tmp_path / f"sample_{i:03d}"
            sample_dir.mkdir()

            # Create a simple cube mesh
            mesh = pv.Cube()
            mesh.save(sample_dir / "geometry.stl")

        return tmp_path

    @pytest.fixture
    def stl_with_exclude_pattern(self, tmp_path, pyvista_available):
        """Create STL data with files to exclude."""
        import pyvista as pv

        sample_dir = tmp_path / "sample_000"
        sample_dir.mkdir()

        # Create main geometry
        mesh = pv.Cube()
        mesh.save(sample_dir / "geometry.stl")

        # Create file that should be excluded
        mesh.save(sample_dir / "single_solid_geometry.stl")

        return tmp_path

    def test_import_error_without_pyvista(self, tmp_path):
        """Test that ImportError is raised when pyvista not installed."""
        from physicsnemo.datapipes.readers.vtk import (
            PYVISTA_AVAILABLE,
            VTKReader,
        )

        if PYVISTA_AVAILABLE:
            pytest.skip("PyVista is installed, cannot test ImportError")

        with pytest.raises(ImportError, match="PyVista is required"):
            VTKReader(tmp_path)

    def test_basic_loading(self, pyvista_available, stl_data_dir):
        """Test basic STL loading."""
        from physicsnemo.datapipes.readers.vtk import VTKReader

        reader = VTKReader(stl_data_dir)

        assert len(reader) == 3

        data, metadata = reader[0]

        assert isinstance(data, TensorDict)
        assert "stl_coordinates" in data
        assert "stl_faces" in data
        assert "stl_centers" in data
        assert "surface_normals" in data

    def test_field_selection(self, pyvista_available, stl_data_dir):
        """Test loading specific fields."""
        from physicsnemo.datapipes.readers.vtk import VTKReader

        reader = VTKReader(stl_data_dir, keys_to_read=["stl_coordinates", "stl_faces"])

        data, metadata = reader[0]

        assert "stl_coordinates" in data
        assert "stl_faces" in data
        assert "stl_centers" not in data
        assert "surface_normals" not in data

    def test_exclude_patterns(self, pyvista_available, stl_with_exclude_pattern):
        """Test file exclusion patterns."""
        from physicsnemo.datapipes.readers.vtk import VTKReader

        reader = VTKReader(stl_with_exclude_pattern, exclude_patterns=["single_solid"])

        # Should only load the main geometry, not the excluded one
        data, metadata = reader[0]
        assert "stl_coordinates" in data

    def test_path_not_found_raises(self, pyvista_available, tmp_path):
        """Test that nonexistent path raises FileNotFoundError."""
        from physicsnemo.datapipes.readers.vtk import VTKReader

        with pytest.raises(FileNotFoundError):
            VTKReader(tmp_path / "nonexistent")

    def test_path_not_directory_raises(self, pyvista_available, tmp_path):
        """Test that file path raises ValueError."""
        from physicsnemo.datapipes.readers.vtk import VTKReader

        file_path = tmp_path / "file.txt"
        file_path.write_text("test")

        with pytest.raises(ValueError, match="directory"):
            VTKReader(file_path)

    def test_no_vtk_directories_raises(self, pyvista_available, tmp_path):
        """Test that empty directory raises ValueError."""
        from physicsnemo.datapipes.readers.vtk import VTKReader

        # Create empty subdirectory
        (tmp_path / "empty_sample").mkdir()

        with pytest.raises(ValueError, match="No directories containing VTK"):
            VTKReader(tmp_path)

    def test_sample_metadata(self, pyvista_available, stl_data_dir):
        """Test metadata includes source info."""
        from physicsnemo.datapipes.readers.vtk import VTKReader

        reader = VTKReader(stl_data_dir)

        data, metadata = reader[0]

        assert "source_file" in metadata
        assert "source_filename" in metadata
        assert "index" in metadata
        assert metadata["index"] == 0

    def test_stl_data_shapes(self, pyvista_available, stl_data_dir):
        """Test that STL data has correct shapes."""
        from physicsnemo.datapipes.readers.vtk import VTKReader

        reader = VTKReader(stl_data_dir)

        data, metadata = reader[0]

        # Coordinates should be (N, 3)
        assert data["stl_coordinates"].ndim == 2
        assert data["stl_coordinates"].shape[1] == 3

        # Normals should be (M, 3)
        assert data["surface_normals"].ndim == 2
        assert data["surface_normals"].shape[1] == 3

        # Centers should be (M, 3)
        assert data["stl_centers"].ndim == 2
        assert data["stl_centers"].shape[1] == 3

    def test_stl_areas_computed(self, pyvista_available, stl_data_dir):
        """Test that STL areas are computed."""
        from physicsnemo.datapipes.readers.vtk import VTKReader

        reader = VTKReader(stl_data_dir, keys_to_read=["stl_areas"])

        data, metadata = reader[0]

        assert "stl_areas" in data
        # Areas should be 1D
        assert data["stl_areas"].ndim == 1
        # All areas should be positive
        assert (data["stl_areas"] > 0).all()

    def test_supports_coordinated_subsampling_false(
        self, pyvista_available, stl_data_dir
    ):
        """Test _supports_coordinated_subsampling returns False."""
        from physicsnemo.datapipes.readers.vtk import VTKReader

        reader = VTKReader(stl_data_dir)

        assert reader._supports_coordinated_subsampling is False

    def test_negative_indexing(self, pyvista_available, stl_data_dir):
        """Test negative indexing."""
        from physicsnemo.datapipes.readers.vtk import VTKReader

        reader = VTKReader(stl_data_dir)

        last_data, _ = reader[-1]
        also_last, _ = reader[2]

        torch.testing.assert_close(
            last_data["stl_coordinates"], also_last["stl_coordinates"]
        )

    def test_repr(self, pyvista_available, stl_data_dir):
        """Test string representation."""
        from physicsnemo.datapipes.readers.vtk import VTKReader

        reader = VTKReader(stl_data_dir)

        repr_str = repr(reader)

        assert "VTKReader" in repr_str
        assert "len=3" in repr_str

    def test_iteration(self, pyvista_available, stl_data_dir):
        """Test iteration over reader."""
        from physicsnemo.datapipes.readers.vtk import VTKReader

        reader = VTKReader(stl_data_dir)

        samples = list(reader)

        assert len(samples) == 3
        for i, (data, metadata) in enumerate(samples):
            assert metadata["index"] == i

    def test_pin_memory(self, pyvista_available, stl_data_dir):
        """Test pin_memory option."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for pin_memory test")

        from physicsnemo.datapipes.readers.vtk import VTKReader

        reader = VTKReader(stl_data_dir, pin_memory=True)

        data, _ = reader[0]

        assert data["stl_coordinates"].is_pinned()

    def test_field_names_property(self, pyvista_available, stl_data_dir):
        """Test field_names property."""
        from physicsnemo.datapipes.readers.vtk import VTKReader

        reader = VTKReader(stl_data_dir)

        field_names = reader.field_names

        assert "stl_coordinates" in field_names
        assert "stl_faces" in field_names

    def test_vtp_not_implemented(self, pyvista_available, stl_data_dir):
        """Test that VTP reading raises NotImplementedError."""
        from physicsnemo.datapipes.readers.vtk import VTKReader

        # Request VTP-specific keys
        reader = VTKReader(stl_data_dir, keys_to_read=["surface_mesh_centers"])

        # This should not load VTP since there are no .vtp files
        data, _ = reader[0]

        # surface_mesh_centers is a VTP key, should not be present
        assert "surface_mesh_centers" not in data
