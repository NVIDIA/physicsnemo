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


"""Tests for data readers."""

import pytest
import torch
from tensordict import TensorDict

import physicsnemo.datapipes as dp

# ============================================================================
# NumpyReader - Directory mode
# ============================================================================


def test_numpy_load_from_directory(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir, file_pattern="sample_*.npz")

    assert len(reader) == 10
    assert "positions" in reader.field_names
    assert "features" in reader.field_names


def test_numpy_get_sample(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    data, metadata = reader[0]

    assert isinstance(data, TensorDict)
    assert data["positions"].shape == (100, 3)
    assert data["features"].shape == (100, 8)
    assert data["positions"].dtype == torch.float32


def test_numpy_sample_metadata(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    data, metadata = reader[0]

    assert "index" in metadata
    assert metadata["index"] == 0
    assert "source_filename" in metadata


def test_numpy_negative_indexing(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)

    last_data, _ = reader[-1]
    also_last_data, _ = reader[9]

    torch.testing.assert_close(last_data["positions"], also_last_data["positions"])


def test_numpy_index_out_of_range(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)

    with pytest.raises(IndexError):
        _ = reader[100]


def test_numpy_iteration(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)

    samples = list(reader)
    assert len(samples) == 10
    for i, (data, metadata) in enumerate(samples):
        assert metadata["index"] == i


def test_numpy_select_fields(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir, fields=["positions"])
    data, metadata = reader[0]

    assert "positions" in data
    assert "features" not in data


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_numpy_pin_memory(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir, pin_memory=True)
    data, metadata = reader[0]

    assert data["positions"].is_pinned()


def test_numpy_context_manager(numpy_data_dir):
    with dp.NumpyReader(numpy_data_dir) as reader:
        data, metadata = reader[0]
        assert "positions" in data


# ============================================================================
# NumpyReader - Single file mode
# ============================================================================


def test_numpy_load_npz(numpy_npz_file):
    reader = dp.NumpyReader(numpy_npz_file)

    assert len(reader) == 15
    assert "images" in reader.field_names
    assert "labels" in reader.field_names


def test_numpy_get_sample_npz(numpy_npz_file):
    reader = dp.NumpyReader(numpy_npz_file)
    data, metadata = reader[5]

    assert data["images"].shape == (32, 32)
    assert data["labels"].item() == 5


# ============================================================================
# HDF5Reader
# ============================================================================


def test_hdf5_load_from_directory(hdf5_data_dir):
    reader = dp.HDF5Reader(hdf5_data_dir, file_pattern="sample_*.h5")

    assert len(reader) == 10
    assert "mesh" in reader.field_names
    assert "pressure" in reader.field_names


def test_hdf5_get_sample(hdf5_data_dir):
    reader = dp.HDF5Reader(hdf5_data_dir)
    data, metadata = reader[0]

    assert data["mesh"].shape == (200, 3)
    assert data["pressure"].shape == (200,)
    assert data["velocity"].shape == (200, 3)


def test_hdf5_single_file_mode(hdf5_single_file):
    reader = dp.HDF5Reader(hdf5_single_file)

    assert len(reader) == 25
    assert "inputs" in reader.field_names


def test_hdf5_get_sample_single_file(hdf5_single_file):
    reader = dp.HDF5Reader(hdf5_single_file)
    data, metadata = reader[10]

    assert data["inputs"].shape == (64,)
    assert data["targets"].shape == (10,)


def test_hdf5_select_fields(hdf5_data_dir):
    reader = dp.HDF5Reader(hdf5_data_dir, fields=["mesh"])
    data, metadata = reader[0]

    assert "mesh" in data
    assert "pressure" not in data


def test_hdf5_close(hdf5_single_file):
    reader = dp.HDF5Reader(hdf5_single_file)
    _ = reader[0]
    reader.close()
    # Should not raise on close


# ============================================================================
# ZarrReader
# ============================================================================


def test_zarr_load_from_directory(zarr_data_dir):
    reader = dp.ZarrReader(zarr_data_dir, group_pattern="sample_*.zarr")

    assert len(reader) == 10
    assert "field_a" in reader.field_names
    assert "field_b" in reader.field_names


def test_zarr_get_sample(zarr_data_dir):
    reader = dp.ZarrReader(zarr_data_dir)
    data, metadata = reader[0]

    assert data["field_a"].shape == (50, 50)
    assert data["field_b"].shape == (50,)


def test_zarr_single_group_mode(zarr_single_group):
    reader = dp.ZarrReader(zarr_single_group)

    assert len(reader) == 30


def test_zarr_get_sample_single_group(zarr_single_group):
    reader = dp.ZarrReader(zarr_single_group)
    data, metadata = reader[5]

    assert data["data"].shape == (16, 16)
    assert data["mask"].shape == (16, 16)


# ============================================================================
# Reader errors
# ============================================================================


def test_numpy_empty_directory(temp_dir):
    with pytest.raises(ValueError, match="No files matching"):
        dp.NumpyReader(temp_dir, file_pattern="*.npz")


def test_numpy_unsupported_extension(temp_dir):
    # Create a file with wrong extension
    (temp_dir / "data.txt").write_text("hello")

    with pytest.raises(ValueError, match="Unsupported file type"):
        dp.NumpyReader(temp_dir / "data.txt")


# ============================================================================
# Reader repr
# ============================================================================


def test_numpy_reader_repr(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    repr_str = repr(reader)

    assert "NumpyReader" in repr_str
    assert "len=10" in repr_str


def test_hdf5_reader_repr(hdf5_data_dir):
    reader = dp.HDF5Reader(hdf5_data_dir)
    repr_str = repr(reader)

    assert "HDF5Reader" in repr_str
    assert "directory" in repr_str
