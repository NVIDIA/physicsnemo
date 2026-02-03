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

"""Shared fixtures for datapipe tests."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict

# ============================================================================
# Sample Fixtures
# ============================================================================


@pytest.fixture
def simple_sample():
    """A simple sample with basic tensors."""
    data = TensorDict(
        {
            "x": torch.randn(100, 3),
            "y": torch.randn(100),
        }
    )
    return data, {}


@pytest.fixture
def sample_with_metadata():
    """A sample with metadata."""
    data = TensorDict({"pressure": torch.randn(50), "velocity": torch.randn(50, 3)})
    metadata = {"filename": "test.h5", "index": 42}
    return data, metadata


@pytest.fixture
def batch_of_samples():
    """Multiple samples for collation tests."""
    return [
        (
            TensorDict({"x": torch.randn(10, 3), "y": torch.randn(10)}),
            {"idx": i},
        )
        for i in range(4)
    ]


@pytest.fixture
def ragged_samples():
    """Samples with different sizes (for ConcatCollator)."""
    return [
        (TensorDict({"points": torch.randn(100, 3)}), {"idx": 0}),
        (TensorDict({"points": torch.randn(150, 3)}), {"idx": 1}),
        (TensorDict({"points": torch.randn(80, 3)}), {"idx": 2}),
    ]


# ============================================================================
# Synthetic Data Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory that's cleaned up after the test."""
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path)


@pytest.fixture
def numpy_data_dir(temp_dir):
    """Create a directory with .npz files for NumpyReader tests."""
    for i in range(10):
        np.savez(
            temp_dir / f"sample_{i:03d}.npz",
            positions=np.random.randn(100, 3).astype(np.float32),
            features=np.random.randn(100, 8).astype(np.float32),
            label=np.array([i], dtype=np.int64),
        )
    return temp_dir


@pytest.fixture
def numpy_npz_file(temp_dir):
    """Create a single .npz file with multiple arrays."""
    path = temp_dir / "data.npz"
    np.savez(
        path,
        images=np.random.randn(15, 32, 32).astype(np.float32),
        labels=np.arange(15, dtype=np.int64),
    )
    return path


@pytest.fixture
def hdf5_data_dir(temp_dir):
    """Create a directory with .h5 files for HDF5Reader tests."""
    h5py = pytest.importorskip("h5py")

    for i in range(10):
        with h5py.File(temp_dir / f"sample_{i:03d}.h5", "w") as f:
            f.create_dataset("mesh", data=np.random.randn(200, 3).astype(np.float32))
            f.create_dataset("pressure", data=np.random.randn(200).astype(np.float32))
            f.create_dataset(
                "velocity", data=np.random.randn(200, 3).astype(np.float32)
            )

    return temp_dir


@pytest.fixture
def hdf5_single_file(temp_dir):
    """Create a single .h5 file with samples indexed along first dim."""
    h5py = pytest.importorskip("h5py")

    path = temp_dir / "data.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("inputs", data=np.random.randn(25, 64).astype(np.float32))
        f.create_dataset("targets", data=np.random.randn(25, 10).astype(np.float32))

    return path


@pytest.fixture
def zarr_data_dir(temp_dir):
    """Create a directory with .zarr groups for ZarrReader tests."""
    zarr = pytest.importorskip("zarr")

    for i in range(10):
        group_path = temp_dir / f"sample_{i:03d}.zarr"
        root = zarr.open(group_path, mode="w")
        root.create_array("field_a", data=np.random.randn(50, 50).astype(np.float32))
        root.create_array("field_b", data=np.random.randn(50).astype(np.float32))

    return temp_dir


@pytest.fixture
def zarr_single_group(temp_dir):
    """Create a single .zarr group with samples indexed along first dim."""
    zarr = pytest.importorskip("zarr")

    path = temp_dir / "data.zarr"
    root = zarr.open(path, mode="w")
    root.create_array("data", data=np.random.randn(30, 16, 16).astype(np.float32))
    root.create_array(
        "mask", data=np.random.randint(0, 2, (30, 16, 16)).astype(np.uint8)
    )

    return path


# ============================================================================
# Device Fixtures
# ============================================================================


@pytest.fixture
def cuda_available():
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return True
