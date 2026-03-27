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

"""
Unit tests for FloodForecaster dataset classes.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

# Conditionally include CUDA in device parametrization only if available
_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _DEVICES.append("cuda:0")

# Add the FloodForecaster example to the path
_examples_dir = Path(__file__).parent.parent.parent / "examples" / "weather" / "flood_modeling" / "flood_forecaster"
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

from datasets import (
    FloodDatasetWithQueryPoints,
    FloodRolloutTestDatasetNew,
    NormalizedDataset,
    NormalizedRolloutTestDataset,
)

from . import common

Tensor = torch.Tensor


@pytest.fixture
def sample_tensors():
    """Create sample tensors for dataset."""
    n_samples = 10
    n_cells = 100
    n_history = 3

    return {
        "geometry": torch.rand(n_samples, n_cells, 2),
        "static": torch.rand(n_samples, n_cells, 7),
        "boundary": torch.rand(n_samples, n_history, n_cells, 1),
        "dynamic": torch.rand(n_samples, n_history, n_cells, 3),
        "target": torch.rand(n_samples, n_cells, 3),
    }


@pytest.mark.parametrize("device", _DEVICES)
def test_normalized_dataset_constructor(sample_tensors, device):
    """Test NormalizedDataset constructor and basic properties."""
    ds = NormalizedDataset(
        geometry=sample_tensors["geometry"],
        static=sample_tensors["static"],
        boundary=sample_tensors["boundary"],
        dynamic=sample_tensors["dynamic"],
        target=sample_tensors["target"],
        query_res=[8, 8],
    )

    common.check_datapipe_iterable(ds)
    assert len(ds) == 10
    assert isinstance(ds, Dataset)


@pytest.mark.parametrize("device", _DEVICES)
def test_normalized_dataset_getitem(sample_tensors, device):
    """Test NormalizedDataset __getitem__ returns correct structure."""
    ds = NormalizedDataset(
        geometry=sample_tensors["geometry"],
        static=sample_tensors["static"],
        boundary=sample_tensors["boundary"],
        dynamic=sample_tensors["dynamic"],
        target=sample_tensors["target"],
        query_res=[8, 8],
    )

    sample = ds[0]

    assert isinstance(sample, dict)
    assert "geometry" in sample
    assert "static" in sample
    assert "boundary" in sample
    assert "dynamic" in sample
    assert "target" in sample
    assert "query_points" in sample

    # Check shapes
    assert sample["geometry"].shape == (100, 2)
    assert sample["static"].shape == (100, 7)
    assert sample["boundary"].shape == (3, 100, 1)
    assert sample["dynamic"].shape == (3, 100, 3)
    assert sample["target"].shape == (100, 3)
    assert sample["query_points"].shape == (8, 8, 2)


@pytest.mark.parametrize("query_res", [[4, 4], [8, 8], [16, 16]])
def test_normalized_dataset_query_points(sample_tensors, query_res):
    """Test query points generation for different resolutions."""
    ds = NormalizedDataset(
        geometry=sample_tensors["geometry"],
        static=sample_tensors["static"],
        boundary=sample_tensors["boundary"],
        dynamic=sample_tensors["dynamic"],
        target=sample_tensors["target"],
        query_res=query_res,
    )

    sample = ds[0]
    query_points = sample["query_points"]

    assert query_points.shape == (*query_res, 2)
    # Values should be in [0, 1] range (normalized coordinates)
    assert query_points.min() >= 0
    assert query_points.max() <= 1


@pytest.fixture
def rollout_samples():
    """Create sample rollout data."""
    n_samples = 5
    n_cells = 100
    n_timesteps = 20

    return [
        {
            "run_id": f"run_{i}",
            "geometry": torch.rand(n_cells, 2),
            "static": torch.rand(n_cells, 7),
            "boundary": torch.rand(n_timesteps, n_cells, 1),
            "dynamic": torch.rand(n_timesteps, n_cells, 3),
            "cell_area": torch.rand(n_cells),
        }
        for i in range(n_samples)
    ]


@pytest.mark.parametrize("device", _DEVICES)
def test_normalized_rollout_dataset_constructor(rollout_samples, device):
    """Test NormalizedRolloutTestDataset constructor."""
    ds = NormalizedRolloutTestDataset(rollout_samples, query_res=[8, 8])

    common.check_datapipe_iterable(ds)
    assert len(ds) == 5
    assert isinstance(ds, Dataset)


@pytest.mark.parametrize("device", _DEVICES)
def test_normalized_rollout_dataset_getitem(rollout_samples, device):
    """Test NormalizedRolloutTestDataset __getitem__ returns correct structure."""
    ds = NormalizedRolloutTestDataset(rollout_samples, query_res=[8, 8])
    sample = ds[0]

    assert isinstance(sample, dict)
    assert "run_id" in sample
    assert "geometry" in sample
    assert "static" in sample
    assert "boundary" in sample
    assert "dynamic" in sample
    assert "query_points" in sample
    assert "cell_area" in sample

    # Check run_id is preserved
    assert sample["run_id"] == "run_0"
    assert sample["cell_area"].shape == (100,)


# Tests for file-based dataset classes
@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory with mock data files."""
    import numpy as np

    data_dir = tmp_path / "test_data"
    data_dir.mkdir()

    # Create train.txt
    train_file = data_dir / "train.txt"
    train_file.write_text("run_001\nrun_002\n")

    # Create XY file
    n_cells = 100
    xy_data = np.random.rand(n_cells, 2).astype(np.float32)
    xy_file = data_dir / "M40_XY.txt"
    np.savetxt(xy_file, xy_data, delimiter="\t")

    # Create static files
    static_files = ["M40_CA.txt", "M40_CE.txt", "M40_CS.txt", "M40_FA.txt", "M40_A.txt", "M40_CU.txt"]
    for fname in static_files:
        static_data = np.random.rand(n_cells, 1).astype(np.float32)
        np.savetxt(data_dir / fname, static_data, delimiter="\t")

    # Create dynamic files for each run
    n_timesteps = 20
    for run_id in ["run_001", "run_002"]:
        for var in ["WD", "VX", "VY"]:
            # Shape: (n_timesteps, n_cells)
            dyn_data = np.random.rand(n_timesteps, n_cells).astype(np.float32)
            fname = f"M40_{var}_{run_id}.txt"
            np.savetxt(data_dir / fname, dyn_data, delimiter="\t")

    # Create boundary files
    for run_id in ["run_001", "run_002"]:
        # Shape: (n_timesteps, 1) or (n_timesteps, 2)
        bc_data = np.random.rand(n_timesteps, 1).astype(np.float32)
        fname = f"M40_US_InF_{run_id}.txt"
        np.savetxt(data_dir / fname, bc_data, delimiter="\t")

    return data_dir


@pytest.fixture
def temp_rollout_dir(tmp_path):
    """Create a temporary rollout data directory with mock data files."""
    import numpy as np

    data_dir = tmp_path / "rollout_data"
    data_dir.mkdir()

    # Create test.txt
    test_file = data_dir / "test.txt"
    test_file.write_text("run_001\nrun_002\n")

    # Create XY file
    n_cells = 100
    xy_data = np.random.rand(n_cells, 2).astype(np.float32)
    xy_file = data_dir / "M40_XY.txt"
    np.savetxt(xy_file, xy_data, delimiter="\t")

    # Create static files including cell area
    static_files = ["M40_CA.txt", "M40_CE.txt", "M40_CS.txt", "M40_FA.txt", "M40_A.txt", "M40_CU.txt"]
    for fname in static_files:
        static_data = np.random.rand(n_cells, 1).astype(np.float32)
        np.savetxt(data_dir / fname, static_data, delimiter="\t")

    # Create dynamic files for each run (need enough timesteps for rollout)
    n_timesteps = 30  # Enough for n_history + rollout_length
    for run_id in ["run_001", "run_002"]:
        for var in ["WD", "VX", "VY"]:
            dyn_data = np.random.rand(n_timesteps, n_cells).astype(np.float32)
            fname = f"M40_{var}_{run_id}.txt"
            np.savetxt(data_dir / fname, dyn_data, delimiter="\t")

    # Create boundary files
    for run_id in ["run_001", "run_002"]:
        bc_data = np.random.rand(n_timesteps, 1).astype(np.float32)
        fname = f"M40_US_InF_{run_id}.txt"
        np.savetxt(data_dir / fname, bc_data, delimiter="\t")

    return data_dir


@pytest.mark.parametrize("device", _DEVICES)
def test_flood_dataset_with_query_points_init(temp_data_dir, device):
    """Test FloodDatasetWithQueryPoints initialization."""
    # Note: xy_file should not be included in static_files to avoid duplication
    # The xy_file is loaded separately for geometry, while static_files are for additional features
    dataset = FloodDatasetWithQueryPoints(
        data_root=str(temp_data_dir),
        n_history=3,
        xy_file="M40_XY.txt",
        static_files=["M40_CA.txt", "M40_CE.txt"],  # Exclude xy_file to avoid duplication
        dynamic_patterns={
            "WD": "M40_WD_{}.txt",
            "VX": "M40_VX_{}.txt",
            "VY": "M40_VY_{}.txt",
        },
        boundary_patterns={"inflow": "M40_US_InF_{}.txt"},
    )

    common.check_datapipe_iterable(dataset)
    assert len(dataset) > 0
    assert dataset.xy_coords is not None
    assert len(dataset.run_ids) == 2


@pytest.mark.parametrize("device", _DEVICES)
def test_flood_dataset_with_query_points_getitem(temp_data_dir, device):
    """Test FloodDatasetWithQueryPoints __getitem__ returns correct structure."""
    dataset = FloodDatasetWithQueryPoints(
        data_root=str(temp_data_dir),
        n_history=3,
        xy_file="M40_XY.txt",
        static_files=["M40_CA.txt", "M40_CE.txt"],
        dynamic_patterns={
            "WD": "M40_WD_{}.txt",
            "VX": "M40_VX_{}.txt",
            "VY": "M40_VY_{}.txt",
        },
        boundary_patterns={"inflow": "M40_US_InF_{}.txt"},
    )

    if len(dataset) > 0:
        sample = dataset[0]

        assert isinstance(sample, dict)
        assert "geometry" in sample
        assert "static" in sample
        assert "boundary" in sample
        assert "dynamic" in sample
        assert "target" in sample
        assert "run_id" in sample
        assert "time_index" in sample

        # Check shapes
        assert sample["geometry"].shape == (100, 2)
        assert sample["static"].shape == (100, 2)  # 2 static files
        assert sample["boundary"].shape == (3, 100, 1)  # n_history, n_cells, bc_dim
        assert sample["dynamic"].shape == (3, 100, 3)  # n_history, n_cells, 3 channels
        assert sample["target"].shape == (100, 3)  # n_cells, 3 channels


@pytest.mark.parametrize("device", _DEVICES)
def test_flood_dataset_noise_types(temp_data_dir, device):
    """Test different noise types in FloodDatasetWithQueryPoints."""
    noise_types = ["none", "only_last", "correlated", "uncorrelated", "random_walk"]

    for noise_type in noise_types:
        dataset = FloodDatasetWithQueryPoints(
            data_root=str(temp_data_dir),
            n_history=3,
            xy_file="M40_XY.txt",
            static_files=["M40_CA.txt"],
            dynamic_patterns={
                "WD": "M40_WD_{}.txt",
                "VX": "M40_VX_{}.txt",
                "VY": "M40_VY_{}.txt",
            },
            boundary_patterns={"inflow": "M40_US_InF_{}.txt"},
            noise_type=noise_type,
            noise_std=[0.01, 0.001, 0.001],
        )

        if len(dataset) > 0:
            sample = dataset[0]
            # Should not crash
            assert "dynamic" in sample


def test_flood_dataset_missing_data_root():
    """Test FloodDatasetWithQueryPoints fails with missing data root."""
    with pytest.raises(FileNotFoundError):
        FloodDatasetWithQueryPoints(
            data_root="/nonexistent/path",
            n_history=3,
            xy_file="M40_XY.txt",
        )


def test_flood_dataset_missing_train_file(tmp_path):
    """Test FloodDatasetWithQueryPoints fails with missing train.txt."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="train.txt"):
        FloodDatasetWithQueryPoints(
            data_root=str(data_dir),
            n_history=3,
            xy_file="M40_XY.txt",
        )


def test_flood_dataset_invalid_noise_std(temp_data_dir):
    """Test FloodDatasetWithQueryPoints validates noise_std length."""
    with pytest.raises(ValueError, match="exactly 3 floats"):
        FloodDatasetWithQueryPoints(
            data_root=str(temp_data_dir),
            n_history=3,
            xy_file="M40_XY.txt",
            noise_std=[0.01, 0.001],  # Only 2 values
        )


@pytest.mark.parametrize("device", _DEVICES)
def test_flood_rollout_test_dataset_new_init(temp_rollout_dir, device):
    """Test FloodRolloutTestDatasetNew initialization."""
    dataset = FloodRolloutTestDatasetNew(
        rollout_data_root=str(temp_rollout_dir),
        n_history=3,
        rollout_length=10,
        xy_file="M40_XY.txt",
        static_files=["M40_CA.txt", "M40_CE.txt"],
        dynamic_patterns={
            "WD": "M40_WD_{}.txt",
            "VX": "M40_VX_{}.txt",
            "VY": "M40_VY_{}.txt",
        },
        boundary_patterns={"inflow": "M40_US_InF_{}.txt"},
    )

    common.check_datapipe_iterable(dataset)
    assert len(dataset) > 0
    assert dataset.xy_coords is not None
    assert len(dataset.valid_run_ids) > 0


@pytest.mark.parametrize("device", _DEVICES)
def test_flood_rollout_test_dataset_new_getitem(temp_rollout_dir, device):
    """Test FloodRolloutTestDatasetNew __getitem__ returns correct structure."""
    dataset = FloodRolloutTestDatasetNew(
        rollout_data_root=str(temp_rollout_dir),
        n_history=3,
        rollout_length=10,
        xy_file="M40_XY.txt",
        static_files=["M40_CA.txt", "M40_CE.txt"],
        dynamic_patterns={
            "WD": "M40_WD_{}.txt",
            "VX": "M40_VX_{}.txt",
            "VY": "M40_VY_{}.txt",
        },
        boundary_patterns={"inflow": "M40_US_InF_{}.txt"},
    )

    if len(dataset) > 0:
        sample = dataset[0]

        assert isinstance(sample, dict)
        assert "run_id" in sample
        assert "geometry" in sample
        assert "static" in sample
        assert "boundary" in sample
        assert "dynamic" in sample

        # Check shapes
        assert sample["geometry"].shape == (100, 2)
        assert sample["static"].shape == (100, 2)  # 2 static files
        assert sample["dynamic"].shape == (30, 100, 3)  # Full time series
        assert sample["boundary"].shape == (30, 100, 1)  # Full time series


def test_flood_rollout_test_dataset_missing_data_root():
    """Test FloodRolloutTestDatasetNew fails with missing data root."""
    with pytest.raises(FileNotFoundError):
        FloodRolloutTestDatasetNew(
            rollout_data_root="/nonexistent/path",
            n_history=3,
            rollout_length=10,
            xy_file="M40_XY.txt",
        )


def test_flood_rollout_test_dataset_insufficient_timesteps(tmp_path):
    """Test that runs with insufficient timesteps are filtered out."""
    import numpy as np

    data_dir = tmp_path / "rollout_data"
    data_dir.mkdir()

    test_file = data_dir / "test.txt"
    test_file.write_text("run_001\n")

    xy_file = data_dir / "M40_XY.txt"
    np.savetxt(xy_file, np.random.rand(100, 2), delimiter="\t")

    # Create static files
    for fname in ["M40_CA.txt", "M40_CE.txt"]:
        np.savetxt(data_dir / fname, np.random.rand(100, 1), delimiter="\t")

    # Create dynamic files with insufficient timesteps
    n_timesteps = 5  # Too few for n_history=3 + rollout_length=10
    for var in ["WD", "VX", "VY"]:
        dyn_data = np.random.rand(n_timesteps, 100).astype(np.float32)
        np.savetxt(data_dir / f"M40_{var}_run_001.txt", dyn_data, delimiter="\t")

    # Create boundary file
    bc_data = np.random.rand(n_timesteps, 1).astype(np.float32)
    np.savetxt(data_dir / "M40_US_InF_run_001.txt", bc_data, delimiter="\t")

    with pytest.raises(ValueError, match="No hydrographs have enough time steps"):
        FloodRolloutTestDatasetNew(
            rollout_data_root=str(data_dir),
            n_history=3,
            rollout_length=10,
            xy_file="M40_XY.txt",
            static_files=["M40_CA.txt"],
            dynamic_patterns={
                "WD": "M40_WD_{}.txt",
                "VX": "M40_VX_{}.txt",
                "VY": "M40_VY_{}.txt",
            },
            boundary_patterns={"inflow": "M40_US_InF_{}.txt"},
        )

