# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import numpy as np
import pytest
import torch
import zarr
from pytest_utils import import_or_fail
from scipy.spatial import ConvexHull

Tensor = torch.Tensor

# DEFINING GLOBAL VARIABLES HERE
# this is for checking against normalizations
# but also a consolidated place to update / manage them
DATA_XMIN = -2.0
DATA_XMAX = 3.0
DATA_YMIN = -4.0
DATA_YMAX = 1.0
DATA_ZMIN = -0.5
DATA_ZMAX = 4.0

# These variables aren't meaningful in any sense,
# except that they are al unique and we can check
# against them.
SURF_BBOX_XMIN = -2.5
SURF_BBOX_XMAX = 3.5
SURF_BBOX_YMIN = -4.25
SURF_BBOX_YMAX = 1.25
SURF_BBOX_ZMIN = 0.0
SURF_BBOX_ZMAX = 2.00

VOL_BBOX_XMIN = -3.5
VOL_BBOX_XMAX = 3.5
VOL_BBOX_YMIN = -2.25
VOL_BBOX_YMAX = 2.25
VOL_BBOX_ZMIN = -0.32
VOL_BBOX_ZMAX = 3.00


def random_sample_on_unit_sphere(n_points):
    # Random points on the sphere:
    phi = np.random.uniform(0, 2 * np.pi, n_points)
    cos_theta = np.random.uniform(-1, 1, n_points)
    theta = np.arccos(cos_theta)

    # Convert to x/y/z and stack:
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    # Shift the entire sphere to Z > 0
    z = np.cos(theta) + 1
    points = np.stack([x, y, z], axis=1)
    return points


def synthetic_domino_data(
    out_format: Literal["zarr", "npy", "npz"],
    n_examples: int = 3,
    N_mesh_points: int = 1000,
    N_surface_samples: int = 5000,
    N_volume_samples_max: int = 20000,
):
    """Generate synthetic domino data and save to temporary directory structure using zarr."""

    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())

    # Create subdirectory for the specific format
    format_dir = temp_dir / out_format
    format_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_examples):
        # We are generating a mesh on a random sphere.
        stl_points = random_sample_on_unit_sphere(N_mesh_points)
        print(f"stl_points.shape: {stl_points.shape}")
        # Generate the triangles with ConvexHull:
        hull = ConvexHull(stl_points)
        faces = hull.simplices  # (M, 3)

        # If you ever need to visualize this, here's the pyvista code:
        # faces_flat = np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten()
        #
        # mesh = pv.PolyData(points, faces_flat)
        # mesh.plot(show_edges=True, color="lightblue")

        # Get the triangle verts
        tri_pts = stl_points[faces]  # (M, 3, 3)

        # Compute the vectors for two edges:
        vec1 = tri_pts[:, 1] - tri_pts[:, 0]
        vec2 = tri_pts[:, 2] - tri_pts[:, 0]

        cross = np.cross(vec1, vec2)
        areas = 0.5 * np.linalg.norm(cross, axis=1)  # (M)

        centroids = tri_pts.mean(axis=1)  # (M, 3)

        out_dict = {
            "stl_coordinates": stl_points.astype(np.float32),
            "stl_faces": faces.astype(np.int32).flatten(),
            "stl_centers": centroids.astype(np.float32),
            "stl_areas": areas.astype(np.float32),
            "air_density": np.float32(1.225),  # Standard air density
            "stream_velocity": np.float32(10.0),  # Some velocity value
        }

        # Now, we will randomly sample for the surface and volume data.
        # We will just do random sphere sampling again for the surface,
        # but this time the other variables are just random.

        out_dict["surface_mesh_centers"] = random_sample_on_unit_sphere(
            N_surface_samples
        ).astype(np.float32)
        out_dict["surface_areas"] = np.random.uniform(
            0.01, 1.0, N_surface_samples
        ).astype(np.float32)
        # The normal, on a unit sphere, is just the value of the point itself:
        out_dict["surface_normals"] = out_dict["surface_mesh_centers"]
        out_dict["surface_fields"] = np.random.randn(N_surface_samples, 4).astype(
            np.float32
        )

        # For volume data, we're going to sample in a rectangular volume
        # and then drop everything with |r| <= 1
        volume_mesh_centers_x = np.random.uniform(
            DATA_XMIN, DATA_XMAX, (N_volume_samples_max,)
        ).astype(np.float32)
        volume_mesh_centers_y = np.random.uniform(
            DATA_YMIN, DATA_YMAX, (N_volume_samples_max,)
        ).astype(np.float32)
        volume_mesh_centers_z = np.random.uniform(
            DATA_ZMIN, DATA_ZMAX, (N_volume_samples_max,)
        ).astype(np.float32)

        volume_points = np.stack(
            [volume_mesh_centers_x, volume_mesh_centers_y, volume_mesh_centers_z],
            axis=1,
        )

        norm = np.linalg.norm(volume_points - np.asarray([[0.0, 0.0, 1.0]]), axis=1)
        accepted_points = volume_points[norm > 1.0]

        out_dict["volume_mesh_centers"] = accepted_points
        out_dict["volume_fields"] = np.random.randn(accepted_points.shape[0], 5)

        # Now, save the output:
        if out_format == "zarr":
            # Save data in zarr format for each model type (all keys for all types)
            zarr_path = format_dir / f"fake_drivaer_ml_data_{i}.zarr"

            # Create zarr group and save all data
            root = zarr.open(str(zarr_path), mode="w")
            for key, value in out_dict.items():
                root[key] = value
        elif out_format == "npz":
            npz_path = format_dir / f"fake_drivaer_ml_data_{i}.npz"
            np.savez(npz_path, **out_dict)
        elif out_format == "npy":
            npy_path = format_dir / f"fake_drivaer_ml_data_{i}.npy"
            np.save(npy_path, out_dict)

    # Return temp_dir after processing all examples
    return temp_dir


@pytest.fixture
def zarr_dataset():
    """Fixture to generate a synthetic Zarr dataset."""

    data_dir = synthetic_domino_data(n_examples=3, out_format="zarr")
    yield data_dir / "zarr/"
    # Cleanup temporary directory
    shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture
def npz_dataset():
    """Fixture to generate a synthetic npz dataset."""

    data_dir = synthetic_domino_data(n_examples=3, out_format="npz")
    yield data_dir / "npz/"
    # Cleanup temporary directory
    shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture
def npy_dataset():
    """Fixture to generate a synthetic npy dataset."""

    data_dir = synthetic_domino_data(n_examples=3, out_format="npy")
    yield data_dir / "npy/"
    # Cleanup temporary directory
    shutil.rmtree(data_dir, ignore_errors=True)


@dataclass
class ConcreteBoundingBox:
    """
    Really simple bounding box to mimic a structured config; Don't use elsewhere.
    """

    min: List[float]
    max: List[float]


def bounding_boxes():
    """Common bounding box configurations for tests."""
    return {
        "volume": ConcreteBoundingBox(
            min=[VOL_BBOX_XMIN, VOL_BBOX_YMIN, VOL_BBOX_ZMIN],
            max=[VOL_BBOX_XMAX, VOL_BBOX_YMAX, VOL_BBOX_ZMAX],
        ),
        "surface": ConcreteBoundingBox(
            min=[SURF_BBOX_XMIN, SURF_BBOX_YMIN, SURF_BBOX_ZMIN],
            max=[SURF_BBOX_XMAX, SURF_BBOX_YMAX, SURF_BBOX_ZMAX],
        ),
    }


def create_basic_dataset(data_dir, model_type, **kwargs):
    """Helper function to create a basic DoMINODataPipe with default settings."""
    from physicsnemo.datapipes.cae.domino_datapipe import DoMINODataPipe

    # assert model_type in ["volume", "surface", "combined"]

    input_path = data_dir

    bounding_box = bounding_boxes()

    default_kwargs = {
        "phase": "test",
        "grid_resolution": [64, 64, 64],
        "volume_points_sample": 1234,
        "surface_points_sample": 1234,
        "geom_points_sample": 2345,
        "num_surface_neighbors": 5,
        "bounding_box_dims": bounding_box["volume"],
        "bounding_box_dims_surf": bounding_box["surface"],
        "normalize_coordinates": True,
        "sampling": False,
        "sample_in_bbox": False,
        "positional_encoding": False,
        "scaling_type": None,
        "volume_factors": None,
        "surface_factors": None,
        "caching": False,
        "compute_scaling_factors": False,
        "gpu_preprocessing": True,
        "gpu_output": True,
    }

    default_kwargs.update(kwargs)

    return DoMINODataPipe(
        input_path=input_path, model_type=model_type, **default_kwargs
    )


def validate_sample_structure(sample, model_type, gpu_output):
    """Helper function to validate the structure of a dataset sample."""
    assert isinstance(sample, dict)

    # Common keys that should always be present
    expected_keys = ["geometry_coordinates", "length_scale", "surface_min_max"]

    # Model-specific keys
    volume_keys = [
        "volume_mesh_centers",
        "volume_fields",
        "grid",
        "sdf_grid",
        "sdf_nodes",
    ]
    surface_keys = [
        "surface_mesh_centers",
        "surface_normals",
        "surface_areas",
        "surface_fields",
    ]

    if model_type in ["volume", "combined"]:
        expected_keys.extend(volume_keys)
    if model_type in ["surface", "combined"]:
        expected_keys.extend(surface_keys)

    # Check that required keys are present and are torch tensors on correct device
    for key in expected_keys:
        if key in sample:  # Some keys may be None if compute_scaling_factors=True
            if sample[key] is not None:
                assert isinstance(sample[key], torch.Tensor), (
                    f"Key {key} should be torch.Tensor"
                )
                expected_device = "cuda" if gpu_output else "cpu"
                assert sample[key].device.type == expected_device, (
                    f"Key {key} on wrong device"
                )


# Core test - smaller matrix focusing on essential device/model combinations
@import_or_fail(["warp", "cupy", "cuml"])
@pytest.mark.parametrize("data_dir", ["zarr_dataset", "npz_dataset", "npy_dataset"])
@pytest.mark.parametrize("gpu_preprocessing", [True, False])
@pytest.mark.parametrize("gpu_output", [True, False])
@pytest.mark.parametrize("model_type", ["surface", "volume", "combined"])
def test_domino_datapipe_core(
    data_dir, gpu_preprocessing, gpu_output, model_type, pytestconfig, request
):
    """Core test for basic functionality with different device and model configurations."""

    data_dir = request.getfixturevalue(data_dir)
    print(f"data_dir: {data_dir}")
    dataset = create_basic_dataset(
        data_dir, model_type, gpu_preprocessing=gpu_preprocessing, gpu_output=gpu_output
    )

    assert len(dataset) > 0
    sample = dataset[0]
    validate_sample_structure(sample, model_type, gpu_output)


# Feature-specific tests
@import_or_fail(["warp", "cupy", "cuml"])
@pytest.mark.parametrize("model_type", ["combined"])
@pytest.mark.parametrize("normalize_coordinates", [True, False])
@pytest.mark.parametrize("sample_in_bbox", [True, False])
def test_domino_datapipe_coordinate_normalization(
    zarr_dataset, model_type, normalize_coordinates, sample_in_bbox, pytestconfig
):
    """Test coordinate normalization functionality."""
    dataset = create_basic_dataset(
        zarr_dataset,
        model_type,
        gpu_preprocessing=True,
        normalize_coordinates=normalize_coordinates,
        sample_in_bbox=sample_in_bbox,
    )

    sample = dataset[0]
    validate_sample_structure(sample, model_type, gpu_output=True)

    v_coords = sample["volume_mesh_centers"]
    s_coords = sample["surface_mesh_centers"]

    v_min = torch.min(v_coords, dim=0).values
    v_max = torch.max(v_coords, dim=0).values
    s_min = torch.min(s_coords, dim=0).values
    s_max = torch.max(s_coords, dim=0).values

    print(f"{normalize_coordinates} v_coords: {v_min} to {v_max}")
    print(f"{normalize_coordinates} s_coords: {s_min} to {s_max}")
    # If normalization is enabled, coordinates should be in [-2, 2] range
    if normalize_coordinates:
        if sample_in_bbox:
            # In this case, the values are rescaled, but only the ones
            # that were already inside the box should be present.

            # That means that all values should be between -1 and 1
            assert v_min[0] >= -1
            assert v_min[1] >= -1
            assert v_min[2] >= -1
            assert v_max[0] <= 1
            assert v_max[1] <= 1
            assert v_max[2] <= 1

        else:
            # When normalizing the coordinates, the values of the bbox
            # for surface and volume will get shifted: everything outside
            # of the bbox will have |val| > 1.0, while inside will have < 1.
            # This leads to both a rescale and a shift.

            # For testing purposes, we'll expect this to shift the extrema values
            # For example, in x, if the max value is 5 and the bbox is [-1, 2],
            # the new value will be shifted to
            # 2 * (val - min_val) / field_range - 1
            # So, field_range = (2 - -1) = 3
            # new_val = 2 * (5 - -1)/ 3 - 1 = 3

            vol_x_rescale = 1 / (VOL_BBOX_XMAX - VOL_BBOX_XMIN)
            vol_y_rescale = 1 / (VOL_BBOX_YMAX - VOL_BBOX_YMIN)
            vol_z_rescale = 1 / (VOL_BBOX_ZMAX - VOL_BBOX_ZMIN)

            assert v_min[0] >= 2 * (DATA_XMIN - VOL_BBOX_XMIN) * vol_x_rescale - 1
            assert v_min[1] >= 2 * (DATA_YMIN - VOL_BBOX_YMIN) * vol_y_rescale - 1
            assert v_min[2] >= 2 * (DATA_ZMIN - VOL_BBOX_ZMIN) * vol_z_rescale - 1
            assert v_max[0] <= 2 * (DATA_XMAX - VOL_BBOX_XMIN) * vol_x_rescale - 1
            assert v_max[1] <= 2 * (DATA_YMAX - VOL_BBOX_YMIN) * vol_y_rescale - 1
            assert v_max[2] <= 2 * (DATA_ZMAX - VOL_BBOX_ZMIN) * vol_z_rescale - 1

            surf_x_rescale = 1 / (SURF_BBOX_XMAX - SURF_BBOX_XMIN)
            surf_y_rescale = 1 / (SURF_BBOX_YMAX - SURF_BBOX_YMIN)
            surf_z_rescale = 1 / (SURF_BBOX_ZMAX - SURF_BBOX_ZMIN)

            assert s_min[0] >= 2 * (DATA_XMIN - SURF_BBOX_XMIN) * surf_x_rescale - 1
            assert s_min[1] >= 2 * (DATA_YMIN - SURF_BBOX_YMIN) * surf_y_rescale - 1
            assert s_min[2] >= 2 * (DATA_ZMIN - SURF_BBOX_ZMIN) * surf_z_rescale - 1
            assert s_max[0] <= 2 * (DATA_XMAX - SURF_BBOX_XMIN) * surf_x_rescale - 1
            assert s_max[1] <= 2 * (DATA_YMAX - SURF_BBOX_YMIN) * surf_y_rescale - 1
            assert s_max[2] <= 2 * (DATA_ZMAX - SURF_BBOX_ZMIN) * surf_z_rescale - 1

    else:
        if sample_in_bbox:
            # We've sampled in the bbox but NOT normalized.
            # So, the values should exclusively be in the BBOX ranges:
            assert v_min[0] >= VOL_BBOX_XMIN
            assert v_min[1] >= VOL_BBOX_YMIN
            assert v_min[2] >= VOL_BBOX_ZMIN
            assert v_max[0] <= VOL_BBOX_XMAX
            assert v_max[1] <= VOL_BBOX_YMAX
            assert v_max[2] <= VOL_BBOX_ZMAX

            assert s_min[0] >= SURF_BBOX_XMIN
            assert s_min[1] >= SURF_BBOX_YMIN
            assert s_min[2] >= SURF_BBOX_ZMIN
            assert s_max[0] <= SURF_BBOX_XMAX
            assert s_max[1] <= SURF_BBOX_YMAX
            assert s_max[2] <= SURF_BBOX_ZMAX

        else:
            # Not sampling, and also
            # Not normalizing, values should be in data range only:
            assert v_min[0] >= DATA_XMIN and v_max[0] <= DATA_XMAX
            assert v_min[1] >= DATA_YMIN and v_max[1] <= DATA_YMAX
            assert v_min[2] >= DATA_ZMIN and v_max[2] <= DATA_ZMAX
            assert s_min[0] >= DATA_XMIN and s_max[0] <= DATA_XMAX
            assert s_min[1] >= DATA_YMIN and s_max[1] <= DATA_YMAX
            # Surface points always should be > 0
            assert s_min[2] >= 0 and s_max[2] <= DATA_ZMAX


@import_or_fail(["warp", "cupy", "cuml"])
@pytest.mark.parametrize("model_type", ["combined"])
@pytest.mark.parametrize("sampling", [True, False])
def test_domino_datapipe_sampling(zarr_dataset, model_type, sampling, pytestconfig):
    """Test point sampling functionality."""
    sample_points = 4321
    dataset = create_basic_dataset(
        zarr_dataset,
        model_type,
        gpu_preprocessing=False,
        sampling=sampling,
        volume_points_sample=sample_points,
        surface_points_sample=sample_points,
    )

    sample = dataset[0]
    validate_sample_structure(sample, model_type, gpu_output=True)

    if model_type in ["volume", "combined"]:
        for key in ["volume_mesh_centers", "volume_fields"]:
            if sampling:
                assert sample[key].shape[0] == sample_points
            else:
                assert sample[key].shape[0] == sample["volume_mesh_centers"].shape[0]

    # Model-specific keys
    if model_type in ["surface", "combined"]:
        for key in [
            "surface_mesh_centers",
            "surface_normals",
            "surface_areas",
            "surface_fields",
        ]:
            if sampling:
                assert sample[key].shape[0] == sample_points
            else:
                assert sample[key].shape[0] == sample["surface_mesh_centers"].shape[0]
        for key in [
            "surface_mesh_neighbors",
            "surface_neighbors_normals",
            "surface_neighbors_areas",
        ]:
            if sampling:
                assert sample[key].shape[0] == sample_points
                assert sample[key].shape[1] == dataset.config.num_surface_neighbors - 1
            else:
                assert sample[key].shape[0] == sample["surface_mesh_neighbors"].shape[0]
                assert sample[key].shape[1] == dataset.config.num_surface_neighbors - 1


@import_or_fail(["warp", "cupy", "cuml"])
@pytest.mark.parametrize("model_type", ["combined"])
@pytest.mark.parametrize(
    "positional_encoding",
    [
        True,
    ],
)
def test_domino_datapipe_positional_encoding(
    zarr_dataset, model_type, positional_encoding, pytestconfig
):
    """Test positional encoding functionality."""
    dataset = create_basic_dataset(
        zarr_dataset,
        model_type,
        gpu_preprocessing=False,
        positional_encoding=positional_encoding,
    )

    sample = dataset[0]
    validate_sample_structure(sample, model_type, gpu_output=True)

    # Check for positional encoding keys
    if positional_encoding:
        pos_keys = ["pos_volume_closest", "pos_volume_center_of_mass"]
        for key in pos_keys:
            if key in sample:
                assert sample[key] is not None


@import_or_fail(["warp", "cupy", "cuml"])
@pytest.mark.parametrize("model_type", ["volume"])
@pytest.mark.parametrize("scaling_type", [None, "min_max_scaling", "mean_std_scaling"])
def test_domino_datapipe_scaling(zarr_dataset, model_type, scaling_type, pytestconfig):
    """Test field scaling functionality."""
    if scaling_type == "min_max_scaling":
        volume_factors = [10.0, -10.0]  # [max, min]
    elif scaling_type == "mean_std_scaling":
        volume_factors = [0.0, 1.0]  # [mean, std]
    else:
        volume_factors = None

    dataset = create_basic_dataset(
        zarr_dataset,
        model_type,
        gpu_preprocessing=False,
        scaling_type=scaling_type,
        volume_factors=volume_factors,
    )

    sample = dataset[0]
    validate_sample_structure(sample, model_type, gpu_output=True)


# Caching tests
@import_or_fail(["warp", "cupy", "cuml"])
@pytest.mark.parametrize("model_type", ["volume"])
def test_domino_datapipe_caching_config(zarr_dataset, model_type, pytestconfig):
    """Test DoMINODataPipe with caching=True configuration."""
    dataset = create_basic_dataset(
        zarr_dataset,
        model_type,
        gpu_preprocessing=False,
        caching=True,
        sampling=False,  # Required for caching
        compute_scaling_factors=False,  # Required for caching
        resample_surfaces=False,  # Required for caching
    )

    sample = dataset[0]
    validate_sample_structure(sample, model_type, gpu_output=True)


@import_or_fail(["warp", "cupy", "cuml"])
def test_cached_domino_dataset(zarr_dataset, tmp_path, pytestconfig):
    """Test CachedDoMINODataset functionality."""
    from physicsnemo.datapipes.cae.domino_datapipe import CachedDoMINODataset

    # Create some mock cached data files
    for i in range(3):
        cached_data = {
            "geometry_coordinates": np.random.randn(1000, 3),
            "volume_mesh_centers": np.random.randn(5000, 3),
            "volume_fields": np.random.randn(5000, 2),
            "surface_mesh_centers": np.random.randn(2000, 3),
            "surface_fields": np.random.randn(2000, 2),
            "surface_normals": np.random.randn(2000, 3),
            "surface_areas": np.random.rand(2000),
            "neighbor_indices": np.random.randint(0, 2000, (2000, 5)),
        }
        np.save(tmp_path / f"cached_{i}.npz", cached_data)

    dataset = CachedDoMINODataset(
        data_path=tmp_path,
        phase="test",
        sampling=True,
        volume_points_sample=1234,
        surface_points_sample=567,
        geom_points_sample=890,
        model_type="combined",
    )

    assert len(dataset) > 0

    sample = dataset[0]

    # Check that sampling worked
    assert sample["volume_mesh_centers"].shape[0] <= 1234
    assert sample["surface_mesh_centers"].shape[0] <= 567
    assert sample["geometry_coordinates"].shape[0] <= 890


# Configuration validation tests
@import_or_fail(["warp", "cupy", "cuml"])
def test_domino_datapipe_invalid_caching_config(zarr_dataset, pytestconfig):
    """Test that invalid caching configurations raise appropriate errors."""

    # Test: caching=True with sampling=True should fail
    with pytest.raises(ValueError, match="Sampling should be False for caching"):
        create_basic_dataset(zarr_dataset, "volume", caching=True, sampling=True)

    # Test: caching=True with compute_scaling_factors=True should fail
    with pytest.raises(
        ValueError, match="Compute scaling factors should be False for caching"
    ):
        create_basic_dataset(
            zarr_dataset, "volume", caching=True, compute_scaling_factors=True
        )

    # Test: caching=True with resample_surfaces=True should fail
    with pytest.raises(
        ValueError, match="Resample surface should be False for caching"
    ):
        create_basic_dataset(
            zarr_dataset, "volume", caching=True, resample_surfaces=True
        )


@import_or_fail(["warp", "cupy", "cuml"])
def test_domino_datapipe_invalid_phase(pytestconfig):
    """Test that invalid phase values raise appropriate errors."""
    from physicsnemo.datapipes.cae.domino_datapipe import DoMINODataConfig

    with pytest.raises(ValueError, match="phase should be one of"):
        DoMINODataConfig(data_path=tempfile.mkdtemp(), phase="invalid_phase")


@import_or_fail(["warp", "cupy", "cuml"])
def test_domino_datapipe_invalid_scaling_type(pytestconfig):
    """Test that invalid scaling_type values raise appropriate errors."""
    from physicsnemo.datapipes.cae.domino_datapipe import DoMINODataConfig

    with pytest.raises(ValueError, match="scaling_type should be one of"):
        DoMINODataConfig(
            data_path=tempfile.mkdtemp(), phase="train", scaling_type="invalid_scaling"
        )


@import_or_fail(["warp", "cupy", "cuml"])
def test_domino_datapipe_file_format_support(zarr_dataset, pytestconfig):
    """Test support for different file formats (.zarr, .npz, .npy)."""
    # This test assumes the data directory has files in these formats
    # If not available, we can mock the file reading
    dataset = create_basic_dataset(zarr_dataset, "volume", gpu_preprocessing=False)

    # Just verify we can load at least one sample
    assert len(dataset) > 0
    sample = dataset[0]
    validate_sample_structure(sample, "volume", gpu_output=True)


# Surface-specific tests (when GPU preprocessing issues are resolved)
@import_or_fail(["warp", "cupy", "cuml"])
@pytest.mark.parametrize("surface_sampling_algorithm", ["area_weighted", "random"])
def test_domino_datapipe_surface_sampling(
    zarr_dataset, surface_sampling_algorithm, pytestconfig
):
    """Test surface sampling algorithms."""
    dataset = create_basic_dataset(
        zarr_dataset,
        "surface",
        gpu_preprocessing=False,  # Avoid known GPU issues
        sampling=True,
        surface_sampling_algorithm=surface_sampling_algorithm,
    )

    sample = dataset[0]
    validate_sample_structure(sample, "surface", gpu_output=True)


if __name__ == "__main__":
    out_dir = synthetic_domino_data(
        out_format="zarr",
    )
    print(out_dir)
