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

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Sequence

import numpy as np
import pytest
import torch
import zarr

from physicsnemo.core.version_check import check_version_spec
from physicsnemo.datapipes.cae.cae_dataset import CAEDataset
from physicsnemo.datapipes.cae.domino_datapipe import (
    CachedDoMINODataset,
    DoMINODataConfig,
    DoMINODataPipe,
)
from test.conftest import requires_module

if not check_version_spec("scipy", hard_fail=False):
    pytest.skip(
        "These tests require torch >= 2.6.0",
        allow_module_level=True,
    )
else:
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


def create_basic_dataset(
    data_dir,
    model_type,
    gpu_preprocessing: bool = False,
    gpu_output: bool = False,
    normalize_coordinates: bool = False,
    sample_in_bbox: bool = False,
    sampling: bool = False,
    volume_points_sample: int = 1234,
    surface_points_sample: int = 1234,
    surface_sampling_algorithm: str = "random",
    caching: bool = False,
    scaling_type: Optional[Literal["min_max_scaling", "mean_std_scaling"]] = None,
    volume_factors: Optional[Sequence] = None,
    surface_factors: Optional[Sequence] = None,
):
    """Helper function to create a basic DoMINODataPipe with default settings."""

    # assert model_type in ["volume", "surface", "combined"]

    input_path = data_dir

    bounding_box = bounding_boxes()

    keys_to_read = [
        "stl_coordinates",
        "stl_faces",
        "stl_centers",
        "stl_areas",
    ]

    if model_type == "volume" or model_type == "combined":
        keys_to_read += [
            "volume_mesh_centers",
            "volume_fields",
        ]

    if model_type == "surface" or model_type == "combined":
        keys_to_read += [
            "surface_mesh_centers",
            "surface_areas",
            "surface_normals",
            "surface_fields",
        ]

    keys_to_read_if_available = {
        "global_params_values": torch.tensor([1.225, 10.0]),
        "global_params_reference": torch.tensor([1.225, 10.0]),
    }

    dataset = CAEDataset(
        data_dir=input_path,
        keys_to_read=keys_to_read,
        keys_to_read_if_available=keys_to_read_if_available,
        output_device=torch.device("cuda")
        if gpu_preprocessing
        else torch.device("cpu"),
        preload_depth=0,
        pin_memory=False,
        device_mesh=None,
        placements=None,
    )

    default_kwargs = {
        "phase": "test",
        "grid_resolution": [64, 64, 64],
        "volume_points_sample": volume_points_sample,
        "surface_points_sample": surface_points_sample,
        "geom_points_sample": 500,
        "num_surface_neighbors": 5,
        "bounding_box_dims": bounding_box["volume"],
        "bounding_box_dims_surf": bounding_box["surface"],
        "normalize_coordinates": normalize_coordinates,
        "sampling": sampling,
        "sample_in_bbox": sample_in_bbox,
        "scaling_type": scaling_type,
        "volume_factors": volume_factors,
        "surface_factors": surface_factors,
        "caching": caching,
        "gpu_preprocessing": gpu_preprocessing,
        "gpu_output": gpu_output,
        "surface_sampling_algorithm": surface_sampling_algorithm,
    }

    pipe = DoMINODataPipe(
        input_path=input_path, model_type=model_type, **default_kwargs
    )

    pipe.set_dataset(dataset)
    return pipe


def validate_sample_structure(sample, model_type, gpu_output):
    """Helper function to validate the structure of a dataset sample."""
    assert isinstance(sample, dict)

    # Common keys that should always be present
    expected_keys = ["geometry_coordinates"]

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
@requires_module(["warp", "cupy", "cuml"])
@pytest.mark.parametrize("data_dir", ["zarr_dataset", "npz_dataset", "npy_dataset"])
@pytest.mark.parametrize("gpu_preprocessing", [True, False])
@pytest.mark.parametrize("gpu_output", [True, False])
@pytest.mark.parametrize("model_type", ["surface", "volume", "combined"])
def test_domino_datapipe_core(
    data_dir, gpu_preprocessing, gpu_output, model_type, pytestconfig, request
):
    """Core test for basic functionality with different device and model configurations."""

    data_dir = request.getfixturevalue(data_dir)
    dataset = create_basic_dataset(
        data_dir,
        model_type,
        gpu_preprocessing=gpu_preprocessing,
        gpu_output=gpu_output,
        normalize_coordinates=False,
        sample_in_bbox=False,
        sampling=False,
    )

    assert len(dataset) > 0
    sample = dataset[0]
    validate_sample_structure(sample, model_type, gpu_output)


# Feature-specific tests
@requires_module(["warp", "cupy", "cuml"])
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
        gpu_output=True,
        normalize_coordinates=normalize_coordinates,
        sample_in_bbox=sample_in_bbox,
        sampling=False,
    )

    sample = dataset[0]
    validate_sample_structure(sample, model_type, gpu_output=True)

    # Check all the volume coordinates:
    for volume_key in ["volume_mesh_centers"]:
        coords = sample[volume_key]
        check_tensor_normalization(
            coords, normalize_coordinates, sample_in_bbox, is_surface=False
        )

    # Check all the surface coordinates:
    for surface_key in ["surface_mesh_centers", "surface_mesh_neighbors"]:
        coords = sample[surface_key]
        if surface_key == "surface_mesh_neighbors":
            coords = coords.reshape((1, -1, 3))
        check_tensor_normalization(
            coords, normalize_coordinates, sample_in_bbox, is_surface=True
        )


def check_tensor_normalization(
    tensor, normalize_coordinates, sample_in_bbox, is_surface
):
    """Check if a tensor is normalized properly."""

    # Batch size is 1 here, but in principle this could be a loop:
    t_min = torch.min(tensor[0], dim=0).values
    t_max = torch.max(tensor[0], dim=0).values

    # If normalization is enabled, coordinates should be in [-2, 2] range
    if normalize_coordinates:
        if sample_in_bbox:
            # In this case, the values are rescaled, but only the ones
            # that were already inside the box should be present.

            # That means that all values should be between -1 and 1
            assert t_min[0] >= -1
            assert t_min[1] >= -1
            assert t_min[2] >= -1
            assert t_max[0] <= 1
            assert t_max[1] <= 1
            assert t_max[2] <= 1

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

            if is_surface:
                x_rescale = 1 / (SURF_BBOX_XMAX - SURF_BBOX_XMIN)
                y_rescale = 1 / (SURF_BBOX_YMAX - SURF_BBOX_YMIN)
                z_rescale = 1 / (SURF_BBOX_ZMAX - SURF_BBOX_ZMIN)
                target_min_x = 2 * (DATA_XMIN - SURF_BBOX_XMIN) * x_rescale - 1
                target_min_y = 2 * (DATA_YMIN - SURF_BBOX_YMIN) * y_rescale - 1
                target_min_z = 2 * (DATA_ZMIN - SURF_BBOX_ZMIN) * z_rescale - 1
                target_max_x = 2 * (DATA_XMAX - SURF_BBOX_XMIN) * x_rescale - 1
                target_max_y = 2 * (DATA_YMAX - SURF_BBOX_YMIN) * y_rescale - 1
                target_max_z = 2 * (DATA_ZMAX - SURF_BBOX_ZMIN) * z_rescale - 1
            else:
                x_rescale = 1 / (VOL_BBOX_XMAX - VOL_BBOX_XMIN)
                y_rescale = 1 / (VOL_BBOX_YMAX - VOL_BBOX_YMIN)
                z_rescale = 1 / (VOL_BBOX_ZMAX - VOL_BBOX_ZMIN)
                target_min_x = 2 * (DATA_XMIN - VOL_BBOX_XMIN) * x_rescale - 1
                target_min_y = 2 * (DATA_YMIN - VOL_BBOX_YMIN) * y_rescale - 1
                target_min_z = 2 * (DATA_ZMIN - VOL_BBOX_ZMIN) * z_rescale - 1
                target_max_x = 2 * (DATA_XMAX - VOL_BBOX_XMIN) * x_rescale - 1
                target_max_y = 2 * (DATA_YMAX - VOL_BBOX_YMIN) * y_rescale - 1
                target_max_z = 2 * (DATA_ZMAX - VOL_BBOX_ZMIN) * z_rescale - 1

            assert t_min[0] >= target_min_x
            assert t_min[1] >= target_min_y
            assert t_min[2] >= target_min_z
            assert t_max[0] <= target_max_x
            assert t_max[1] <= target_max_y
            assert t_max[2] <= target_max_z

    else:
        if sample_in_bbox:
            # We've sampled in the bbox but NOT normalized.
            # So, the values should exclusively be in the BBOX ranges:

            if is_surface:
                assert t_min[0] >= SURF_BBOX_XMIN
                assert t_min[1] >= SURF_BBOX_YMIN
                assert t_min[2] >= SURF_BBOX_ZMIN
                assert t_max[0] <= SURF_BBOX_XMAX
                assert t_max[1] <= SURF_BBOX_YMAX
                assert t_max[2] <= SURF_BBOX_ZMAX
            else:
                assert t_min[0] >= VOL_BBOX_XMIN
                assert t_min[1] >= VOL_BBOX_YMIN
                assert t_min[2] >= VOL_BBOX_ZMIN
                assert t_max[0] <= VOL_BBOX_XMAX
                assert t_max[1] <= VOL_BBOX_YMAX
                assert t_max[2] <= VOL_BBOX_ZMAX

        else:
            # Not sampling, and also
            # Not normalizing, values should be in data range only:
            assert t_min[0] >= DATA_XMIN and t_max[0] <= DATA_XMAX
            assert t_min[1] >= DATA_YMIN and t_max[1] <= DATA_YMAX

            if is_surface:
                # Surface points always should be > 0
                assert t_min[2] >= 0 and t_max[2] <= DATA_ZMAX
            else:
                assert t_min[2] >= DATA_ZMIN and t_max[2] <= DATA_ZMAX

    return True


@pytest.mark.parametrize("model_type", ["surface"])
@pytest.mark.parametrize("normalize_coordinates", [True, False])
@pytest.mark.parametrize("sample_in_bbox", [True, False])
def test_domino_datapipe_surface_normalization(
    zarr_dataset, pytestconfig, model_type, normalize_coordinates, sample_in_bbox
):
    """Test normalization functionality.

    This test is meant to make sure all the peripheral outputs are
    normalized properly. FOcus on surface here.

    We could do them all in one test but it gets unweildy, and if there
    are failures it helps nail down exactly where.
    """
    cuda = torch.cuda.is_available()

    dataset = create_basic_dataset(
        zarr_dataset,
        model_type,
        gpu_preprocessing=cuda,
        gpu_output=cuda,
        normalize_coordinates=normalize_coordinates,
        sampling=True,
        sample_in_bbox=sample_in_bbox,
    )

    # Here's a list of values to check, and the behavior we expect:

    # surf_grid - normalized by s_min, s_max
    sample = dataset[0]
    surf_grid = sample["surf_grid"]

    # If normalizing, surf_grid should be between -1 and 1.
    # Otherwise, should be between s_min and s_max
    if not normalize_coordinates:
        target_min = torch.tensor([SURF_BBOX_XMIN, SURF_BBOX_YMIN, SURF_BBOX_ZMIN])
        target_max = torch.tensor([SURF_BBOX_XMAX, SURF_BBOX_YMAX, SURF_BBOX_ZMAX])
    else:
        target_min = torch.tensor([-1.0, -1.0, -1.0])
        target_max = torch.tensor([1.0, 1.0, 1.0])

    target_min = target_min.to(surf_grid.device)
    target_max = target_max.to(surf_grid.device)

    # Flatten all the grid coords:
    surf_grid = surf_grid.reshape((-1, 3))

    assert torch.all(surf_grid >= target_min)
    assert torch.all(surf_grid <= target_max)

    # sdf_surf_grid - should have max values less than || s_max - s_min||

    max_norm_allowed = torch.norm(target_max - target_min)

    sdf_surf_grid = sample["sdf_surf_grid"]
    assert torch.all(sdf_surf_grid <= max_norm_allowed)
    # (Negative values are ok but we don't really check that.)

    # surface_min_max should only be in the dict if normaliztion is on:
    if normalize_coordinates:
        assert "surface_min_max" in sample
        s_mm = sample["surface_min_max"]
        assert s_mm.shape == (1, 2, 3)

        assert torch.allclose(
            s_mm[0, 0],
            torch.tensor([SURF_BBOX_XMIN, SURF_BBOX_YMIN, SURF_BBOX_ZMIN]).to(
                s_mm.device
            ),
        )
        assert torch.allclose(
            s_mm[0, 1],
            torch.tensor([SURF_BBOX_XMAX, SURF_BBOX_YMAX, SURF_BBOX_ZMAX]).to(
                s_mm.device
            ),
        )

    else:
        assert "surface_min_max" not in sample

    # For the rest of the values, checks are straightforward:

    assert torch.all(sample["surface_areas"] > 0)
    assert torch.all(sample["surface_neighbors_areas"] > 0)

    # No checks implemented on the following, yet:
    # - pos_surface_center_of_mass


@pytest.mark.parametrize("model_type", ["volume"])
@pytest.mark.parametrize("normalize_coordinates", [True, False])
@pytest.mark.parametrize("sample_in_bbox", [True, False])
def test_domino_datapipe_volume_normalization(
    zarr_dataset, pytestconfig, model_type, normalize_coordinates, sample_in_bbox
):
    """Test normalization functionality.

    This test is meant to make sure all the peripheral outputs are
    normalized properly. FOcus on volume here.

    We could do them all in one test but it gets unweildy, and if there
    are failures it helps nail down exactly where.
    """
    cuda = torch.cuda.is_available()

    dataset = create_basic_dataset(
        zarr_dataset,
        model_type,
        gpu_preprocessing=cuda,
        gpu_output=cuda,
        normalize_coordinates=normalize_coordinates,
        sampling=True,
        sample_in_bbox=sample_in_bbox,
    )

    # Here's a list of values to check, and the behavior we expect:

    # grid - normalized by s_min, s_max
    sample = dataset[0]
    grid = sample["grid"]

    # If normalizing, surf_grid should be between -1 and 1.
    # Otherwise, should be between s_min and s_max
    if not normalize_coordinates:
        target_min = torch.tensor([VOL_BBOX_XMIN, VOL_BBOX_YMIN, VOL_BBOX_ZMIN])
        target_max = torch.tensor([VOL_BBOX_XMAX, VOL_BBOX_YMAX, VOL_BBOX_ZMAX])
    else:
        target_min = torch.tensor([-1.0, -1.0, -1.0])
        target_max = torch.tensor([1.0, 1.0, 1.0])

    target_min = target_min.to(grid.device)
    target_max = target_max.to(grid.device)

    # Flatten all the grid coords:
    grid = grid.reshape((-1, 3))

    assert torch.all(grid >= target_min)
    assert torch.all(grid <= target_max)

    # sdf_grid - should have max values less than || s_max - s_min||

    max_norm_allowed = torch.norm(target_max - target_min)

    sdf_grid = sample["sdf_grid"]
    assert torch.all(sdf_grid <= max_norm_allowed)
    # (Negative values are ok but we don't really check that.)

    # surface_min_max should only be in the dict if normaliztion is on:
    if normalize_coordinates:
        assert "volume_min_max" in sample
        s_mm = sample["volume_min_max"]
        assert s_mm.shape == (1, 2, 3)

        assert torch.allclose(
            s_mm[0, 0],
            torch.tensor([VOL_BBOX_XMIN, VOL_BBOX_YMIN, VOL_BBOX_ZMIN]).to(s_mm.device),
        )
        assert torch.allclose(
            s_mm[0, 1],
            torch.tensor([VOL_BBOX_XMAX, VOL_BBOX_YMAX, VOL_BBOX_ZMAX]).to(s_mm.device),
        )

    else:
        assert "volume_min_max" not in sample

    sdf_nodes = sample["sdf_nodes"]
    pos_volume_closest_norm = torch.norm(sample["pos_volume_closest"], dim=-1).reshape(
        sdf_nodes.shape
    )
    assert torch.allclose(pos_volume_closest_norm, sdf_nodes)
    # No checks implemented on the following, yet:
    # - pos_volume_center_of_mass

    # The center of mass should be inside the mesh.  So, the displacement
    # from the center of mass should be exclusively larger than the sdf:
    pos_volume_center_of_mass_norm = torch.norm(
        sample["pos_volume_center_of_mass"], dim=-1
    ).reshape(sdf_nodes.shape)
    assert torch.all(pos_volume_center_of_mass_norm > sdf_nodes)


@requires_module(["warp", "cupy", "cuml"])
@pytest.mark.parametrize("model_type", ["combined"])
@pytest.mark.parametrize("sampling", [True, False])
def test_domino_datapipe_sampling(zarr_dataset, model_type, sampling, pytestconfig):
    """Test point sampling functionality."""
    sample_points = 4321

    use_cuda = torch.cuda.is_available()

    dataset = create_basic_dataset(
        zarr_dataset,
        model_type,
        gpu_preprocessing=use_cuda,
        gpu_output=use_cuda,
        normalize_coordinates=False,
        sample_in_bbox=False,
        sampling=sampling,
        volume_points_sample=sample_points,
        surface_points_sample=sample_points,
    )

    sample = dataset[0]
    validate_sample_structure(sample, model_type, gpu_output=use_cuda)

    if model_type in ["volume", "combined"]:
        for key in ["volume_mesh_centers", "volume_fields"]:
            if sampling:
                assert sample[key].shape[1] == sample_points
            else:
                assert sample[key].shape[1] == sample["volume_mesh_centers"].shape[1]

    # Model-specific keys
    if model_type in ["surface", "combined"]:
        for key in [
            "surface_mesh_centers",
            "surface_normals",
            "surface_areas",
            "surface_fields",
        ]:
            if sampling:
                assert sample[key].shape[1] == sample_points
            else:
                assert sample[key].shape[1] == sample["surface_mesh_centers"].shape[1]
        for key in [
            "surface_mesh_neighbors",
            "surface_neighbors_normals",
            "surface_neighbors_areas",
        ]:
            if sampling:
                assert sample[key].shape[1] == sample_points
                assert sample[key].shape[2] == dataset.config.num_surface_neighbors - 1
            else:
                assert sample[key].shape[1] == sample["surface_mesh_neighbors"].shape[1]
                assert sample[key].shape[2] == dataset.config.num_surface_neighbors - 1


@requires_module(["warp", "cupy", "cuml"])
@pytest.mark.parametrize("model_type", ["volume", "surface", "combined"])
@pytest.mark.parametrize("scaling_type", [None, "min_max_scaling", "mean_std_scaling"])
def test_domino_datapipe_scaling(zarr_dataset, model_type, scaling_type, pytestconfig):
    """Test field scaling functionality."""
    use_cuda = torch.cuda.is_available()

    if model_type in ["volume", "combined"]:
        volume_factors = torch.tensor(
            [
                [10.0, -10.0, 10.0, 10.0, 10.0],
                [10.0, -10.0, 10.0, 10.0, 10.0],
            ]
        )
    else:
        volume_factors = None
    if model_type in ["surface", "combined"]:
        surface_factors = torch.tensor(
            [
                [10.0, -10.0, 10.0, 10.0],
                [10.0, -10.0, 10.0, 10.0],
            ]
        )
    else:
        surface_factors = None

    dataset = create_basic_dataset(
        zarr_dataset,
        model_type,
        gpu_preprocessing=use_cuda,
        gpu_output=use_cuda,
        scaling_type=scaling_type,
        volume_factors=volume_factors,
        surface_factors=surface_factors,
    )

    sample = dataset[0]
    validate_sample_structure(sample, model_type, gpu_output=use_cuda)


# Caching tests
@requires_module(["warp", "cupy", "cuml"])
@pytest.mark.parametrize("model_type", ["volume"])
def test_domino_datapipe_caching_config(zarr_dataset, model_type, pytestconfig):
    """Test DoMINODataPipe with caching=True configuration."""
    use_cuda = torch.cuda.is_available()
    dataset = create_basic_dataset(
        zarr_dataset,
        model_type,
        gpu_preprocessing=use_cuda,
        gpu_output=use_cuda,
        caching=True,
        sampling=False,  # Required for caching
    )

    sample = dataset[0]
    validate_sample_structure(sample, model_type, gpu_output=use_cuda)


@requires_module(["warp", "cupy", "cuml"])
def test_cached_domino_dataset(zarr_dataset, tmp_path, pytestconfig):
    """Test CachedDoMINODataset functionality."""

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
@requires_module(["warp", "cupy", "cuml"])
def test_domino_datapipe_invalid_caching_config(zarr_dataset, pytestconfig):
    """Test that invalid caching configurations raise appropriate errors."""

    use_cuda = torch.cuda.is_available()
    # Test: caching=True with sampling=True should fail
    with pytest.raises(ValueError, match="Sampling should be False for caching"):
        create_basic_dataset(
            zarr_dataset,
            "volume",
            caching=True,
            sampling=True,
            gpu_preprocessing=use_cuda,
            gpu_output=use_cuda,
        )


@requires_module(["warp", "cupy", "cuml"])
def test_domino_datapipe_invalid_phase(pytestconfig):
    """Test that invalid phase values raise appropriate errors."""

    with pytest.raises(ValueError, match="phase should be one of"):
        DoMINODataConfig(data_path=tempfile.mkdtemp(), phase="invalid_phase")


@requires_module(["warp", "cupy", "cuml"])
def test_domino_datapipe_invalid_scaling_type(pytestconfig):
    """Test that invalid scaling_type values raise appropriate errors."""

    with pytest.raises(ValueError, match="scaling_type should be one of"):
        DoMINODataConfig(
            data_path=tempfile.mkdtemp(), phase="train", scaling_type="invalid_scaling"
        )


@requires_module(["warp", "cupy", "cuml"])
def test_domino_datapipe_file_format_support(zarr_dataset, pytestconfig):
    """Test support for different file formats (.zarr, .npz, .npy)."""
    # This test assumes the data directory has files in these formats
    # If not available, we can mock the file reading
    use_cuda = torch.cuda.is_available()
    dataset = create_basic_dataset(
        zarr_dataset, "volume", gpu_preprocessing=use_cuda, gpu_output=use_cuda
    )

    # Just verify we can load at least one sample
    assert len(dataset) > 0
    sample = dataset[0]
    validate_sample_structure(sample, "volume", gpu_output=use_cuda)


# Surface-specific tests (when GPU preprocessing issues are resolved)
@requires_module(["warp", "cupy", "cuml"])
@pytest.mark.parametrize("surface_sampling_algorithm", ["area_weighted", "random"])
def test_domino_datapipe_surface_sampling(
    zarr_dataset, surface_sampling_algorithm, pytestconfig
):
    """Test surface sampling algorithms."""

    gpu = torch.cuda.is_available()

    dataset = create_basic_dataset(
        zarr_dataset,
        "surface",
        gpu_preprocessing=gpu,
        gpu_output=gpu,
        sampling=True,
        surface_sampling_algorithm=surface_sampling_algorithm,
    )

    sample = dataset[0]
    validate_sample_structure(sample, "surface", gpu_output=True)
