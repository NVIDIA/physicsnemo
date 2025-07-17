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

from dataclasses import dataclass
from typing import List

import pytest
import torch
from pytest_utils import import_or_fail

Tensor = torch.Tensor


@dataclass
class ConcreteBoundingBox:
    """
    Really simple bounding box to mimic a structured config;  Don't use elsewhere.
    """

    min: List[float]
    max: List[float]


@pytest.fixture
def data_dir(nfs_data_dir):
    return nfs_data_dir.joinpath("datasets/domino/")


@import_or_fail(["warp", "cupy", "cuml"])
@pytest.mark.parametrize("gpu_preprocessing", [True, False])
@pytest.mark.parametrize("gpu_output", [True, False])
@pytest.mark.parametrize("model_type", ["surface", "volume", "combined"])
def test_domino_datapipe(
    data_dir, gpu_preprocessing, gpu_output, model_type, tmp_path, pytestconfig
):
    if gpu_preprocessing and model_type in ["surface", "combined"]:
        pytest.xfail(
            "Known cuda/cuml issue with GPU preprocessing for surface data (cuml nearest neighbors)"
        )

    from physicsnemo.datapipes.cae.domino_datapipe import DoMINODataPipe

    if model_type == "surface":
        input_path = data_dir / "surface/"
    elif model_type == "volume":
        input_path = data_dir / "volume/"
    elif model_type == "combined":
        input_path = data_dir / "combined/"

    print(f"input_path: {input_path}")

    bounding_box = ConcreteBoundingBox(min=[-3.5, -2.25, -0.32], max=[8.5, 2.25, 3.00])
    bounding_box_surface = ConcreteBoundingBox(
        min=[-1.1, -1.2, -0.32], max=[4.5, 1.2, 1.2]
    )

    dataset = DoMINODataPipe(
        input_path=input_path,
        model_type=model_type,
        gpu_preprocessing=gpu_preprocessing,
        gpu_output=gpu_output,
        phase="test",
        grid_resolution=[64, 64, 64],
        normalize_coordinates=True,
        sampling=True,
        sample_in_bbox=True,
        volume_points_sample=1234,
        surface_points_sample=1234,
        geom_points_sample=2345,
        positional_encoding=False,
        volume_factors=None,
        surface_factors=None,
        scaling_type=None,
        bounding_box_dims=bounding_box,
        bounding_box_dims_surf=bounding_box_surface,
    )

    assert len(dataset) > 0

    # Check if datapipe is iterable. This will iterate over the dataset
    # and cache graphs, if requested.

    print(f"len(dataset): {len(dataset)}")
    sample = dataset[0]

    # Make sure that all keys for the model are present, they are torch tensors, and on the correct device.

    # Always check these keys

    print(f"sample.keys(): {sample.keys()}")

    keys_to_read_if_available = ["global_params_values", "global_params_reference"]

    volume_keys = ["volume_mesh_centers", "volume_fields"]
    surface_keys = [
        "surface_mesh_centers",
        "surface_normals",
        "surface_areas",
        "surface_fields",
    ]

    if model_type == "volume" or model_type == "combined":
        for key in volume_keys:
            assert key in sample
            assert isinstance(sample[key], torch.Tensor)
            assert sample[key].device.type == "cuda" if gpu_output else "cpu"

    if model_type == "surface" or model_type == "combined":
        for key in surface_keys:
            assert key in sample
            assert isinstance(sample[key], torch.Tensor)
            assert sample[key].device.type == "cuda" if gpu_output else "cpu"

    for key in keys_to_read_if_available:
        if key in sample:
            assert isinstance(sample[key], torch.Tensor)
            assert sample[key].device.type == "cuda" if gpu_output else "cpu"
