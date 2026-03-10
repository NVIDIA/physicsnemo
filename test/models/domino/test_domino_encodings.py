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

import math

import pytest
import torch

from .utils import validate_output_shape_and_values


@pytest.mark.parametrize("fourier_features", [True, False])
@pytest.mark.parametrize("num_modes", [3, 5, 10])
def test_fourier_mlp(device, fourier_features, num_modes):
    """Test FourierMLP with various configurations"""
    from physicsnemo.nn import FourierMLP

    model = FourierMLP(
        input_features=3,
        base_layer=64,
        fourier_features=fourier_features,
        num_modes=num_modes,
        activation="relu",
    ).to(device)

    x = torch.randn(2, 100, 3).to(device)
    output = model(x)

    validate_output_shape_and_values(output, (2, 100, 64))


def test_fourier_encode_vectorized(device):
    """Test fourier encoding function"""
    from physicsnemo.nn import fourier_encode

    coords = torch.randn(4, 20, 3).to(device)
    freqs = torch.exp(torch.linspace(0, math.pi, 5)).to(device)

    output = fourier_encode(coords, freqs)

    # Output should be [batch, points, D * 2 * F] = [4, 20, 3 * 2 * 5] = [4, 20, 30]
    validate_output_shape_and_values(output, (4, 20, 30))


def test_local_geometry_encoding(device):
    """Test LocalGeometryEncoding"""
    from physicsnemo.models.domino.encodings import LocalGeometryEncoding
    from physicsnemo.nn import get_activation

    BATCH_SIZE = 1

    N_ENCODING_CHANNELS = 3
    N_NEIGHBORS = 32
    N_MESH_POINTS = 50
    GRID_RESOLUTION = (32, 32, 32)

    model = LocalGeometryEncoding(
        radius=0.1,
        neighbors_in_radius=N_NEIGHBORS,
        total_neighbors_in_radius=N_ENCODING_CHANNELS * N_NEIGHBORS,
        base_layer=128,
        activation=get_activation("relu"),
        grid_resolution=GRID_RESOLUTION,
    ).to(device)

    encoding_g = torch.randn(BATCH_SIZE, N_ENCODING_CHANNELS, *GRID_RESOLUTION).to(
        device
    )
    volume_mesh_centers = torch.randn(BATCH_SIZE, N_MESH_POINTS, 3).to(device)
    p_grid = torch.randn(BATCH_SIZE, *GRID_RESOLUTION, 3).to(device)

    output = model(encoding_g, volume_mesh_centers, p_grid)

    validate_output_shape_and_values(output, (BATCH_SIZE, N_MESH_POINTS, 32))


@pytest.mark.parametrize("geo_encoding_type", ["both", "stl", "sdf"])
def test_multi_geometry_encoding(device, geo_encoding_type):
    """Test MultiGeometryEncoding with different encoding types"""
    from physicsnemo.models.domino.encodings import MultiGeometryEncoding
    from physicsnemo.models.domino.model import get_activation

    BATCH_SIZE = 1
    N_MESH_POINTS = 50
    GRID_RESOLUTION = (32, 32, 32)

    radii = [0.05, 0.1]
    neighbors_in_radius = [16, 32]

    model = MultiGeometryEncoding(
        radii=radii,
        neighbors_in_radius=neighbors_in_radius,
        geo_encoding_type=geo_encoding_type,
        base_layer=64,
        n_upstream_radii=2,
        activation=get_activation("relu"),
        grid_resolution=GRID_RESOLUTION,
    ).to(device)

    if geo_encoding_type == "both":
        num_channels = len(radii) + 1
    elif geo_encoding_type == "stl":
        num_channels = len(radii)
    else:  # sdf
        num_channels = 1

    encoding_g = torch.randn(BATCH_SIZE, num_channels, *GRID_RESOLUTION).to(device)
    volume_mesh_centers = torch.randn(BATCH_SIZE, N_MESH_POINTS, 3).to(device)
    p_grid = torch.randn(BATCH_SIZE, *GRID_RESOLUTION, 3).to(device)

    output = model(encoding_g, volume_mesh_centers, p_grid)

    expected_output_dim = sum(neighbors_in_radius)

    validate_output_shape_and_values(
        output, (BATCH_SIZE, N_MESH_POINTS, expected_output_dim)
    )
