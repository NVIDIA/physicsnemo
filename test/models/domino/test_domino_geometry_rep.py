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

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from .utils import validate_output_shape_and_values


@pytest.mark.parametrize("act", ["relu", "gelu"])
@pytest.mark.parametrize("fourier_features", [True, False])
def test_geo_conv_out(device, act, fourier_features):
    """Test GeoConvOut layer"""
    from physicsnemo.models.domino.geometry_rep import GeoConvOut

    @dataclass
    class TestParams:
        base_neurons: int = 32
        base_neurons_in: int = 8
        fourier_features: bool = False
        neighbors_in_radius: int = 8
        num_modes: int = 5
        activation: str = act

    params = TestParams()
    params.fourier_features = fourier_features

    input_features = 3

    grid_resolution = [32, 32, 32]

    layer = GeoConvOut(
        input_features=input_features,
        neighbors_in_radius=params.neighbors_in_radius,
        model_parameters=params,
        grid_resolution=grid_resolution,
    ).to(device)

    x = torch.randn(1, np.prod(grid_resolution), params.neighbors_in_radius, 3).to(
        device
    )
    grid = torch.randn(1, *grid_resolution, 3).to(device)

    output = layer(x, grid)

    validate_output_shape_and_values(
        output, (1, params.base_neurons_in, *grid_resolution)
    )


@pytest.mark.parametrize("act", ["relu", "gelu"])
def test_geo_processor(device, act):
    """Test GeoProcessor CNN"""
    from physicsnemo.models.domino.geometry_rep import GeoProcessor

    @dataclass
    class TestParams:
        base_filters: int = 8
        activation: str = act

    params = TestParams()

    processor = GeoProcessor(
        input_filters=4, output_filters=2, model_parameters=params
    ).to(device)

    x = torch.randn(2, 4, 16, 16, 16).to(device)
    output = processor(x)

    validate_output_shape_and_values(output, (2, 2, 16, 16, 16))


@pytest.mark.parametrize("geometry_encoding_type", ["both", "stl", "sdf"])
@pytest.mark.parametrize("processor_type", ["unet", "conv"])
def test_geometry_rep(
    device, geometry_encoding_type, processor_type, base_model_params
):
    """Test GeometryRep module with different configurations"""
    from physicsnemo.models.domino.geometry_rep import GeometryRep

    # Modify params for this test
    params = base_model_params()
    params.geometry_encoding_type = geometry_encoding_type
    params.geometry_rep.geo_processor.processor_type = processor_type
    params.geometry_rep.geo_processor.self_attention = False
    params.geometry_rep.geo_processor.cross_attention = False
    params.interp_res = (16, 16, 16)  # Smaller for faster testing

    radii = [0.1, 0.2]
    neighbors_in_radius = [8, 16]

    geo_rep = GeometryRep(
        input_features=3,
        radii=radii,
        neighbors_in_radius=neighbors_in_radius,
        hops=1,
        model_parameters=params,
    ).to(device)

    # Test inputs
    x = torch.randn(1, 20, 3).to(device)
    p_grid = torch.randn(1, 16, 16, 16, 3).to(device)
    sdf = torch.randn(1, 16, 16, 16).to(device)

    output = geo_rep(x, p_grid, sdf)

    # Determine expected output channels
    if geometry_encoding_type == "both":
        expected_channels = len(radii) + 1  # STL channels + SDF channel
    elif geometry_encoding_type == "stl":
        expected_channels = len(radii)
    else:  # sdf
        expected_channels = 1

    validate_output_shape_and_values(output, (1, expected_channels, 16, 16, 16))
