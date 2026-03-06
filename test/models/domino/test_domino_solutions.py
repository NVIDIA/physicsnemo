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

import pytest
import torch
import torch.nn as nn

from .utils import validate_output_shape_and_values


@pytest.mark.parametrize("num_variables", [1, 3, 5])
@pytest.mark.parametrize("num_sample_points", [1, 3, 7])
@pytest.mark.parametrize("encode_parameters", [True, False])
def test_solution_calculator_volume(
    device, num_variables, num_sample_points, encode_parameters
):
    """Test SolutionCalculatorVolume with various configurations"""
    from physicsnemo.models.domino.mlps import AggregationModel
    from physicsnemo.models.domino.solutions import SolutionCalculatorVolume
    from physicsnemo.nn import FourierMLP, get_activation

    activation = get_activation("relu")

    # Create parameter model if needed
    parameter_model = (
        FourierMLP(
            input_features=2,
            base_layer=32,
            fourier_features=True,
            num_modes=3,
            activation=activation,
        ).to(device)
        if encode_parameters
        else None
    )

    # Create aggregation models
    aggregation_model = nn.ModuleList(
        [
            AggregationModel(
                input_features=64 + 32 + 32 + (32 if encode_parameters else 0),
                output_features=1,
                base_layer=64,
                activation=activation,
            ).to(device)
            for _ in range(num_variables)
        ]
    )

    # Create basis functions
    nn_basis = nn.ModuleList(
        [
            FourierMLP(
                input_features=3,
                base_layer=32,
                fourier_features=False,
                num_modes=5,
                activation=activation,
            ).to(device)
            for _ in range(num_variables)
        ]
    )

    model = SolutionCalculatorVolume(
        num_variables=num_variables,
        num_sample_points=num_sample_points,
        noise_intensity=50.0,
        encode_parameters=encode_parameters,
        return_volume_neighbors=False,
        parameter_model=parameter_model,
        aggregation_model=aggregation_model,
        nn_basis=nn_basis,
    ).to(device)

    # Test data
    volume_mesh_centers = torch.randn(2, 30, 3).to(device)
    encoding_g = torch.randn(2, 30, 32).to(device)
    encoding_node = torch.randn(2, 30, 64).to(device)
    global_params_values = torch.randn(2, 2, 1).to(device)
    global_params_reference = torch.randn(2, 2, 1).to(device)

    output = model(
        volume_mesh_centers,
        encoding_g,
        encoding_node,
        global_params_values,
        global_params_reference,
    )

    validate_output_shape_and_values(output, (2, 30, num_variables))


@pytest.mark.parametrize("num_variables", [1, 3, 5])
@pytest.mark.parametrize("use_surface_normals", [True, False])
@pytest.mark.parametrize("use_surface_area", [True, False])
def test_solution_calculator_surface(
    device, num_variables, use_surface_normals, use_surface_area
):
    """Test SolutionCalculatorSurface with various configurations"""
    from physicsnemo.models.domino.mlps import AggregationModel
    from physicsnemo.models.domino.solutions import SolutionCalculatorSurface
    from physicsnemo.nn import FourierMLP, get_activation

    activation = get_activation("relu")

    # Determine input features based on surface configuration
    input_features = 3
    if use_surface_normals:
        input_features += 3
    if use_surface_area:
        input_features += 1

    # Create aggregation models
    aggregation_model = nn.ModuleList(
        [
            AggregationModel(
                input_features=64 + 32 + 32,
                output_features=1,
                base_layer=64,
                activation=activation,
            ).to(device)
            for _ in range(num_variables)
        ]
    )

    # Create basis functions
    nn_basis = nn.ModuleList(
        [
            FourierMLP(
                input_features=input_features,
                base_layer=32,
                fourier_features=False,
                num_modes=5,
                activation=activation,
            ).to(device)
            for _ in range(num_variables)
        ]
    )

    model = SolutionCalculatorSurface(
        num_variables=num_variables,
        num_sample_points=3,
        encode_parameters=False,
        use_surface_normals=use_surface_normals,
        use_surface_area=use_surface_area,
        parameter_model=None,
        aggregation_model=aggregation_model,
        nn_basis=nn_basis,
    ).to(device)

    # Test data
    surface_mesh_centers = torch.randn(2, 30, 3).to(device)
    encoding_g = torch.randn(2, 30, 32).to(device)
    encoding_node = torch.randn(2, 30, 64).to(device)
    surface_mesh_neighbors = torch.randn(2, 30, 2, 3).to(device)
    surface_normals = torch.randn(2, 30, 3).to(device)
    surface_neighbors_normals = torch.randn(2, 30, 2, 3).to(device)
    surface_areas = torch.rand(2, 30, 1).to(device) + 1e-6
    surface_neighbors_areas = torch.rand(2, 30, 2, 1).to(device) + 1e-6
    global_params_values = torch.randn(2, 2, 1).to(device)
    global_params_reference = torch.randn(2, 2, 1).to(device)

    output = model(
        surface_mesh_centers,
        encoding_g,
        encoding_node,
        surface_mesh_neighbors,
        surface_normals,
        surface_neighbors_normals,
        surface_areas,
        surface_neighbors_areas,
        global_params_values,
        global_params_reference,
    )

    validate_output_shape_and_values(output, (2, 30, num_variables))


@pytest.mark.parametrize("r", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("num_points", [10, 50, 100])
def test_sample_sphere(device, r, num_points):
    """Test sphere sampling function"""
    from physicsnemo.models.domino.solutions import sample_sphere

    center = torch.randn(2, 30, 3).to(device)
    output = sample_sphere(center, r, num_points)

    validate_output_shape_and_values(output, (2, 30, num_points, 3))

    # Check that points are within the sphere radius
    distances = torch.norm(output - center.unsqueeze(2), dim=-1)
    assert (distances <= r + 1e-6).all(), "Some sampled points are outside the sphere"


def test_sample_sphere_shell(device):
    """Test spherical shell sampling function"""
    from physicsnemo.models.domino.solutions import sample_sphere_shell

    center = torch.randn(2, 30, 3).to(device)
    r_inner, r_outer = 0.5, 1.5
    num_points = 50

    output = sample_sphere_shell(center, r_inner, r_outer, num_points)

    validate_output_shape_and_values(output, (2, 30, num_points, 3))

    # Check that points are within the shell
    distances = torch.norm(output - center.unsqueeze(2), dim=-1)
    assert (distances >= r_inner - 1e-6).all(), (
        "Some sampled points are inside inner radius"
    )
    assert (distances <= r_outer + 1e-6).all(), (
        "Some sampled points are outside outer radius"
    )
