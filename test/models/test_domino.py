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
from pathlib import Path
from typing import Sequence

import pytest
import torch
from pytest_utils import import_or_fail

# from . import common
from .common.fwdaccuracy import save_output
from .common.utils import compare_output


def validate_domino(
    model,
    input_dict,
    file_name,
    device,
    rtol=1e-3,
    atol=1e-3,
):
    # Perform a foward pass of the model
    output = model.forward(input_dict)
    if file_name is None:
        file_name = model.meta.name + "_output.pth"
    file_name = (
        Path(__file__).parents[0].resolve() / Path("data") / Path(file_name.lower())
    )
    # If file does not exist, we will create it then error
    # Model should then reproduce it on next pytest run
    if not file_name.exists():
        save_output(output, file_name)
        raise IOError(
            f"Output check file {str(file_name)} wasn't found so one was created. Please re-run the test."
        )
    # Load tensor dictionary and check
    else:
        tensor_dict = torch.load(str(file_name))
        output_target = tuple([value.to(device) for value in tensor_dict.values()])

        return compare_output(output, output_target, rtol, atol)


@import_or_fail("warp")
@pytest.mark.parametrize("device", ["cuda:0"])
def test_domino_forward(device, pytestconfig):
    """Test domino forward pass"""

    from physicsnemo.models.domino.model import DoMINO

    torch.manual_seed(0)

    @dataclass
    class model_params:
        @dataclass
        class geometry_rep:
            @dataclass
            class geo_conv:
                base_neurons: int = 32
                base_neurons_out: int = 1
                radii: Sequence = (0.1, 0.5)
                hops: int = 1

            @dataclass
            class geo_processor:
                base_filters: int = 8

            @dataclass
            class geo_processor_sdf:
                base_filters: int = 8

            base_filters: int = 8
            geo_conv = geo_conv
            geo_processor = geo_processor
            geo_processor_sdf = geo_processor_sdf

        @dataclass
        class geometry_local:
            base_layer: int = 128
            neighbors_in_radius: Sequence = (64,)
            radii: Sequence = (0.05,)

        @dataclass
        class nn_basis_functions:
            base_layer: int = 128

        @dataclass
        class aggregation_model:
            base_layer: int = 128

        @dataclass
        class position_encoder:
            base_neurons: int = 128

        @dataclass
        class parameter_model:
            base_layer: int = 128
            scaling_params: Sequence = (30.0, 1.226)

        model_type: str = "combined"
        interp_res: Sequence = (128, 64, 48)
        use_sdf_in_basis_func: bool = True
        positional_encoding: bool = False
        surface_neighbors: bool = True
        num_surface_neighbors: int = 21
        use_surface_normals: bool = True
        use_only_normals: bool = True
        encode_parameters: bool = False
        geometry_rep = geometry_rep
        nn_basis_functions = nn_basis_functions
        aggregation_model = aggregation_model
        position_encoder = position_encoder
        geometry_local = geometry_local

    model = DoMINO(
        input_features=3,
        output_features_vol=4,
        output_features_surf=4,
        model_parameters=model_params,
    ).to(device)

    bsize = 1
    nx, ny, nz = model_params.interp_res
    num_neigh = model_params.num_surface_neighbors
    radii = model_params.geometry_rep.geo_conv.radii
    print(radii)

    pos_normals_closest_vol = torch.randn(bsize, 100, 3).to(device)
    pos_normals_com_vol = torch.randn(bsize, 100, 3).to(device)
    pos_normals_com_surface = torch.randn(bsize, 100, 3).to(device)
    geom_centers = torch.randn(bsize, 100, 3).to(device)
    grid = torch.randn(bsize, nx, ny, nz, 3).to(device)
    surf_grid = torch.randn(bsize, nx, ny, nz, 3).to(device)
    sdf_grid = torch.randn(bsize, nx, ny, nz).to(device)
    sdf_surf_grid = torch.randn(bsize, nx, ny, nz).to(device)
    sdf_nodes = torch.randn(bsize, 100, 1).to(device)
    surface_coordinates = torch.randn(bsize, 100, 3).to(device)
    surface_neighbors = torch.randn(bsize, 100, num_neigh, 3).to(device)
    surface_normals = torch.randn(bsize, 100, 3).to(device)
    surface_neighbors_normals = torch.randn(bsize, 100, num_neigh, 3).to(device)
    surface_sizes = torch.randn(bsize, 100, 3).to(device)
    surface_neighbors_sizes = torch.randn(bsize, 100, num_neigh, 3).to(device)
    volume_coordinates = torch.randn(bsize, 100, 3).to(device)
    vol_grid_max_min = torch.randn(bsize, 2, 3).to(device)
    surf_grid_max_min = torch.randn(bsize, 2, 3).to(device)
    stream_velocity = torch.randn(bsize, 1).to(device)
    air_density = torch.randn(bsize, 1).to(device)
    input_dict = {
        "pos_volume_closest": pos_normals_closest_vol,
        "pos_volume_center_of_mass": pos_normals_com_vol,
        "pos_surface_center_of_mass": pos_normals_com_surface,
        "geometry_coordinates": geom_centers,
        "grid": grid,
        "surf_grid": surf_grid,
        "sdf_grid": sdf_grid,
        "sdf_surf_grid": sdf_surf_grid,
        "sdf_nodes": sdf_nodes,
        "surface_mesh_centers": surface_coordinates,
        "surface_mesh_neighbors": surface_neighbors,
        "surface_normals": surface_normals,
        "surface_neighbors_normals": surface_neighbors_normals,
        "surface_areas": surface_sizes,
        "surface_neighbors_areas": surface_neighbors_sizes,
        "volume_mesh_centers": volume_coordinates,
        "volume_min_max": vol_grid_max_min,
        "surface_min_max": surf_grid_max_min,
        "stream_velocity": stream_velocity,
        "air_density": air_density,
    }

    # assert common.validate_forward_accuracy(
    #     model, input_dict, file_name=f"domino_output.pth"
    # )
    assert validate_domino(
        model, input_dict, file_name="domino_output.pth", device=device
    )
