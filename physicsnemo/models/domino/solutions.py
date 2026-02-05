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

"""
This code contains the DoMINO model architecture.
The DoMINO class contains an architecture to model both surface and
volume quantities together as well as separately (controlled using
the config.yaml file)
"""

from collections import defaultdict

import torch
import torch.nn as nn


def apply_parameter_encoding(
    mesh_centers: torch.Tensor,
    global_params_values: torch.Tensor,
    global_params_reference: torch.Tensor,
) -> torch.Tensor:
    processed_parameters = []
    for k in range(global_params_values.shape[1]):
        param = torch.unsqueeze(global_params_values[:, k, :], 1)
        ref = torch.unsqueeze(global_params_reference[:, k, :], 1)
        param = param.expand(
            param.shape[0],
            mesh_centers.shape[1],
            param.shape[2],
        )
        param = param / ref
        processed_parameters.append(param)
    processed_parameters = torch.cat(processed_parameters, axis=-1)

    return processed_parameters


def sample_sphere(center, r, num_points):
    """Uniformly sample points in a 3D sphere around the center.

    This method generates random points within a sphere of radius r centered
    at each point in the input tensor. The sampling is uniform in volume,
    meaning points are more likely to be sampled in the outer regions of the sphere.

    Args:
        center: Tensor of shape (batch_size, num_points, 3) containing center coordinates
        r: Radius of the sphere for sampling
        num_points: Number of points to sample per center

    Returns:
        Tensor of shape (batch_size, num_points, num_samples, 3) containing
        the sampled points around each center
    """
    # Adjust the center points to the final shape:
    unsqueezed_center = center.unsqueeze(2).expand(-1, -1, num_points, -1)

    # Generate directions like the centers:
    directions = torch.randn_like(unsqueezed_center)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    # Generate radii like the centers:
    radii = r * torch.pow(torch.rand_like(unsqueezed_center), 1 / 3)

    output = unsqueezed_center + directions * radii
    return output


def sample_sphere_shell(center, r_inner, r_outer, num_points):
    """Uniformly sample points in a 3D spherical shell around a center.

    This method generates random points within a spherical shell (annulus)
    between inner radius r_inner and outer radius r_outer centered at each
    point in the input tensor. The sampling is uniform in volume within the shell.

    Args:
        center: Tensor of shape (batch_size, num_points, 3) containing center coordinates
        r_inner: Inner radius of the spherical shell
        r_outer: Outer radius of the spherical shell
        num_points: Number of points to sample per center

    Returns:
        Tensor of shape (batch_size, num_points, num_samples, 3) containing
        the sampled points within the spherical shell around each center
    """

    unsqueezed_center = center.unsqueeze(2).expand(-1, -1, num_points, -1)

    # Generate directions like the centers:
    directions = torch.randn_like(unsqueezed_center)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    radii = torch.rand_like(unsqueezed_center) * (r_outer**3 - r_inner**3) + r_inner**3
    radii = torch.pow(radii, 1 / 3)

    output = unsqueezed_center + directions * radii

    return output


class SolutionCalculatorVolume(nn.Module):
    """
    Module to calculate the output solution of the DoMINO Model for volume data.
    """

    def __init__(
        self,
        num_variables: int,
        num_sample_points: int,
        noise_intensity: float,
        encode_parameters: bool,
        return_volume_neighbors: bool,
        parameter_model: nn.Module | None,
        aggregation_model: nn.ModuleList,
        nn_basis: nn.ModuleList,
    ):
        super().__init__()

        self.num_variables = num_variables
        self.num_sample_points = num_sample_points
        self.noise_intensity = noise_intensity
        self.encode_parameters = encode_parameters
        self.return_volume_neighbors = return_volume_neighbors
        self.parameter_model = parameter_model
        self.aggregation_model = aggregation_model
        self.nn_basis = nn_basis

        if self.encode_parameters:
            if self.parameter_model is None:
                raise ValueError(
                    "Parameter model is required when encode_parameters is True"
                )

    def forward(
        self,
        volume_mesh_centers: torch.Tensor,
        encoding_g: torch.Tensor,
        encoding_node: torch.Tensor,
        global_params_values: torch.Tensor,
        global_params_reference: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass of the SolutionCalculator module.
        """
        if self.encode_parameters:
            param_encoding = apply_parameter_encoding(
                volume_mesh_centers, global_params_values, global_params_reference
            )
            param_encoding = self.parameter_model(param_encoding)

        volume_m_c_perturbed = [volume_mesh_centers.unsqueeze(2)]

        if self.return_volume_neighbors:
            num_hop1 = self.num_sample_points
            num_hop2 = (
                self.num_sample_points // 2 if self.num_sample_points != 1 else 1
            )  # This is per 1 hop node
            neighbors = defaultdict(list)

            volume_m_c_hop1 = sample_sphere(
                volume_mesh_centers, 1 / self.noise_intensity, num_hop1
            )
            # 1 hop neighbors
            for i in range(num_hop1):
                idx = len(volume_m_c_perturbed)
                volume_m_c_perturbed.append(volume_m_c_hop1[:, :, i : i + 1, :])
                neighbors[0].append(idx)

            # 2 hop neighbors
            for i in range(num_hop1):
                parent_idx = i + 1  # Skipping the first point, which is the original
                parent_point = volume_m_c_perturbed[parent_idx]

                children = sample_sphere_shell(
                    parent_point.squeeze(2),
                    1 / self.noise_intensity,
                    2 / self.noise_intensity,
                    num_hop2,
                )

                for c in range(num_hop2):
                    idx = len(volume_m_c_perturbed)
                    volume_m_c_perturbed.append(children[:, :, c : c + 1, :])
                    neighbors[parent_idx].append(idx)

            volume_m_c_perturbed = torch.cat(volume_m_c_perturbed, dim=2)
            neighbors = dict(neighbors)
            field_neighbors = {i: [] for i in range(self.num_variables)}
        else:
            volume_m_c_sample = sample_sphere(
                volume_mesh_centers, 1 / self.noise_intensity, self.num_sample_points
            )
            for i in range(self.num_sample_points):
                volume_m_c_perturbed.append(volume_m_c_sample[:, :, i : i + 1, :])

            volume_m_c_perturbed = torch.cat(volume_m_c_perturbed, dim=2)

        for f in range(self.num_variables):
            for p in range(volume_m_c_perturbed.shape[2]):
                volume_m_c = volume_m_c_perturbed[:, :, p, :]
                if p != 0:
                    dist = torch.norm(
                        volume_m_c - volume_mesh_centers, dim=-1, keepdim=True
                    )
                basis_f = self.nn_basis[f](volume_m_c)
                output = torch.cat((basis_f, encoding_node, encoding_g), dim=-1)
                if self.encode_parameters:
                    output = torch.cat((output, param_encoding), dim=-1)
                if p == 0:
                    output_center = self.aggregation_model[f](output)
                else:
                    if p == 1:
                        output_neighbor = self.aggregation_model[f](output) * (
                            1.0 / dist
                        )
                        dist_sum = 1.0 / dist
                    else:
                        output_neighbor += self.aggregation_model[f](output) * (
                            1.0 / dist
                        )
                        dist_sum += 1.0 / dist
                if self.return_volume_neighbors:
                    field_neighbors[f].append(self.aggregation_model[f](output))

            if self.return_volume_neighbors:
                field_neighbors[f] = torch.stack(field_neighbors[f], dim=2)

            if self.num_sample_points > 1:
                output_res = (
                    0.5 * output_center + 0.5 * output_neighbor / dist_sum
                )  # This only applies to the main point, and not the preturbed points
            else:
                output_res = output_center
            if f == 0:
                output_all = output_res
            else:
                output_all = torch.cat((output_all, output_res), axis=-1)

        if self.return_volume_neighbors:
            field_neighbors = torch.cat(
                [field_neighbors[i] for i in range(self.num_variables)], dim=3
            )
            return output_all, volume_m_c_perturbed, field_neighbors, neighbors
        else:
            return output_all


class SolutionCalculatorSurface(nn.Module):
    """
    Module to calculate the output solution of the DoMINO Model for surface data.
    """

    def __init__(
        self,
        num_variables: int,
        num_sample_points: int,
        encode_parameters: bool,
        use_surface_normals: bool,
        use_surface_area: bool,
        parameter_model: nn.Module | None,
        aggregation_model: nn.ModuleList,
        nn_basis: nn.ModuleList,
    ):
        super().__init__()
        self.num_variables = num_variables
        self.num_sample_points = num_sample_points
        self.encode_parameters = encode_parameters
        self.use_surface_normals = use_surface_normals
        self.use_surface_area = use_surface_area
        self.parameter_model = parameter_model
        self.aggregation_model = aggregation_model
        self.nn_basis = nn_basis

        if self.encode_parameters:
            if self.parameter_model is None:
                raise ValueError(
                    "Parameter model is required when encode_parameters is True"
                )

    def forward(
        self,
        surface_mesh_centers: torch.Tensor,
        encoding_g: torch.Tensor,
        encoding_node: torch.Tensor,
        surface_mesh_neighbors: torch.Tensor,
        surface_normals: torch.Tensor,
        surface_neighbors_normals: torch.Tensor,
        surface_areas: torch.Tensor,
        surface_neighbors_areas: torch.Tensor,
        global_params_values: torch.Tensor,
        global_params_reference: torch.Tensor,
    ) -> torch.Tensor:
        """Function to approximate solution given the neighborhood information"""

        if self.encode_parameters:
            param_encoding = apply_parameter_encoding(
                surface_mesh_centers, global_params_values, global_params_reference
            )
            param_encoding = self.parameter_model(param_encoding)

        centers_inputs = [
            surface_mesh_centers,
        ]
        neighbors_inputs = [
            surface_mesh_neighbors,
        ]

        if self.use_surface_normals:
            centers_inputs.append(surface_normals)
            if self.num_sample_points > 1:
                neighbors_inputs.append(surface_neighbors_normals)

        if self.use_surface_area:
            centers_inputs.append(torch.log(surface_areas) / 10)
            if self.num_sample_points > 1:
                neighbors_inputs.append(torch.log(surface_neighbors_areas) / 10)

        surface_mesh_centers = torch.cat(centers_inputs, dim=-1)
        surface_mesh_neighbors = torch.cat(neighbors_inputs, dim=-1)

        for f in range(self.num_variables):
            for p in range(self.num_sample_points):
                if p == 0:
                    volume_m_c = surface_mesh_centers
                else:
                    volume_m_c = surface_mesh_neighbors[:, :, p - 1] + 1e-6
                    noise = surface_mesh_centers - volume_m_c
                    dist = torch.norm(noise, dim=-1, keepdim=True)

                basis_f = self.nn_basis[f](volume_m_c)
                output = torch.cat((basis_f, encoding_node, encoding_g), dim=-1)
                if self.encode_parameters:
                    output = torch.cat((output, param_encoding), dim=-1)
                if p == 0:
                    output_center = self.aggregation_model[f](output)
                else:
                    if p == 1:
                        output_neighbor = self.aggregation_model[f](output) * (
                            1.0 / dist
                        )
                        dist_sum = 1.0 / dist
                    else:
                        output_neighbor += self.aggregation_model[f](output) * (
                            1.0 / dist
                        )
                        dist_sum += 1.0 / dist
            if self.num_sample_points > 1:
                output_res = 0.5 * output_center + 0.5 * output_neighbor / dist_sum
            else:
                output_res = output_center
            if f == 0:
                output_all = output_res
            else:
                output_all = torch.cat((output_all, output_res), dim=-1)

        return output_all
