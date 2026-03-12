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

r"""
DoMINO Solution Calculator Modules.

This module contains the solution calculation layers for the DoMINO model architecture,
which compute the final output predictions for both surface and volume quantities.
"""

from collections import defaultdict

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.core import Module


def apply_parameter_encoding(
    mesh_centers: Float[torch.Tensor, "batch num_points 3"],
    global_params_values: Float[torch.Tensor, "batch num_params 1"],
    global_params_reference: Float[torch.Tensor, "batch num_params 1"],
) -> Float[torch.Tensor, "batch num_points num_params"]:
    r"""
    Apply parameter encoding to mesh centers.

    Normalizes global parameters by their reference values and expands them
    to match the spatial dimensions of the mesh centers.

    Parameters
    ----------
    mesh_centers : torch.Tensor
        Mesh center coordinates of shape :math:`(B, N, 3)`.
    global_params_values : torch.Tensor
        Global parameter values of shape :math:`(B, N_{params}, 1)`.
    global_params_reference : torch.Tensor
        Global parameter reference values of shape :math:`(B, N_{params}, 1)`.

    Returns
    -------
    torch.Tensor
        Processed parameters of shape :math:`(B, N, N_{params})`.
    """
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


def sample_sphere(
    center: Float[torch.Tensor, "batch num_points 3"],
    r: float,
    num_points: int,
) -> Float[torch.Tensor, "batch num_points num_samples 3"]:
    r"""
    Uniformly sample points in a 3D sphere around the center.

    This function generates random points within a sphere of radius ``r`` centered
    at each point in the input tensor. The sampling is uniform in volume,
    meaning points are more likely to be sampled in the outer regions of the sphere.

    Parameters
    ----------
    center : torch.Tensor
        Tensor of shape :math:`(B, N, 3)` containing center coordinates.
    r : float
        Radius of the sphere for sampling.
    num_points : int
        Number of points to sample per center.

    Returns
    -------
    torch.Tensor
        Tensor of shape :math:`(B, N, K, 3)` containing the sampled points
        around each center, where :math:`K` is ``num_points``.
    """
    # Expand center points to final shape
    unsqueezed_center = center.unsqueeze(2).expand(-1, -1, num_points, -1)

    # Generate random directions
    directions = torch.randn_like(unsqueezed_center)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    # Generate random radii with cubic root for uniform volume sampling
    radii = r * torch.pow(torch.rand_like(unsqueezed_center), 1 / 3)

    output = unsqueezed_center + directions * radii
    return output


def sample_sphere_shell(
    center: Float[torch.Tensor, "batch num_points 3"],
    r_inner: float,
    r_outer: float,
    num_points: int,
) -> Float[torch.Tensor, "batch num_points num_samples 3"]:
    r"""
    Uniformly sample points in a 3D spherical shell around a center.

    This function generates random points within a spherical shell (annulus)
    between inner radius ``r_inner`` and outer radius ``r_outer`` centered at each
    point in the input tensor. The sampling is uniform in volume within the shell.

    Parameters
    ----------
    center : torch.Tensor
        Tensor of shape :math:`(B, N, 3)` containing center coordinates.
    r_inner : float
        Inner radius of the spherical shell.
    r_outer : float
        Outer radius of the spherical shell.
    num_points : int
        Number of points to sample per center.

    Returns
    -------
    torch.Tensor
        Tensor of shape :math:`(B, N, K, 3)` containing the sampled points
        within the spherical shell around each center, where :math:`K` is ``num_points``.
    """
    unsqueezed_center = center.unsqueeze(2).expand(-1, -1, num_points, -1)

    # Generate random directions
    directions = torch.randn_like(unsqueezed_center)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    # Generate random radii uniformly in shell volume
    radii = torch.rand_like(unsqueezed_center) * (r_outer**3 - r_inner**3) + r_inner**3
    radii = torch.pow(radii, 1 / 3)

    output = unsqueezed_center + directions * radii

    return output


class SolutionCalculatorVolume(Module):
    r"""
    Module to calculate the output solution of the DoMINO Model for volume data.

    This module computes field predictions at volume mesh points by combining
    basis functions, positional encodings, and geometry encodings through an
    aggregation model. It supports neighbor-based averaging for improved accuracy.

    Parameters
    ----------
    num_variables : int
        Number of output field variables to predict.
    num_sample_points : int
        Number of neighbor sample points to use for averaging.
    noise_intensity : float
        Controls the sampling radius as :math:`1 / \text{noise\_intensity}`.
    encode_parameters : bool
        Whether to include parameter encoding in the aggregation.
    return_volume_neighbors : bool
        Whether to return neighbor information for visualization/debugging.
    parameter_model : nn.Module | None
        The parameter encoding model (required if ``encode_parameters=True``).
    aggregation_model : nn.ModuleList
        List of aggregation models, one per output variable.
    nn_basis : nn.ModuleList
        List of neural network basis function models, one per output variable.

    Forward
    -------
    volume_mesh_centers : torch.Tensor
        Volume mesh center coordinates of shape :math:`(B, N_{vol}, 3)`.
    encoding_g : torch.Tensor
        Geometry encoding of shape :math:`(B, N_{vol}, D_{geo})`.
    encoding_node : torch.Tensor
        Node positional encoding of shape :math:`(B, N_{vol}, D_{pos})`.
    global_params_values : torch.Tensor
        Global parameter values of shape :math:`(B, N_{params}, 1)`.
    global_params_reference : torch.Tensor
        Global parameter reference values of shape :math:`(B, N_{params}, 1)`.

    Outputs
    -------
    torch.Tensor | tuple
        If ``return_volume_neighbors=False``:
            Output tensor of shape :math:`(B, N_{vol}, N_{vars})`.
        If ``return_volume_neighbors=True``:
            Tuple of (output, perturbed_centers, field_neighbors, neighbors_dict).

    See Also
    --------
    :class:`~physicsnemo.models.domino.mlps.AggregationModel` : Used for final prediction.
    :class:`SolutionCalculatorSurface` : Similar module for surface data.
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
        super().__init__(meta=None)

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
        volume_mesh_centers: Float[torch.Tensor, "batch num_vol 3"],
        encoding_g: Float[torch.Tensor, "batch num_vol geo_features"],
        encoding_node: Float[torch.Tensor, "batch num_vol pos_features"],
        global_params_values: Float[torch.Tensor, "batch num_params 1"],
        global_params_reference: Float[torch.Tensor, "batch num_params 1"],
    ) -> (
        Float[torch.Tensor, "batch num_vol num_vars"]
        | tuple[
            Float[torch.Tensor, "batch num_vol num_vars"],
            Float[torch.Tensor, "batch num_vol num_samples 3"],
            Float[torch.Tensor, "batch num_vol num_samples num_vars"],
            dict,
        ]
    ):
        r"""
        Forward pass of the SolutionCalculatorVolume module.

        Parameters
        ----------
        volume_mesh_centers : torch.Tensor
            Volume mesh center coordinates of shape :math:`(B, N_{vol}, 3)`.
        encoding_g : torch.Tensor
            Geometry encoding of shape :math:`(B, N_{vol}, D_{geo})`.
        encoding_node : torch.Tensor
            Node positional encoding of shape :math:`(B, N_{vol}, D_{pos})`.
        global_params_values : torch.Tensor
            Global parameter values of shape :math:`(B, N_{params}, 1)`.
        global_params_reference : torch.Tensor
            Global parameter reference values of shape :math:`(B, N_{params}, 1)`.

        Returns
        -------
        torch.Tensor | tuple
            Output predictions, optionally with neighbor information.
        """
        # Input validation
        if not torch.compiler.is_compiling():
            if volume_mesh_centers.ndim != 3 or volume_mesh_centers.shape[-1] != 3:
                raise ValueError(
                    f"Expected volume_mesh_centers of shape (B, N, 3), "
                    f"got shape {tuple(volume_mesh_centers.shape)}"
                )
            if encoding_g.ndim != 3:
                raise ValueError(
                    f"Expected encoding_g to be 3D (B, N, D), "
                    f"got {encoding_g.ndim}D with shape {tuple(encoding_g.shape)}"
                )
            if encoding_node.ndim != 3:
                raise ValueError(
                    f"Expected encoding_node to be 3D (B, N, D), "
                    f"got {encoding_node.ndim}D with shape {tuple(encoding_node.shape)}"
                )

        # Compute parameter encoding if enabled
        if self.encode_parameters:
            param_encoding = apply_parameter_encoding(
                volume_mesh_centers, global_params_values, global_params_reference
            )
            param_encoding = self.parameter_model(param_encoding)

        # Initialize perturbed centers with original centers
        volume_m_c_perturbed = [volume_mesh_centers.unsqueeze(2)]

        if self.return_volume_neighbors:
            # Sample neighbors in a hierarchical pattern (1-hop and 2-hop)
            num_hop1 = self.num_sample_points
            num_hop2 = self.num_sample_points // 2 if self.num_sample_points != 1 else 1
            neighbors = defaultdict(list)

            # Sample 1-hop neighbors
            volume_m_c_hop1 = sample_sphere(
                volume_mesh_centers, 1 / self.noise_intensity, num_hop1
            )
            for i in range(num_hop1):
                idx = len(volume_m_c_perturbed)
                volume_m_c_perturbed.append(volume_m_c_hop1[:, :, i : i + 1, :])
                neighbors[0].append(idx)

            # Sample 2-hop neighbors from each 1-hop neighbor
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
            # Sample neighbors uniformly in sphere
            volume_m_c_sample = sample_sphere(
                volume_mesh_centers, 1 / self.noise_intensity, self.num_sample_points
            )
            for i in range(self.num_sample_points):
                volume_m_c_perturbed.append(volume_m_c_sample[:, :, i : i + 1, :])

            volume_m_c_perturbed = torch.cat(volume_m_c_perturbed, dim=2)

        # Compute predictions for each variable
        for f in range(self.num_variables):
            for p in range(volume_m_c_perturbed.shape[2]):
                volume_m_c = volume_m_c_perturbed[:, :, p, :]

                # Compute distance for neighbor weighting (skip for center point)
                if p != 0:
                    dist = torch.norm(
                        volume_m_c - volume_mesh_centers, dim=-1, keepdim=True
                    )

                # Compute basis functions and aggregate features
                basis_f = self.nn_basis[f](volume_m_c)
                output = torch.cat((basis_f, encoding_node, encoding_g), dim=-1)
                if self.encode_parameters:
                    output = torch.cat((output, param_encoding), dim=-1)

                # Apply aggregation model with inverse distance weighting
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

            # Combine center prediction with neighbor-averaged prediction
            if self.num_sample_points > 1:
                output_res = 0.5 * output_center + 0.5 * output_neighbor / dist_sum
            else:
                output_res = output_center

            # Concatenate predictions for all variables
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


class SolutionCalculatorSurface(Module):
    r"""
    Module to calculate the output solution of the DoMINO Model for surface data.

    This module computes field predictions at surface mesh points by combining
    basis functions, positional encodings, geometry encodings, and optionally
    surface normals and areas through an aggregation model.

    Parameters
    ----------
    num_variables : int
        Number of output field variables to predict.
    num_sample_points : int
        Number of neighbor sample points to use for averaging.
    encode_parameters : bool
        Whether to include parameter encoding in the aggregation.
    use_surface_normals : bool
        Whether to include surface normals in the basis function input.
    use_surface_area : bool
        Whether to include surface areas in the basis function input.
    parameter_model : nn.Module | None
        The parameter encoding model (required if ``encode_parameters=True``).
    aggregation_model : nn.ModuleList
        List of aggregation models, one per output variable.
    nn_basis : nn.ModuleList
        List of neural network basis function models, one per output variable.

    Forward
    -------
    surface_mesh_centers : torch.Tensor
        Surface mesh center coordinates of shape :math:`(B, N_{surf}, 3)`.
    encoding_g : torch.Tensor
        Geometry encoding of shape :math:`(B, N_{surf}, D_{geo})`.
    encoding_node : torch.Tensor
        Node positional encoding of shape :math:`(B, N_{surf}, D_{pos})`.
    surface_mesh_neighbors : torch.Tensor
        Surface mesh neighbor coordinates of shape :math:`(B, N_{surf}, K, 3)`.
    surface_normals : torch.Tensor
        Surface normals of shape :math:`(B, N_{surf}, 3)`.
    surface_neighbors_normals : torch.Tensor
        Surface neighbor normals of shape :math:`(B, N_{surf}, K, 3)`.
    surface_areas : torch.Tensor
        Surface cell areas of shape :math:`(B, N_{surf}, 1)`.
    surface_neighbors_areas : torch.Tensor
        Surface neighbor areas of shape :math:`(B, N_{surf}, K, 1)`.
    global_params_values : torch.Tensor
        Global parameter values of shape :math:`(B, N_{params}, 1)`.
    global_params_reference : torch.Tensor
        Global parameter reference values of shape :math:`(B, N_{params}, 1)`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, N_{surf}, N_{vars})`.

    See Also
    --------
    :class:`~physicsnemo.models.domino.mlps.AggregationModel` : Used for final prediction.
    :class:`SolutionCalculatorVolume` : Similar module for volume data.
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
        super().__init__(meta=None)
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
        surface_mesh_centers: Float[torch.Tensor, "batch num_surf 3"],
        encoding_g: Float[torch.Tensor, "batch num_surf geo_features"],
        encoding_node: Float[torch.Tensor, "batch num_surf pos_features"],
        surface_mesh_neighbors: Float[torch.Tensor, "batch num_surf num_neighbors 3"],
        surface_normals: Float[torch.Tensor, "batch num_surf 3"],
        surface_neighbors_normals: Float[
            torch.Tensor, "batch num_surf num_neighbors 3"
        ],
        surface_areas: Float[torch.Tensor, "batch num_surf 1"],
        surface_neighbors_areas: Float[torch.Tensor, "batch num_surf num_neighbors 1"],
        global_params_values: Float[torch.Tensor, "batch num_params 1"],
        global_params_reference: Float[torch.Tensor, "batch num_params 1"],
    ) -> Float[torch.Tensor, "batch num_surf num_vars"]:
        r"""
        Function to approximate solution given the neighborhood information.

        Parameters
        ----------
        surface_mesh_centers : torch.Tensor
            Surface mesh center coordinates of shape :math:`(B, N_{surf}, 3)`.
        encoding_g : torch.Tensor
            Geometry encoding of shape :math:`(B, N_{surf}, D_{geo})`.
        encoding_node : torch.Tensor
            Node positional encoding of shape :math:`(B, N_{surf}, D_{pos})`.
        surface_mesh_neighbors : torch.Tensor
            Surface mesh neighbor coordinates of shape :math:`(B, N_{surf}, K, 3)`.
        surface_normals : torch.Tensor
            Surface normals of shape :math:`(B, N_{surf}, 3)`.
        surface_neighbors_normals : torch.Tensor
            Surface neighbor normals of shape :math:`(B, N_{surf}, K, 3)`.
        surface_areas : torch.Tensor
            Surface cell areas of shape :math:`(B, N_{surf}, 1)`.
        surface_neighbors_areas : torch.Tensor
            Surface neighbor areas of shape :math:`(B, N_{surf}, K, 1)`.
        global_params_values : torch.Tensor
            Global parameter values of shape :math:`(B, N_{params}, 1)`.
        global_params_reference : torch.Tensor
            Global parameter reference values of shape :math:`(B, N_{params}, 1)`.

        Returns
        -------
        torch.Tensor
            Output predictions of shape :math:`(B, N_{surf}, N_{vars})`.
        """
        # Input validation
        if not torch.compiler.is_compiling():
            if surface_mesh_centers.ndim != 3 or surface_mesh_centers.shape[-1] != 3:
                raise ValueError(
                    f"Expected surface_mesh_centers of shape (B, N, 3), "
                    f"got shape {tuple(surface_mesh_centers.shape)}"
                )
            if encoding_g.ndim != 3:
                raise ValueError(
                    f"Expected encoding_g to be 3D (B, N, D), "
                    f"got {encoding_g.ndim}D with shape {tuple(encoding_g.shape)}"
                )
            if encoding_node.ndim != 3:
                raise ValueError(
                    f"Expected encoding_node to be 3D (B, N, D), "
                    f"got {encoding_node.ndim}D with shape {tuple(encoding_node.shape)}"
                )
            if (
                surface_mesh_neighbors.ndim != 4
                or surface_mesh_neighbors.shape[-1] != 3
            ):
                raise ValueError(
                    f"Expected surface_mesh_neighbors of shape (B, N, K, 3), "
                    f"got shape {tuple(surface_mesh_neighbors.shape)}"
                )

        # Compute parameter encoding if enabled
        if self.encode_parameters:
            param_encoding = apply_parameter_encoding(
                surface_mesh_centers, global_params_values, global_params_reference
            )
            param_encoding = self.parameter_model(param_encoding)

        # Build input features for centers
        centers_inputs = [
            surface_mesh_centers,
        ]
        neighbors_inputs = [
            surface_mesh_neighbors,
        ]

        # Optionally add surface normals
        if self.use_surface_normals:
            centers_inputs.append(surface_normals)
            if self.num_sample_points > 1:
                neighbors_inputs.append(surface_neighbors_normals)

        # Optionally add surface areas (log-scaled for numerical stability)
        if self.use_surface_area:
            centers_inputs.append(torch.log(surface_areas) / 10)
            if self.num_sample_points > 1:
                neighbors_inputs.append(torch.log(surface_neighbors_areas) / 10)

        # Concatenate all input features
        surface_mesh_centers = torch.cat(centers_inputs, dim=-1)
        surface_mesh_neighbors = torch.cat(neighbors_inputs, dim=-1)

        # Compute predictions for each variable
        for f in range(self.num_variables):
            for p in range(self.num_sample_points):
                if p == 0:
                    # Use center point
                    volume_m_c = surface_mesh_centers
                else:
                    # Use neighbor points with small offset for numerical stability
                    volume_m_c = surface_mesh_neighbors[:, :, p - 1] + 1e-6
                    noise = surface_mesh_centers - volume_m_c
                    dist = torch.norm(noise, dim=-1, keepdim=True)

                # Compute basis functions and aggregate features
                basis_f = self.nn_basis[f](volume_m_c)
                output = torch.cat((basis_f, encoding_node, encoding_g), dim=-1)
                if self.encode_parameters:
                    output = torch.cat((output, param_encoding), dim=-1)

                # Apply aggregation model with inverse distance weighting
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

            # Combine center prediction with neighbor-averaged prediction
            if self.num_sample_points > 1:
                output_res = 0.5 * output_center + 0.5 * output_neighbor / dist_sum
            else:
                output_res = output_center

            # Concatenate predictions for all variables
            if f == 0:
                output_all = output_res
            else:
                output_all = torch.cat((output_all, output_res), dim=-1)

        return output_all
