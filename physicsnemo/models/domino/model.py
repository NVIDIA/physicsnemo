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
DoMINO Model Architecture.

The DoMINO class contains an architecture to model both surface and
volume quantities together as well as separately (controlled using
the config.yaml file).
"""

from typing import Any

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.core import Module
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.models.unet import UNet
from physicsnemo.nn import FourierMLP, get_activation

from .config import DEFAULT_MODEL_PARAMS, Config
from .encodings import (
    MultiGeometryEncoding,
)
from .geometry_rep import GeometryRep, scale_sdf
from .mlps import AggregationModel
from .solutions import SolutionCalculatorSurface, SolutionCalculatorVolume


class DoMINO(Module):
    r"""
    DoMINO model architecture for predicting both surface and volume quantities.

    The DoMINO (Deep Operational Modal Identification and Nonlinear Optimization) model
    is designed to model both surface and volume physical quantities in aerodynamic
    simulations. It can operate in three modes:

    1. Surface-only: Predicting only surface quantities
    2. Volume-only: Predicting only volume quantities
    3. Combined: Predicting both surface and volume quantities

    The model uses a combination of:

    - Geometry representation modules via :class:`~physicsnemo.models.domino.geometry_rep.GeometryRep`
    - Neural network basis functions via :class:`~physicsnemo.nn.FourierMLP`
    - Parameter encoding
    - Local and global geometry processing
    - Aggregation models for final prediction

    Parameters
    ----------
    input_features : int
        Number of point input features (typically 3 for x, y, z coordinates).
    output_features_vol : int, optional, default=None
        Number of output features in volume. Set to ``None`` for surface-only mode.
    output_features_surf : int, optional, default=None
        Number of output features on surface. Set to ``None`` for volume-only mode.
    global_features : int, optional, default=2
        Number of global parameter features for conditioning.
    model_parameters : Any, optional, default=None
        Model parameters controlled by config.yaml. Contains nested configuration
        for geometry representation, neural network basis functions, aggregation
        model, position encoder, and geometry local settings.

    Forward
    -------
    data_dict : dict[str, torch.Tensor]
        Dictionary containing input tensors with the following keys:

        - ``"geometry_coordinates"``: Geometry centers of shape :math:`(B, N_{geo}, 3)`
        - ``"grid"``: Computational domain grid of shape :math:`(B, N_x, N_y, N_z, 3)`
        - ``"surf_grid"``: Surface bounding box grid of shape :math:`(B, N_x, N_y, N_z, 3)`
        - ``"sdf_grid"``: SDF on volume grid of shape :math:`(B, N_x, N_y, N_z)`
        - ``"sdf_surf_grid"``: SDF on surface grid of shape :math:`(B, N_x, N_y, N_z)`
        - ``"sdf_nodes"``: SDF at volume mesh nodes of shape :math:`(B, N_{vol}, 1)`
        - ``"pos_volume_closest"``: Closest surface point to volume nodes of shape :math:`(B, N_{vol}, 3)`
        - ``"pos_volume_center_of_mass"``: Center of mass to volume nodes of shape :math:`(B, N_{vol}, 3)`
        - ``"pos_surface_center_of_mass"``: Center of mass to surface nodes of shape :math:`(B, N_{surf}, 3)`
        - ``"surface_mesh_centers"``: Surface mesh center coordinates of shape :math:`(B, N_{surf}, 3)`
        - ``"surface_mesh_neighbors"``: Surface mesh neighbor coordinates of shape :math:`(B, N_{surf}, K, 3)`
        - ``"surface_normals"``: Surface normals of shape :math:`(B, N_{surf}, 3)`
        - ``"surface_neighbors_normals"``: Surface neighbor normals of shape :math:`(B, N_{surf}, K, 3)`
        - ``"surface_areas"``: Surface cell areas of shape :math:`(B, N_{surf})`
        - ``"surface_neighbors_areas"``: Surface neighbor areas of shape :math:`(B, N_{surf}, K)`
        - ``"volume_mesh_centers"``: Volume mesh center coordinates of shape :math:`(B, N_{vol}, 3)`
        - ``"volume_min_max"``: Volume bounding box min/max of shape :math:`(B, 2, 3)`
        - ``"surface_min_max"``: Surface bounding box min/max of shape :math:`(B, 2, 3)`
        - ``"global_params_values"``: Global parameter values of shape :math:`(B, N_{params}, 1)`
        - ``"global_params_reference"``: Global parameter reference values of shape :math:`(B, N_{params}, 1)`

    Outputs
    -------
    tuple[torch.Tensor | None, torch.Tensor | None]
        A tuple containing:

        - Volume output tensor of shape :math:`(B, N_{vol}, D_{vol})` or ``None`` if volume-only mode is disabled
        - Surface output tensor of shape :math:`(B, N_{surf}, D_{surf})` or ``None`` if surface-only mode is disabled

    Example
    -------
    >>> from physicsnemo.models.domino.model import DoMINO
    >>> from physicsnemo.models.domino.config import DEFAULT_MODEL_PARAMS
    >>> import torch
    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> cfg = DEFAULT_MODEL_PARAMS  # already has model_type "combined"
    >>> model = DoMINO(
    ...         input_features=3,
    ...         output_features_vol=5,
    ...         output_features_surf=4,
    ...         model_parameters=cfg
    ...     ).to(device)
    >>> bsize = 1
    >>> nx, ny, nz = cfg.interp_res
    >>> num_neigh = cfg.num_neighbors_surface
    >>> global_features = 2
    >>> pos_normals_closest_vol = torch.randn(bsize, 100, 3).to(device)
    >>> pos_normals_com_vol = torch.randn(bsize, 100, 3).to(device)
    >>> pos_normals_com_surface = torch.randn(bsize, 100, 3).to(device)
    >>> geom_centers = torch.randn(bsize, 100, 3).to(device)
    >>> grid = torch.randn(bsize, nx, ny, nz, 3).to(device)
    >>> surf_grid = torch.randn(bsize, nx, ny, nz, 3).to(device)
    >>> sdf_grid = torch.randn(bsize, nx, ny, nz).to(device)
    >>> sdf_surf_grid = torch.randn(bsize, nx, ny, nz).to(device)
    >>> sdf_nodes = torch.randn(bsize, 100, 1).to(device)
    >>> surface_coordinates = torch.randn(bsize, 100, 3).to(device)
    >>> surface_neighbors = torch.randn(bsize, 100, num_neigh, 3).to(device)
    >>> surface_normals = torch.randn(bsize, 100, 3).to(device)
    >>> surface_neighbors_normals = torch.randn(bsize, 100, num_neigh, 3).to(device)
    >>> surface_sizes = torch.rand(bsize, 100).to(device) + 1e-6 # Note this needs to be > 0.0
    >>> surface_neighbors_areas = torch.rand(bsize, 100, num_neigh).to(device) + 1e-6
    >>> volume_coordinates = torch.randn(bsize, 100, 3).to(device)
    >>> vol_grid_max_min = torch.randn(bsize, 2, 3).to(device)
    >>> surf_grid_max_min = torch.randn(bsize, 2, 3).to(device)
    >>> global_params_values = torch.randn(bsize, global_features, 1).to(device)
    >>> global_params_reference = torch.randn(bsize, global_features, 1).to(device)
    >>> input_dict = {
    ...            "pos_volume_closest": pos_normals_closest_vol,
    ...            "pos_volume_center_of_mass": pos_normals_com_vol,
    ...            "pos_surface_center_of_mass": pos_normals_com_surface,
    ...            "geometry_coordinates": geom_centers,
    ...            "grid": grid,
    ...            "surf_grid": surf_grid,
    ...            "sdf_grid": sdf_grid,
    ...            "sdf_surf_grid": sdf_surf_grid,
    ...            "sdf_nodes": sdf_nodes,
    ...            "surface_mesh_centers": surface_coordinates,
    ...            "surface_mesh_neighbors": surface_neighbors,
    ...            "surface_normals": surface_normals,
    ...            "surface_neighbors_normals": surface_neighbors_normals,
    ...            "surface_areas": surface_sizes,
    ...            "surface_neighbors_areas": surface_neighbors_areas,
    ...            "volume_mesh_centers": volume_coordinates,
    ...            "volume_min_max": vol_grid_max_min,
    ...            "surface_min_max": surf_grid_max_min,
    ...            "global_params_reference": global_params_values,
    ...            "global_params_values": global_params_reference,
    ...        }
    >>> output = model(input_dict)
    >>> print(f"{output[0].shape}, {output[1].shape}")
    torch.Size([1, 100, 5]), torch.Size([1, 100, 4])

    Note
    ----
    At least one of ``output_features_vol`` or ``output_features_surf`` must be specified.
    """

    def __init__(
        self,
        input_features: int,
        output_features_vol: int | None = None,
        output_features_surf: int | None = None,
        global_features: int = 2,
        model_parameters: Any = None,
    ):
        super().__init__(meta=ModelMetaData(name="DoMINO"))

        # Convert model_parameters to Config, using defaults if None
        if model_parameters is None:
            model_parameters = DEFAULT_MODEL_PARAMS
        elif not isinstance(model_parameters, Config):
            model_parameters = Config.from_hydra(model_parameters)

        self.output_features_vol = output_features_vol
        self.output_features_surf = output_features_surf
        self.num_sample_points_surface = model_parameters.num_neighbors_surface
        self.num_sample_points_volume = model_parameters.num_neighbors_volume
        self.combined_vol_surf = model_parameters.combine_volume_surface
        self.activation_processor = (
            model_parameters.geometry_rep.geo_processor.activation
        )

        if self.combined_vol_surf:
            h = 8
            in_channels = (
                2
                + len(model_parameters.geometry_rep.geo_conv.volume_radii)
                + len(model_parameters.geometry_rep.geo_conv.surface_radii)
            )
            out_channels_surf = 1 + len(
                model_parameters.geometry_rep.geo_conv.surface_radii
            )
            out_channels_vol = 1 + len(
                model_parameters.geometry_rep.geo_conv.volume_radii
            )
            self.combined_unet_surf = UNet(
                in_channels=in_channels,
                out_channels=out_channels_surf,
                model_depth=3,
                feature_map_channels=[
                    h,
                    2 * h,
                    4 * h,
                ],
                num_conv_blocks=1,
                kernel_size=3,
                stride=1,
                conv_activation=self.activation_processor,
                padding=1,
                padding_mode="zeros",
                pooling_type="MaxPool3d",
                pool_size=2,
                normalization="layernorm",
                use_attn_gate=True,
                attn_decoder_feature_maps=[4 * h, 2 * h],
                attn_feature_map_channels=[2 * h, h],
                attn_intermediate_channels=4 * h,
                gradient_checkpointing=True,
            )
            self.combined_unet_vol = UNet(
                in_channels=in_channels,
                out_channels=out_channels_vol,
                model_depth=3,
                feature_map_channels=[
                    h,
                    2 * h,
                    4 * h,
                ],
                num_conv_blocks=1,
                kernel_size=3,
                stride=1,
                conv_activation=self.activation_processor,
                padding=1,
                padding_mode="zeros",
                pooling_type="MaxPool3d",
                pool_size=2,
                normalization="layernorm",
                use_attn_gate=True,
                attn_decoder_feature_maps=[4 * h, 2 * h],
                attn_feature_map_channels=[2 * h, h],
                attn_intermediate_channels=4 * h,
                gradient_checkpointing=True,
            )
        self.global_features = global_features

        if self.output_features_vol is None and self.output_features_surf is None:
            raise ValueError(
                "At least one of `output_features_vol` or `output_features_surf` must be specified"
            )
        if hasattr(model_parameters, "solution_calculation_mode"):
            if model_parameters.solution_calculation_mode not in [
                "one-loop",
                "two-loop",
            ]:
                raise ValueError(
                    f"Invalid solution_calculation_mode: {model_parameters.solution_calculation_mode}, select 'one-loop' or 'two-loop'."
                )
            self.solution_calculation_mode = model_parameters.solution_calculation_mode
        else:
            self.solution_calculation_mode = "two-loop"
        self.num_variables_vol = output_features_vol
        self.num_variables_surf = output_features_surf
        self.grid_resolution = model_parameters.interp_res
        self.use_surface_normals = model_parameters.use_surface_normals
        self.use_surface_area = model_parameters.use_surface_area
        self.encode_parameters = model_parameters.encode_parameters
        self.geo_encoding_type = model_parameters.geometry_encoding_type

        if self.use_surface_normals:
            if not self.use_surface_area:
                input_features_surface = input_features + 3
            else:
                input_features_surface = input_features + 4
        else:
            input_features_surface = input_features

        if self.encode_parameters:
            # Defining the parameter model
            base_layer_p = model_parameters.parameter_model.base_layer
            self.parameter_model = FourierMLP(
                input_features=self.global_features,
                fourier_features=model_parameters.parameter_model.fourier_features,
                num_modes=model_parameters.parameter_model.num_modes,
                base_layer=model_parameters.parameter_model.base_layer,
                activation=get_activation(model_parameters.parameter_model.activation),
            )
        else:
            base_layer_p = 0

        self.geo_rep_volume = GeometryRep(
            input_features=input_features,
            radii=model_parameters.geometry_rep.geo_conv.volume_radii,
            neighbors_in_radius=model_parameters.geometry_rep.geo_conv.volume_neighbors_in_radius,
            hops=model_parameters.geometry_rep.geo_conv.volume_hops,
            sdf_scaling_factor=model_parameters.geometry_rep.geo_processor.volume_sdf_scaling_factor,
            model_parameters=model_parameters,
        )

        self.geo_rep_surface = GeometryRep(
            input_features=input_features,
            radii=model_parameters.geometry_rep.geo_conv.surface_radii,
            neighbors_in_radius=model_parameters.geometry_rep.geo_conv.surface_neighbors_in_radius,
            hops=model_parameters.geometry_rep.geo_conv.surface_hops,
            sdf_scaling_factor=model_parameters.geometry_rep.geo_processor.surface_sdf_scaling_factor,
            model_parameters=model_parameters,
        )

        # Basis functions for surface and volume
        base_layer_nn = model_parameters.nn_basis_functions.base_layer
        if self.output_features_surf is not None:
            self.nn_basis_surf = nn.ModuleList()
            for _ in range(
                self.num_variables_surf
            ):  # Have the same basis function for each variable
                self.nn_basis_surf.append(
                    FourierMLP(
                        input_features=input_features_surface,
                        base_layer=model_parameters.nn_basis_functions.base_layer,
                        fourier_features=model_parameters.nn_basis_functions.fourier_features,
                        num_modes=model_parameters.nn_basis_functions.num_modes,
                        activation=get_activation(
                            model_parameters.nn_basis_functions.activation
                        ),
                    )
                )

        if self.output_features_vol is not None:
            self.nn_basis_vol = nn.ModuleList()
            for _ in range(
                self.num_variables_vol
            ):  # Have the same basis function for each variable
                self.nn_basis_vol.append(
                    FourierMLP(
                        input_features=input_features,
                        base_layer=model_parameters.nn_basis_functions.base_layer,
                        fourier_features=model_parameters.nn_basis_functions.fourier_features,
                        num_modes=model_parameters.nn_basis_functions.num_modes,
                        activation=get_activation(
                            model_parameters.nn_basis_functions.activation
                        ),
                    )
                )

        # Positional encoding
        position_encoder_base_neurons = model_parameters.position_encoder.base_neurons
        self.activation = get_activation(model_parameters.activation)
        self.use_sdf_in_basis_func = model_parameters.use_sdf_in_basis_func
        self.sdf_scaling_factor = (
            model_parameters.geometry_rep.geo_processor.volume_sdf_scaling_factor
        )
        if self.output_features_vol is not None:
            inp_pos_vol = (
                7 + len(self.sdf_scaling_factor)
                if model_parameters.use_sdf_in_basis_func
                else 3
            )

            self.fc_p_vol = FourierMLP(
                input_features=inp_pos_vol,
                fourier_features=model_parameters.position_encoder.fourier_features,
                num_modes=model_parameters.position_encoder.num_modes,
                base_layer=model_parameters.position_encoder.base_neurons,
                activation=get_activation(model_parameters.position_encoder.activation),
            )

        if self.output_features_surf is not None:
            inp_pos_surf = 3

            self.fc_p_surf = FourierMLP(
                input_features=inp_pos_surf,
                fourier_features=model_parameters.position_encoder.fourier_features,
                num_modes=model_parameters.position_encoder.num_modes,
                base_layer=model_parameters.position_encoder.base_neurons,
                activation=get_activation(model_parameters.position_encoder.activation),
            )

        # Create a set of local geometry encodings for the surface data
        self.surface_local_geo_encodings = MultiGeometryEncoding(
            radii=model_parameters.geometry_local.surface_radii,
            neighbors_in_radius=model_parameters.geometry_local.surface_neighbors_in_radius,
            geo_encoding_type=self.geo_encoding_type,
            n_upstream_radii=len(model_parameters.geometry_rep.geo_conv.surface_radii),
            base_layer=512,
            activation=get_activation(model_parameters.local_point_conv.activation),
            grid_resolution=self.grid_resolution,
        )

        # Create a set of local geometry encodings for the volume data
        self.volume_local_geo_encodings = MultiGeometryEncoding(
            radii=model_parameters.geometry_local.volume_radii,
            neighbors_in_radius=model_parameters.geometry_local.volume_neighbors_in_radius,
            geo_encoding_type=self.geo_encoding_type,
            n_upstream_radii=len(model_parameters.geometry_rep.geo_conv.volume_radii),
            base_layer=512,
            activation=get_activation(model_parameters.local_point_conv.activation),
            grid_resolution=self.grid_resolution,
        )

        # Aggregation model for surface
        if self.output_features_surf is not None:
            base_layer_geo_surf = 0
            for j in model_parameters.geometry_local.surface_neighbors_in_radius:
                base_layer_geo_surf += j

            self.agg_model_surf = nn.ModuleList()
            for _ in range(self.num_variables_surf):
                self.agg_model_surf.append(
                    AggregationModel(
                        input_features=position_encoder_base_neurons
                        + base_layer_nn
                        + base_layer_geo_surf
                        + base_layer_p,
                        output_features=1,
                        base_layer=model_parameters.aggregation_model.base_layer,
                        activation=get_activation(
                            model_parameters.aggregation_model.activation
                        ),
                    )
                )

            self.solution_calculator_surf = SolutionCalculatorSurface(
                num_variables=self.num_variables_surf,
                num_sample_points=self.num_sample_points_surface,
                use_surface_normals=self.use_surface_normals,
                use_surface_area=self.use_surface_area,
                encode_parameters=self.encode_parameters,
                parameter_model=self.parameter_model
                if self.encode_parameters
                else None,
                aggregation_model=self.agg_model_surf,
                nn_basis=self.nn_basis_surf,
            )

        # Aggregation model for volume
        if self.output_features_vol is not None:
            base_layer_geo_vol = 0
            for j in model_parameters.geometry_local.volume_neighbors_in_radius:
                base_layer_geo_vol += j

            self.agg_model_vol = nn.ModuleList()
            for _ in range(self.num_variables_vol):
                self.agg_model_vol.append(
                    AggregationModel(
                        input_features=position_encoder_base_neurons
                        + base_layer_nn
                        + base_layer_geo_vol
                        + base_layer_p,
                        output_features=1,
                        base_layer=model_parameters.aggregation_model.base_layer,
                        activation=get_activation(
                            model_parameters.aggregation_model.activation
                        ),
                    )
                )
            if hasattr(model_parameters, "return_volume_neighbors"):
                return_volume_neighbors = model_parameters.return_volume_neighbors
            else:
                return_volume_neighbors = False

            self.solution_calculator_vol = SolutionCalculatorVolume(
                num_variables=self.num_variables_vol,
                num_sample_points=self.num_sample_points_volume,
                noise_intensity=50,
                return_volume_neighbors=return_volume_neighbors,
                encode_parameters=self.encode_parameters,
                parameter_model=self.parameter_model
                if self.encode_parameters
                else None,
                aggregation_model=self.agg_model_vol,
                nn_basis=self.nn_basis_vol,
            )

    def forward(
        self,
        data_dict: dict[str, Float[torch.Tensor, "..."]],
    ) -> tuple[
        Float[torch.Tensor, "batch num_vol out_vol"] | None,
        Float[torch.Tensor, "batch num_surf out_surf"] | None,
    ]:
        r"""
        Perform forward pass of the DoMINO model.

        Parameters
        ----------
        data_dict : dict[str, torch.Tensor]
            Dictionary containing input tensors. See class docstring for required keys.

        Returns
        -------
        tuple[torch.Tensor | None, torch.Tensor | None]
            Tuple of (volume_output, surface_output). Either may be ``None`` if
            the corresponding output mode is disabled.
        """
        # Input validation
        if not torch.compiler.is_compiling():
            required_keys = [
                "geometry_coordinates",
                "surf_grid",
                "sdf_surf_grid",
                "global_params_values",
                "global_params_reference",
            ]
            if self.output_features_vol is not None:
                required_keys.extend(
                    [
                        "grid",
                        "sdf_grid",
                        "sdf_nodes",
                        "pos_volume_closest",
                        "pos_volume_center_of_mass",
                        "volume_mesh_centers",
                    ]
                )
            if self.output_features_surf is not None:
                required_keys.extend(
                    [
                        "pos_surface_center_of_mass",
                        "surface_mesh_centers",
                        "surface_mesh_neighbors",
                        "surface_normals",
                        "surface_neighbors_normals",
                        "surface_areas",
                        "surface_neighbors_areas",
                    ]
                )

            missing_keys = [k for k in required_keys if k not in data_dict]
            if missing_keys:
                raise ValueError(f"Missing required keys in data_dict: {missing_keys}")

        # Load STL inputs, bounding box grids, precomputed SDF and scaling factors
        # STL nodes
        geo_centers = data_dict["geometry_coordinates"]

        # Bounding box grid
        s_grid = data_dict["surf_grid"]
        sdf_surf_grid = data_dict["sdf_surf_grid"]

        # Parameters
        global_params_values = data_dict["global_params_values"]
        global_params_reference = data_dict["global_params_reference"]

        if self.output_features_vol is not None:
            # Represent geometry on computational grid
            # Computational domain grid
            p_grid = data_dict["grid"]
            sdf_grid = data_dict["sdf_grid"]

            # Normalize geometry coordinates based on computational domain
            if "volume_min_max" in data_dict.keys():
                vol_max = data_dict["volume_min_max"][:, 1]
                vol_min = data_dict["volume_min_max"][:, 0]
                geo_centers_vol = (
                    2.0 * (geo_centers - vol_min) / (vol_max - vol_min) - 1
                )
            else:
                geo_centers_vol = geo_centers

            # Compute geometry encoding for volume
            encoding_g_vol = self.geo_rep_volume(geo_centers_vol, p_grid, sdf_grid)

            # SDF on volume mesh nodes
            sdf_nodes = data_dict["sdf_nodes"]
            scaled_sdf_nodes = [
                scale_sdf(sdf_nodes, scaling) for scaling in self.sdf_scaling_factor
            ]
            scaled_sdf_nodes = torch.cat(scaled_sdf_nodes, dim=-1)

            # Positional encoding based on closest point on surface to a volume node
            pos_volume_closest = data_dict["pos_volume_closest"]
            # Positional encoding based on center of mass of geometry to volume node
            pos_volume_center_of_mass = data_dict["pos_volume_center_of_mass"]

            # Build volume node encoding
            if self.use_sdf_in_basis_func:
                encoding_node_vol = torch.cat(
                    (
                        sdf_nodes,
                        scaled_sdf_nodes,
                        pos_volume_closest,
                        pos_volume_center_of_mass,
                    ),
                    dim=-1,
                )
            else:
                encoding_node_vol = pos_volume_center_of_mass

            # Calculate positional encoding on volume nodes
            encoding_node_vol = self.fc_p_vol(encoding_node_vol)

        if self.output_features_surf is not None:
            # Represent geometry on bounding box
            # Normalize geometry coordinates based on surface bounding box
            if "surface_min_max" in data_dict.keys():
                surf_max = data_dict["surface_min_max"][:, 1]
                surf_min = data_dict["surface_min_max"][:, 0]
                geo_centers_surf = (
                    2.0 * (geo_centers - surf_min) / (surf_max - surf_min) - 1
                )
            else:
                geo_centers_surf = geo_centers

            # Compute geometry encoding for surface
            encoding_g_surf = self.geo_rep_surface(
                geo_centers_surf, s_grid, sdf_surf_grid
            )

            # Positional encoding based on center of mass of geometry to surface node
            pos_surface_center_of_mass = data_dict["pos_surface_center_of_mass"]
            encoding_node_surf = pos_surface_center_of_mass

            # Calculate positional encoding on surface centers
            encoding_node_surf = self.fc_p_surf(encoding_node_surf)

        # Combine volume and surface geometry encodings if both are present
        if (
            self.output_features_surf is not None
            and self.output_features_vol is not None
            and self.combined_vol_surf
        ):
            encoding_g = torch.cat((encoding_g_vol, encoding_g_surf), axis=1)
            encoding_g_surf = self.combined_unet_surf(encoding_g)
            encoding_g_vol = self.combined_unet_vol(encoding_g)

        if self.output_features_vol is not None:
            # Calculate local geometry encoding for volume
            volume_mesh_centers = data_dict["volume_mesh_centers"]
            encoding_g_vol = self.volume_local_geo_encodings(
                0.5 * encoding_g_vol,
                volume_mesh_centers,
                p_grid,
            )

            # Approximate solution on volume nodes
            output_vol = self.solution_calculator_vol(
                volume_mesh_centers,
                encoding_g_vol,
                encoding_node_vol,
                global_params_values,
                global_params_reference,
            )

        else:
            output_vol = None

        if self.output_features_surf is not None:
            # Load surface mesh data
            surface_mesh_centers = data_dict["surface_mesh_centers"]
            surface_normals = data_dict["surface_normals"]
            surface_areas = data_dict["surface_areas"]

            # Neighbors of sampled points on surface
            surface_mesh_neighbors = data_dict["surface_mesh_neighbors"]
            surface_neighbors_normals = data_dict["surface_neighbors_normals"]
            surface_neighbors_areas = data_dict["surface_neighbors_areas"]
            surface_areas = torch.unsqueeze(surface_areas, -1)
            surface_neighbors_areas = torch.unsqueeze(surface_neighbors_areas, -1)

            # Calculate local geometry encoding for surface
            encoding_g_surf = self.surface_local_geo_encodings(
                0.5 * encoding_g_surf, surface_mesh_centers, s_grid
            )

            # Approximate solution on surface cell centers
            output_surf = self.solution_calculator_surf(
                surface_mesh_centers,
                encoding_g_surf,
                encoding_node_surf,
                surface_mesh_neighbors,
                surface_normals,
                surface_neighbors_normals,
                surface_areas,
                surface_neighbors_areas,
                global_params_values,
                global_params_reference,
            )
        else:
            output_surf = None

        return output_vol, output_surf
