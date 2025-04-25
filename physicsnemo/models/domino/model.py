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

"""
This code contains the DoMINO model architecture.
The DoMINO class contains an architecture to model both surface and
volume quantities together as well as separately (controlled using
the config.yaml file)
"""

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from physicsnemo.models.layers.ball_query import BallQueryLayer


def fourier_encode(coords, num_freqs):
    """Function to caluculate fourier features"""
    # Create a range of frequencies
    freqs = torch.exp(torch.linspace(0, math.pi, num_freqs))
    # Generate sine and cosine features
    features = [torch.sin(coords * f) for f in freqs] + [
        torch.cos(coords * f) for f in freqs
    ]
    return torch.cat(features, dim=-1)


def calculate_pos_encoding(nx, d=8):
    """Function to caluculate positional encoding"""
    vec = []
    for k in range(int(d / 2)):
        vec.append(torch.sin(nx / 10000 ** (2 * (k) / d)))
        vec.append(torch.cos(nx / 10000 ** (2 * (k) / d)))
    return vec


def scale_sdf(sdf: torch.Tensor) -> torch.Tensor:
    """
    Scale a signed distance function (SDF) to emphasize surface regions.

    This function applies a non-linear scaling to the SDF values that compresses
    the range while preserving the sign, effectively giving more weight to points
    near surfaces where |SDF| is small.

    Args:
        sdf: Tensor containing signed distance function values

    Returns:
        Tensor with scaled SDF values in range [-1, 1]
    """
    return sdf / (0.4 + torch.abs(sdf))


class BQWarp(nn.Module):
    """
    Warp-based ball-query layer for finding neighboring points within a specified radius.

    This layer uses an accelerated ball query implementation to efficiently find points
    within a specified radius of query points.
    """

    def __init__(
        self,
        grid_resolution=None,
        radius: float = 0.25,
        neighbors_in_radius: int = 10,
    ):
        """
        Initialize the BQWarp layer.

        Args:
            grid_resolution: Resolution of the grid in each dimension [nx, ny, nz]
            radius: Radius for ball query operation
            neighbors_in_radius: Maximum number of neighbors to return within radius
        """
        super().__init__()
        if grid_resolution is None:
            grid_resolution = [256, 96, 64]
        self.ball_query_layer = BallQueryLayer(neighbors_in_radius, radius)
        self.grid_resolution = grid_resolution

    def forward(
        self, x: torch.Tensor, p_grid: torch.Tensor, reverse_mapping: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs ball query operation to find neighboring points and their features.

        This method uses the Warp-accelerated ball query implementation to find points
        within a specified radius. It can operate in two modes:
        - Forward mapping: Find points from x that are near p_grid points (reverse_mapping=False)
        - Reverse mapping: Find points from p_grid that are near x points (reverse_mapping=True)

        Args:
            x: Tensor of shape (batch_size, num_points, 3+features) containing point coordinates
               and their features
            p_grid: Tensor of shape (batch_size, grid_x, grid_y, grid_z, 3) containing grid point
                   coordinates
            reverse_mapping: Boolean flag to control the direction of the mapping:
                            - True: Find p_grid points near x points
                            - False: Find x points near p_grid points

        Returns:
            tuple containing:
                - mapping: Tensor containing indices of neighboring points
                - outputs: Tensor containing coordinates of the neighboring points
        """
        batch_size = x.shape[0]
        nx, ny, nz = self.grid_resolution

        p_grid = torch.reshape(p_grid, (batch_size, nx * ny * nz, 3))
        # p1 = nx * ny * nz
        # p2 = x.shape[1]

        if reverse_mapping:
            # lengths1 = torch.full((batch_size,), p1, dtype=torch.int32)
            # lengths2 = torch.full((batch_size,), p2, dtype=torch.int32)
            mapping, num_neighbors, outputs = self.ball_query_layer(
                p_grid,
                x,
                # lengths1,
                # lengths2,
            )
        else:
            # lengths1 = torch.full((batch_size,), p2, dtype=torch.int32)
            # lengths2 = torch.full((batch_size,), p1, dtype=torch.int32)
            mapping, num_neighbors, outputs = self.ball_query_layer(
                x,
                p_grid,
                # lengths1,
                # lengths2,
            )

        return mapping, outputs


class GeoConvOut(nn.Module):
    """
    Geometry layer to project STL geometry data onto regular grids.
    """

    def __init__(
        self,
        input_features: int,
        model_parameters,
        grid_resolution=None,
    ):
        """
        Initialize the GeoConvOut layer.

        Args:
            input_features: Number of input feature dimensions
            model_parameters: Configuration parameters for the model
            grid_resolution: Resolution of the output grid [nx, ny, nz]
        """
        super().__init__()
        if grid_resolution is None:
            grid_resolution = [256, 96, 64]
        base_neurons = model_parameters.base_neurons

        self.fc1 = nn.Linear(input_features, base_neurons)
        self.fc2 = nn.Linear(base_neurons, int(base_neurons / 2))
        self.fc3 = nn.Linear(int(base_neurons / 2), model_parameters.base_neurons_out)

        self.grid_resolution = grid_resolution

        self.activation = F.relu

    def forward(
        self, x: torch.Tensor, radius: float = 0.025, neighbors_in_radius: int = 10
    ) -> torch.Tensor:
        """
        Process and project geometric features onto a 3D grid.
        """
        batch_size = x.shape[0]
        nx, ny, nz = (
            self.grid_resolution[0],
            self.grid_resolution[1],
            self.grid_resolution[2],
        )

        mask = abs(x - 0) > 1e-6
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = F.tanh(self.fc3(x))
        mask = mask[:, :, :, 0:1].expand(
            mask.shape[0], mask.shape[1], mask.shape[2], x.shape[-1]
        )

        x = torch.sum(x * mask, 2)

        x = torch.reshape(x, (batch_size, x.shape[-1], nx, ny, nz))
        return x


class GeoProcessor(nn.Module):
    """Geometry processing layer using CNNs"""

    def __init__(self, input_filters: int, model_parameters):
        """
        Initialize the GeoProcessor network.

        Args:
            input_filters: Number of input channels
            model_parameters: Configuration parameters for the model
        """
        super().__init__()
        base_filters = model_parameters.base_filters
        self.conv1 = nn.Conv3d(
            input_filters, base_filters, kernel_size=3, padding="same"
        )
        self.conv2 = nn.Conv3d(
            base_filters, 2 * base_filters, kernel_size=3, padding="same"
        )
        self.conv3 = nn.Conv3d(
            2 * base_filters, 4 * base_filters, kernel_size=3, padding="same"
        )
        self.conv3_1 = nn.Conv3d(
            4 * base_filters, 4 * base_filters, kernel_size=3, padding="same"
        )
        self.conv4 = nn.Conv3d(
            4 * base_filters, 2 * base_filters, kernel_size=3, padding="same"
        )
        self.conv5 = nn.Conv3d(
            4 * base_filters, base_filters, kernel_size=3, padding="same"
        )
        self.conv6 = nn.Conv3d(
            2 * base_filters, input_filters, kernel_size=3, padding="same"
        )
        self.conv7 = nn.Conv3d(
            2 * input_filters, input_filters, kernel_size=3, padding="same"
        )
        self.conv8 = nn.Conv3d(input_filters, 1, kernel_size=3, padding="same")
        self.avg_pool = torch.nn.AvgPool3d((2, 2, 2))
        self.max_pool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.activation = F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process geometry information through the 3D CNN network.

        The network follows an encoder-decoder architecture with skip connections:
        1. Downsampling path (encoder) with three levels of max pooling
        2. Processing loop in the bottleneck
        3. Upsampling path (decoder) with skip connections from the encoder

        Args:
            x: Input tensor containing grid-represented geometry of shape
               (batch_size, input_filters, nx, ny, nz)

        Returns:
            Processed geometry features of shape (batch_size, 1, nx, ny, nz)
        """
        # Encoder
        x0 = x
        x = self.conv1(x)
        x = self.activation(x)
        x = self.max_pool(x)

        x1 = x
        x = self.conv2(x)
        x = self.activation(x)
        x = self.max_pool(x)

        x2 = x
        x = self.conv3(x)
        x = self.activation(x)
        x = self.max_pool(x)

        # Processor loop
        x = F.relu(self.conv3_1(x))

        # Decoder
        x = self.conv4(x)
        x = self.activation(x)
        x = self.upsample(x)
        x = torch.cat((x, x2), axis=1)

        x = self.conv5(x)
        x = self.activation(x)
        x = self.upsample(x)
        x = torch.cat((x, x1), axis=1)

        x = self.conv6(x)
        x = self.activation(x)
        x = self.upsample(x)
        x = torch.cat((x, x0), axis=1)

        x = self.activation(self.conv7(x))
        x = self.conv8(x)

        return x


class GeometryRep(nn.Module):
    """
    Geometry representation module that processes STL geometry data.

    This module constructs a multiscale representation of geometry by:
    1. Computing short-range geometry encoding for local features
    2. Computing long-range geometry encoding for global context
    3. Processing signed distance field (SDF) data for surface information

    The combined encoding enables the model to reason about both local and global
    geometric properties.
    """

    def __init__(self, input_features: int, radii, model_parameters=None):
        """
        Initialize the GeometryRep module.

        Args:
            input_features: Number of input feature dimensions
            model_parameters: Configuration parameters for the model
        """
        super().__init__()
        geometry_rep = model_parameters.geometry_rep
        self.geo_encoding_type = model_parameters.geometry_encoding_type

        self.bq_warp = nn.ModuleList()
        self.geo_processors = nn.ModuleList()
        for j, p in enumerate(radii):
            self.bq_warp.append(
                BQWarp(
                    grid_resolution=model_parameters.interp_res,
                    radius=radii[j],
                )
            )
            self.geo_processors.append(
                GeoProcessor(
                    input_filters=geometry_rep.geo_conv.base_neurons_out,
                    model_parameters=geometry_rep.geo_processor,
                )
            )

        self.geo_conv_out = nn.ModuleList()
        for j, p in enumerate(radii):
            self.geo_conv_out.append(
                GeoConvOut(
                    input_features=input_features,
                    model_parameters=geometry_rep.geo_conv,
                    grid_resolution=model_parameters.interp_res,
                )
            )

        self.geo_processor_sdf = GeoProcessor(
            input_filters=6, model_parameters=geometry_rep.geo_processor
        )
        self.activation = F.relu
        self.radii = radii
        self.hops = geometry_rep.geo_conv.hops

    def forward(
        self, x: torch.Tensor, p_grid: torch.Tensor, sdf: torch.Tensor
    ) -> torch.Tensor:
        """
        Process geometry data to create a comprehensive representation.

        This method combines short-range, long-range, and SDF-based geometry
        encodings to create a rich representation of the geometry.

        Args:
            x: Input tensor containing geometric point data
            p_grid: Grid points for sampling
            sdf: Signed distance field tensor

        Returns:
            Comprehensive geometry encoding that concatenates short-range,
            SDF-based, and long-range features
        """
        if self.geo_encoding_type == "both" or self.geo_encoding_type == "stl":
            # Calculate multi-scale geoemtry dependency
            x_encoding = []
            for j, p in enumerate(self.radii):
                mapping, k_short = self.bq_warp[j](x, p_grid)
                x_encoding_inter = self.geo_conv_out[j](k_short)
                # Propagate information in the geometry enclosed BBox
                for _ in range(self.hops):
                    dx = self.geo_processors[j](x_encoding_inter) / self.hops
                    x_encoding_inter = x_encoding_inter + dx
                x_encoding.append(x_encoding_inter)
            x_encoding = torch.cat(x_encoding, axis=1)

        if self.geo_encoding_type == "both" or self.geo_encoding_type == "sdf":
            # Expand SDF
            sdf = torch.unsqueeze(sdf, 1)
            # Scaled sdf to emphasize near surface
            scaled_sdf = scale_sdf(sdf)
            # Binary sdf
            binary_sdf = torch.where(sdf >= 0, 0.0, 1.0)
            # Gradients of SDF
            sdf_x, sdf_y, sdf_z = torch.gradient(sdf, dim=[2, 3, 4])

            # Process SDF and its computed features
            sdf = torch.cat((sdf, scaled_sdf, binary_sdf, sdf_x, sdf_y, sdf_z), 1)
            sdf_encoding = self.geo_processor_sdf(sdf)

        if self.geo_encoding_type == "both":
            # Geometry encoding comprised of short-range, long-range and SDF features
            encoding_g = torch.cat((x_encoding, sdf_encoding), 1)
        elif self.geo_encoding_type == "sdf":
            encoding_g = sdf_encoding
        elif self.geo_encoding_type == "stl":
            encoding_g = x_encoding

        return encoding_g


class NNBasisFunctions(nn.Module):
    """Basis function layer for point clouds"""

    def __init__(self, input_features: int, model_parameters=None):
        super(NNBasisFunctions, self).__init__()
        base_layer = model_parameters.base_layer
        self.fourier_features = model_parameters.fourier_features
        self.num_modes = model_parameters.num_modes

        if self.fourier_features:
            input_features_calculated = (
                input_features + input_features * self.num_modes * 2
            )
        else:
            input_features_calculated = input_features

        self.fc1 = nn.Linear(input_features_calculated, base_layer)
        self.fc2 = nn.Linear(base_layer, int(base_layer))
        self.fc3 = nn.Linear(int(base_layer), int(base_layer))
        self.bn1 = nn.BatchNorm1d(base_layer)
        self.bn2 = nn.BatchNorm1d(int(base_layer))
        self.bn3 = nn.BatchNorm1d(int(base_layer))

        self.activation = F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform point features into a basis function representation.

        Args:
            x: Input tensor containing point features

        Returns:
            Tensor containing basis function coefficients
        """
        if self.fourier_features:
            facets = torch.cat((x, fourier_encode(x, self.num_modes)), axis=-1)
        else:
            facets = x
        facets = self.activation(self.fc1(facets))
        facets = self.activation(self.fc2(facets))
        facets = self.fc3(facets)

        return facets


class ParameterModel(nn.Module):
    """
    Neural network module to encode simulation parameters.

    This module encodes physical parameters such as inlet velocity and air density
    into a learned latent representation that can be incorporated into the model's
    prediction process.
    """

    def __init__(self, input_features: int, model_parameters=None):
        """
        Initialize the parameter encoding network.

        Args:
            input_features: Number of input parameters to encode
            model_parameters: Configuration parameters for the model
        """
        super(ParameterModel, self).__init__()
        self.fourier_features = model_parameters.fourier_features
        self.num_modes = model_parameters.num_modes

        if self.fourier_features:
            input_features_calculated = (
                input_features + input_features * self.num_modes * 2
            )
        else:
            input_features_calculated = input_features

        base_layer = model_parameters.base_layer
        self.fc1 = nn.Linear(input_features_calculated, base_layer)
        self.fc2 = nn.Linear(base_layer, int(base_layer))
        self.fc3 = nn.Linear(int(base_layer), int(base_layer))
        self.bn1 = nn.BatchNorm1d(base_layer)
        self.bn2 = nn.BatchNorm1d(int(base_layer))
        self.bn3 = nn.BatchNorm1d(int(base_layer))

        self.activation = F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode physical parameters into a latent representation.

        Args:
            x: Input tensor containing physical parameters (e.g., inlet velocity, air density)

        Returns:
            Tensor containing encoded parameter representation
        """
        if self.fourier_features:
            params = torch.cat((x, fourier_encode(x, self.num_modes)), axis=-1)
        else:
            params = x
        params = self.activation(self.fc1(params))
        params = self.activation(self.fc2(params))
        params = self.fc3(params)

        return params


class AggregationModel(nn.Module):
    """
    Neural network module to aggregate local geometry encoding with basis functions.

    This module combines basis function representations with geometry encodings
    to predict the final output quantities. It serves as the final prediction layer
    that integrates all available information sources.
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        model_parameters=None,
        new_change: bool = True,
    ):
        """
        Initialize the aggregation model.

        Args:
            input_features: Number of input feature dimensions
            output_features: Number of output feature dimensions
            model_parameters: Configuration parameters for the model
            new_change: Flag to enable newer implementation (default: True)
        """
        super(AggregationModel, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.new_change = new_change
        base_layer = model_parameters.base_layer
        self.fc1 = nn.Linear(self.input_features, base_layer)
        self.fc2 = nn.Linear(base_layer, int(base_layer))
        self.fc3 = nn.Linear(int(base_layer), int(base_layer))
        self.fc4 = nn.Linear(int(base_layer), int(base_layer))
        self.fc5 = nn.Linear(int(base_layer), self.output_features)
        self.bn1 = nn.BatchNorm1d(base_layer)
        self.bn2 = nn.BatchNorm1d(int(base_layer))
        self.bn3 = nn.BatchNorm1d(int(base_layer))
        self.bn4 = nn.BatchNorm1d(int(base_layer))
        self.activation = F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the combined input features to predict output quantities.

        This method applies a series of fully connected layers to the input,
        which typically contains a combination of basis functions, geometry
        encodings, and potentially parameter encodings.

        Args:
            x: Input tensor containing combined features

        Returns:
            Tensor containing predicted output quantities
        """
        out = self.activation(self.fc1(x))
        out = self.activation(self.fc2(out))
        out = self.activation(self.fc3(out))
        out = self.activation(self.fc4(out))

        out = self.fc5(out)

        return out


class LocalPointConv(nn.Module):
    """Layer for local geometry point kernel"""

    def __init__(
        self,
        input_features,
        base_layer,
        output_features,
        model_parameters=None,
        new_change=True,
    ):
        super(LocalPointConv, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.fc1 = nn.Linear(self.input_features, base_layer)
        self.fc2 = nn.Linear(base_layer, self.output_features)
        self.activation = F.relu

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(out)

        return out


# @dataclass
# class MetaData(ModelMetaData):
#     name: str = "DoMINO"
#     # Optimization
#     jit: bool = False
#     cuda_graphs: bool = True
#     amp: bool = True
#     # Inference
#     onnx_cpu: bool = True
#     onnx_gpu: bool = True
#     onnx_runtime: bool = True
#     # Physics informed
#     var_dim: int = 1
#     func_torch: bool = False
#     auto_grad: bool = False


class DoMINO(nn.Module):
    """
    DoMINO model architecture for predicting both surface and volume quantities.

    The DoMINO (Deep Operational Modal Identification and Nonlinear Optimization) model
    is designed to model both surface and volume physical quantities in aerodynamic
    simulations. It can operate in three modes:
    1. Surface-only: Predicting only surface quantities
    2. Volume-only: Predicting only volume quantities
    3. Combined: Predicting both surface and volume quantities

    The model uses a combination of:
    - Geometry representation modules
    - Neural network basis functions
    - Parameter encoding
    - Local and global geometry processing
    - Aggregation models for final prediction

    Parameters
    ----------
    input_features : int
        Number of point input features
    output_features_vol : int, optional
        Number of output features in volume
    output_features_surf : int, optional
        Number of output features on surface
    model_parameters
        Model parameters controlled by config.yaml

    Example
    -------
    >>> from physicsnemo.models.domino.model import DoMINO
    >>> import torch, os
    >>> from hydra import compose, initialize
    >>> from omegaconf import OmegaConf
    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> cfg = OmegaConf.register_new_resolver("eval", eval)
    >>> with initialize(version_base="1.3", config_path="examples/cfd/external_aerodynamics/domino/src/conf"):
    ...    cfg = compose(config_name="config")
    >>> cfg.model.model_type = "combined"
    >>> model = DoMINO(
    ...         input_features=3,
    ...         output_features_vol=5,
    ...         output_features_surf=4,
    ...         model_parameters=cfg.model
    ...     ).to(device)

    Warp ...
    >>> bsize = 1
    >>> nx, ny, nz = 128, 64, 48
    >>> num_neigh = 7
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
    >>> surface_sizes = torch.randn(bsize, 100, 3).to(device)
    >>> surface_neighbors_areas = torch.rand(bsize, 100, num_neigh).to(device)
    >>> volume_coordinates = torch.randn(bsize, 100, 3).to(device)
    >>> vol_grid_max_min = torch.randn(bsize, 2, 3).to(device)
    >>> surf_grid_max_min = torch.randn(bsize, 2, 3).to(device)
    >>> stream_velocity = torch.randn(bsize, 1).to(device)
    >>> air_density = torch.randn(bsize, 1).to(device)
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
    ...            "stream_velocity": stream_velocity,
    ...            "air_density": air_density,
    ...        }
    >>> output = model(input_dict)
    Module ...
    >>> print(f"{output[0].shape}, {output[1].shape}")
    torch.Size([1, 100, 5]), torch.Size([1, 100, 4])
    """

    def __init__(
        self,
        input_features: int,
        output_features_vol: int | None = None,
        output_features_surf: int | None = None,
        model_parameters=None,
    ):
        """
        Initialize the DoMINO model.

        Args:
            input_features: Number of input feature dimensions for point data
            output_features_vol: Number of output features for volume quantities (None for surface-only mode)
            output_features_surf: Number of output features for surface quantities (None for volume-only mode)
            model_parameters: Configuration parameters for the model

        Raises:
            ValueError: If both output_features_vol and output_features_surf are None
        """
        super().__init__()
        self.input_features = input_features
        self.output_features_vol = output_features_vol
        self.output_features_surf = output_features_surf

        if self.output_features_vol is None and self.output_features_surf is None:
            raise ValueError(
                "At least one of `output_features_vol` or `output_features_surf` must be specified"
            )

        self.num_variables_vol = output_features_vol
        self.num_variables_surf = output_features_surf
        self.grid_resolution = model_parameters.interp_res
        self.surface_neighbors = model_parameters.surface_neighbors
        self.use_surface_normals = model_parameters.use_surface_normals
        self.use_surface_area = model_parameters.use_surface_area
        self.encode_parameters = model_parameters.encode_parameters
        self.param_scaling_factors = model_parameters.parameter_model.scaling_params
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
            self.parameter_model = ParameterModel(
                input_features=2, model_parameters=model_parameters.parameter_model
            )
        else:
            base_layer_p = 0

        self.geo_rep_volume = GeometryRep(
            input_features=input_features,
            radii=model_parameters.geometry_rep.geo_conv.volume_radii,
            model_parameters=model_parameters,
        )

        self.geo_rep_surface = GeometryRep(
            input_features=input_features,
            radii=model_parameters.geometry_rep.geo_conv.surface_radii,
            model_parameters=model_parameters,
        )

        self.geo_rep_surface1 = GeometryRep(
            input_features=input_features,
            radii=model_parameters.geometry_rep.geo_conv.volume_radii,
            model_parameters=model_parameters,
        )

        # Basis functions for surface and volume
        base_layer_nn = model_parameters.nn_basis_functions.base_layer
        if self.output_features_surf is not None:
            self.nn_basis_surf = nn.ModuleList()
            for _ in range(self.num_variables_surf):
                self.nn_basis_surf.append(
                    NNBasisFunctions(
                        input_features=input_features_surface,
                        model_parameters=model_parameters.nn_basis_functions,
                    )
                )

        if self.output_features_vol is not None:
            self.nn_basis_vol = nn.ModuleList()
            for _ in range(self.num_variables_vol):
                self.nn_basis_vol.append(
                    NNBasisFunctions(
                        input_features=input_features,
                        model_parameters=model_parameters.nn_basis_functions,
                    )
                )

        # Positional encoding
        position_encoder_base_neurons = model_parameters.position_encoder.base_neurons
        self.activation = F.relu
        self.use_sdf_in_basis_func = model_parameters.use_sdf_in_basis_func
        if self.output_features_vol is not None:
            if model_parameters.positional_encoding:
                inp_pos_vol = 25 if model_parameters.use_sdf_in_basis_func else 12
            else:
                inp_pos_vol = 7 if model_parameters.use_sdf_in_basis_func else 3

            self.fc_p_vol = nn.Linear(inp_pos_vol, position_encoder_base_neurons)

        if self.output_features_surf is not None:
            if model_parameters.positional_encoding:
                inp_pos_surf = 12
            else:
                inp_pos_surf = 3

            self.fc_p_surf = nn.Linear(inp_pos_surf, position_encoder_base_neurons)

        # Positional encoding hidden layers
        self.fc_p1 = nn.Linear(
            position_encoder_base_neurons, position_encoder_base_neurons
        )
        self.fc_p2 = nn.Linear(
            position_encoder_base_neurons, position_encoder_base_neurons
        )

        # base_layer_geo = model_parameters.geometry_local.base_layer

        # BQ for surface
        self.surface_neighbors_in_radius = (
            model_parameters.geometry_local.surface_neighbors_in_radius
        )
        self.surface_radius = model_parameters.geometry_local.surface_radii
        self.surface_bq_warp = nn.ModuleList()
        self.surface_local_point_conv = nn.ModuleList()

        for ct, j in enumerate(self.surface_radius):
            if self.geo_encoding_type == "both":
                total_neighbors_in_radius = self.surface_neighbors_in_radius[ct] * (
                    len(model_parameters.geometry_rep.geo_conv.surface_radii) + 1
                )
            elif self.geo_encoding_type == "stl":
                total_neighbors_in_radius = self.surface_neighbors_in_radius[ct] * (
                    len(model_parameters.geometry_rep.geo_conv.surface_radii)
                )
            elif self.geo_encoding_type == "sdf":
                total_neighbors_in_radius = self.surface_neighbors_in_radius[ct]

            self.surface_bq_warp.append(
                BQWarp(
                    grid_resolution=model_parameters.interp_res,
                    radius=self.surface_radius[ct],
                    neighbors_in_radius=self.surface_neighbors_in_radius[ct],
                )
            )
            self.surface_local_point_conv.append(
                LocalPointConv(
                    input_features=total_neighbors_in_radius,
                    base_layer=512,
                    output_features=self.surface_neighbors_in_radius[ct],
                )
            )

        # BQ for volume
        self.volume_neighbors_in_radius = (
            model_parameters.geometry_local.volume_neighbors_in_radius
        )
        self.volume_radius = model_parameters.geometry_local.volume_radii
        self.volume_bq_warp = nn.ModuleList()
        self.volume_local_point_conv = nn.ModuleList()

        for ct, j in enumerate(self.volume_radius):
            if self.geo_encoding_type == "both":
                total_neighbors_in_radius = self.volume_neighbors_in_radius[ct] * (
                    len(model_parameters.geometry_rep.geo_conv.volume_radii) + 1
                )
            elif self.geo_encoding_type == "stl":
                total_neighbors_in_radius = self.volume_neighbors_in_radius[ct] * (
                    len(model_parameters.geometry_rep.geo_conv.volume_radii)
                )
            elif self.geo_encoding_type == "sdf":
                total_neighbors_in_radius = self.volume_neighbors_in_radius[ct]

            self.volume_bq_warp.append(
                BQWarp(
                    grid_resolution=model_parameters.interp_res,
                    radius=self.volume_radius[ct],
                    neighbors_in_radius=self.volume_neighbors_in_radius[ct],
                )
            )
            self.volume_local_point_conv.append(
                LocalPointConv(
                    input_features=total_neighbors_in_radius,
                    base_layer=512,
                    output_features=self.volume_neighbors_in_radius[ct],
                )
            )

        # Transmitting surface to volume
        self.surf_to_vol_conv1 = nn.Conv3d(
            len(model_parameters.geometry_rep.geo_conv.volume_radii) + 1,
            16,
            kernel_size=3,
            padding="same",
        )
        self.surf_to_vol_conv2 = nn.Conv3d(
            16,
            len(model_parameters.geometry_rep.geo_conv.volume_radii) + 1,
            kernel_size=3,
            padding="same",
        )

        # Aggregation model
        if self.output_features_surf is not None:
            # Surface
            base_layer_geo_surf = 0
            for j in self.surface_neighbors_in_radius:
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
                        model_parameters=model_parameters.aggregation_model,
                    )
                )

        if self.output_features_vol is not None:
            # Volume
            base_layer_geo_vol = 0
            for j in self.volume_neighbors_in_radius:
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
                        model_parameters=model_parameters.aggregation_model,
                    )
                )

    def position_encoder(
        self,
        encoding_node: torch.Tensor,
        eval_mode: Literal["surface", "volume"] = "volume",
    ) -> torch.Tensor:
        """
        Compute positional encoding for input points.

        Args:
            encoding_node: Tensor containing node position information
            eval_mode: Mode of evaluation, either "volume" or "surface"

        Returns:
            Tensor containing positional encoding features
        """
        if eval_mode == "volume":
            x = self.activation(self.fc_p_vol(encoding_node))
        elif eval_mode == "surface":
            x = self.activation(self.fc_p_surf(encoding_node))
        else:
            raise ValueError(
                f"`eval_mode` must be 'surface' or 'volume', got {eval_mode=}"
            )
        x = self.activation(self.fc_p1(x))
        x = self.fc_p2(x)
        return x

    def geo_encoding_local(
        self, encoding_g, volume_mesh_centers, p_grid, mode="volume"
    ):
        """Function to calculate local geometry encoding from global encoding"""

        if mode == "volume":
            radius = self.volume_radius
            bq_warp = self.volume_bq_warp
            point_conv = self.volume_local_point_conv
        elif mode == "surface":
            radius = self.surface_radius
            bq_warp = self.surface_bq_warp
            point_conv = self.surface_local_point_conv

        batch_size = volume_mesh_centers.shape[0]
        nx, ny, nz = (
            self.grid_resolution[0],
            self.grid_resolution[1],
            self.grid_resolution[2],
        )

        encoding_outer = []
        for p, q in enumerate(radius):
            p_grid = torch.reshape(p_grid, (batch_size, nx * ny * nz, 3))
            mapping, outputs = bq_warp[p](
                volume_mesh_centers, p_grid, reverse_mapping=False
            )
            mapping = mapping.type(torch.int64)
            mask = mapping != 0

            encoding_g_inner = []
            for j in range(encoding_g.shape[1]):
                geo_encoding = torch.reshape(
                    encoding_g[:, j], (batch_size, 1, nx * ny * nz)
                )
                geo_encoding_sampled = torch.index_select(
                    geo_encoding, 2, mapping.flatten()
                )
                geo_encoding_sampled = torch.reshape(geo_encoding_sampled, mask.shape)
                geo_encoding_sampled = geo_encoding_sampled * mask

                encoding_g_inner.append(geo_encoding_sampled)
            encoding_g_inner = torch.cat(encoding_g_inner, axis=2)
            encoding_g_inner = point_conv[p](encoding_g_inner)
            encoding_outer.append(encoding_g_inner)

        encoding_g = torch.cat(encoding_outer, axis=-1)

        return encoding_g

    def calculate_solution_with_neighbors(
        self,
        surface_mesh_centers,
        encoding_g,
        encoding_node,
        surface_mesh_neighbors,
        surface_normals,
        surface_neighbors_normals,
        surface_areas,
        surface_neighbors_areas,
        inlet_velocity,
        air_density,
    ):
        """Function to approximate solution given the neighborhood information"""
        num_variables = self.num_variables_surf
        nn_basis = self.nn_basis_surf
        agg_model = self.agg_model_surf
        num_sample_points = surface_mesh_neighbors.shape[2] + 1

        if self.encode_parameters:
            inlet_velocity = torch.unsqueeze(inlet_velocity, 1)
            inlet_velocity = inlet_velocity.expand(
                inlet_velocity.shape[0],
                surface_mesh_centers.shape[1],
                inlet_velocity.shape[2],
            )
            inlet_velocity = inlet_velocity / self.param_scaling_factors[0]

            air_density = torch.unsqueeze(air_density, 1)
            air_density = air_density.expand(
                air_density.shape[0],
                surface_mesh_centers.shape[1],
                air_density.shape[2],
            )
            air_density = air_density / self.param_scaling_factors[1]

            params = torch.cat((inlet_velocity, air_density), axis=-1)
            param_encoding = self.parameter_model(params)

        if self.use_surface_normals:
            if not self.use_surface_area:
                surface_mesh_centers = torch.cat(
                    (surface_mesh_centers, surface_normals),
                    axis=-1,
                )
                surface_mesh_neighbors = torch.cat(
                    (
                        surface_mesh_neighbors,
                        surface_neighbors_normals,
                    ),
                    axis=-1,
                )

            else:
                surface_mesh_centers = torch.cat(
                    (
                        surface_mesh_centers,
                        surface_normals,
                        torch.log(surface_areas) / 10,
                    ),
                    axis=-1,
                )
                surface_mesh_neighbors = torch.cat(
                    (
                        surface_mesh_neighbors,
                        surface_neighbors_normals,
                        torch.log(surface_neighbors_areas) / 10,
                    ),
                    axis=-1,
                )

        for f in range(num_variables):
            for p in range(num_sample_points):
                if p == 0:
                    volume_m_c = surface_mesh_centers
                else:
                    volume_m_c = surface_mesh_neighbors[:, :, p - 1] + 1e-6
                    noise = surface_mesh_centers - volume_m_c
                    dist = torch.sqrt(
                        noise[:, :, 0:1] ** 2.0
                        + noise[:, :, 1:2] ** 2.0
                        + noise[:, :, 2:3] ** 2.0
                    )
                basis_f = nn_basis[f](volume_m_c)
                output = torch.cat((basis_f, encoding_node, encoding_g), axis=-1)
                if self.encode_parameters:
                    output = torch.cat((output, param_encoding), axis=-1)
                if p == 0:
                    output_center = agg_model[f](output)
                else:
                    if p == 1:
                        output_neighbor = agg_model[f](output) * (1.0 / dist)
                        dist_sum = 1.0 / dist
                    else:
                        output_neighbor += agg_model[f](output) * (1.0 / dist)
                        dist_sum += 1.0 / dist
            if num_sample_points > 1:
                output_res = 0.5 * output_center + 0.5 * output_neighbor / dist_sum
            else:
                output_res = output_center
            if f == 0:
                output_all = output_res
            else:
                output_all = torch.cat((output_all, output_res), axis=-1)

        return output_all

    def calculate_solution(
        self,
        volume_mesh_centers,
        encoding_g,
        encoding_node,
        inlet_velocity,
        air_density,
        eval_mode,
        num_sample_points=10,
        noise_intensity=50,
    ):
        """Function to approximate solution sampling the neighborhood information"""
        if eval_mode == "volume":
            num_variables = self.num_variables_vol
            nn_basis = self.nn_basis_vol
            agg_model = self.agg_model_vol
        elif eval_mode == "surface":
            num_variables = self.num_variables_surf
            nn_basis = self.nn_basis_surf
            agg_model = self.agg_model_surf

        if self.encode_parameters:
            inlet_velocity = torch.unsqueeze(inlet_velocity, 1)
            inlet_velocity = inlet_velocity.expand(
                inlet_velocity.shape[0],
                volume_mesh_centers.shape[1],
                inlet_velocity.shape[2],
            )
            inlet_velocity = inlet_velocity / self.param_scaling_factors[0]

            air_density = torch.unsqueeze(air_density, 1)
            air_density = air_density.expand(
                air_density.shape[0], volume_mesh_centers.shape[1], air_density.shape[2]
            )
            air_density = air_density / self.param_scaling_factors[1]

            params = torch.cat((inlet_velocity, air_density), axis=-1)
            param_encoding = self.parameter_model(params)

        for f in range(num_variables):
            for p in range(num_sample_points):
                if p == 0:
                    volume_m_c = volume_mesh_centers
                else:
                    noise = torch.rand_like(volume_mesh_centers)
                    noise = 2 * (noise - 0.5)
                    noise = noise / noise_intensity
                    dist = torch.sqrt(
                        noise[:, :, 0:1] ** 2.0
                        + noise[:, :, 1:2] ** 2.0
                        + noise[:, :, 2:3] ** 2.0
                    )
                    volume_m_c = volume_mesh_centers + noise
                basis_f = nn_basis[f](volume_m_c)
                output = torch.cat((basis_f, encoding_node, encoding_g), axis=-1)
                if self.encode_parameters:
                    output = torch.cat((output, param_encoding), axis=-1)
                if p == 0:
                    output_center = agg_model[f](output)
                else:
                    if p == 1:
                        output_neighbor = agg_model[f](output) * (1.0 / dist)
                        dist_sum = 1.0 / dist
                    else:
                        output_neighbor += agg_model[f](output) * (1.0 / dist)
                        dist_sum += 1.0 / dist
            if num_sample_points > 1:
                output_res = 0.5 * output_center + 0.5 * output_neighbor / dist_sum
            else:
                output_res = output_center
            if f == 0:
                output_all = output_res
            else:
                output_all = torch.cat((output_all, output_res), axis=-1)

        return output_all

    def forward(
        self,
        data_dict,
    ):
        # Loading STL inputs, bounding box grids, precomputed SDF and scaling factors

        # STL nodes
        geo_centers = data_dict["geometry_coordinates"]

        # Bounding box grid
        s_grid = data_dict["surf_grid"]
        sdf_surf_grid = data_dict["sdf_surf_grid"]
        # Scaling factors
        surf_max = data_dict["surface_min_max"][:, 1]
        surf_min = data_dict["surface_min_max"][:, 0]

        # Parameters
        stream_velocity = data_dict["stream_velocity"]
        air_density = data_dict["air_density"]

        if self.output_features_vol is not None:
            # Represent geometry on computational grid
            # Computational domain grid
            p_grid = data_dict["grid"]
            sdf_grid = data_dict["sdf_grid"]
            # Scaling factors
            vol_max = data_dict["volume_min_max"][:, 1]
            vol_min = data_dict["volume_min_max"][:, 0]

            # Normalize based on computational domain
            geo_centers_vol = 2.0 * (geo_centers - vol_min) / (vol_max - vol_min) - 1
            encoding_g_vol = self.geo_rep_volume(geo_centers_vol, p_grid, sdf_grid)

            # Normalize based on BBox around surface (car)
            geo_centers_surf = (
                2.0 * (geo_centers - surf_min) / (surf_max - surf_min) - 1
            )

            encoding_g_surf = self.geo_rep_surface1(
                geo_centers_surf, s_grid, sdf_surf_grid
            )

            encoding_g_vol += encoding_g_surf

            # SDF on volume mesh nodes
            sdf_nodes = data_dict["sdf_nodes"]
            # Positional encoding based on closest point on surface to a volume node
            pos_volume_closest = data_dict["pos_volume_closest"]
            # Positional encoding based on center of mass of geometry to volume node
            pos_volume_center_of_mass = data_dict["pos_volume_center_of_mass"]
            if self.use_sdf_in_basis_func:
                encoding_node_vol = torch.cat(
                    (sdf_nodes, pos_volume_closest, pos_volume_center_of_mass), axis=-1
                )
            else:
                encoding_node_vol = pos_volume_center_of_mass

            # Calculate positional encoding on volume nodes
            encoding_node_vol = self.position_encoder(
                encoding_node_vol, eval_mode="volume"
            )

        if self.output_features_surf is not None:
            # Represent geometry on bounding box
            geo_centers_surf = (
                2.0 * (geo_centers - surf_min) / (surf_max - surf_min) - 1
            )
            encoding_g_surf = self.geo_rep_surface(
                geo_centers_surf, s_grid, sdf_surf_grid
            )

            # Positional encoding based on center of mass of geometry to surface node
            pos_surface_center_of_mass = data_dict["pos_surface_center_of_mass"]
            encoding_node_surf = pos_surface_center_of_mass

            # Calculate positional encoding on surface centers
            encoding_node_surf = self.position_encoder(
                encoding_node_surf, eval_mode="surface"
            )

        if self.output_features_vol is not None:
            # Calculate local geometry encoding for volume
            # Sampled points on volume
            volume_mesh_centers = data_dict["volume_mesh_centers"]
            encoding_g_vol = self.geo_encoding_local(
                0.5 * encoding_g_vol, volume_mesh_centers, p_grid, mode="volume"
            )

            # Approximate solution on volume node
            output_vol = self.calculate_solution(
                volume_mesh_centers,
                encoding_g_vol,
                encoding_node_vol,
                stream_velocity,
                air_density,
                eval_mode="volume",
            )
        else:
            output_vol = None

        if self.output_features_surf is not None:
            # Sampled points on surface
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
            encoding_g_surf = self.geo_encoding_local(
                0.5 * encoding_g_surf, surface_mesh_centers, s_grid, mode="surface"
            )

            # Approximate solution on surface cell center
            if not self.surface_neighbors:
                output_surf = self.calculate_solution(
                    surface_mesh_centers,
                    encoding_g_surf,
                    encoding_node_surf,
                    stream_velocity,
                    air_density,
                    eval_mode="surface",
                    num_sample_points=1,
                    noise_intensity=500,
                )
            else:
                output_surf = self.calculate_solution_with_neighbors(
                    surface_mesh_centers,
                    encoding_g_surf,
                    encoding_node_surf,
                    surface_mesh_neighbors,
                    surface_normals,
                    surface_neighbors_normals,
                    surface_areas,
                    surface_neighbors_areas,
                    stream_velocity,
                    air_density,
                )
        else:
            output_surf = None

        return output_vol, output_surf
