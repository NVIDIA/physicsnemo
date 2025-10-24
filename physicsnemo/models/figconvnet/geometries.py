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

# ruff: noqa: S101,F722
import enum
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


def grid_init(bb_max, bb_min, resolution):
    """grid_init."""

    # Define grid points
    grid = torch.meshgrid(
        torch.linspace(bb_min[0], bb_max[0], resolution[0]),
        torch.linspace(bb_min[1], bb_max[1], resolution[1]),
        torch.linspace(bb_min[2], bb_max[2], resolution[2]),
    )
    grid = torch.stack(grid, dim=-1)  # (n_x, n_y, n_z, 3)
    return grid


class PointFeatures:
    """
    PointFeatures class represents the features defined on a set of points in 3D space.
    The vertices are the set of 3D coordinates that define the points and the features
    are defined at each point.

    The point features have BxNx3 and BxNxC shape where B is the batch size, N is the
    number of points, and C is the number of channels in the features.
    """

    _shape_hint = None  # default value
    vertices: Float[Tensor, "B N 3"]
    features: Float[Tensor, "B N C"]
    num_channels: int = None
    num_points: int = None

    def __class_getitem__(cls, item: str):
        # Create a new subclass with the shape hint set
        class _PointFeaturesSubclass(cls):
            _shape_hint = tuple(item.split())

        return _PointFeaturesSubclass

    def __init__(self, vertices, features):
        self.vertices = vertices
        self.features = features
        self.check()
        self.batch_size = len(self.vertices)
        self.num_points = self.vertices.shape[1]
        self.num_channels = self.features.shape[-1]

    @property
    def device(self):
        return self.vertices.device

    def check(self):
        assert self.vertices.ndim == 3
        assert self.features.ndim == 3
        assert self.vertices.shape[0] == self.features.shape[0]
        assert self.vertices.shape[1] == self.features.shape[1]
        assert self.vertices.shape[2] == 3

    def to(self, device):
        self.vertices = self.vertices.to(device)
        self.features = self.features.to(device)
        return self

    def expand_batch_size(self, batch_size: int):
        if batch_size == 1:
            return self

        # contiguous tensor is required for view operation
        self.vertices = self.vertices.expand(batch_size, -1, -1).contiguous()
        self.features = self.features.expand(batch_size, -1, -1).contiguous()
        self.batch_size = batch_size
        return self

    def voxel_down_sample(self, voxel_size: float):
        down_vertices = []
        down_features = []
        for vert, feat in zip(self.vertices, self.features):
            assert len(vert.shape) == 2
            assert vert.shape[1] == 3
            int_coords = torch.floor((vert) / voxel_size).int()
            # use unique to get the index of the in coord torch.unique(int_coords, dim=0, return_inverse=True)
            _, unique_indices = np.unique(
                int_coords.cpu().numpy(), axis=0, return_index=True
            )
            unique_indices = torch.from_numpy(unique_indices).to(self.vertices.device)
            down_vertices.append(vert[unique_indices])
            down_features.append(feat[unique_indices])
        # Clip the length of the downsampled vertices to the minimum length of the batch
        min_len = min([len(vert) for vert in down_vertices])
        down_vertices = torch.stack([vert[:min_len] for vert in down_vertices], dim=0)
        down_features = torch.stack([feat[:min_len] for feat in down_features], dim=0)
        return PointFeatures(down_vertices, down_features)

    def contiguous(self):
        self.vertices = self.vertices.contiguous()
        self.features = self.features.contiguous()
        return self

    def __add__(self, other):
        assert self.batch_size == other.batch_size
        assert self.num_channels == other.num_channels
        return PointFeatures(self.vertices, self.features + other.features)

    def __mul__(self, other):
        assert self.batch_size == other.batch_size
        assert self.num_channels == other.num_channels
        return PointFeatures(self.vertices, self.features * other.features)

    def __len__(self):
        return self.batch_size

    def __repr__(self) -> str:
        return f"PointFeatures(vertices={self.vertices.shape}, features={self.features.shape})"


class GridFeaturesMemoryFormat(enum.Enum):
    """Memory format used for GridFeatures class.

    The memory format defines how the grid features are stored in memory.

    b_x_y_z_c: Batch, X, Y, Z, Channels (3D Grid)
    b_c_x_y_z: Batch, Channels, X, Y, Z (3D Grid)
    b_zc_x_y: Batch, Z * Channels, X, Y (2D Grid)
    b_xc_y_z: Batch, X * Channels, Y, Z (2D Grid)
    b_yc_x_z: Batch, Y * Channels, X, Z (2D Grid)
    """

    b_x_y_z_c = enum.auto()
    b_c_x_y_z = enum.auto()

    # flattened 3D memory format
    b_zc_x_y = enum.auto()
    b_xc_y_z = enum.auto()
    b_yc_x_z = enum.auto()


grid_mem_format2str_format = {
    GridFeaturesMemoryFormat.b_x_y_z_c: "b_x_y_z_c",
    GridFeaturesMemoryFormat.b_c_x_y_z: "b_c_x_y_z",
    GridFeaturesMemoryFormat.b_zc_x_y: "b_zc_x_y",
    GridFeaturesMemoryFormat.b_xc_y_z: "b_xc_y_z",
    GridFeaturesMemoryFormat.b_yc_x_z: "b_yc_x_z",
}


def convert_to_b_x_y_z_c(tensor, from_memory_format, num_channels):
    if from_memory_format == GridFeaturesMemoryFormat.b_zc_x_y:
        B, D_C, H, W = tensor.shape
        D, rem = divmod(D_C, num_channels)
        assert rem == 0, "Number of channels does not match."
        return tensor.reshape(B, D, num_channels, H, W).permute(0, 3, 4, 1, 2)
    elif from_memory_format == GridFeaturesMemoryFormat.b_xc_y_z:
        B, H_C, W, D = tensor.shape
        H, rem = divmod(H_C, num_channels)
        assert rem == 0, "Number of channels does not match."
        return tensor.reshape(B, H, num_channels, W, D).permute(0, 1, 3, 4, 2)
    elif from_memory_format == GridFeaturesMemoryFormat.b_yc_x_z:
        B, W_C, H, D = tensor.shape
        W, rem = divmod(W_C, num_channels)
        assert rem == 0, "Number of channels does not match."
        return tensor.reshape(B, W, num_channels, H, D).permute(0, 3, 1, 4, 2)
    elif from_memory_format == GridFeaturesMemoryFormat.b_c_x_y_z:
        return tensor.permute(0, 2, 3, 4, 1)
    else:
        raise ValueError(f"Unsupported memory format: {from_memory_format}")


def convert_from_b_x_y_z_c(tensor, to_memory_format):
    B, H, W, D, C = tensor.shape
    if to_memory_format == GridFeaturesMemoryFormat.b_zc_x_y:
        return tensor.permute(0, 3, 4, 1, 2).reshape(B, D * C, H, W)
    elif to_memory_format == GridFeaturesMemoryFormat.b_xc_y_z:
        return tensor.permute(0, 1, 4, 2, 3).reshape(B, H * C, W, D)
    elif to_memory_format == GridFeaturesMemoryFormat.b_yc_x_z:
        return tensor.permute(0, 2, 4, 1, 3).reshape(B, W * C, H, D)
    elif to_memory_format == GridFeaturesMemoryFormat.b_c_x_y_z:
        return tensor.permute(0, 4, 1, 2, 3)
    else:
        raise ValueError(f"Unsupported memory format: {to_memory_format}")


class GridFeatures:
    """
    Dense features defined on a grid. The vertices are the set of 3D coordinates
    that define grid points and the features are defined at each grid point.

    The grid features have BxCxHxWxD shape where B is the batch size, C is the number
    of channels, H, W, D are the spatial dimensions of the grid.

    In Factorized Implicit Global ConvNet (FIGConvNet), we use different memory formats
    to store the grid features that compresses one low resolution spatial dimension
    into the channel dimension.
    """

    memory_format: GridFeaturesMemoryFormat

    def __init__(
        self,
        vertices,
        features,
        memory_format: GridFeaturesMemoryFormat = GridFeaturesMemoryFormat.b_x_y_z_c,
        grid_shape: Optional[Tuple[int, int, int]] = None,
        num_channels: Optional[int] = None,
    ) -> None:
        self.memory_format = memory_format
        self.vertices = vertices
        self.features = features
        self.check()

        if memory_format == GridFeaturesMemoryFormat.b_x_y_z_c:
            B, H, W, D, C = self.features.shape
            self.grid_shape = (H, W, D)
            self.num_channels = C
        else:
            assert grid_shape is not None, "grid_shape must be provided."
            assert num_channels is not None, "num_channels must be provided."

            self.grid_shape = grid_shape
            self.num_channels = num_channels
            self.memory_format = memory_format
        self.batch_size = len(features)

    @staticmethod
    def from_conv_output(
        conv_output, vertices, memory_format, grid_shape, num_channels
    ):
        """Initialize GridFeatures from the output of a convolutional layer.

        Args:
            conv_output: Output tensor from a convolutional layer.
            vertices: The vertices to associate with the grid features.
            memory_format: The memory format of the grid features.
            num_spatial_dims: Tuple of spatial dimensions (H, W, D).
            num_channels: The number of output channels from the convolution.

        Returns:
            GridFeatures: A new GridFeatures object with the given memory format.
        """
        # Infer spatial dimensions based on the memory format
        rem = 0
        if memory_format == GridFeaturesMemoryFormat.b_zc_x_y:
            B, DC, H, W = conv_output.shape
            D, rem = divmod(DC, num_channels)
            assert D == grid_shape[2], "Spatial dimension D does not match."
        elif memory_format == GridFeaturesMemoryFormat.b_xc_y_z:
            B, HC, W, D = conv_output.shape
            H, rem = divmod(HC, num_channels)
            assert H == grid_shape[0], "Spatial dimension H does not match."
        elif memory_format == GridFeaturesMemoryFormat.b_yc_x_z:
            B, WC, H, D = conv_output.shape
            W, rem = divmod(WC, num_channels)
            assert W == grid_shape[1], "Spatial dimension W does not match."
        elif memory_format == GridFeaturesMemoryFormat.b_c_x_y_z:
            B, C, H, W, D = conv_output.shape
            assert C == num_channels, "Number of channels does not match."
        else:
            raise ValueError("Unsupported memory format.")
        assert rem == 0, "Number of channels does not match."

        # The channel dimension is now the last dimension after reshaping.
        return GridFeatures(
            vertices=vertices,
            features=conv_output,
            memory_format=memory_format,
            grid_shape=grid_shape,
            num_channels=num_channels,
        )

    def channel_size(self, memory_format: GridFeaturesMemoryFormat = None):
        if memory_format is None:
            memory_format = self.memory_format
        if memory_format == GridFeaturesMemoryFormat.b_x_y_z_c:
            return self.num_channels
        elif memory_format == GridFeaturesMemoryFormat.b_c_x_y_z:
            return self.num_channels
        elif memory_format == GridFeaturesMemoryFormat.b_xc_y_z:
            return self.num_channels * self.grid_shape[0]
        elif memory_format == GridFeaturesMemoryFormat.b_yc_x_z:
            return self.num_channels * self.grid_shape[1]
        elif memory_format == GridFeaturesMemoryFormat.b_zc_x_y:
            return self.num_channels * self.grid_shape[2]

    def check(self):
        assert self.vertices.ndim == 5
        assert self.vertices.shape[-1] == 3

        spatial_dims = self.vertices.shape[-4:-1]
        if self.memory_format == GridFeaturesMemoryFormat.b_x_y_z_c:
            # Last 4 dim of vertices should be H,W,D,3 which should match the features
            assert self.features.ndim == 5
            assert spatial_dims == self.features.shape[1:4]
            assert self.vertices.shape[-1] == 3
        elif self.memory_format == GridFeaturesMemoryFormat.b_c_x_y_z:
            assert self.features.ndim == 5
            # pass the check for strided outputs
            # assert self.vertices.shape[0] == self.features.shape[1]
            # assert self.vertices.shape[1] == self.features.shape[2]
            # assert self.vertices.shape[2] == self.features.shape[3]
        else:
            assert self.features.ndim == 4

    @property
    def batch_features(self) -> Float[Tensor, "B C H W D"]:
        if self.memory_format == GridFeaturesMemoryFormat.b_x_y_z_c:
            return self.features.permute(0, 4, 1, 2, 3)
        elif self.memory_format == GridFeaturesMemoryFormat.b_c_x_y_z:
            return self.features
        else:
            raise ValueError("Unsupported memory format.")

    @property
    def point_features(self) -> PointFeatures:
        """Convert the grid features to point features."""
        if self.memory_format == GridFeaturesMemoryFormat.b_c_x_y_z:
            permuted_features = self.features.permute(0, 2, 3, 4, 1)
        elif self.memory_format == GridFeaturesMemoryFormat.b_x_y_z_c:
            # Crop the features to the grid_shape
            grid_shape = self.grid_shape
            if (
                self.features.shape[1] > grid_shape[0]
                or self.features.shape[2] > grid_shape[1]
                or self.features.shape[3] > grid_shape[2]
            ):
                permuted_features = self.features[
                    :, : grid_shape[0], : grid_shape[1], : grid_shape[2]
                ]
            else:
                permuted_features = self.features

        return PointFeatures(
            self.vertices.flatten(1, 3), permuted_features.flatten(1, 3)
        )

    @property
    def resolution(self) -> Tuple[int, int, int]:
        return self.grid_shape

    def to(
        self, device=None, memory_format: GridFeaturesMemoryFormat = None
    ):  # -> GridFeatures
        """
        Convert the GridFeatures to the given device and memory format.
        """
        assert device is not None or memory_format is not None
        if device is not None:
            self.vertices = self.vertices.to(device)
            self.features = self.features.to(device)

        if memory_format is not None:
            # Step 1: Convert to x_y_z_c format if not already in that format
            if self.memory_format != GridFeaturesMemoryFormat.b_x_y_z_c:
                self.features = convert_to_b_x_y_z_c(
                    self.features, self.memory_format, self.num_channels
                )

            # Step 2: Convert to the desired memory format from x_y_z_c
            if memory_format != GridFeaturesMemoryFormat.b_x_y_z_c:
                self.features = convert_from_b_x_y_z_c(self.features, memory_format)

            self.memory_format = memory_format  # Update the memory format
        return self

    def __repr__(self) -> str:
        return f"GridFeatures(vertices={self.vertices.shape}, features={self.features.shape})"

    def __add__(self, other):
        assert self.batch_size == other.batch_size
        assert self.features.shape == other.features.shape
        return GridFeatures(
            self.vertices,
            self.features + other.features,
            self.memory_format,
            grid_shape=self.grid_shape,
            num_channels=self.num_channels,
        )

    def strided_vertices(self, resolution: Tuple[int, int, int]):
        assert self.vertices.ndim == 5
        assert len(resolution) == 3
        if self.resolution == resolution:
            return self.vertices

        # Compute the stride
        if (
            self.resolution[0] % resolution[0] == 0
            and self.resolution[1] % resolution[1] == 0
            and self.resolution[2] % resolution[2] == 0
        ):
            stride = (
                self.resolution[0] // resolution[0],
                self.resolution[1] // resolution[1],
                self.resolution[2] // resolution[2],
            )
            vertices = self.vertices[:, :: stride[0], :: stride[1], :: stride[2]]
        else:
            # Use grid_sample 3D to interpolate the vertices
            # vertices have shape (H, W, D, 3)
            # create sample points using AABB
            grid_points = grid_init(
                bb_max=(1, 1, 1), bb_min=(-1, -1, -1), resolution=resolution
            )  # (res[0], res[1], res[2], 3)
            grid_points = grid_points.unsqueeze(0).to(self.vertices.device)

            sampled_vertices = F.grid_sample(
                self.vertices.permute(
                    0, 4, 1, 2, 3
                ),  # move the coordinates to the channel dim (1st)
                grid_points.expand(self.batch_size, -1, -1, -1, -1),
                align_corners=True,
            )
            vertices = sampled_vertices.permute(
                0, 2, 3, 4, 1
            )  # move the coordinates back to the last dim

        assert vertices.shape[-4:-1] == resolution
        return vertices
