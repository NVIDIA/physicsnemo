# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
r"""Geometry data structures for FIGConvNet.

This module provides core data structures for representing point-based and
grid-based features used in the Factorized Implicit Global Convolutional
Network (FIGConvNet) architecture.

The main classes are:

- :class:`PointFeatures`: Features defined on a set of 3D points
- :class:`GridFeatures`: Dense features defined on a regular 3D grid
- :class:`GridFeaturesMemoryFormat`: Enum defining memory layouts for grid features
"""

import enum
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


def grid_init(
    bb_max: Tuple[float, float, float],
    bb_min: Tuple[float, float, float],
    resolution: Tuple[int, int, int],
) -> Tensor:
    r"""Initialize a regular 3D grid of points within a bounding box.

    Creates a meshgrid of 3D coordinates spanning from ``bb_min`` to ``bb_max``
    with the specified resolution.

    Parameters
    ----------
    bb_max : Tuple[float, float, float]
        Maximum coordinates (x, y, z) of the bounding box.
    bb_min : Tuple[float, float, float]
        Minimum coordinates (x, y, z) of the bounding box.
    resolution : Tuple[int, int, int]
        Number of grid points along each axis (nx, ny, nz).

    Returns
    -------
    torch.Tensor
        Grid coordinates of shape :math:`(n_x, n_y, n_z, 3)` where the last
        dimension contains (x, y, z) coordinates.

    Examples
    --------
    >>> grid = grid_init((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (10, 10, 10))
    >>> grid.shape
    torch.Size([10, 10, 10, 3])
    """
    # Define grid points using meshgrid
    grid = torch.meshgrid(
        torch.linspace(bb_min[0], bb_max[0], resolution[0]),
        torch.linspace(bb_min[1], bb_max[1], resolution[1]),
        torch.linspace(bb_min[2], bb_max[2], resolution[2]),
        indexing="ij",
    )
    grid = torch.stack(grid, dim=-1)  # (n_x, n_y, n_z, 3)
    return grid


class PointFeatures:
    r"""Features defined on a set of 3D points.

    This class represents point cloud data with associated per-point features.
    The vertices define the 3D positions and features store the feature vectors
    at each point.

    Parameters
    ----------
    vertices : torch.Tensor
        3D coordinates of shape :math:`(B, N, 3)` where :math:`B` is batch size
        and :math:`N` is number of points.
    features : torch.Tensor
        Feature vectors of shape :math:`(B, N, C)` where :math:`C` is the
        number of feature channels.

    Attributes
    ----------
    vertices : torch.Tensor
        Point coordinates of shape :math:`(B, N, 3)`.
    features : torch.Tensor
        Point features of shape :math:`(B, N, C)`.
    batch_size : int
        Number of samples in the batch.
    num_points : int
        Number of points per sample.
    num_channels : int
        Number of feature channels.

    Examples
    --------
    >>> import torch
    >>> vertices = torch.randn(2, 1000, 3)  # 2 batches, 1000 points each
    >>> features = torch.randn(2, 1000, 64)  # 64 feature channels
    >>> pf = PointFeatures(vertices, features)
    >>> pf.batch_size
    2
    >>> pf.num_points
    1000

    Note
    ----
    The class supports arithmetic operations (``+``, ``*``) that operate on
    the features while preserving the vertices.
    """

    _shape_hint = None  # default value for type hint support
    vertices: Float[Tensor, "B N 3"]
    features: Float[Tensor, "B N C"]
    num_channels: int = None
    num_points: int = None

    def __class_getitem__(cls, item: str):
        """Support for parameterized type hints like PointFeatures["B N C"]."""

        # Create a new subclass with the shape hint set
        class _PointFeaturesSubclass(cls):
            _shape_hint = tuple(item.split())

        return _PointFeaturesSubclass

    def __init__(
        self,
        vertices: Float[Tensor, "batch num_points 3"],
        features: Float[Tensor, "batch num_points channels"],
    ) -> None:
        self.vertices = vertices
        self.features = features
        self.check()
        self.batch_size = len(self.vertices)
        self.num_points = self.vertices.shape[1]
        self.num_channels = self.features.shape[-1]

    @property
    def device(self) -> torch.device:
        """Return the device where tensors are stored."""
        return self.vertices.device

    def check(self) -> None:
        r"""Validate tensor shapes and consistency.

        Raises
        ------
        ValueError
            If tensor shapes are invalid or inconsistent.
        """
        if self.vertices.ndim != 3:
            raise ValueError(f"Expected 3D vertices tensor, got {self.vertices.ndim}D")
        if self.features.ndim != 3:
            raise ValueError(f"Expected 3D features tensor, got {self.features.ndim}D")
        if self.vertices.shape[0] != self.features.shape[0]:
            raise ValueError(
                f"Batch size mismatch: vertices {self.vertices.shape[0]} vs features {self.features.shape[0]}"
            )
        if self.vertices.shape[1] != self.features.shape[1]:
            raise ValueError(
                f"Number of points mismatch: vertices {self.vertices.shape[1]} vs features {self.features.shape[1]}"
            )
        if self.vertices.shape[2] != 3:
            raise ValueError(f"Expected 3D coordinates, got {self.vertices.shape[2]}D")

    def to(self, device: torch.device) -> "PointFeatures":
        r"""Move tensors to specified device.

        Parameters
        ----------
        device : torch.device
            Target device (e.g., 'cuda', 'cpu').

        Returns
        -------
        PointFeatures
            Self, with tensors moved to the target device.
        """
        self.vertices = self.vertices.to(device)
        self.features = self.features.to(device)
        return self

    def expand_batch_size(self, batch_size: int) -> "PointFeatures":
        r"""Expand tensors to a larger batch size.

        Parameters
        ----------
        batch_size : int
            Target batch size.

        Returns
        -------
        PointFeatures
            Self, with expanded batch dimension.
        """
        if batch_size == 1:
            return self

        # contiguous tensor is required for view operation
        self.vertices = self.vertices.expand(batch_size, -1, -1).contiguous()
        self.features = self.features.expand(batch_size, -1, -1).contiguous()
        self.batch_size = batch_size
        return self

    def voxel_down_sample(self, voxel_size: float) -> "PointFeatures":
        r"""Downsample points using voxel grid filtering.

        Groups points into voxel cells and keeps one point per cell.

        Parameters
        ----------
        voxel_size : float
            Size of the voxel grid cells.

        Returns
        -------
        PointFeatures
            Downsampled point features with reduced number of points.
        """
        down_vertices = []
        down_features = []

        # Process each batch element separately
        for vert, feat in zip(self.vertices, self.features):
            if len(vert.shape) != 2:
                raise ValueError(f"Expected 2D vertex tensor, got {len(vert.shape)}D")
            if vert.shape[1] != 3:
                raise ValueError(f"Expected 3D coordinates, got {vert.shape[1]}D")

            # Compute voxel indices for each point
            int_coords = torch.floor((vert) / voxel_size).int()

            # Get unique voxel indices
            _, unique_indices = np.unique(
                int_coords.cpu().numpy(), axis=0, return_index=True
            )
            unique_indices = torch.from_numpy(unique_indices).to(self.vertices.device)

            down_vertices.append(vert[unique_indices])
            down_features.append(feat[unique_indices])

        # Clip to minimum length across batch for consistent tensor shape
        min_len = min([len(vert) for vert in down_vertices])
        down_vertices = torch.stack([vert[:min_len] for vert in down_vertices], dim=0)
        down_features = torch.stack([feat[:min_len] for feat in down_features], dim=0)

        return PointFeatures(down_vertices, down_features)

    def contiguous(self) -> "PointFeatures":
        r"""Make tensors contiguous in memory.

        Returns
        -------
        PointFeatures
            Self, with contiguous tensors.
        """
        self.vertices = self.vertices.contiguous()
        self.features = self.features.contiguous()
        return self

    def __add__(self, other: "PointFeatures") -> "PointFeatures":
        """Add features element-wise, preserving vertices."""
        if self.batch_size != other.batch_size:
            raise ValueError(
                f"Batch size mismatch: {self.batch_size} vs {other.batch_size}"
            )
        if self.num_channels != other.num_channels:
            raise ValueError(
                f"Channel mismatch: {self.num_channels} vs {other.num_channels}"
            )
        return PointFeatures(self.vertices, self.features + other.features)

    def __mul__(self, other: "PointFeatures") -> "PointFeatures":
        """Multiply features element-wise, preserving vertices."""
        if self.batch_size != other.batch_size:
            raise ValueError(
                f"Batch size mismatch: {self.batch_size} vs {other.batch_size}"
            )
        if self.num_channels != other.num_channels:
            raise ValueError(
                f"Channel mismatch: {self.num_channels} vs {other.num_channels}"
            )
        return PointFeatures(self.vertices, self.features * other.features)

    def __len__(self) -> int:
        """Return the batch size."""
        return self.batch_size

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PointFeatures(vertices={self.vertices.shape}, features={self.features.shape})"


class GridFeaturesMemoryFormat(str, enum.Enum):
    r"""Memory format for GridFeatures storage.

    This enum defines different memory layouts for storing 3D grid features.
    The factorized formats compress one spatial dimension into the channel
    dimension, enabling efficient 2D convolution operations.

    This class inherits from both ``str`` and ``enum.Enum`` to enable JSON
    serialization. Enum values can be directly serialized with ``json.dumps()``
    and deserialized using ``GridFeaturesMemoryFormat(value)``.

    Attributes
    ----------
    b_x_y_z_c : str
        Standard 3D format: Batch, X, Y, Z, Channels. Shape: :math:`(B, X, Y, Z, C)`
    b_c_x_y_z : str
        PyTorch 3D conv format: Batch, Channels, X, Y, Z. Shape: :math:`(B, C, X, Y, Z)`
    b_zc_x_y : str
        Factorized format with Z compressed: Batch, Z*Channels, X, Y.
        Shape: :math:`(B, Z \cdot C, X, Y)`
    b_xc_y_z : str
        Factorized format with X compressed: Batch, X*Channels, Y, Z.
        Shape: :math:`(B, X \cdot C, Y, Z)`
    b_yc_x_z : str
        Factorized format with Y compressed: Batch, Y*Channels, X, Z.
        Shape: :math:`(B, Y \cdot C, X, Z)`

    Note
    ----
    The factorized formats (``b_zc_x_y``, ``b_xc_y_z``, ``b_yc_x_z``) are key to
    the FIGConvNet architecture. They allow representing 3D features as 2D
    feature maps, enabling efficient 2D convolutions that implicitly operate
    globally along the compressed dimension.
    """

    b_x_y_z_c = "b_x_y_z_c"
    b_c_x_y_z = "b_c_x_y_z"

    # Factorized (compressed) 3D to 2D memory formats
    b_zc_x_y = "b_zc_x_y"
    b_xc_y_z = "b_xc_y_z"
    b_yc_x_z = "b_yc_x_z"


# Mapping from memory format to string representation
grid_mem_format2str_format = {
    GridFeaturesMemoryFormat.b_x_y_z_c: "b_x_y_z_c",
    GridFeaturesMemoryFormat.b_c_x_y_z: "b_c_x_y_z",
    GridFeaturesMemoryFormat.b_zc_x_y: "b_zc_x_y",
    GridFeaturesMemoryFormat.b_xc_y_z: "b_xc_y_z",
    GridFeaturesMemoryFormat.b_yc_x_z: "b_yc_x_z",
}


def convert_to_b_x_y_z_c(
    tensor: Tensor,
    from_memory_format: GridFeaturesMemoryFormat,
    num_channels: int,
) -> Tensor:
    r"""Convert tensor from any format to b_x_y_z_c format.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor in the specified memory format.
    from_memory_format : GridFeaturesMemoryFormat
        Current memory format of the tensor.
    num_channels : int
        Number of feature channels.

    Returns
    -------
    torch.Tensor
        Tensor in b_x_y_z_c format with shape :math:`(B, X, Y, Z, C)`.

    Raises
    ------
    ValueError
        If the memory format is unsupported.
    """
    if from_memory_format == GridFeaturesMemoryFormat.b_zc_x_y:
        B, D_C, H, W = tensor.shape
        D, rem = divmod(D_C, num_channels)
        if rem != 0:
            raise ValueError("Number of channels does not match.")
        return tensor.reshape(B, D, num_channels, H, W).permute(0, 3, 4, 1, 2)

    elif from_memory_format == GridFeaturesMemoryFormat.b_xc_y_z:
        B, H_C, W, D = tensor.shape
        H, rem = divmod(H_C, num_channels)
        if rem != 0:
            raise ValueError("Number of channels does not match.")
        return tensor.reshape(B, H, num_channels, W, D).permute(0, 1, 3, 4, 2)

    elif from_memory_format == GridFeaturesMemoryFormat.b_yc_x_z:
        B, W_C, H, D = tensor.shape
        W, rem = divmod(W_C, num_channels)
        if rem != 0:
            raise ValueError("Number of channels does not match.")
        return tensor.reshape(B, W, num_channels, H, D).permute(0, 3, 1, 4, 2)

    elif from_memory_format == GridFeaturesMemoryFormat.b_c_x_y_z:
        return tensor.permute(0, 2, 3, 4, 1)

    else:
        raise ValueError(f"Unsupported memory format: {from_memory_format}")


def convert_from_b_x_y_z_c(
    tensor: Tensor,
    to_memory_format: GridFeaturesMemoryFormat,
) -> Tensor:
    r"""Convert tensor from b_x_y_z_c format to target format.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor in b_x_y_z_c format with shape :math:`(B, X, Y, Z, C)`.
    to_memory_format : GridFeaturesMemoryFormat
        Target memory format.

    Returns
    -------
    torch.Tensor
        Tensor in the target memory format.

    Raises
    ------
    ValueError
        If the target memory format is unsupported.
    """
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
    r"""Dense features defined on a regular 3D grid.

    This class represents volumetric feature data on a regular grid. It supports
    multiple memory formats, including factorized formats where one spatial
    dimension is compressed into the channel dimension for efficient 2D convolutions.

    Parameters
    ----------
    vertices : torch.Tensor
        Grid vertex coordinates of shape :math:`(B, X, Y, Z, 3)`.
    features : torch.Tensor
        Grid features. Shape depends on ``memory_format``:

        - ``b_x_y_z_c``: :math:`(B, X, Y, Z, C)`
        - ``b_c_x_y_z``: :math:`(B, C, X, Y, Z)`
        - ``b_zc_x_y``: :math:`(B, Z \cdot C, X, Y)`
        - ``b_xc_y_z``: :math:`(B, X \cdot C, Y, Z)`
        - ``b_yc_x_z``: :math:`(B, Y \cdot C, X, Z)`

    memory_format : GridFeaturesMemoryFormat, optional
        Memory layout of the features. Default is ``b_x_y_z_c``.
    grid_shape : Tuple[int, int, int], optional
        Grid resolution (X, Y, Z). Required for compressed formats.
    num_channels : int, optional
        Number of feature channels. Required for compressed formats.

    Attributes
    ----------
    vertices : torch.Tensor
        Grid vertex coordinates.
    features : torch.Tensor
        Grid feature values.
    memory_format : GridFeaturesMemoryFormat
        Current memory format.
    grid_shape : Tuple[int, int, int]
        Grid resolution (X, Y, Z).
    num_channels : int
        Number of feature channels.
    batch_size : int
        Batch size.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.models.figconvnet.geometries import GridFeatures
    >>> vertices = torch.randn(2, 32, 32, 32, 3)
    >>> features = torch.randn(2, 32, 32, 32, 64)
    >>> gf = GridFeatures(vertices, features)
    >>> gf.resolution
    (32, 32, 32)

    Note
    ----
    The factorized memory formats are central to the FIGConvNet architecture,
    enabling efficient implicit global convolutions through 2D operations.

    See Also
    --------
    :class:`GridFeaturesMemoryFormat`
    :class:`PointFeatures`
    """

    memory_format: GridFeaturesMemoryFormat

    def __init__(
        self,
        vertices: Tensor,
        features: Tensor,
        memory_format: GridFeaturesMemoryFormat = GridFeaturesMemoryFormat.b_x_y_z_c,
        grid_shape: Optional[Tuple[int, int, int]] = None,
        num_channels: Optional[int] = None,
    ) -> None:
        self.memory_format = memory_format
        self.vertices = vertices
        self.features = features
        self.check()

        # Infer grid shape and channels from tensor shape for standard format
        if memory_format == GridFeaturesMemoryFormat.b_x_y_z_c:
            B, H, W, D, C = self.features.shape
            self.grid_shape = (H, W, D)
            self.num_channels = C
        else:
            # For compressed formats, shape info must be provided
            if grid_shape is None:
                raise ValueError("grid_shape must be provided for compressed formats.")
            if num_channels is None:
                raise ValueError(
                    "num_channels must be provided for compressed formats."
                )
            self.grid_shape = grid_shape
            self.num_channels = num_channels
            self.memory_format = memory_format

        self.batch_size = len(features)

    @staticmethod
    def from_conv_output(
        conv_output: Tensor,
        vertices: Tensor,
        memory_format: GridFeaturesMemoryFormat,
        grid_shape: Tuple[int, int, int],
        num_channels: int,
    ) -> "GridFeatures":
        r"""Create GridFeatures from convolutional layer output.

        This factory method properly interprets the output of 2D convolutions
        operating on factorized grid representations.

        Parameters
        ----------
        conv_output : torch.Tensor
            Output tensor from a convolutional layer.
        vertices : torch.Tensor
            Grid vertex coordinates.
        memory_format : GridFeaturesMemoryFormat
            Memory format of the convolution output.
        grid_shape : Tuple[int, int, int]
            Original grid resolution (X, Y, Z).
        num_channels : int
            Number of output channels from the convolution.

        Returns
        -------
        GridFeatures
            New GridFeatures object with the given memory format.

        Raises
        ------
        ValueError
            If the memory format is unsupported or dimensions don't match.
        """
        # Verify spatial dimensions match the expected memory format
        rem = 0
        if memory_format == GridFeaturesMemoryFormat.b_zc_x_y:
            B, DC, H, W = conv_output.shape
            D, rem = divmod(DC, num_channels)
            if D != grid_shape[2]:
                raise ValueError("Spatial dimension D does not match.")

        elif memory_format == GridFeaturesMemoryFormat.b_xc_y_z:
            B, HC, W, D = conv_output.shape
            H, rem = divmod(HC, num_channels)
            if H != grid_shape[0]:
                raise ValueError("Spatial dimension H does not match.")

        elif memory_format == GridFeaturesMemoryFormat.b_yc_x_z:
            B, WC, H, D = conv_output.shape
            W, rem = divmod(WC, num_channels)
            if W != grid_shape[1]:
                raise ValueError("Spatial dimension W does not match.")

        elif memory_format == GridFeaturesMemoryFormat.b_c_x_y_z:
            B, C, H, W, D = conv_output.shape
            if C != num_channels:
                raise ValueError("Number of channels does not match.")

        else:
            raise ValueError("Unsupported memory format.")

        if rem != 0:
            raise ValueError("Number of channels does not match.")

        return GridFeatures(
            vertices=vertices,
            features=conv_output,
            memory_format=memory_format,
            grid_shape=grid_shape,
            num_channels=num_channels,
        )

    def channel_size(
        self, memory_format: Optional[GridFeaturesMemoryFormat] = None
    ) -> int:
        r"""Get the effective channel dimension size for a memory format.

        For factorized formats, this includes the compressed spatial dimension.

        Parameters
        ----------
        memory_format : GridFeaturesMemoryFormat, optional
            Memory format to compute channel size for. Uses current format if None.

        Returns
        -------
        int
            Effective channel dimension size.
        """
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

    def check(self) -> None:
        r"""Validate tensor shapes and consistency.

        Raises
        ------
        ValueError
            If tensor shapes are invalid.
        """
        if self.vertices.ndim != 5:
            raise ValueError(f"Expected 5D vertices tensor, got {self.vertices.ndim}D")
        if self.vertices.shape[-1] != 3:
            raise ValueError(f"Expected 3D coordinates, got {self.vertices.shape[-1]}D")

        spatial_dims = self.vertices.shape[-4:-1]

        if self.memory_format == GridFeaturesMemoryFormat.b_x_y_z_c:
            if self.features.ndim != 5:
                raise ValueError(
                    f"Expected 5D features tensor for b_x_y_z_c format, got {self.features.ndim}D"
                )
            if spatial_dims != self.features.shape[1:4]:
                raise ValueError(
                    f"Spatial dimensions mismatch: vertices {spatial_dims} vs features {self.features.shape[1:4]}"
                )
            if self.vertices.shape[-1] != 3:
                raise ValueError("Expected 3D coordinates for vertices")

        elif self.memory_format == GridFeaturesMemoryFormat.b_c_x_y_z:
            if self.features.ndim != 5:
                raise ValueError(
                    f"Expected 5D features tensor for b_c_x_y_z format, got {self.features.ndim}D"
                )
        else:
            # Factorized formats have 4D feature tensors
            if self.features.ndim != 4:
                raise ValueError(
                    f"Expected 4D features tensor for factorized format, got {self.features.ndim}D"
                )

    @property
    def batch_features(self) -> Float[Tensor, "B C H W D"]:
        r"""Get features in b_c_x_y_z format.

        Returns
        -------
        torch.Tensor
            Features of shape :math:`(B, C, X, Y, Z)`.

        Raises
        ------
        ValueError
            If current format cannot be converted.
        """
        if self.memory_format == GridFeaturesMemoryFormat.b_x_y_z_c:
            return self.features.permute(0, 4, 1, 2, 3)
        elif self.memory_format == GridFeaturesMemoryFormat.b_c_x_y_z:
            return self.features
        else:
            raise ValueError("Unsupported memory format.")

    @property
    def point_features(self) -> PointFeatures:
        r"""Convert grid features to point features.

        Flattens the grid structure to create a PointFeatures object where
        each grid cell becomes a point.

        Returns
        -------
        PointFeatures
            Point features with flattened grid data.
        """
        if self.memory_format == GridFeaturesMemoryFormat.b_c_x_y_z:
            permuted_features = self.features.permute(0, 2, 3, 4, 1)
        elif self.memory_format == GridFeaturesMemoryFormat.b_x_y_z_c:
            # Crop features to grid_shape if needed
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
        else:
            raise ValueError(
                f"Cannot convert {self.memory_format} directly to point features"
            )

        return PointFeatures(
            self.vertices.flatten(1, 3), permuted_features.flatten(1, 3)
        )

    @property
    def resolution(self) -> Tuple[int, int, int]:
        r"""Get the grid resolution.

        Returns
        -------
        Tuple[int, int, int]
            Grid resolution (X, Y, Z).
        """
        return self.grid_shape

    def to(
        self,
        device: Optional[torch.device] = None,
        memory_format: Optional[GridFeaturesMemoryFormat] = None,
    ) -> "GridFeatures":
        r"""Move to device and/or convert memory format.

        Parameters
        ----------
        device : torch.device, optional
            Target device.
        memory_format : GridFeaturesMemoryFormat, optional
            Target memory format.

        Returns
        -------
        GridFeatures
            Self, with updated device and/or format.
        """
        if device is None and memory_format is None:
            raise ValueError("At least one of device or memory_format must be provided")

        # Move to device if specified
        if device is not None:
            self.vertices = self.vertices.to(device)
            self.features = self.features.to(device)

        # Convert memory format if specified
        if memory_format is not None:
            # Step 1: Convert to b_x_y_z_c format (canonical intermediate)
            if self.memory_format != GridFeaturesMemoryFormat.b_x_y_z_c:
                self.features = convert_to_b_x_y_z_c(
                    self.features, self.memory_format, self.num_channels
                )

            # Step 2: Convert from b_x_y_z_c to target format
            if memory_format != GridFeaturesMemoryFormat.b_x_y_z_c:
                self.features = convert_from_b_x_y_z_c(self.features, memory_format)

            self.memory_format = memory_format

        return self

    def __repr__(self) -> str:
        """Return string representation."""
        return f"GridFeatures(vertices={self.vertices.shape}, features={self.features.shape})"

    def __add__(self, other: "GridFeatures") -> "GridFeatures":
        """Add features element-wise, preserving vertices and format."""
        if self.batch_size != other.batch_size:
            raise ValueError(
                f"Batch size mismatch: {self.batch_size} vs {other.batch_size}"
            )
        if self.features.shape != other.features.shape:
            raise ValueError(
                f"Feature shape mismatch: {self.features.shape} vs {other.features.shape}"
            )
        return GridFeatures(
            self.vertices,
            self.features + other.features,
            self.memory_format,
            grid_shape=self.grid_shape,
            num_channels=self.num_channels,
        )

    def strided_vertices(self, resolution: Tuple[int, int, int]) -> Tensor:
        r"""Get vertices at a different resolution via striding or interpolation.

        Parameters
        ----------
        resolution : Tuple[int, int, int]
            Target resolution (X, Y, Z).

        Returns
        -------
        torch.Tensor
            Vertices of shape :math:`(B, X', Y', Z', 3)` at target resolution.
        """
        if self.vertices.ndim != 5:
            raise ValueError(f"Expected 5D vertices tensor, got {self.vertices.ndim}D")
        if len(resolution) != 3:
            raise ValueError(f"Expected 3D resolution tuple, got {len(resolution)}D")

        if self.resolution == resolution:
            return self.vertices

        # Try integer striding first
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
            # Use grid_sample for non-integer resampling
            grid_points = grid_init(
                bb_max=(1, 1, 1), bb_min=(-1, -1, -1), resolution=resolution
            )  # (res[0], res[1], res[2], 3)
            grid_points = grid_points.unsqueeze(0).to(self.vertices.device)

            # Interpolate vertex coordinates
            sampled_vertices = F.grid_sample(
                self.vertices.permute(0, 4, 1, 2, 3),  # move coords to channel dim
                grid_points.expand(self.batch_size, -1, -1, -1, -1),
                align_corners=True,
            )
            vertices = sampled_vertices.permute(0, 2, 3, 4, 1)  # move coords back

        if vertices.shape[-4:-1] != resolution:
            raise ValueError(
                f"Output vertices resolution mismatch: {vertices.shape[-4:-1]} vs {resolution}"
            )
        return vertices
