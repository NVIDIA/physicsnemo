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

r"""Point-grid feature operations for FIGConvNet.

This module provides operations for converting between point-based and
grid-based feature representations, which are essential for the FIGConvNet
architecture.

The main classes are:

- :class:`AABBGridFeatures`: Grid features initialized from a bounding box
- :class:`PointFeatureToGrid`: Convert point features to grid features
- :class:`GridFeatureToPoint`: Convert grid features to point features
- :class:`GridFeatureCat`: Concatenate grid features along channel dimension
"""

# ruff: noqa: S101
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Int
from torch import Tensor
from torch.nn import functional as F

from physicsnemo.models.figconvnet.components.encodings import SinusoidalEncoding
from physicsnemo.models.figconvnet.components.reductions import REDUCTION_TYPES
from physicsnemo.models.figconvnet.geometries import (
    GridFeatures,
    GridFeaturesMemoryFormat,
    PointFeatures,
    grid_init,
)
from physicsnemo.models.figconvnet.point_feature_conv import (
    PointFeatureCat,
    PointFeatureConv,
    PointFeatureTransform,
)
from physicsnemo.utils.profiling import Profiler

prof = Profiler()


class AABBGridFeatures(GridFeatures):
    r"""Grid features initialized from an axis-aligned bounding box.

    Creates grid features with vertices spanning the specified bounding box
    and features initialized using sinusoidal positional encoding.

    Parameters
    ----------
    aabb_max : Tuple[float, float, float]
        Maximum coordinates of the bounding box.
    aabb_min : Tuple[float, float, float]
        Minimum coordinates of the bounding box.
    resolution : Union[torch.Tensor, List[int]]
        Grid resolution along each axis.
    pos_encode_dim : int, optional, default=32
        Dimension of sinusoidal positional encoding.

    Note
    ----
    This class is primarily used internally to create query grids for
    point-to-grid conversion operations.
    """

    def __init__(
        self,
        aabb_max: Tuple[float, float, float],
        aabb_min: Tuple[float, float, float],
        resolution: Union[Int[Tensor, "3"], List[int]],
        pos_encode_dim: int = 32,
    ):
        grid = grid_init(aabb_max, aabb_min, resolution)
        feat = SinusoidalEncoding(pos_encode_dim, data_range=aabb_max[0] - aabb_min[0])(
            grid
        )
        super().__init__(grid.unsqueeze(0), feat.view(1, *resolution, -1))


class PointFeatureToGrid(nn.Module):
    r"""Convert point features to grid features.

    This module projects point cloud features onto a regular 3D grid using
    graph convolution. For each grid cell, features are aggregated from
    nearby points using the specified neighbor search and reduction operations.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    out_channels : int
        Number of output feature channels.
    aabb_max : Tuple[float, float, float]
        Maximum coordinates of the bounding box.
    aabb_min : Tuple[float, float, float]
        Minimum coordinates of the bounding box.
    voxel_size : float, optional, default=None
        Voxel size for grid construction. Either this or ``resolution`` must be provided.
    resolution : Union[torch.Tensor, List[int]], optional, default=None
        Grid resolution. Either this or ``voxel_size`` must be provided.
    use_rel_pos : bool, optional, default=True
        Whether to use relative positions in convolution.
    use_rel_pos_encode : bool, optional, default=False
        Whether to use sinusoidal positional encoding.
    pos_encode_dim : int, optional, default=32
        Dimension of positional encoding.
    reductions : List[REDUCTION_TYPES], optional, default=["mean"]
        Reduction operations for aggregating features.
    neighbor_search_type : Literal["radius", "knn"], optional, default="radius"
        Type of neighbor search.
    knn_k : int, optional, default=16
        Number of neighbors for KNN search.
    radius : float, optional, default=sqrt(3)
        Search radius (diagonal of a unit cube by default).

    Forward
    -------
    point_features : PointFeatures
        Input point features of shape :math:`(B, N, C_{in})`.

    Outputs
    -------
    GridFeatures
        Grid features of shape :math:`(B, X, Y, Z, C_{out})`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.models.figconvnet.geometries import PointFeatures
    >>> converter = PointFeatureToGrid(
    ...     in_channels=32, out_channels=64,
    ...     aabb_max=(1, 1, 1), aabb_min=(0, 0, 0),
    ...     resolution=[32, 32, 32]
    ... )
    >>> vertices = torch.randn(2, 1000, 3)
    >>> features = torch.randn(2, 1000, 32)
    >>> pf = PointFeatures(vertices, features)
    >>> gf = converter(pf)

    See Also
    --------
    :class:`GridFeatureToPoint`
    :class:`~physicsnemo.models.figconvnet.point_feature_conv.PointFeatureConv`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aabb_max: Tuple[float, float, float],
        aabb_min: Tuple[float, float, float],
        voxel_size: Optional[float] = None,
        resolution: Optional[Union[Int[Tensor, "3"], List[int]]] = None,
        use_rel_pos: bool = True,
        use_rel_pos_encode: bool = False,
        pos_encode_dim: int = 32,
        reductions: List[REDUCTION_TYPES] = ["mean"],
        neighbor_search_type: Literal["radius", "knn"] = "radius",
        knn_k: int = 16,
        radius: float = np.sqrt(3),  # diagonal of a unit cube
    ) -> None:
        super().__init__()
        if resolution is None:
            if voxel_size is None:
                raise ValueError("Either resolution or voxel_size must be provided")
            resolution = (
                int((aabb_max[0] - aabb_min[0]) / voxel_size),
                int((aabb_max[1] - aabb_min[1]) / voxel_size),
                int((aabb_max[2] - aabb_min[2]) / voxel_size),
            )
        if voxel_size is None:
            if resolution is None:
                raise ValueError("Either resolution or voxel_size must be provided")
        if isinstance(resolution, Tensor):
            resolution = resolution.tolist()
        self.resolution = resolution
        for i in range(3):
            if aabb_max[i] <= aabb_min[i]:
                raise ValueError(
                    f"aabb_max[{i}] ({aabb_max[i]}) must be greater than aabb_min[{i}] ({aabb_min[i]})"
                )
        self.grid_features = AABBGridFeatures(
            aabb_max, aabb_min, resolution, pos_encode_dim=pos_encode_dim
        )
        # Find per axis scaler that scales the vertices to [0, resolution[0]] x [0, resolution[1]] x [0, resolution[2]]
        vertices_scaler = torch.FloatTensor(
            [
                resolution[0] / (aabb_max[0] - aabb_min[0]),
                resolution[1] / (aabb_max[1] - aabb_min[1]),
                resolution[2] / (aabb_max[2] - aabb_min[2]),
            ]
        )
        self.conv = PointFeatureConv(
            radius=radius,
            in_channels=in_channels,
            out_channels=out_channels,
            provided_in_channels=3 * pos_encode_dim,
            use_rel_pos=use_rel_pos,
            use_rel_pos_encode=use_rel_pos_encode,
            pos_encode_dim=pos_encode_dim,
            neighbor_search_vertices_scaler=vertices_scaler,
            out_point_feature_type="provided",
            reductions=reductions,
            neighbor_search_type=neighbor_search_type,
            knn_k=knn_k,
        )

    @prof
    def forward(self, point_features: PointFeatures) -> GridFeatures:
        # match the batch size of points
        self.grid_features.to(device=point_features.vertices.device)
        grid_point_features = self.grid_features.point_features.expand_batch_size(
            point_features.batch_size
        )

        out_point_features = self.conv(
            point_features,
            grid_point_features,
        )

        B, _, C = out_point_features.features.shape
        grid_feature = GridFeatures(
            out_point_features.vertices.reshape(B, *self.resolution, 3),
            out_point_features.features.view(B, *self.resolution, C),
        )
        return grid_feature


class GridFeatureToPoint(nn.Module):
    r"""Convert grid features to point features.

    This module samples grid features at arbitrary point locations using
    either graph convolution or trilinear interpolation.

    Parameters
    ----------
    grid_in_channels : int
        Number of input channels in grid features.
    point_in_channels : int
        Number of input channels in point features.
    out_channels : int
        Number of output feature channels.
    aabb_max : Tuple[float, float, float]
        Maximum coordinates of the bounding box.
    aabb_min : Tuple[float, float, float]
        Minimum coordinates of the bounding box.
    hidden_dim : int, optional, default=None
        Hidden dimension for graph convolution.
    use_rel_pos : bool, optional, default=True
        Whether to use relative positions.
    use_rel_pos_embed : bool, optional, default=False
        Whether to use sinusoidal positional encoding.
    pos_embed_dim : int, optional, default=32
        Dimension of positional encoding.
    sample_method : Literal["graphconv", "interp"], optional, default="graphconv"
        Sampling method. "graphconv" uses graph convolution, "interp" uses
        trilinear interpolation.
    neighbor_search_type : Literal["radius", "knn"], optional, default="radius"
        Type of neighbor search for graph convolution.
    knn_k : int, optional, default=16
        Number of neighbors for KNN search.
    reductions : List[REDUCTION_TYPES], optional, default=["mean"]
        Reduction operations for graph convolution.

    Forward
    -------
    grid_features : GridFeatures
        Input grid features.
    point_features : PointFeatures
        Point features defining query locations.

    Outputs
    -------
    PointFeatures
        Sampled point features of shape :math:`(B, N, C_{out})`.

    See Also
    --------
    :class:`PointFeatureToGrid`
    :class:`GridFeatureToPointGraphConv`
    :class:`GridFeatureToPointInterp`
    """

    def __init__(
        self,
        grid_in_channels: int,
        point_in_channels: int,
        out_channels: int,
        aabb_max: Tuple[float, float, float],
        aabb_min: Tuple[float, float, float],
        hidden_dim: Optional[int] = None,
        use_rel_pos: bool = True,
        use_rel_pos_embed: bool = False,
        pos_embed_dim: int = 32,
        sample_method: Literal["graphconv", "interp"] = "graphconv",
        neighbor_search_type: Literal["radius", "knn"] = "radius",
        knn_k: int = 16,
        reductions: List[REDUCTION_TYPES] = ["mean"],
    ) -> None:
        super().__init__()
        self.sample_method = sample_method
        if sample_method == "graphconv":
            self.conv = GridFeatureToPointGraphConv(
                grid_in_channels,
                point_in_channels,
                out_channels,
                aabb_max,
                aabb_min,
                hidden_dim=hidden_dim,
                use_rel_pos=use_rel_pos,
                use_rel_pos_embed=use_rel_pos_embed,
                pos_embed_dim=pos_embed_dim,
                neighbor_search_type=neighbor_search_type,
                knn_k=knn_k,
                reductions=reductions,
            )
        elif sample_method == "interp":
            self.conv = GridFeatureToPointInterp(
                aabb_max,
                aabb_min,
                cat_in_point_features=True,
            )
            self.transform = PointFeatureTransform(
                nn.Sequential(
                    nn.Linear(grid_in_channels + point_in_channels, out_channels),
                    nn.LayerNorm(out_channels),
                )
            )
        else:
            raise NotImplementedError

    def forward(
        self, grid_features: GridFeatures, point_features: PointFeatures
    ) -> PointFeatures:
        r"""Sample grid features at point locations.

        Parameters
        ----------
        grid_features : GridFeatures
            Input grid features.
        point_features : PointFeatures
            Point features defining query locations.

        Returns
        -------
        PointFeatures
            Sampled point features.
        """
        out_point_features = self.conv(grid_features, point_features)
        if self.sample_method == "interp":
            out_point_features = self.transform(out_point_features)
        return out_point_features


class GridFeatureToPointGraphConv(nn.Module):
    r"""Convert grid features to point features using graph convolution.

    Samples grid features at point locations by finding neighboring grid
    cells and aggregating their features through a graph convolution.

    Parameters
    ----------
    grid_in_channels : int
        Number of input channels in grid features.
    point_in_channels : int
        Number of input channels in point features.
    out_channels : int
        Number of output feature channels.
    aabb_max : Tuple[float, float, float]
        Maximum coordinates of the bounding box.
    aabb_min : Tuple[float, float, float]
        Minimum coordinates of the bounding box.
    hidden_dim : int, optional, default=None
        Hidden dimension for the convolution MLP.
    use_rel_pos : bool, optional, default=True
        Whether to use relative positions.
    use_rel_pos_embed : bool, optional, default=False
        Whether to use sinusoidal positional encoding.
    pos_embed_dim : int, optional, default=32
        Dimension of positional encoding.
    neighbor_search_type : Literal["radius", "knn"], optional, default="radius"
        Type of neighbor search.
    knn_k : int, optional, default=16
        Number of neighbors for KNN search.
    reductions : List[REDUCTION_TYPES], optional, default=["mean"]
        Reduction operations for aggregation.

    Forward
    -------
    grid_features : GridFeatures
        Input grid features.
    point_features : PointFeatures
        Point features defining query locations.

    Outputs
    -------
    PointFeatures
        Sampled point features.
    """

    def __init__(
        self,
        grid_in_channels: int,
        point_in_channels: int,
        out_channels: int,
        aabb_max: Tuple[float, float, float],
        aabb_min: Tuple[float, float, float],
        hidden_dim: Optional[int] = None,
        use_rel_pos: bool = True,
        use_rel_pos_embed: bool = False,
        pos_embed_dim: int = 32,
        neighbor_search_type: Literal["radius", "knn"] = "radius",
        knn_k: int = 16,
        reductions: List[REDUCTION_TYPES] = ["mean"],
    ) -> None:
        super().__init__()
        self.aabb_max = aabb_max
        self.aabb_min = aabb_min
        self.conv = PointFeatureConv(
            radius=np.sqrt(3),  # diagonal of a unit cube
            in_channels=grid_in_channels,
            out_channels=out_channels,
            provided_in_channels=point_in_channels,
            hidden_dim=hidden_dim,
            use_rel_pos=use_rel_pos,
            use_rel_pos_encode=use_rel_pos_embed,
            pos_encode_dim=pos_embed_dim,
            out_point_feature_type="provided",
            neighbor_search_type=neighbor_search_type,
            knn_k=knn_k,
            reductions=reductions,
        )

    def forward(
        self, grid_features: GridFeatures, point_features: PointFeatures
    ) -> PointFeatures:
        r"""Sample grid features using graph convolution.

        Parameters
        ----------
        grid_features : GridFeatures
            Input grid features.
        point_features : PointFeatures
            Point features defining query locations.

        Returns
        -------
        PointFeatures
            Sampled point features.
        """
        resolution = grid_features.resolution
        # Find per axis scaler that scales the vertices to [0, resolution[0]] x [0, resolution[1]] x [0, resolution[2]]
        vertices_scaler = torch.FloatTensor(
            [
                resolution[0] / (self.aabb_max[0] - self.aabb_min[0]),
                resolution[1] / (self.aabb_max[1] - self.aabb_min[1]),
                resolution[2] / (self.aabb_max[2] - self.aabb_min[2]),
            ]
        )
        out_point_features = self.conv(
            grid_features.point_features.contiguous(),
            point_features,
            neighbor_search_vertices_scaler=vertices_scaler,
        )
        return out_point_features


class GridFeatureToPointInterp(nn.Module):
    r"""Convert grid features to point features using trilinear interpolation.

    Samples grid features at point locations using PyTorch's grid_sample
    function for efficient trilinear interpolation.

    Parameters
    ----------
    aabb_max : Tuple[float, float, float]
        Maximum coordinates of the bounding box.
    aabb_min : Tuple[float, float, float]
        Minimum coordinates of the bounding box.
    cat_in_point_features : bool, optional, default=True
        Whether to concatenate input point features with sampled features.

    Forward
    -------
    grid_features : GridFeatures
        Input grid features.
    point_features : PointFeatures
        Point features defining query locations.

    Outputs
    -------
    PointFeatures
        Sampled point features (optionally concatenated with input features).

    Note
    ----
    This method is faster than graph convolution but may produce smoother
    interpolated values.
    """

    def __init__(
        self,
        aabb_max: Tuple[float, float, float],
        aabb_min: Tuple[float, float, float],
        cat_in_point_features: bool = True,
    ) -> None:
        super().__init__()
        self.aabb_max = torch.Tensor(aabb_max)
        self.aabb_min = torch.Tensor(aabb_min)
        self.cat_in_point_features = cat_in_point_features
        self.cat = PointFeatureCat()

    def to(self, *args, **kwargs):
        self.aabb_max = self.aabb_max.to(*args, **kwargs)
        self.aabb_min = self.aabb_min.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(
        self, grid_features: GridFeatures, point_features: PointFeatures
    ) -> PointFeatures:
        r"""Sample grid features using trilinear interpolation.

        Parameters
        ----------
        grid_features : GridFeatures
            Input grid features.
        point_features : PointFeatures
            Point features defining query locations.

        Returns
        -------
        PointFeatures
            Sampled point features.
        """
        # Use F.grid_sample to interpolate grid features to point features
        grid_features.to(memory_format=GridFeaturesMemoryFormat.b_c_x_y_z)
        xyz = point_features.vertices  # N x 3
        self.to(device=xyz.device)
        normalized_xyz = (xyz - self.aabb_min) / (self.aabb_max - self.aabb_min) * 2 - 1
        normalized_xyz = normalized_xyz.view(1, 1, 1, -1, 3)
        batch_grid_features = grid_features.batch_features  # B x C x X x Y x Z
        # interpolate
        batch_point_features = (
            F.grid_sample(
                batch_grid_features,
                normalized_xyz,
                align_corners=True,
            )
            .squeeze()
            .permute(1, 0)
        )  # N x C

        out_point_features = PointFeatures(
            point_features.vertices,
            batch_point_features,
        )
        if self.cat_in_point_features:
            out_point_features = self.cat(point_features, out_point_features)
        return out_point_features


class GridFeatureCat(nn.Module):
    r"""Concatenate two GridFeatures along the channel dimension.

    Parameters
    ----------
    None

    Forward
    -------
    grid_features : GridFeatures
        First grid features.
    other_grid_features : GridFeatures
        Second grid features.

    Outputs
    -------
    GridFeatures
        Concatenated grid features.

    Note
    ----
    Both inputs must have the same memory format and spatial dimensions.
    """

    def forward(
        self, grid_features: GridFeatures, other_grid_features: GridFeatures
    ) -> GridFeatures:
        r"""Concatenate grid features.

        Parameters
        ----------
        grid_features : GridFeatures
            First grid features.
        other_grid_features : GridFeatures
            Second grid features.

        Returns
        -------
        GridFeatures
            Concatenated grid features.
        """
        if grid_features.memory_format != other_grid_features.memory_format:
            raise ValueError(
                f"Memory format mismatch: {grid_features.memory_format} vs {other_grid_features.memory_format}"
            )
        # assert torch.allclose(grid_features.vertices, other_grid_features.vertices)

        orig_memory_format = grid_features.memory_format
        grid_features.to(memory_format=GridFeaturesMemoryFormat.b_c_x_y_z)
        other_grid_features.to(memory_format=GridFeaturesMemoryFormat.b_c_x_y_z)
        cat_grid_features = GridFeatures(
            vertices=grid_features.vertices,
            features=torch.cat(
                [grid_features.features, other_grid_features.features], dim=0
            ),
            memory_format=grid_features.memory_format,
            grid_shape=grid_features.grid_shape,
            num_channels=grid_features.num_channels + other_grid_features.num_channels,
        )
        cat_grid_features.to(memory_format=orig_memory_format)
        return cat_grid_features
