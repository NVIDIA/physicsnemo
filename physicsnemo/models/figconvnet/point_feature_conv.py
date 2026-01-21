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

r"""Point feature convolution operations for FIGConvNet.

This module provides point-based convolution operations that aggregate
features from neighboring points using various reduction operations.

The main classes are:

- :class:`PointFeatureTransform`: Apply transforms to point features
- :class:`PointFeatureCat`: Concatenate point features
- :class:`PointFeatureMLP`: MLP for point feature transformation
- :class:`PointFeatureConv`: Point-based graph convolution
- :class:`PointFeatureConvBlock`: Residual block with point convolutions
"""

# ruff: noqa: S101,F722
from typing import List, Literal, Optional

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from physicsnemo.models.figconvnet.components.encodings import SinusoidalEncoding
from physicsnemo.models.figconvnet.components.mlp import MLPBlock
from physicsnemo.models.figconvnet.components.reductions import (
    REDUCTION_TYPES,
    row_reduction,
)
from physicsnemo.models.figconvnet.geometries import PointFeatures
from physicsnemo.models.figconvnet.neighbor_ops import (
    batched_neighbor_knn_search,
    batched_neighbor_radius_search,
)


class PointFeatureTransform(nn.Module):
    r"""Apply a transform to point feature tensors.

    Wraps an arbitrary transform (e.g., MLP, normalization) to operate on
    the feature tensors within PointFeatures objects while preserving vertices.

    Parameters
    ----------
    feature_transform : nn.Module
        Transform module to apply to the feature tensor.

    Forward
    -------
    point_features : PointFeatures
        Input point features.

    Outputs
    -------
    PointFeatures
        Transformed point features with same vertices.

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> transform = PointFeatureTransform(nn.Linear(64, 128))
    """

    def __init__(
        self,
        feature_transform: nn.Module,
    ):
        super().__init__()
        self.feature_transform = feature_transform

    def forward(self, point_features: PointFeatures) -> PointFeatures:
        r"""Apply transform to point features.

        Parameters
        ----------
        point_features : PointFeatures
            Input point features.

        Returns
        -------
        PointFeatures
            Transformed point features.
        """
        return PointFeatures(
            point_features.vertices, self.feature_transform(point_features.features)
        )


class PointFeatureCat(nn.Module):
    r"""Concatenate two PointFeatures along the channel dimension.

    Parameters
    ----------
    None

    Forward
    -------
    point_features : PointFeatures
        First point features with shape :math:`(B, N, C_1)`.
    point_features2 : PointFeatures
        Second point features with shape :math:`(B, N, C_2)`.

    Outputs
    -------
    PointFeatures
        Concatenated point features with shape :math:`(B, N, C_1 + C_2)`.

    Note
    ----
    Both inputs must have the same vertices (same point locations).
    """

    def forward(
        self,
        point_features: PointFeatures,
        point_features2: PointFeatures,
    ) -> PointFeatures:
        r"""Concatenate point features.

        Parameters
        ----------
        point_features : PointFeatures
            First point features.
        point_features2 : PointFeatures
            Second point features.

        Returns
        -------
        PointFeatures
            Concatenated point features.
        """
        return PointFeatures(
            point_features.vertices,
            torch.cat([point_features.features, point_features2.features], dim=-1),
        )


class PointFeatureMLP(PointFeatureTransform):
    r"""MLP for point feature transformation.

    A simple two-layer MLP applied to point features.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    hidden_channels : int, optional, default=None
        Number of hidden channels. Defaults to ``multiplier * out_channels``.
    multiplier : int, optional, default=2
        Multiplier for computing default hidden channels.
    nonlinearity : nn.Module, optional, default=nn.GELU
        Activation function class.

    Forward
    -------
    point_features : PointFeatures
        Input point features with shape :math:`(B, N, C_{in})`.

    Outputs
    -------
    PointFeatures
        Transformed point features with shape :math:`(B, N, C_{out})`.

    Examples
    --------
    >>> import torch
    >>> mlp = PointFeatureMLP(64, 128)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int] = None,
        multiplier: int = 2,
        nonlinearity: nn.Module = nn.GELU,
    ):
        if hidden_channels is None:
            hidden_channels = multiplier * out_channels
        PointFeatureTransform.__init__(
            self,
            nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nonlinearity(),
                nn.Linear(hidden_channels, out_channels),
            ),
        )


class PointFeatureConv(nn.Module):
    r"""Point-based graph convolution.

    This module performs convolution on point clouds by aggregating features
    from neighboring points. It supports both radius-based and KNN neighbor
    search, with configurable edge features and reduction operations.

    The convolution operates as follows:

    1. Find neighbors for each output point (via radius or KNN search)
    2. Compute edge features from input and output point features
    3. Optionally add relative position information
    4. Transform edge features through an MLP
    5. Aggregate using specified reduction operations
    6. Transform aggregated features through output MLP

    Parameters
    ----------
    radius : float
        Search radius for neighbor finding (used when ``neighbor_search_type="radius"``).
    edge_transform_mlp : nn.Module, optional, default=None
        MLP for transforming edge features. Auto-created if None.
    out_transform_mlp : nn.Module, optional, default=None
        MLP for transforming aggregated features. Auto-created if None.
    in_channels : int, optional, default=8
        Number of input feature channels.
    out_channels : int, optional, default=32
        Number of output feature channels.
    hidden_dim : int, optional, default=None
        Hidden dimension for MLPs. Defaults to ``channel_multiplier * out_channels``.
    channel_multiplier : int, optional, default=2
        Multiplier for computing default hidden dimension.
    use_rel_pos : bool, optional, default=True
        Whether to include relative position in edge features.
    use_rel_pos_encode : bool, optional, default=False
        Whether to use sinusoidal encoding for relative positions.
    pos_encode_dim : int, optional, default=32
        Dimension of positional encoding.
    pos_encode_range : float, optional, default=4
        Range for positional encoding.
    reductions : List[REDUCTION_TYPES], optional, default=["mean"]
        Reduction operations for aggregating neighbor features.
    downsample_voxel_size : float, optional, default=None
        Voxel size for downsampling output points.
    out_point_feature_type : Literal["provided", "downsample", "same"], optional, default="same"
        How to determine output point locations.
    provided_in_channels : int, optional, default=None
        Input channels for provided output points.
    neighbor_search_vertices_scaler : torch.Tensor, optional, default=None
        Scaling factors for anisotropic neighbor search.
    neighbor_search_type : Literal["radius", "knn"], optional, default="radius"
        Type of neighbor search.
    radius_search_method : Literal["open3d", "warp"], optional, default="warp"
        Backend for radius search.
    knn_k : int, optional, default=None
        Number of neighbors for KNN search.

    Forward
    -------
    in_point_features : PointFeatures
        Input point features of shape :math:`(B, N, C_{in})`.
    out_point_features : PointFeatures, optional
        Output point features defining query locations (for "provided" type).
    neighbor_search_vertices_scaler : torch.Tensor, optional
        Per-call override for vertex scaling.

    Outputs
    -------
    PointFeatures
        Output point features of shape :math:`(B, M, C_{out})`.

    Note
    ----
    This is the core convolution operation for point-based neural networks,
    implementing a message-passing scheme on point clouds.

    See Also
    --------
    :class:`PointFeatureConvBlock`
    :func:`~physicsnemo.models.figconvnet.neighbor_ops.batched_neighbor_radius_search`
    """

    def __init__(
        self,
        radius: float,
        edge_transform_mlp: Optional[nn.Module] = None,
        out_transform_mlp: Optional[nn.Module] = None,
        in_channels: int = 8,
        out_channels: int = 32,
        hidden_dim: Optional[int] = None,
        channel_multiplier: int = 2,
        use_rel_pos: bool = True,
        use_rel_pos_encode: bool = False,
        pos_encode_dim: int = 32,
        pos_encode_range: float = 4,
        reductions: List[REDUCTION_TYPES] = ["mean"],
        downsample_voxel_size: Optional[float] = None,
        out_point_feature_type: Literal["provided", "downsample", "same"] = "same",
        provided_in_channels: Optional[int] = None,
        neighbor_search_vertices_scaler: Optional[Float[Tensor, "3"]] = None,
        neighbor_search_type: Literal["radius", "knn"] = "radius",
        radius_search_method: Literal["open3d", "warp"] = "warp",
        knn_k: Optional[int] = None,
    ):
        """If use_relative_position_encoding is True, the positional encoding vertex coordinate
        difference is added to the edge features.

        downsample_voxel_size: If not None, the input point cloud will be downsampled.

        out_point_feature_type: If "upsample", the output point features will be upsampled to the input point cloud size.

        use_rel_pos: If True, the relative position of the neighbor points will be used as the edge features.
        use_rel_pos_encode: If True, the encoding relative position of the neighbor points will be used as the edge features.

        if neighbor_search_vertices_scaler is not None, find neighbors using the
        scaled vertices. This allows finding neighbors with an axis aligned
        ellipsoidal neighborhood.
        """
        super().__init__()
        if not isinstance(reductions, (tuple, list)) or len(reductions) == 0:
            raise ValueError(
                f"reductions must be a list or tuple of length > 0, got {reductions}"
            )
        if out_point_feature_type == "provided":
            if downsample_voxel_size is not None:
                raise ValueError("downsample_voxel_size is only used for downsample")
            if provided_in_channels is None:
                raise ValueError(
                    "provided_in_channels must be provided for provided type"
                )
        elif out_point_feature_type == "downsample":
            if downsample_voxel_size is None:
                raise ValueError(
                    "downsample_voxel_size must be provided for downsample"
                )
            if provided_in_channels is not None:
                raise ValueError(
                    "provided_in_channels must be None for downsample type"
                )
        elif out_point_feature_type == "same":
            if downsample_voxel_size is not None:
                raise ValueError("downsample_voxel_size is only used for downsample")
            if provided_in_channels is not None:
                raise ValueError("provided_in_channels must be None for same type")
        if downsample_voxel_size is not None and downsample_voxel_size > radius:
            raise ValueError(
                f"downsample_voxel_size {downsample_voxel_size} must be <= radius {radius}"
            )
        self.reductions = reductions
        self.downsample_voxel_size = downsample_voxel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_rel_pos = use_rel_pos
        self.use_rel_pos_encode = use_rel_pos_encode
        self.out_point_feature_type = out_point_feature_type
        self.neighbor_search_vertices_scaler = neighbor_search_vertices_scaler
        self.neighbor_search_type = neighbor_search_type
        self.radius_search_method = radius_search_method
        if neighbor_search_type == "radius":
            self.radius_or_k = radius
        elif neighbor_search_type == "knn":
            if knn_k is None:
                raise ValueError("knn_k must be provided for knn neighbor search type")
            self.radius_or_k = knn_k
        else:
            raise ValueError(
                f"neighbor_search_type must be radius or knn, got {neighbor_search_type}"
            )
        self.positional_encoding = SinusoidalEncoding(
            pos_encode_dim, data_range=pos_encode_range
        )
        # When down voxel size is not None, there will be out_point_features will be provided as an additional input
        if provided_in_channels is None:
            provided_in_channels = in_channels
        if hidden_dim is None:
            hidden_dim = channel_multiplier * out_channels
        if edge_transform_mlp is None:
            edge_in_channels = in_channels + provided_in_channels
            if use_rel_pos_encode:
                edge_in_channels += pos_encode_dim * 3
            elif use_rel_pos:
                edge_in_channels += 3
            edge_transform_mlp = MLPBlock(
                in_channels=edge_in_channels,
                hidden_channels=hidden_dim,
                out_channels=out_channels,
            )
        self.edge_transform_mlp = edge_transform_mlp
        if out_transform_mlp is None:
            out_transform_mlp = MLPBlock(
                in_channels=out_channels * len(reductions),
                hidden_channels=hidden_dim,
                out_channels=out_channels,
            )
        self.out_transform_mlp = out_transform_mlp

    def __repr__(self):
        out_str = f"{self.__class__.__name__}(in_channels={self.in_channels} out_channels={self.out_channels} search_type={self.neighbor_search_type} reductions={self.reductions}"
        if self.downsample_voxel_size is not None:
            out_str += f" down_voxel_size={self.downsample_voxel_size}"
        if self.use_rel_pos_encode:
            out_str += f" rel_pos_encode={self.use_rel_pos_encode}"
        out_str += ")"
        return out_str

    def forward(
        self,
        in_point_features: PointFeatures,
        out_point_features: Optional[PointFeatures] = None,
        neighbor_search_vertices_scaler: Optional[Float[Tensor, "3"]] = None,
    ) -> PointFeatures:
        r"""Apply point convolution.

        Parameters
        ----------
        in_point_features : PointFeatures
            Input point features of shape :math:`(B, N, C_{in})`.
        out_point_features : PointFeatures, optional
            Output point locations. Required for "provided" type, ignored otherwise.
        neighbor_search_vertices_scaler : torch.Tensor, optional
            Scaling factors for anisotropic neighbor search.

        Returns
        -------
        PointFeatures
            Output point features of shape :math:`(B, M, C_{out})`.
        """
        if self.out_point_feature_type == "provided":
            if out_point_features is None:
                raise ValueError(
                    "out_point_features must be provided for the provided type"
                )
        elif self.out_point_feature_type == "downsample":
            if out_point_features is not None:
                raise ValueError("out_point_features must be None for downsample type")
            out_point_features = in_point_features.voxel_down_sample(
                self.downsample_voxel_size
            )
        elif self.out_point_feature_type == "same":
            if out_point_features is not None:
                raise ValueError("out_point_features must be None for same type")
            out_point_features = in_point_features

        in_num_channels = in_point_features.num_channels
        out_num_channels = out_point_features.num_channels
        expected_edge_channels = (
            in_num_channels
            + out_num_channels
            + self.use_rel_pos_encode * self.positional_encoding.num_channels * 3
            + (not self.use_rel_pos_encode) * self.use_rel_pos * 3
        )
        if expected_edge_channels != self.edge_transform_mlp.in_channels:
            raise ValueError(
                f"input features shape {in_point_features.features.shape} and {out_point_features.features.shape} "
                f"does not match the edge_transform_mlp input features {self.edge_transform_mlp.in_channels}"
            )
        # Get the neighbors
        in_vertices = in_point_features.vertices
        out_vertices = out_point_features.vertices
        if self.neighbor_search_vertices_scaler is not None:
            neighbor_search_vertices_scaler = self.neighbor_search_vertices_scaler
        if neighbor_search_vertices_scaler is not None:
            in_vertices = in_vertices * neighbor_search_vertices_scaler.to(in_vertices)
            out_vertices = out_vertices * neighbor_search_vertices_scaler.to(
                out_vertices
            )

        if self.neighbor_search_type == "knn":
            device = in_vertices.device
            neighbors_index = batched_neighbor_knn_search(
                in_vertices, out_vertices, self.radius_or_k
            )
            # B x M x K index
            neighbors_index = neighbors_index.long().to(device).view(-1)
            # M row splits
            neighbors_row_splits = (
                torch.arange(
                    0, out_vertices.shape[0] * out_vertices.shape[1] + 1, device=device
                )
                * self.radius_or_k
            )
            rep_in_features = in_point_features.features.view(-1, in_num_channels)[
                neighbors_index
            ]
            num_reps = self.radius_or_k
        elif self.neighbor_search_type == "radius":
            neighbors = batched_neighbor_radius_search(
                in_vertices,
                out_vertices,
                radius=self.radius_or_k,
                search_method=self.radius_search_method,
            )
            neighbors_index = neighbors.neighbors_index.long()
            rep_in_features = in_point_features.features.view(-1, in_num_channels)[
                neighbors_index
            ]
            neighbors_row_splits = neighbors.neighbors_row_splits
            num_reps = neighbors_row_splits[1:] - neighbors_row_splits[:-1]
        else:
            raise ValueError(
                f"neighbor_search_type must be radius or knn, got {self.neighbor_search_type}"
            )
        # repeat the self features using num_reps
        self_features = torch.repeat_interleave(
            out_point_features.features.view(-1, out_num_channels).contiguous(),
            num_reps,
            dim=0,
        )
        edge_features = [rep_in_features, self_features]
        if self.use_rel_pos or self.use_rel_pos_encode:
            in_rep_vertices = in_point_features.vertices.view(-1, 3)[neighbors_index]
            self_vertices = torch.repeat_interleave(
                out_point_features.vertices.view(-1, 3).contiguous(), num_reps, dim=0
            )
            if self.use_rel_pos_encode:
                edge_features.append(
                    self.positional_encoding(
                        in_rep_vertices.view(-1, 3) - self_vertices.view(-1, 3)
                    )
                )
            elif self.use_rel_pos:
                edge_features.append(in_rep_vertices - self_vertices)
        edge_features = torch.cat(edge_features, dim=1)
        edge_features = self.edge_transform_mlp(edge_features)
        # if in_weight is not None:
        #     assert in_weight.shape[0] == in_point_features.features.shape[0]
        #     rep_weights = in_weight[neighbors_index]
        #     edge_features = edge_features * rep_weights.squeeze().unsqueeze(-1)

        out_features = [
            row_reduction(edge_features, neighbors_row_splits, reduction=reduction)
            for reduction in self.reductions
        ]
        out_features = torch.cat(out_features, dim=-1)
        out_features = self.out_transform_mlp(out_features)
        # Convert back to the original shape
        out_features = out_features.view(
            out_point_features.batch_size,
            out_point_features.num_points,
            out_features.shape[-1],
        )
        return PointFeatures(out_point_features.vertices, out_features)


class PointFeatureConvBlock(nn.Module):
    r"""Residual block with two point convolutions.

    A standard residual block architecture for point features consisting of:

    1. First point convolution (with optional downsampling)
    2. Normalization and activation
    3. Second point convolution (same resolution)
    4. Normalization
    5. Skip connection
    6. Final activation

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    radius : float
        Search radius for neighbor finding.
    use_rel_pos : bool, optional, default=True
        Whether to use relative positions.
    use_rel_pos_encode : bool, optional, default=False
        Whether to use sinusoidal positional encoding.
    pos_encode_dim : int, optional, default=32
        Dimension of positional encoding.
    reductions : List[REDUCTION_TYPES], optional, default=["mean"]
        Reduction operations for aggregation.
    downsample_voxel_size : float, optional, default=None
        Voxel size for downsampling.
    pos_encode_range : float, optional, default=4
        Range for positional encoding.
    out_point_feature_type : Literal["provided", "downsample", "same"], optional, default="same"
        How to determine output point locations.
    provided_in_channels : int, optional, default=None
        Input channels for provided output points.
    neighbor_search_type : Literal["radius", "knn"], optional, default="radius"
        Type of neighbor search.
    knn_k : int, optional, default=None
        Number of neighbors for KNN search.

    Forward
    -------
    in_point_features : PointFeatures
        Input point features of shape :math:`(B, N, C_{in})`.
    out_point_features : PointFeatures, optional
        Output point features (for "provided" type).

    Outputs
    -------
    PointFeatures
        Output point features of shape :math:`(B, M, C_{out})`.

    Note
    ----
    The skip connection is automatically adjusted based on the output type
    and channel dimensions.

    See Also
    --------
    :class:`PointFeatureConv`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        radius: float,
        use_rel_pos: bool = True,
        use_rel_pos_encode: bool = False,
        pos_encode_dim: int = 32,
        reductions: List[REDUCTION_TYPES] = ["mean"],
        downsample_voxel_size: Optional[float] = None,
        pos_encode_range: float = 4,
        out_point_feature_type: Literal["provided", "downsample", "same"] = "same",
        provided_in_channels: Optional[int] = None,
        neighbor_search_type: Literal["radius", "knn"] = "radius",
        knn_k: Optional[int] = None,
    ):
        super().__init__()
        self.downsample_voxel_size = downsample_voxel_size
        self.out_point_feature_type = out_point_feature_type
        self.conv1 = PointFeatureConv(
            in_channels=in_channels,
            out_channels=out_channels,
            radius=radius,
            use_rel_pos=use_rel_pos,
            use_rel_pos_encode=use_rel_pos_encode,
            pos_encode_dim=pos_encode_dim,
            reductions=reductions,
            downsample_voxel_size=downsample_voxel_size,
            pos_encode_range=pos_encode_range,
            out_point_feature_type=out_point_feature_type,
            provided_in_channels=provided_in_channels,
            neighbor_search_type=neighbor_search_type,
            knn_k=knn_k,
        )
        self.conv2 = PointFeatureConv(
            in_channels=out_channels,
            out_channels=out_channels,
            radius=radius,
            use_rel_pos=use_rel_pos,
            pos_encode_range=pos_encode_range,
            reductions=reductions,
            out_point_feature_type="same",
            neighbor_search_type=neighbor_search_type,
            knn_k=knn_k,
        )
        self.norm1 = PointFeatureTransform(nn.LayerNorm(out_channels))
        self.norm2 = PointFeatureTransform(nn.LayerNorm(out_channels))
        if out_point_feature_type == "provided":
            self.shortcut = PointFeatureMLP(
                in_channels=provided_in_channels, out_channels=out_channels
            )
        elif in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = PointFeatureMLP(in_channels, out_channels)
        # if down_sample_voxel_size is None, just use MLP for shortcut, otherwise, use conv to downsample
        self.nonlinear = PointFeatureTransform(nn.GELU())

    def forward(
        self,
        in_point_features: PointFeatures,
        out_point_features: Optional[PointFeatures] = None,
    ) -> PointFeatures:
        r"""Apply residual block to point features.

        Parameters
        ----------
        in_point_features : PointFeatures
            Input point features.
        out_point_features : PointFeatures, optional
            Output point locations for "provided" type.

        Returns
        -------
        PointFeatures
            Processed point features.
        """
        if self.out_point_feature_type == "provided":
            if out_point_features is None:
                raise ValueError(
                    "out_point_features must be provided for the provided type"
                )
            out = self.conv1(in_point_features, out_point_features)
        elif self.out_point_feature_type == "downsample":
            if out_point_features is not None:
                raise ValueError("out_point_features must be None for downsample type")
            out_point_features = in_point_features.voxel_down_sample(
                self.downsample_voxel_size
            )
            out = self.conv1(in_point_features)
        elif self.out_point_feature_type == "same":
            if out_point_features is not None:
                raise ValueError("out_point_features must be None for same type")
            out_point_features = in_point_features
            out = self.conv1(in_point_features)
        out = self.nonlinear(self.norm1(out))
        out = self.norm2(self.conv2(out))
        return self.nonlinear(out + self.shortcut(out_point_features))
