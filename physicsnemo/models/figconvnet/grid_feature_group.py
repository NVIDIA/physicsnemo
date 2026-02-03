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

r"""Grid feature group operations for FIGConvNet.

This module provides classes for managing groups of factorized grid features
and their inter-grid communication in the FIGConvNet architecture.

The main classes are:

- :class:`GridFeatureGroup`: Container for multiple GridFeatures objects
- :class:`GridFeaturesGroupIntraCommunication`: Inter-grid feature communication
- :class:`GridFeatureConv2DBlocksAndIntraCommunication`: Combined conv + communication block
- :class:`GridFeatureGroupToPoint`: Convert grid features back to point features
- :class:`GridFeatureGroupPool`: Pool grid features to scalar values
"""

# ruff: noqa: S101,F722
from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from physicsnemo.models.figconvnet.components.reductions import REDUCTION_TYPES
from physicsnemo.models.figconvnet.geometries import (
    GridFeatures,
    GridFeaturesMemoryFormat,
    PointFeatures,
)
from physicsnemo.models.figconvnet.point_feature_grid_conv import (
    GridFeatureConv2d,
    GridFeatureConv2dBlock,
    GridFeaturePadToMatch,
    GridFeatureTransform,
    LayerNorm2d,
)
from physicsnemo.models.figconvnet.point_feature_grid_ops import (
    GridFeatureCat,
    GridFeatureToPoint,
)
from physicsnemo.utils.profiling import profile


class GridFeatureGroup:
    r"""Container for a set of factorized GridFeatures.

    This class represents a group of GridFeatures with different factorized
    representations (e.g., XY, XZ, YZ planes). Together, these factorized
    features can implicitly represent high-resolution 3D features.

    For example, with resolutions:
    - ``(high_res, high_res, low_res)`` for XY plane
    - ``(high_res, low_res, high_res)`` for XZ plane
    - ``(low_res, high_res, high_res)`` for YZ plane

    These can synthesize features at ``(high_res, high_res, high_res)`` through
    the :class:`GridFeatureGroupToPoint` module.

    Parameters
    ----------
    grid_features : List[GridFeatures]
        List of GridFeatures objects, one for each factorized representation.

    Attributes
    ----------
    grid_features : List[GridFeatures]
        The stored list of GridFeatures.

    Examples
    --------
    >>> from physicsnemo.models.figconvnet.geometries import GridFeatures
    >>> import torch
    >>> # Create three factorized grid features
    >>> gf1 = GridFeatures(torch.randn(2, 128, 128, 2, 3), torch.randn(2, 128, 128, 2, 32))
    >>> gf2 = GridFeatures(torch.randn(2, 128, 2, 128, 3), torch.randn(2, 128, 2, 128, 32))
    >>> gf3 = GridFeatures(torch.randn(2, 2, 128, 128, 3), torch.randn(2, 2, 128, 128, 32))
    >>> group = GridFeatureGroup([gf1, gf2, gf3])
    >>> len(group)
    3

    See Also
    --------
    :class:`~physicsnemo.models.figconvnet.geometries.GridFeatures`
    :class:`GridFeatureGroupToPoint`
    """

    grid_features: List[GridFeatures]

    def __init__(self, grid_features: List[GridFeatures]) -> None:
        if len(grid_features) == 0:
            raise ValueError("GridFeatureGroup requires at least one GridFeatures")
        self.grid_features = grid_features

    def to(
        self,
        device: Union[torch.device, str] = None,
        memory_format: GridFeaturesMemoryFormat = None,
    ) -> "GridFeatureGroup":
        r"""Move all grid features to device and/or convert memory format.

        Parameters
        ----------
        device : Union[torch.device, str], optional
            Target device.
        memory_format : GridFeaturesMemoryFormat, optional
            Target memory format for all grid features.

        Returns
        -------
        GridFeatureGroup
            Self, with updated device and/or format.
        """
        if device is None and memory_format is None:
            raise ValueError("At least one of device or memory_format must be provided")
        if device is not None:
            for grid_features in self.grid_features:
                grid_features.to(device=device)

        if memory_format is not None:
            for grid_features in self.grid_features:
                grid_features.to(memory_format=memory_format)
        return self

    def __getitem__(self, index: int) -> GridFeatures:
        """Get grid features at the specified index."""
        return self.grid_features[index]

    def __len__(self) -> int:
        """Return the number of grid features in the group."""
        return len(self.grid_features)

    def __iter__(self):
        """Iterate over grid features."""
        return iter(self.grid_features)

    def __repr__(self) -> str:
        """Return string representation."""
        out_str = "GridFeaturesGroup("
        for grid_features in self.grid_features:
            out_str += f"\n\t{grid_features}"
        out_str += "\n)"
        return out_str

    def __add__(self, other: "GridFeatureGroup") -> "GridFeatureGroup":
        """Element-wise addition of corresponding grid features."""
        if len(self) != len(other):
            raise ValueError(
                f"GridFeatureGroup size mismatch: {len(self)} vs {len(other)}"
            )
        grid_features = [item + other[i] for i, item in enumerate(self)]
        return GridFeatureGroup(grid_features)


class GridFeaturesGroupIntraCommunication(nn.Module):
    r"""Inter-grid communication for GridFeatureGroup.

    This module enables communication between the factorized grid features in a
    GridFeatureGroup. The communication is achieved by sampling features from
    one grid at the vertex locations of another grid, effectively sharing
    information across the different factorized representations.

    Mathematically, for a set of grid features :math:`\mathcal{G} = \{G_1, G_2, ..., G_n\}`,
    the communication for each :math:`G_i` is computed as:

    .. math::

        G_i(v) = G_i(v) \circ \sum_{j \neq i} G_j(v)

    where :math:`v` are the vertices of :math:`G_i` and :math:`\circ` is either
    element-wise sum or multiplication depending on ``communication_type``.

    Parameters
    ----------
    communication_type : Literal["sum", "mul"], optional, default="sum"
        Type of communication operation. ``"sum"`` adds sampled features,
        ``"mul"`` multiplies them element-wise.

    Forward
    -------
    grid_features_group : GridFeatureGroup
        Input group of grid features.

    Outputs
    -------
    GridFeatureGroup
        Grid features with inter-grid communication applied.

    Note
    ----
    This operation modifies the input GridFeatureGroup in-place for efficiency.
    """

    def __init__(self, communication_type: Literal["sum", "mul"] = "sum") -> None:
        super().__init__()
        self.communication_type = communication_type

    @profile
    def forward(self, grid_features_group: GridFeatureGroup) -> GridFeatureGroup:
        r"""Apply inter-grid communication.

        Parameters
        ----------
        grid_features_group : GridFeatureGroup
            Input grid features group.

        Returns
        -------
        GridFeatureGroup
            Grid features with communication applied.
        """
        # Convert grid_features to b_c_x_y_z format for grid_sample
        orig_memory_formats = []
        for grid_features in grid_features_group:
            orig_memory_formats.append(grid_features.memory_format)
            grid_features.to(memory_format=GridFeaturesMemoryFormat.b_c_x_y_z)

        # Verify all grid features have the same channel size
        channel_size = grid_features_group[0].features.shape[0]
        for grid_features in grid_features_group:
            if grid_features.features.shape[0] != channel_size:
                raise ValueError(
                    f"Channel size mismatch: {grid_features.features.shape[0]} != {channel_size}"
                )

        # Store original features to avoid in-place modification during iteration
        orig_features = [
            torch.clone(grid_features.features) for grid_features in grid_features_group
        ]

        # Compute normalized vertex coordinates for grid_sample
        normalized_bxyzs = []
        with torch.no_grad():
            for i in range(len(grid_features_group)):
                vertices = grid_features_group[i].vertices

                # Handle strided vertices if resolution doesn't match feature shape
                if grid_features_group[i].resolution != orig_features[i].shape[2:]:
                    vertices = grid_features_group[i].strided_vertices(
                        orig_features[i].shape[2:]
                    )

                if vertices.ndim != 5:
                    raise ValueError(
                        f"Vertices must be BxHxWxDx3 format (5D), got {vertices.ndim}D"
                    )

                # Normalize coordinates to [-1, 1] for grid_sample
                bxyz = vertices.flatten(1, 3)
                bxyz_min = torch.min(bxyz, dim=1, keepdim=True)[0]
                bxyz_max = torch.max(bxyz, dim=1, keepdim=True)[0]
                normalized_bxyz = (bxyz - bxyz_min) / (bxyz_max - bxyz_min) * 2 - 1
                normalized_bxyzs.append(normalized_bxyz.view(vertices.shape))

        # Apply inter-grid communication: sample features from grid j at grid i's vertices
        for i in range(len(grid_features_group)):
            for j in range(len(grid_features_group)):
                if i == j:
                    continue

                # Sample features from grid j at grid i's vertex locations
                sampled_features = torch.nn.functional.grid_sample(
                    orig_features[j],  # (B, C, X, Y, Z)
                    normalized_bxyzs[i],  # (B, X, Y, Z, 3)
                    align_corners=True,
                )

                # Apply communication operation
                if self.communication_type == "sum":
                    grid_features_group[i].features += sampled_features
                elif self.communication_type == "mul":
                    grid_features_group[i].features *= sampled_features
                else:
                    raise NotImplementedError(
                        f"Unknown communication type: {self.communication_type}"
                    )

        # Convert back to original memory formats
        for i, grid_features in enumerate(grid_features_group):
            grid_features.to(memory_format=orig_memory_formats[i])

        return grid_features_group


class GridFeatureGroupIntraCommunications(nn.Module):
    r"""Multiple inter-grid communication operations.

    Extension of :class:`GridFeaturesGroupIntraCommunication` that supports
    multiple communication types. When multiple types are specified, the
    outputs are concatenated along the channel dimension.

    Parameters
    ----------
    communication_types : List[Literal["sum", "mul"]], optional, default=["sum"]
        List of communication types to apply.

    Forward
    -------
    grid_features_group : GridFeatureGroup
        Input group of grid features.

    Outputs
    -------
    GridFeatureGroup
        Grid features with all communication types applied and concatenated.

    See Also
    --------
    :class:`GridFeaturesGroupIntraCommunication`
    """

    def __init__(
        self, communication_types: List[Literal["sum", "mul"]] = ["sum"]
    ) -> None:
        super().__init__()
        self.intra_communications = nn.ModuleList()
        self.grid_cat = GridFeatureGroupCat()
        for communication_type in communication_types:
            self.intra_communications.append(
                GridFeaturesGroupIntraCommunication(
                    communication_type=communication_type
                )
            )

    def forward(self, grid_features_group: GridFeatureGroup) -> GridFeatureGroup:
        r"""Apply multiple communication operations.

        Parameters
        ----------
        grid_features_group : GridFeatureGroup
            Input grid features group.

        Returns
        -------
        GridFeatureGroup
            Grid features with communications applied.
        """
        if len(self.intra_communications) == 1:
            return self.intra_communications[0](grid_features_group)
        elif len(self.intra_communications) == 2:
            # Concatenate outputs from both communication types
            return self.grid_cat(
                self.intra_communications[0](grid_features_group),
                self.intra_communications[1](grid_features_group),
            )
        else:
            raise NotImplementedError(
                f"Only 1 or 2 communication types supported, got {len(self.intra_communications)}"
            )


class GridFeatureGroupConv2dNorm(nn.Module):
    r"""2D convolution with normalization for GridFeatureGroup.

    Applies a 2D convolution followed by normalization to each grid feature
    in the group independently.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution kernel size.
    compressed_spatial_dims : Tuple[int]
        Compressed spatial dimension for each factorized grid.
    stride : int, optional, default=1
        Convolution stride.
    up_stride : int, optional, default=None
        Upsampling stride for transposed convolution.
    norm : nn.Module, optional, default=LayerNorm2d
        Normalization layer class.

    Forward
    -------
    grid_features_group : GridFeatureGroup
        Input group of grid features.

    Outputs
    -------
    GridFeatureGroup
        Processed grid features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        compressed_spatial_dims: Tuple[int],
        stride: int = 1,
        up_stride: Optional[int] = None,
        norm: nn.Module = LayerNorm2d,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        for compressed_spatial_dim in compressed_spatial_dims:
            self.convs.append(
                nn.Sequential(
                    GridFeatureConv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        compressed_spatial_dim=compressed_spatial_dim,
                        stride=stride,
                        up_stride=up_stride,
                    ),
                    GridFeatureTransform(norm(out_channels * compressed_spatial_dim)),
                )
            )

    def forward(self, grid_features_group: GridFeatureGroup) -> GridFeatureGroup:
        r"""Apply convolution and normalization to each grid feature.

        Parameters
        ----------
        grid_features_group : GridFeatureGroup
            Input grid features group.

        Returns
        -------
        GridFeatureGroup
            Processed grid features.
        """
        if len(grid_features_group) != len(self.convs):
            raise ValueError(
                f"GridFeatureGroup size mismatch: {len(grid_features_group)} vs {len(self.convs)}"
            )
        grid_feats = []
        for grid_feat, conv in zip(grid_features_group, self.convs):
            grid_feats.append(conv(grid_feat))
        return GridFeatureGroup(grid_feats)


class GridFeatureGroupTransform(nn.Module):
    r"""Apply a transform to all grid features in a group.

    Parameters
    ----------
    transform : nn.Module
        Transform module to apply to feature tensors.
    in_place : bool, optional, default=True
        Whether to modify features in-place.

    Forward
    -------
    grid_feature_group : GridFeatureGroup
        Input group of grid features.

    Outputs
    -------
    GridFeatureGroup
        Transformed grid features.
    """

    def __init__(self, transform: nn.Module, in_place: bool = True) -> None:
        super().__init__()
        self.transform = transform
        self.in_place = in_place

    def forward(self, grid_feature_group: GridFeatureGroup) -> GridFeatureGroup:
        r"""Apply transform to each grid feature.

        Parameters
        ----------
        grid_feature_group : GridFeatureGroup
            Input grid features group.

        Returns
        -------
        GridFeatureGroup
            Transformed grid features.
        """
        if not self.in_place:
            grid_feature_group = GridFeatureGroup(
                [grid_feature.clone() for grid_feature in grid_feature_group]
            )
        for grid_feature in grid_feature_group:
            grid_feature.features = self.transform(grid_feature.features)
        return grid_feature_group


class GridFeatureConv2DBlocksAndIntraCommunication(nn.Module):
    r"""Combined convolution block with inter-grid communication.

    This is the core building block of the FIGConvNet architecture. It applies:

    1. Factorized 2D convolutions to each grid in the group
    2. Inter-grid communication to share information between factorized grids
    3. Optional channel projection if multiple communication types are used
    4. Non-linear activation

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution kernel size.
    compressed_spatial_dims : Tuple[int]
        Compressed spatial dimension for each factorized grid.
    stride : int, optional, default=1
        Downsampling stride.
    up_stride : int, optional, default=None
        Upsampling stride for transposed convolution.
    communication_types : List[Literal["sum", "mul"]], optional, default=["sum"]
        Types of inter-grid communication.

    Forward
    -------
    grid_features_group : GridFeatureGroup
        Input group of grid features.

    Outputs
    -------
    GridFeatureGroup
        Processed grid features.

    Note
    ----
    This block implements the factorized implicit global convolution proposed
    in the FIGConvNet paper. The 2D convolutions operating on factorized grids
    are equivalent to 3D global convolutions along the compressed dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        compressed_spatial_dims: Tuple[int],
        stride: int = 1,
        up_stride: Optional[int] = None,
        communication_types: List[Literal["sum", "mul"]] = ["sum"],
    ):
        super().__init__()

        # Create convolution blocks for each factorized grid
        self.convs = nn.ModuleList()
        for compressed_spatial_dim in compressed_spatial_dims:
            self.convs.append(
                GridFeatureConv2dBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    compressed_spatial_dim=compressed_spatial_dim,
                    stride=stride,
                    up_stride=up_stride,
                    apply_nonlinear_at_end=False,
                )
            )

        # Inter-grid communication module
        self.intra_communications = GridFeatureGroupIntraCommunications(
            communication_types=communication_types
        )

        # Channel projection if using multiple communication types
        if isinstance(communication_types, str):
            communication_types = [communication_types]
        if len(communication_types) > 1:
            self.proj = GridFeatureGroupConv2dNorm(
                in_channels=out_channels * len(communication_types),
                out_channels=out_channels,
                kernel_size=1,
                compressed_spatial_dims=compressed_spatial_dims,
            )
        else:
            self.proj = nn.Identity()

        # Non-linear activation
        self.nonlinear = GridFeatureGroupTransform(nn.GELU())

    def forward(self, grid_features_group: GridFeatureGroup) -> GridFeatureGroup:
        r"""Apply convolution, communication, and activation.

        Parameters
        ----------
        grid_features_group : GridFeatureGroup
            Input grid features group.

        Returns
        -------
        GridFeatureGroup
            Processed grid features.
        """
        if len(grid_features_group) != len(self.convs):
            raise ValueError(
                f"GridFeatureGroup size mismatch: {len(grid_features_group)} vs {len(self.convs)}"
            )

        # Apply 2D convolutions to each factorized grid
        grid_feats = []
        for grid_feat, conv in zip(grid_features_group, self.convs):
            grid_feats.append(conv(grid_feat))
        grid_features_group = GridFeatureGroup(grid_feats)

        # Apply inter-grid communication
        grid_features_group = self.intra_communications(grid_features_group)

        # Project channels if needed
        grid_features_group = self.proj(grid_features_group)

        # Apply non-linearity
        grid_features_group = self.nonlinear(grid_features_group)

        return grid_features_group


class GridFeatureGroupCat(nn.Module):
    r"""Concatenate two GridFeatureGroups along channel dimension.

    Parameters
    ----------
    None

    Forward
    -------
    group1 : GridFeatureGroup
        First group of grid features.
    group2 : GridFeatureGroup
        Second group of grid features.

    Outputs
    -------
    GridFeatureGroup
        Concatenated grid features.
    """

    def __init__(self):
        super().__init__()
        self.grid_cat = GridFeatureCat()

    def forward(
        self, group1: GridFeatureGroup, group2: GridFeatureGroup
    ) -> GridFeatureGroup:
        r"""Concatenate two grid feature groups.

        Parameters
        ----------
        group1 : GridFeatureGroup
            First group.
        group2 : GridFeatureGroup
            Second group.

        Returns
        -------
        GridFeatureGroup
            Concatenated group.
        """
        if len(group1) != len(group2):
            raise ValueError(
                f"GridFeatureGroup size mismatch: {len(group1)} vs {len(group2)}"
            )
        return GridFeatureGroup(
            [self.grid_cat(g1, g2) for g1, g2 in zip(group1, group2)]
        )


class GridFeatureGroupPadToMatch(nn.Module):
    r"""Pad grid features to match reference resolution.

    Used for skip connections in U-Net architectures where encoder and decoder
    features may have slightly different spatial dimensions due to striding.

    Parameters
    ----------
    None

    Forward
    -------
    grid_features_group_ref : GridFeatureGroup
        Reference group defining target resolution.
    grid_features_group_target : GridFeatureGroup
        Group to be padded/cropped to match reference.

    Outputs
    -------
    GridFeatureGroup
        Padded/cropped grid features matching reference resolution.
    """

    def __init__(self) -> None:
        super().__init__()
        self.match = GridFeaturePadToMatch()

    def forward(
        self,
        grid_features_group_ref: GridFeatureGroup,
        grid_features_group_target: GridFeatureGroup,
    ) -> GridFeatureGroup:
        r"""Pad/crop target to match reference resolution.

        Parameters
        ----------
        grid_features_group_ref : GridFeatureGroup
            Reference group.
        grid_features_group_target : GridFeatureGroup
            Target group to pad/crop.

        Returns
        -------
        GridFeatureGroup
            Target group with matched resolution.
        """
        if len(grid_features_group_ref) != len(grid_features_group_target):
            raise ValueError(
                f"GridFeatureGroup size mismatch: {len(grid_features_group_ref)} vs {len(grid_features_group_target)}"
            )
        grid_features_group_out = [
            self.match(ref, grid_features_group_target[i])
            for i, ref in enumerate(grid_features_group_ref)
        ]
        return GridFeatureGroup(grid_features_group_out)


class GridFeatureGroupToPoint(nn.Module):
    r"""Convert GridFeatureGroup to PointFeatures.

    This module samples features from all factorized grids in a GridFeatureGroup
    at point locations and combines them to produce point-wise features. The
    combination uses both additive and multiplicative aggregation.

    Parameters
    ----------
    grid_in_channels : int
        Number of input channels in grid features.
    point_in_channels : int
        Number of input channels in point features.
    out_channels : int
        Number of output channels (must be even for add/mul split).
    grid_feature_group_size : int
        Number of grids in the group.
    aabb_max : Tuple[float, float, float]
        Maximum coordinates of the bounding box.
    aabb_min : Tuple[float, float, float]
        Minimum coordinates of the bounding box.
    use_rel_pos : bool, optional, default=True
        Whether to use relative positions.
    use_rel_pos_embed : bool, optional, default=False
        Whether to use positional embeddings for relative positions.
    pos_embed_dim : int, optional, default=32
        Dimension of positional embeddings.
    sample_method : Literal["graphconv", "interp"], optional, default="graphconv"
        Method for sampling grid features at point locations.
    neighbor_search_type : Literal["radius", "knn"], optional, default="radius"
        Type of neighbor search for graph convolution.
    knn_k : int, optional, default=16
        Number of neighbors for KNN search.
    reductions : List[REDUCTION_TYPES], optional, default=["mean"]
        Reduction operations for aggregating neighbor features.

    Forward
    -------
    grid_features_group : GridFeatureGroup
        Input group of grid features.
    point_features : PointFeatures
        Input point features defining query locations.

    Outputs
    -------
    PointFeatures
        Output point features sampled from grid features.

    Note
    ----
    The output combines features using both sum and element-wise multiplication
    across all grids, which is why ``out_channels`` must be even (half for add,
    half for multiply).
    """

    def __init__(
        self,
        grid_in_channels: int,
        point_in_channels: int,
        out_channels: int,
        grid_feature_group_size: int,
        aabb_max: Tuple[float, float, float],
        aabb_min: Tuple[float, float, float],
        use_rel_pos: bool = True,
        use_rel_pos_embed: bool = False,
        pos_embed_dim: int = 32,
        sample_method: Literal["graphconv", "interp"] = "graphconv",
        neighbor_search_type: Literal["radius", "knn"] = "radius",
        knn_k: int = 16,
        reductions: List[REDUCTION_TYPES] = ["mean"],
    ) -> None:
        super().__init__()
        self.conv_list = nn.ModuleList()
        if out_channels % 2 != 0:
            raise ValueError("out_channels must be even for add/mul split")

        # Create a grid-to-point converter for each grid in the group
        for i in range(grid_feature_group_size):
            self.conv_list.append(
                GridFeatureToPoint(
                    grid_in_channels=grid_in_channels,
                    point_in_channels=point_in_channels,
                    out_channels=out_channels // 2,
                    aabb_max=aabb_max,
                    aabb_min=aabb_min,
                    use_rel_pos=use_rel_pos,
                    use_rel_pos_embed=use_rel_pos_embed,
                    pos_embed_dim=pos_embed_dim,
                    sample_method=sample_method,
                    neighbor_search_type=neighbor_search_type,
                    knn_k=knn_k,
                    reductions=reductions,
                )
            )

    def forward(
        self, grid_features_group: GridFeatureGroup, point_features: PointFeatures
    ) -> PointFeatures:
        r"""Sample grid features at point locations.

        Parameters
        ----------
        grid_features_group : GridFeatureGroup
            Input grid features to sample from.
        point_features : PointFeatures
            Point features defining query locations.

        Returns
        -------
        PointFeatures
            Sampled point features combining all grids.
        """
        if len(grid_features_group) != len(self.conv_list):
            raise ValueError(
                f"GridFeatureGroup size mismatch: {len(grid_features_group)} vs {len(self.conv_list)}"
            )

        # Sample from first grid
        out_point_features: PointFeatures = self.conv_list[0](
            grid_features_group[0], point_features
        )

        # Initialize additive and multiplicative aggregations
        out_point_features_add: PointFeatures = out_point_features
        out_point_features_mul: PointFeatures = out_point_features

        # Aggregate features from remaining grids
        for i in range(1, len(grid_features_group)):
            curr = self.conv_list[i](grid_features_group[i], point_features)
            out_point_features_add += curr
            out_point_features_mul *= curr

        # Concatenate additive and multiplicative features
        out_point_features = PointFeatures(
            vertices=point_features.vertices,
            features=torch.cat(
                (out_point_features_add.features, out_point_features_mul.features),
                dim=-1,
            ),
        )
        return out_point_features


class AttentionPool(nn.Module):
    r"""Attention-based pooling for sequences.

    Pools a sequence of features to a single feature vector using attention
    weights computed from max-pooled query.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    num_heads : int, optional, default=2
        Number of attention heads.
    dropout : float, optional, default=0.0
        Dropout probability.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, N, C)`.

    Outputs
    -------
    torch.Tensor
        Pooled tensor of shape :math:`(B, C_{out})`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.qkv = nn.Linear(in_channels, out_channels * 3)
        self.out = nn.Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    @profile
    def forward(
        self,
        x: Float[Tensor, "B N C"],
    ) -> Float[Tensor, "B C"]:
        r"""Apply attention pooling.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(B, N, C)`.

        Returns
        -------
        torch.Tensor
            Pooled tensor of shape :math:`(B, C_{out})`.
        """
        B, N, C = x.shape

        # Compute Q, K, V projections
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Max-pool query across sequence
        q = q.max(dim=2, keepdim=True).values

        # Compute attention weights
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention and reshape
        x = (attn @ v).reshape(B, -1)
        x = self.out(x)

        return x


class GridFeaturePool(nn.Module):
    r"""Pool GridFeatures to a single feature vector.

    Applies a 1x1 convolution followed by spatial pooling to reduce grid
    features to a single vector per batch element.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    compressed_spatial_dim : int
        Size of the compressed spatial dimension.
    pooling_type : Literal["max", "mean", "attention"], optional, default="max"
        Type of spatial pooling.

    Forward
    -------
    grid_features : GridFeatures
        Input grid features in compressed format.

    Outputs
    -------
    torch.Tensor
        Pooled features of shape :math:`(B, C_{out})`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        compressed_spatial_dim: int,
        pooling_type: Literal["max", "mean", "attention"] = "max",
    ):
        super().__init__()

        # 1x1 convolution to transform channels
        self.conv = nn.Conv2d(
            in_channels=in_channels * compressed_spatial_dim,
            out_channels=out_channels,
            kernel_size=1,
        )

        # Pooling operation
        if pooling_type == "attention":
            self.pool = AttentionPool(out_channels, out_channels)
        elif pooling_type == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif pooling_type == "mean":
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            raise NotImplementedError(f"Unknown pooling type: {pooling_type}")

        self.pooling_type = pooling_type
        self.norm = nn.LayerNorm(out_channels)

    def forward(
        self,
        grid_features: GridFeatures,
    ) -> Float[Tensor, "B C"]:
        r"""Pool grid features to vector.

        Parameters
        ----------
        grid_features : GridFeatures
            Input grid features.

        Returns
        -------
        torch.Tensor
            Pooled features of shape :math:`(B, C_{out})`.
        """
        features = grid_features.features
        if features.ndim != 4:
            raise ValueError(
                f"Features must be in compressed format with BxCxHxW (4D), got {features.ndim}D"
            )

        # Apply 1x1 conv and flatten spatial dimensions
        features = self.conv(features)
        features = features.flatten(2, 3)

        # Apply pooling
        if self.pooling_type == "attention":
            features = features.transpose(1, 2)
        pooled_feat = self.pool(features)

        # Normalize and return
        return self.norm(pooled_feat.squeeze(-1))


class GridFeatureGroupPool(nn.Module):
    r"""Pool GridFeatureGroup to a single feature vector.

    Pools each grid in the group independently and concatenates the results.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels per grid.
    compressed_spatial_dims : Tuple[int]
        Compressed spatial dimensions for each grid.
    pooling_type : Literal["max", "mean", "attention"], optional, default="max"
        Type of spatial pooling.

    Forward
    -------
    grid_features_group : GridFeatureGroup
        Input group of grid features.

    Outputs
    -------
    torch.Tensor
        Pooled features of shape :math:`(B, n\_grids \cdot C_{out})`.

    Note
    ----
    The output dimension is ``len(compressed_spatial_dims) * out_channels``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        compressed_spatial_dims: Tuple[int],
        pooling_type: Literal["max", "mean", "attention"] = "max",
    ):
        super().__init__()
        self.pools = nn.ModuleList()
        for compressed_spatial_dim in compressed_spatial_dims:
            self.pools.append(
                GridFeaturePool(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    compressed_spatial_dim=compressed_spatial_dim,
                    pooling_type=pooling_type,
                )
            )

    def forward(
        self,
        grid_features_group: GridFeatureGroup,
    ) -> Float[Tensor, "B 3C"]:
        r"""Pool all grids and concatenate.

        Parameters
        ----------
        grid_features_group : GridFeatureGroup
            Input grid features group.

        Returns
        -------
        torch.Tensor
            Concatenated pooled features.
        """
        if len(grid_features_group) != len(self.pools):
            raise ValueError(
                f"GridFeatureGroup size mismatch: {len(grid_features_group)} vs {len(self.pools)}"
            )

        # Pool each grid independently
        pooled_features = []
        for grid_features, pool in zip(grid_features_group, self.pools):
            pooled_features.append(pool(grid_features))

        # Concatenate pooled features
        return torch.cat(pooled_features, dim=-1)
