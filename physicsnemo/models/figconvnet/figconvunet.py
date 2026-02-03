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
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.models.figconvnet.base_model import BaseModel
from physicsnemo.models.figconvnet.components.encodings import SinusoidalEncoding
from physicsnemo.models.figconvnet.components.mlp import MLP
from physicsnemo.models.figconvnet.components.reductions import REDUCTION_TYPES
from physicsnemo.models.figconvnet.geometries import (
    GridFeaturesMemoryFormat,
    PointFeatures,
)
from physicsnemo.models.figconvnet.grid_feature_group import (
    GridFeatureConv2DBlocksAndIntraCommunication,
    GridFeatureGroup,
    GridFeatureGroupPadToMatch,
    GridFeatureGroupPool,
    GridFeatureGroupToPoint,
)
from physicsnemo.models.figconvnet.point_feature_conv import (
    PointFeatureTransform,
)
from physicsnemo.models.figconvnet.point_feature_grid_conv import (
    GridFeatureMemoryFormatConverter,
)
from physicsnemo.models.figconvnet.point_feature_grid_ops import PointFeatureToGrid
from physicsnemo.utils.profiling import profile

memory_format_to_axis_index = {
    GridFeaturesMemoryFormat.b_xc_y_z: 0,
    GridFeaturesMemoryFormat.b_yc_x_z: 1,
    GridFeaturesMemoryFormat.b_zc_x_y: 2,
    GridFeaturesMemoryFormat.b_x_y_z_c: -1,
}


class VerticesToPointFeatures(nn.Module):
    r"""Convert 3D vertices (XYZ coordinates) to point features.

    This module applies sinusoidal positional encoding to the input vertices
    and optionally transforms them through an MLP to produce point features.

    Parameters
    ----------
    embed_dim : int
        Dimension of the sinusoidal positional encoding.
    out_features : int, optional, default=32
        Number of output feature channels after MLP transformation.
    use_mlp : bool, optional, default=True
        Whether to apply an MLP to the encoded vertices.
    pos_embed_range : float, optional, default=2.0
        Range parameter for the sinusoidal encoding.

    Forward
    -------
    vertices : torch.Tensor
        Input vertices of shape :math:`(B, N, 3)` where :math:`B` is the batch
        size, :math:`N` is the number of points, and 3 represents XYZ coordinates.

    Outputs
    -------
    PointFeatures
        Point features object containing the original vertices and computed
        features of shape :math:`(B, N, C_{out})`.

    Examples
    --------
    >>> import torch
    >>> converter = VerticesToPointFeatures(embed_dim=32, out_features=64)
    >>> vertices = torch.randn(2, 1000, 3)
    >>> point_features = converter(vertices)
    >>> point_features.features.shape
    torch.Size([2, 1000, 64])
    """

    def __init__(
        self,
        embed_dim: int,
        out_features: Optional[int] = 32,
        use_mlp: Optional[bool] = True,
        pos_embed_range: Optional[float] = 2.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.out_features = out_features
        self.use_mlp = use_mlp
        self.pos_embed_range = pos_embed_range
        self.pos_embed = SinusoidalEncoding(embed_dim, pos_embed_range)
        if self.use_mlp:
            self.mlp = MLP(3 * embed_dim, out_features, [])

    def forward(self, vertices: Float[Tensor, "batch num_points 3"]) -> PointFeatures:
        r"""Transform vertices to point features.

        Parameters
        ----------
        vertices : torch.Tensor
            Input vertices of shape :math:`(B, N, 3)`.

        Returns
        -------
        PointFeatures
            Point features with encoded vertex information.
        """
        # Input validation (MOD-005)
        if not torch.compiler.is_compiling():
            if vertices.ndim != 3:
                raise ValueError(
                    f"Expected 3D vertices tensor of shape (B, N, 3), "
                    f"got {vertices.ndim}D tensor with shape {tuple(vertices.shape)}"
                )
            if vertices.shape[2] != 3:
                raise ValueError(
                    f"Expected vertices with 3 coordinates (XYZ), "
                    f"got {vertices.shape[2]} coordinates"
                )

        # Apply sinusoidal positional encoding to vertices
        vert_embed = self.pos_embed(vertices)  # (B, N, 3 * embed_dim)

        # Optionally transform through MLP
        if self.use_mlp:
            vert_embed = self.mlp(vert_embed)  # (B, N, out_features)

        return PointFeatures(vertices, vert_embed)


@dataclass
class MetaData(ModelMetaData):
    """Metadata for the FIGConvUNet model."""

    name: str = "FIGConvUNet"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class FIGConvUNet(BaseModel):
    r"""Factorized Implicit Global Convolutional U-Net.

    The FIGConvUNet is a U-Net architecture that uses factorized implicit global
    convolutional layers to create a U-shaped encoder-decoder architecture. The
    key advantage of using FIGConvolution is that it can handle high resolution
    3D data efficiently using a set of factorized 2D grids that implicitly
    represent 3D features.

    The architecture processes point cloud data by:

    1. Converting point features to multiple factorized grid representations
    2. Processing through encoder blocks with downsampling
    3. Optionally computing scalar outputs via pooling
    4. Upsampling through decoder blocks with skip connections
    5. Converting grid features back to point features

    Based on the paper: `FIGConvNet: Fine-Grained Implicit 3D Convolution
    <https://arxiv.org/abs/2303.08042>`_.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels per point.
    out_channels : int
        Number of output feature channels per point.
    kernel_size : int
        Kernel size for the 2D convolutions in the grid feature blocks.
    hidden_channels : List[int]
        List of hidden channel dimensions for each level of the U-Net.
        Length should be ``num_levels + 1``.
    num_levels : int, optional, default=3
        Number of levels in the U-Net encoder/decoder.
    num_down_blocks : Union[int, List[int]], optional, default=1
        Number of convolutional blocks per downsampling level. Can be a single
        int (applied to all levels) or a list of ints per level.
    num_up_blocks : Union[int, List[int]], optional, default=1
        Number of convolutional blocks per upsampling level. Can be a single
        int (applied to all levels) or a list of ints per level.
    mlp_channels : List[int], optional, default=[512, 512]
        Channel dimensions for the MLP used in scalar output prediction.
    aabb_max : Tuple[float, float, float], optional, default=(1.0, 1.0, 1.0)
        Maximum coordinates of the axis-aligned bounding box.
    aabb_min : Tuple[float, float, float], optional, default=(0.0, 0.0, 0.0)
        Minimum coordinates of the axis-aligned bounding box.
    voxel_size : float, optional, default=None
        Voxel size for grid construction. If None, resolution is used directly.
    resolution_memory_format_pairs : List[Tuple], optional
        List of tuples specifying (memory_format, resolution) for each factorized
        grid. Default creates three orthogonal factorized grids.
    use_rel_pos : bool, optional, default=True
        Whether to use relative positions in point-grid convolutions.
    use_rel_pos_embed : bool, optional, default=True
        Whether to use positional embeddings for relative positions.
    pos_encode_dim : int, optional, default=32
        Dimension of positional encoding.
    communication_types : List[Literal["mul", "sum"]], optional, default=["sum"]
        Types of inter-grid communication operations.
    to_point_sample_method : Literal["graphconv", "interp"], optional, default="graphconv"
        Method for sampling grid features to points.
    neighbor_search_type : Literal["knn", "radius"], optional, default="radius"
        Type of neighbor search for point-grid operations.
    knn_k : int, optional, default=16
        Number of neighbors for KNN search.
    reductions : List[REDUCTION_TYPES], optional, default=["mean"]
        Reduction operations for aggregating neighbor features.
    use_scalar_output : bool, optional, default=True
        Whether to compute scalar output (e.g., drag coefficient).
    has_input_features : bool, optional, default=False
        Whether input already has features (True) or just vertices (False).
    pooling_type : Literal["attention", "max", "mean"], optional, default="max"
        Type of pooling for scalar output computation.
    pooling_layers : List[int], optional, default=None
        Which layers to pool from for scalar output. Defaults to [num_levels].

    Forward
    -------
    vertices : torch.Tensor
        Input point cloud vertices of shape :math:`(B, N, 3)`.
    features : torch.Tensor, optional
        Input point features of shape :math:`(B, N, C_{in})`. If None and
        ``has_input_features=False``, features are computed from vertices.

    Outputs
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - Point features of shape :math:`(B, N, C_{out})`
        - Scalar prediction of shape :math:`(B, 1)` if ``use_scalar_output=True``,
          otherwise None

    Examples
    --------
    >>> import torch
    >>> model = FIGConvUNet(
    ...     in_channels=32,
    ...     out_channels=3,
    ...     kernel_size=3,
    ...     hidden_channels=[32, 64, 128, 256],
    ...     num_levels=3,
    ... )
    >>> vertices = torch.randn(2, 10000, 3)
    >>> point_features, scalar_pred = model(vertices)
    >>> point_features.shape
    torch.Size([2, 10000, 3])
    >>> scalar_pred.shape
    torch.Size([2, 1])

    Note
    ----
    This model requires the ``warp`` package for efficient neighbor search
    operations on GPU. For CPU execution, consider using smaller resolutions.

    See Also
    --------
    :class:`~physicsnemo.models.figconvnet.grid_feature_group.GridFeatureGroup`
    :class:`~physicsnemo.models.figconvnet.geometries.PointFeatures`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_channels: List[int],
        num_levels: int = 3,
        num_down_blocks: Union[int, List[int]] = 1,
        num_up_blocks: Union[int, List[int]] = 1,
        mlp_channels: Optional[List[int]] = None,
        aabb_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        aabb_min: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        voxel_size: Optional[float] = None,
        resolution_memory_format_pairs: List[
            Tuple[GridFeaturesMemoryFormat | str, Tuple[int, int, int]]
        ] = [
            (GridFeaturesMemoryFormat.b_xc_y_z, (2, 128, 128)),
            (GridFeaturesMemoryFormat.b_yc_x_z, (128, 2, 128)),
            (GridFeaturesMemoryFormat.b_zc_x_y, (128, 128, 2)),
        ],
        use_rel_pos: bool = True,
        use_rel_pos_embed: bool = True,
        pos_encode_dim: int = 32,
        communication_types: List[Literal["mul", "sum"]] = ["sum"],
        to_point_sample_method: Literal["graphconv", "interp"] = "graphconv",
        neighbor_search_type: Literal["knn", "radius"] = "radius",
        knn_k: int = 16,
        reductions: List[REDUCTION_TYPES] = ["mean"],
        use_scalar_output: bool = True,
        has_input_features: bool = False,
        pooling_type: Literal["attention", "max", "mean"] = "max",
        pooling_layers: List[int] = None,
    ):
        super().__init__(meta=MetaData())

        # Handle default for mlp_channels
        if mlp_channels is None:
            mlp_channels = [512, 512]

        # Store configuration attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_levels = num_levels
        self.use_scalar_output = use_scalar_output
        self.has_input_features = has_input_features

        # Compute AABB dimensions
        self.aabb_length = torch.tensor(aabb_max) - torch.tensor(aabb_min)
        self.min_voxel_edge_length = torch.tensor([np.inf, np.inf, np.inf])

        # Initialize factorized grid conversion modules
        compressed_spatial_dims = []
        self.grid_feature_group_size = len(resolution_memory_format_pairs)
        self.point_feature_to_grids = nn.ModuleList()

        # Build point-to-grid conversion for each factorized plane
        for mem_fmt, res in resolution_memory_format_pairs:
            if isinstance(mem_fmt, str):
                mem_fmt = GridFeaturesMemoryFormat[mem_fmt]
            compressed_axis = memory_format_to_axis_index[mem_fmt]
            compressed_spatial_dims.append(res[compressed_axis])

            # Create point-to-grid conversion pipeline
            to_grid = nn.Sequential(
                PointFeatureToGrid(
                    in_channels=in_channels,
                    out_channels=hidden_channels[0],
                    aabb_max=aabb_max,
                    aabb_min=aabb_min,
                    voxel_size=voxel_size,
                    resolution=res,
                    use_rel_pos=use_rel_pos,
                    use_rel_pos_encode=use_rel_pos_embed,
                    pos_encode_dim=pos_encode_dim,
                    reductions=reductions,
                    neighbor_search_type=neighbor_search_type,
                    knn_k=knn_k,
                ),
                GridFeatureMemoryFormatConverter(
                    memory_format=mem_fmt,
                ),
            )
            self.point_feature_to_grids.append(to_grid)

            # Track minimum voxel size for later use
            current_voxel_size = self.aabb_length / torch.tensor(res)
            self.min_voxel_edge_length = torch.min(
                self.min_voxel_edge_length, current_voxel_size
            )

        self.compressed_spatial_dims = compressed_spatial_dims

        # Initialize encoder (down) and decoder (up) blocks
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        # Convert single int to list if needed
        if isinstance(num_down_blocks, int):
            num_down_blocks = [num_down_blocks] * (num_levels + 1)
        if isinstance(num_up_blocks, int):
            num_up_blocks = [num_up_blocks] * (num_levels + 1)

        # Build U-Net encoder and decoder for each level
        for level in range(num_levels):
            # Build downsampling blocks for this level
            down_block = [
                GridFeatureConv2DBlocksAndIntraCommunication(
                    in_channels=hidden_channels[level],
                    out_channels=hidden_channels[level + 1],
                    kernel_size=kernel_size,
                    stride=2,
                    compressed_spatial_dims=compressed_spatial_dims,
                    communication_types=communication_types,
                )
            ]
            for _ in range(1, num_down_blocks[level]):
                down_block.extend(
                    [
                        GridFeatureConv2DBlocksAndIntraCommunication(
                            in_channels=hidden_channels[level + 1],
                            out_channels=hidden_channels[level + 1],
                            kernel_size=kernel_size,
                            stride=1,
                            compressed_spatial_dims=compressed_spatial_dims,
                            communication_types=communication_types,
                        )
                    ]
                )
            down_block = nn.Sequential(*down_block)
            self.down_blocks.append(down_block)

            # Build upsampling blocks for this level
            up_block = [
                GridFeatureConv2DBlocksAndIntraCommunication(
                    in_channels=hidden_channels[level + 1],
                    out_channels=hidden_channels[level],
                    kernel_size=kernel_size,
                    up_stride=2,
                    compressed_spatial_dims=compressed_spatial_dims,
                    communication_types=communication_types,
                )
            ]
            for _ in range(1, num_up_blocks[level]):
                up_block.extend(
                    [
                        GridFeatureConv2DBlocksAndIntraCommunication(
                            in_channels=hidden_channels[level],
                            out_channels=hidden_channels[level],
                            kernel_size=kernel_size,
                            up_stride=1,
                            compressed_spatial_dims=compressed_spatial_dims,
                            communication_types=communication_types,
                        )
                    ]
                )
            up_block = nn.Sequential(*up_block)
            self.up_blocks.append(up_block)

        # Memory format converter for output
        self.convert_to_orig = GridFeatureMemoryFormatConverter(
            memory_format=GridFeaturesMemoryFormat.b_x_y_z_c
        )

        # Initialize scalar output branch if enabled
        if use_scalar_output:
            if pooling_layers is None:
                pooling_layers = [num_levels]
            else:
                if not isinstance(pooling_layers, list):
                    raise ValueError(
                        f"pooling_layers must be a list, got {type(pooling_layers)}."
                    )
                for layer in pooling_layers:
                    if layer > num_levels:
                        raise ValueError(
                            f"pooling_layer {layer} is greater than num_levels {num_levels}."
                        )

            self.pooling_layers = pooling_layers

            # Build pooling modules for each specified layer
            grid_pools = [
                GridFeatureGroupPool(
                    in_channels=hidden_channels[layer],
                    out_channels=mlp_channels[0],
                    compressed_spatial_dims=self.compressed_spatial_dims,
                    pooling_type=pooling_type,
                )
                for layer in pooling_layers
            ]
            self.grid_pools = nn.ModuleList(grid_pools)

            # MLP for scalar prediction
            self.mlp = MLP(
                mlp_channels[0]
                * len(self.compressed_spatial_dims)
                * len(pooling_layers),
                mlp_channels[-1],
                mlp_channels,
                use_residual=True,
                activation=nn.GELU,
            )
            self.mlp_projection = nn.Linear(mlp_channels[-1], 1)

        # Grid-to-point conversion module
        self.to_point = GridFeatureGroupToPoint(
            grid_in_channels=hidden_channels[0],
            point_in_channels=in_channels,
            out_channels=hidden_channels[0] * 2,
            grid_feature_group_size=self.grid_feature_group_size,
            aabb_max=aabb_max,
            aabb_min=aabb_min,
            use_rel_pos=use_rel_pos,
            use_rel_pos_embed=use_rel_pos_embed,
            pos_embed_dim=pos_encode_dim,
            sample_method=to_point_sample_method,
            neighbor_search_type=neighbor_search_type,
            knn_k=knn_k,
            reductions=reductions,
        )

        # Final projection to output channels
        self.projection = PointFeatureTransform(
            nn.Sequential(
                nn.Linear(hidden_channels[0] * 2, hidden_channels[0] * 2),
                nn.LayerNorm(hidden_channels[0] * 2),
                nn.GELU(),
                nn.Linear(hidden_channels[0] * 2, out_channels),
            )
        )

        # Padding module for skip connections
        self.pad_to_match = GridFeatureGroupPadToMatch()

        # Vertex-to-feature conversion if no input features provided
        if not has_input_features:
            self.vertex_to_point_features = VerticesToPointFeatures(
                embed_dim=pos_encode_dim,
                out_features=hidden_channels[0],
                use_mlp=True,
                pos_embed_range=aabb_max[0] - aabb_min[0],
            )

    @profile
    def _grid_forward(self, point_features: PointFeatures):
        r"""Process point features through the grid-based U-Net.

        Parameters
        ----------
        point_features : PointFeatures
            Input point features to process.

        Returns
        -------
        Tuple[GridFeatureGroup, Optional[torch.Tensor]]
            - Processed grid features
            - Scalar prediction if ``use_scalar_output=True``, else None
        """
        # Convert point features to factorized grid representations
        grid_feature_group = GridFeatureGroup(
            [to_grid(point_features) for to_grid in self.point_feature_to_grids]
        )

        # Encoder: store features at each level for skip connections
        down_grid_feature_groups = [grid_feature_group]
        for down_block in self.down_blocks:
            out_features = down_block(down_grid_feature_groups[-1])
            down_grid_feature_groups.append(out_features)

        # Compute scalar output if enabled
        drag_pred = None
        if self.use_scalar_output:
            # Pool features from specified layers
            pooled_feats = []
            for grid_pool, layer in zip(self.grid_pools, self.pooling_layers):
                pooled_feats.append(grid_pool(down_grid_feature_groups[layer]))

            # Concatenate pooled features
            if len(pooled_feats) > 1:
                pooled_feats = torch.cat(pooled_feats, dim=-1)
            else:
                pooled_feats = pooled_feats[0]

            # Project to scalar output
            drag_pred = self.mlp_projection(self.mlp(pooled_feats))

        # Decoder: upsample with skip connections
        for level in reversed(range(self.num_levels)):
            # Upsample features
            up_grid_features = self.up_blocks[level](
                down_grid_feature_groups[level + 1]
            )

            # Add skip connection from encoder
            padded_down_features = self.pad_to_match(
                up_grid_features, down_grid_feature_groups[level]
            )
            up_grid_features = up_grid_features + padded_down_features
            down_grid_feature_groups[level] = up_grid_features

        # Convert to standard memory format for output
        grid_features = self.convert_to_orig(down_grid_feature_groups[0])

        return grid_features, drag_pred

    @profile
    def forward(
        self,
        vertices: Float[Tensor, "batch num_points 3"],
        features: Optional[Float[Tensor, "batch num_points in_channels"]] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Forward pass of the FIGConvUNet.

        Parameters
        ----------
        vertices : torch.Tensor
            Input point cloud vertices of shape :math:`(B, N, 3)`.
        features : torch.Tensor, optional
            Input point features of shape :math:`(B, N, C_{in})`.

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            - Output point features of shape :math:`(B, N, C_{out})`
            - Scalar prediction of shape :math:`(B, 1)` or None
        """
        # Input validation (MOD-005)
        if not torch.compiler.is_compiling():
            # Validate vertices shape
            if vertices.ndim != 3:
                raise ValueError(
                    f"Expected 3D vertices tensor of shape (B, N, 3), "
                    f"got {vertices.ndim}D tensor with shape {tuple(vertices.shape)}"
                )
            if vertices.shape[2] != 3:
                raise ValueError(
                    f"Expected vertices with 3 coordinates (XYZ), "
                    f"got {vertices.shape[2]} coordinates"
                )

            # Validate features if provided
            if features is not None:
                if features.ndim != 3:
                    raise ValueError(
                        f"Expected 3D features tensor of shape (B, N, C), "
                        f"got {features.ndim}D tensor with shape {tuple(features.shape)}"
                    )
                if features.shape[0] != vertices.shape[0]:
                    raise ValueError(
                        f"Batch size mismatch: vertices has batch size {vertices.shape[0]}, "
                        f"features has batch size {features.shape[0]}"
                    )
                if features.shape[1] != vertices.shape[1]:
                    raise ValueError(
                        f"Number of points mismatch: vertices has {vertices.shape[1]} points, "
                        f"features has {features.shape[1]} points"
                    )
                if features.shape[2] != self.in_channels:
                    raise ValueError(
                        f"Expected {self.in_channels} input feature channels, "
                        f"got {features.shape[2]} channels"
                    )

        # Convert vertices to point features if no features provided
        if features is None:
            point_features = self.vertex_to_point_features(vertices)
        else:
            point_features = PointFeatures(vertices, features)

        # Process through grid-based U-Net
        grid_features, drag_pred = self._grid_forward(point_features)

        # Convert grid features back to point features
        out_point_features = self.to_point(grid_features, point_features)

        # Project to output channels
        out_point_features = self.projection(out_point_features)

        return out_point_features.features, drag_pred
