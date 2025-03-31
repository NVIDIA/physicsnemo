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
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

try:
    from jaxtyping import Float
except ImportError:
    raise ImportError(
        "FIGConvUNet requires jaxtyping package, install using `pip install jaxtyping`"
    )

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

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
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.utils.profiling import profile

memory_format_to_axis_index = {
    GridFeaturesMemoryFormat.b_xc_y_z: 0,
    GridFeaturesMemoryFormat.b_yc_x_z: 1,
    GridFeaturesMemoryFormat.b_zc_x_y: 2,
    GridFeaturesMemoryFormat.b_x_y_z_c: -1,
}


class VerticesToPointFeatures(nn.Module):
    """
    VerticesToPointFeatures module converts the 3D vertices (XYZ coordinates) to point features.

    The module applies sinusoidal encoding to the vertices and optionally applies
    an MLP to the encoded vertices.
    """

    def __init__(
        self,
        embed_dim: int,
        out_features: Optional[int] = 32,
        use_mlp: Optional[bool] = True,
        pos_embed_range: Optional[float] = 2.0,
    ) -> None:
        super().__init__()
        self.pos_embed = SinusoidalEncoding(embed_dim, pos_embed_range)
        self.use_mlp = use_mlp
        if self.use_mlp:
            self.mlp = MLP(3 * embed_dim, out_features, [])

    def forward(self, vertices: Float[Tensor, "B N 3"]) -> PointFeatures:
        assert (
            vertices.ndim == 3
        ), f"Expected 3D vertices of shape BxNx3, got {vertices.shape}"
        vert_embed = self.pos_embed(vertices)
        if self.use_mlp:
            vert_embed = self.mlp(vert_embed)
        return PointFeatures(vertices, vert_embed)


@dataclass
class MetaData(ModelMetaData):
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
    """Factorized Implicit Global Convolutional U-Net.

    The FIGConvUNet is a U-Net architecture that uses factorized implicit global
    convolutional layers to create U-shaped architecture. The advantage of using
    FIGConvolution is that it can handle high resolution 3D data efficiently
    using a set of factorized grids.
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
        mlp_channels: List[int] = [512, 512],
        aabb_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        aabb_min: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        voxel_size: Optional[float] = None,
        resolution_memory_format_pairs: List[
            Tuple[GridFeaturesMemoryFormat, Tuple[int, int, int]]
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
        drag_loss_weight: Optional[float] = None,
        pooling_type: Literal["attention", "max", "mean"] = "max",
        pooling_layers: List[int] = None,
    ):
        super().__init__(meta=MetaData())
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_levels = num_levels
        compressed_spatial_dims = []
        self.grid_feature_group_size = len(resolution_memory_format_pairs)
        self.point_feature_to_grids = nn.ModuleList()
        self.aabb_length = torch.tensor(aabb_max) - torch.tensor(aabb_min)
        self.min_voxel_edge_length = torch.tensor([np.inf, np.inf, np.inf])
        for mem_fmt, res in resolution_memory_format_pairs:
            compressed_axis = memory_format_to_axis_index[mem_fmt]
            compressed_spatial_dims.append(res[compressed_axis])
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
            # Compute voxel size
            voxel_size = self.aabb_length / torch.tensor(res)
            self.min_voxel_edge_length = torch.min(
                self.min_voxel_edge_length, voxel_size
            )
        self.compressed_spatial_dims = compressed_spatial_dims

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        if isinstance(num_down_blocks, int):
            num_down_blocks = [num_down_blocks] * (num_levels + 1)
        if isinstance(num_up_blocks, int):
            num_up_blocks = [num_up_blocks] * (num_levels + 1)

        for level in range(num_levels):
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
                down_block.append(  # noqa: PERF401
                    GridFeatureConv2DBlocksAndIntraCommunication(
                        in_channels=hidden_channels[level + 1],
                        out_channels=hidden_channels[level + 1],
                        kernel_size=kernel_size,
                        stride=1,
                        compressed_spatial_dims=compressed_spatial_dims,
                        communication_types=communication_types,
                    )
                )
            down_block = nn.Sequential(*down_block)
            self.down_blocks.append(down_block)
            # Add up blocks
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
                up_block.append(  # noqa: PERF401
                    GridFeatureConv2DBlocksAndIntraCommunication(
                        in_channels=hidden_channels[level],
                        out_channels=hidden_channels[level],
                        kernel_size=kernel_size,
                        up_stride=1,
                        compressed_spatial_dims=compressed_spatial_dims,
                        communication_types=communication_types,
                    )
                )
            up_block = nn.Sequential(*up_block)
            self.up_blocks.append(up_block)
        self.convert_to_orig = GridFeatureMemoryFormatConverter(
            memory_format=GridFeaturesMemoryFormat.b_x_y_z_c
        )

        if pooling_layers is None:
            pooling_layers = [num_levels]
        else:
            assert isinstance(
                pooling_layers, list
            ), f"pooling_layers must be a list, got {type(pooling_layers)}."
            for layer in pooling_layers:
                assert (
                    layer <= num_levels
                ), f"pooling_layer {layer} is greater than num_levels {num_levels}."
        self.pooling_layers = pooling_layers
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

        self.mlp = MLP(
            mlp_channels[0] * len(self.compressed_spatial_dims) * len(pooling_layers),
            mlp_channels[-1],
            mlp_channels,
            use_residual=True,
            activation=nn.GELU,
        )
        self.mlp_projection = nn.Linear(mlp_channels[-1], 1)
        # nn.Sigmoid(),

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
        self.projection = PointFeatureTransform(
            nn.Sequential(
                nn.Linear(hidden_channels[0] * 2, hidden_channels[0] * 2),
                nn.LayerNorm(hidden_channels[0] * 2),
                nn.GELU(),
                nn.Linear(hidden_channels[0] * 2, out_channels),
            )
        )

        self.pad_to_match = GridFeatureGroupPadToMatch()

        vertex_to_point_features = VerticesToPointFeatures(
            embed_dim=pos_encode_dim,
            out_features=hidden_channels[0],
            use_mlp=True,
            pos_embed_range=aabb_max[0] - aabb_min[0],
        )

        self.vertex_to_point_features = vertex_to_point_features
        if drag_loss_weight is not None:
            self.drag_loss_weight = drag_loss_weight

    @profile
    def _grid_forward(self, point_features: PointFeatures):
        grid_feature_group = GridFeatureGroup(
            [to_grid(point_features) for to_grid in self.point_feature_to_grids]
        )
        down_grid_feature_groups = [grid_feature_group]
        for down_block in self.down_blocks:
            out_features = down_block(down_grid_feature_groups[-1])
            down_grid_feature_groups.append(out_features)

        # Drag prediction
        pooled_feats = []
        for grid_pool, layer in zip(self.grid_pools, self.pooling_layers):
            pooled_feats.append(grid_pool(down_grid_feature_groups[layer]))
        if len(pooled_feats) > 1:
            pooled_feats = torch.cat(pooled_feats, dim=-1)
        else:
            pooled_feats = pooled_feats[0]
        drag_pred = self.mlp_projection(self.mlp(pooled_feats))

        for level in reversed(range(self.num_levels)):
            up_grid_features = self.up_blocks[level](
                down_grid_feature_groups[level + 1]
            )
            padded_down_features = self.pad_to_match(
                up_grid_features, down_grid_feature_groups[level]
            )
            up_grid_features = up_grid_features + padded_down_features
            down_grid_feature_groups[level] = up_grid_features

        grid_features = self.convert_to_orig(down_grid_feature_groups[0])
        return grid_features, drag_pred

    @profile
    def forward(
        self,
        vertices: Float[Tensor, "B N 3"],
        features: Optional[Float[Tensor, "B N C"]] = None,
    ) -> Tensor:
        if features is None:
            point_features = self.vertex_to_point_features(vertices)
        else:
            point_features = PointFeatures(vertices, features)

        grid_features, drag_pred = self._grid_forward(point_features)
        out_point_features = self.to_point(grid_features, point_features)
        out_point_features = self.projection(out_point_features)
        return out_point_features.features, drag_pred
