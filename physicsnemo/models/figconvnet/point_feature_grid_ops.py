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
    """AABBGridFeatures."""

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
    """PointFeatureToGrid."""

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
            assert voxel_size is not None
            resolution = (
                int((aabb_max[0] - aabb_min[0]) / voxel_size),
                int((aabb_max[1] - aabb_min[1]) / voxel_size),
                int((aabb_max[2] - aabb_min[2]) / voxel_size),
            )
        if voxel_size is None:
            assert resolution is not None
        if isinstance(resolution, Tensor):
            resolution = resolution.tolist()
        self.resolution = resolution
        for i in range(3):
            assert aabb_max[i] > aabb_min[i]
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
    """GridFeatureToPoint."""

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
        out_point_features = self.conv(grid_features, point_features)
        if self.sample_method == "interp":
            out_point_features = self.transform(out_point_features)
        return out_point_features


class GridFeatureToPointGraphConv(nn.Module):
    """GridFeatureToPointGraphConv."""

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
    """GridFeatureToPointInterp."""

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
        # Use F.interpolate to interpolate grid features to point features
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
    """GridFeatureCat."""

    def forward(
        self, grid_features: GridFeatures, other_grid_features: GridFeatures
    ) -> GridFeatures:
        assert grid_features.memory_format == other_grid_features.memory_format
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
