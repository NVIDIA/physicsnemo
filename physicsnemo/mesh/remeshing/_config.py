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

"""Configuration objects for remeshing backends."""

import math
from dataclasses import dataclass

_MAX_HASH_GRID_RESOLUTION = 256


@dataclass(frozen=True, slots=True)
class WarpRemeshOptions:
    """Performance and initialization controls for Warp remeshing.

    Parameters
    ----------
    search_radius_scale : float, optional
        Hash-grid query radius relative to ``sqrt(surface_area / n_clusters)``.
        Smaller values inspect fewer candidate centroids but can trigger the
        exact global-search fallback on sparse regions. Default is ``1.6``.
    voxel_width_scale : float, optional
        Spatial-stratification voxel width relative to
        ``sqrt(surface_area / n_clusters)``. Default is ``1.15``.
    hash_grid_resolution : int, optional
        Resolution of each axis of the sparse Warp centroid hash grid. Must not
        exceed ``256`` because its scratch storage grows cubically. Default is
        ``128``.
    farthest_point_threshold : int, optional
        Use farthest-point initialization when ``n_clusters`` is at most this
        value; larger targets use the faster voxel initializer. Set to ``0``
        to always use voxel initialization. Default is ``256``.
    farthest_point_oversampling : int, optional
        Size of the area-weighted FPS candidate pool as a multiple of
        ``n_clusters``. Default is ``4``.

    Notes
    -----
    These values configure host-side orchestration or runtime kernel inputs;
    changing them does not trigger Warp kernel recompilation.
    """

    search_radius_scale: float = 1.6
    voxel_width_scale: float = 1.15
    hash_grid_resolution: int = 128
    farthest_point_threshold: int = 256
    farthest_point_oversampling: int = 4

    def __post_init__(self) -> None:
        for name in ("search_radius_scale", "voxel_width_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"{name} must be a real number, got {type(value).__name__}"
                )
            try:
                valid = math.isfinite(float(value)) and value > 0.0
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(f"{name} must be finite and positive")

        for name, minimum in (
            ("hash_grid_resolution", 1),
            ("farthest_point_threshold", 0),
            ("farthest_point_oversampling", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an integer, got {type(value).__name__}"
                )
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}, got {value}")

        if self.hash_grid_resolution > _MAX_HASH_GRID_RESOLUTION:
            raise ValueError(
                "hash_grid_resolution must be at most "
                f"{_MAX_HASH_GRID_RESOLUTION}, got {self.hash_grid_resolution}"
            )


__all__ = ["WarpRemeshOptions"]
