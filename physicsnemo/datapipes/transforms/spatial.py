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

"""
Spatial transforms for mesh and grid processing.

Provides generic transforms for spatial operations including bounding box
filtering, grid creation, k-NN neighbor computation, and center of mass calculation.
"""

from __future__ import annotations

from typing import Optional

import torch
from tensordict import TensorDict

from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.base import Transform
from physicsnemo.nn.neighbors import knn


@register()
class BoundingBoxFilter(Transform):
    r"""
    Filter points outside a spatial bounding box.

    Removes points that fall outside specified min/max bounds and applies
    the same filtering to dependent arrays to maintain correspondence.
    This is useful for focusing on specific regions of interest or removing
    outliers from simulation data.

    Parameters
    ----------
    input_keys : list[str]
        List of coordinate tensor keys to filter.
    bbox_min : torch.Tensor
        Minimum corner of bounding box, shape :math:`(3,)`.
    bbox_max : torch.Tensor
        Maximum corner of bounding box, shape :math:`(3,)`.
    dependent_keys : list[str], optional
        Optional list of keys to filter using the same mask.
        These maintain correspondence with the filtered coordinates.

    Examples
    --------
    >>> transform = BoundingBoxFilter(
    ...     input_keys=["volume_mesh_centers"],
    ...     bbox_min=torch.tensor([-1.0, -1.0, -1.0]),
    ...     bbox_max=torch.tensor([1.0, 1.0, 1.0]),
    ...     dependent_keys=["volume_fields", "sdf_nodes"]
    ... )
    >>> sample = TensorDict({
    ...     "volume_mesh_centers": torch.randn(10000, 3) * 2,  # Some outside bbox
    ...     "volume_fields": torch.randn(10000, 4)
    ... })
    >>> result = transform(sample)
    >>> # Only points within bbox remain
    """

    def __init__(
        self,
        input_keys: list[str],
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
        *,
        dependent_keys: Optional[list[str]] = None,
    ) -> None:
        """
        Initialize the bounding box filter transform.

        Parameters
        ----------
        input_keys : list[str]
            List of coordinate tensor keys to filter.
        bbox_min : torch.Tensor
            Minimum corner of bounding box, shape :math:`(3,)`.
        bbox_max : torch.Tensor
            Maximum corner of bounding box, shape :math:`(3,)`.
        dependent_keys : list[str], optional
            Optional list of keys to filter using the same mask.
            These maintain correspondence with the filtered coordinates.
        """
        super().__init__()
        self.input_keys = input_keys
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.dependent_keys = dependent_keys or []

    def __call__(self, data: TensorDict) -> TensorDict:
        """
        Apply bounding box filtering to the sample.

        Parameters
        ----------
        data : TensorDict
            Input TensorDict containing coordinate and dependent data.

        Returns
        -------
        TensorDict
            TensorDict with filtered points.
        """
        updates = {}

        for coord_key in self.input_keys:
            if coord_key not in data:
                continue

            coords = data[coord_key]

            # Move bbox to same device
            bbox_min = self.bbox_min.to(coords.device)
            bbox_max = self.bbox_max.to(coords.device)

            # Create mask for points inside bbox
            ids_min = coords > bbox_min
            ids_max = coords < bbox_max
            ids_in_bbox = ids_min & ids_max
            ids_in_bbox = ids_in_bbox.all(dim=-1)

            # Apply mask to coordinates
            updates[coord_key] = coords[ids_in_bbox]

            # Apply same mask to dependent keys
            for dep_key in self.dependent_keys:
                if dep_key in data:
                    updates[dep_key] = data[dep_key][ids_in_bbox]

        return data.update(updates)

    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str
            String representation of the transform.
        """
        return (
            f"BoundingBoxFilter(input_keys={self.input_keys}, "
            f"dependent_keys={self.dependent_keys})"
        )


@register()
class CreateGrid(Transform):
    r"""
    Create a regular 3D spatial grid.

    Generates a uniform grid spanning a bounding box, used for latent space
    representations, interpolation grids, or structured spatial queries.

    Parameters
    ----------
    output_key : str
        Key to store the generated grid.
    resolution : tuple[int, int, int]
        Grid resolution as (nx, ny, nz).
    bbox_min : torch.Tensor
        Minimum corner of bounding box, shape :math:`(3,)`.
    bbox_max : torch.Tensor
        Maximum corner of bounding box, shape :math:`(3,)`.

    Examples
    --------
    >>> transform = CreateGrid(
    ...     output_key="grid",
    ...     resolution=(64, 64, 64),
    ...     bbox_min=torch.tensor([-1.0, -1.0, -1.0]),
    ...     bbox_max=torch.tensor([1.0, 1.0, 1.0])
    ... )
    >>> sample = TensorDict({})
    >>> result = transform(sample)
    >>> print(result["grid"].shape)
    torch.Size([262144, 3])
    """

    def __init__(
        self,
        output_key: str,
        resolution: tuple[int, int, int],
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
    ) -> None:
        """
        Initialize the grid creation transform.

        Parameters
        ----------
        output_key : str
            Key to store the generated grid.
        resolution : tuple[int, int, int]
            Grid resolution as (nx, ny, nz).
        bbox_min : torch.Tensor
            Minimum corner of bounding box, shape :math:`(3,)`.
        bbox_max : torch.Tensor
            Maximum corner of bounding box, shape :math:`(3,)`.
        """
        super().__init__()
        self.output_key = output_key
        self.resolution = resolution
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max

    def __call__(self, data: TensorDict) -> TensorDict:
        """
        Create grid and add to sample.

        Parameters
        ----------
        data : TensorDict
            Input TensorDict.

        Returns
        -------
        TensorDict
            TensorDict with generated grid added.
        """
        device = data.device if data.device is not None else torch.device("cpu")

        # Move bbox to device
        bbox_min = self.bbox_min.to(device)
        bbox_max = self.bbox_max.to(device)

        nx, ny, nz = self.resolution

        # Create 1D arrays for each dimension
        x = torch.linspace(bbox_min[0], bbox_max[0], nx, device=device)
        y = torch.linspace(bbox_min[1], bbox_max[1], ny, device=device)
        z = torch.linspace(bbox_min[2], bbox_max[2], nz, device=device)

        # Create meshgrid
        xv, yv, zv = torch.meshgrid(x, y, z, indexing="ij")

        # Stack into grid of shape (nx*ny*nz, 3)
        grid = torch.stack([xv.flatten(), yv.flatten(), zv.flatten()], dim=-1)

        return data.update({self.output_key: grid})

    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str
            String representation of the transform.
        """
        return f"CreateGrid(output_key={self.output_key}, resolution={self.resolution})"


@register()
class KNearestNeighbors(Transform):
    r"""
    Compute k-nearest neighbors in a point cloud.

    Finds the k nearest neighbors for each query point and extracts
    corresponding coordinates and other attributes. Useful for local
    feature aggregation in mesh networks and spatial interpolation.

    Parameters
    ----------
    points_key : str
        Key for reference points to search, shape :math:`(N, 3)`.
    queries_key : str
        Key for query points, shape :math:`(M, 3)`.
    k : int
        Number of nearest neighbors to find.
    output_prefix : str, default="neighbors"
        Prefix for output keys.
    extract_keys : list[str], optional
        Optional list of keys to extract for neighbors
        (e.g., ``["normals", "areas"]``). If None, only extracts coordinates.

    Examples
    --------
    >>> transform = KNearestNeighbors(
    ...     points_key="surface_mesh_centers",
    ...     queries_key="surface_mesh_centers_subsampled",
    ...     k=11,
    ...     output_prefix="surface_neighbors",
    ...     extract_keys=["surface_normals", "surface_areas"]
    ... )
    >>> sample = TensorDict({
    ...     "surface_mesh_centers": torch.randn(10000, 3),
    ...     "surface_mesh_centers_subsampled": torch.randn(1000, 3),
    ...     "surface_normals": torch.randn(10000, 3),
    ...     "surface_areas": torch.rand(10000)
    ... })
    >>> result = transform(sample)
    >>> # Creates: surface_neighbors_coords, surface_neighbors_normals, etc.
    """

    def __init__(
        self,
        points_key: str,
        queries_key: str,
        k: int,
        *,
        output_prefix: str = "neighbors",
        extract_keys: Optional[list[str]] = None,
        drop_first_neighbor: bool = False,
    ) -> None:
        """
        Initialize the k-NN transform.

        Parameters
        ----------
        points_key : str
            Key for reference points to search, shape :math:`(N, 3)`.
        queries_key : str
            Key for query points, shape :math:`(M, 3)`.
        k : int
            Number of nearest neighbors to find.
        output_prefix : str, default="neighbors"
            Prefix for output keys.
        extract_keys : list[str], optional
            Optional list of keys to extract for neighbors
            (e.g., ``["normals", "areas"]``). If None, only extracts coordinates.
        """
        super().__init__()
        self.points_key = points_key
        self.queries_key = queries_key
        self.k = k
        self.output_prefix = output_prefix
        self.extract_keys = extract_keys or []
        self.drop_first_neighbor = drop_first_neighbor

    def __call__(self, data: TensorDict) -> TensorDict:
        """
        Compute k-NN and extract neighbor features.

        Parameters
        ----------
        data : TensorDict
            Input TensorDict containing points and queries.

        Returns
        -------
        TensorDict
            TensorDict with neighbor indices, distances, and features added.

        Raises
        ------
        KeyError
            If points or queries keys are not found in the data.
        """
        if self.points_key not in data:
            raise KeyError(f"Points key '{self.points_key}' not found")
        if self.queries_key not in data:
            raise KeyError(f"Queries key '{self.queries_key}' not found")

        points = data[self.points_key]
        queries = data[self.queries_key]

        # Compute k-NN
        neighbor_indices, neighbor_distances = knn(
            points=points,
            queries=queries,
            k=self.k,
        )

        updates = {}

        # Store indices and distances
        updates[f"{self.output_prefix}_indices"] = neighbor_indices
        updates[f"{self.output_prefix}_distances"] = neighbor_distances

        # Extract neighbor coordinates (skip first, which is self)
        if self.drop_first_neighbor:
            neighbor_coords = points[neighbor_indices][:, 1:]
        else:
            neighbor_coords = points[neighbor_indices]
        updates[f"{self.output_prefix}_coords"] = neighbor_coords

        # Extract additional features for neighbors
        for key in self.extract_keys:
            if key in data:
                if self.drop_first_neighbor:
                    neighbor_features = data[key][neighbor_indices][:, 1:]
                else:
                    neighbor_features = data[key][neighbor_indices]
                updates[f"{self.output_prefix}_{key}"] = neighbor_features

        return data.update(updates)

    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str
            String representation of the transform.
        """
        return (
            f"KNearestNeighbors(points_key={self.points_key}, "
            f"queries_key={self.queries_key}, k={self.k})"
        )


@register()
class CenterOfMass(Transform):
    r"""
    Compute weighted center of mass for a point cloud.

    Calculates the center of mass using area or mass weights, typically
    applied to mesh data where each point represents a cell with a specific area.

    Parameters
    ----------
    coords_key : str
        Key for coordinates, shape :math:`(N, 3)`.
    areas_key : str
        Key for area weights, shape :math:`(N,)`.
    output_key : str
        Key to store the computed center of mass, shape :math:`(1, 3)`.

    Examples
    --------
    >>> transform = CenterOfMass(
    ...     coords_key="stl_centers",
    ...     areas_key="stl_areas",
    ...     output_key="center_of_mass"
    ... )
    >>> sample = TensorDict({
    ...     "stl_centers": torch.randn(5000, 3),
    ...     "stl_areas": torch.rand(5000)
    ... })
    >>> result = transform(sample)
    >>> print(result["center_of_mass"].shape)
    torch.Size([3])
    """

    def __init__(
        self,
        coords_key: str,
        output_key: str,
        *,
        areas_key: str | None = None,
    ) -> None:
        """
        Initialize the center of mass transform.

        Parameters
        ----------
        coords_key : str
            Key for coordinates, shape :math:`(N, 3)`.
        areas_key : str
            Key for area weights, shape :math:`(N,)`.
        output_key : str
            Key to store the computed center of mass, shape :math:`(1, 3)`.
        """
        super().__init__()
        self.coords_key = coords_key
        self.areas_key = areas_key
        self.output_key = output_key

    def __call__(self, data: TensorDict) -> TensorDict:
        """
        Compute center of mass for the sample.

        Parameters
        ----------
        data : TensorDict
            Input TensorDict containing coordinates and area weights.

        Returns
        -------
        TensorDict
            TensorDict with computed center of mass added.

        Raises
        ------
        KeyError
            If coordinates or areas keys are not found in the data.
        """
        if self.coords_key not in data:
            raise KeyError(f"Coordinates key '{self.coords_key}' not found")

        coords = data[self.coords_key]
        if self.areas_key is not None:
            if self.areas_key not in data:
                raise KeyError(f"Areas key '{self.areas_key}' not found")

            areas = data[self.areas_key]
            # Compute weighted center of mass
            total_area = areas.sum()

            #  Apply the weighting:
            coords = coords * areas.unsqueeze(-1)

            center_of_mass = coords.sum(dim=0) / total_area
        else:
            center_of_mass = coords.mean(dim=0)

        return data.update({self.output_key: center_of_mass})

    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str
            String representation of the transform.
        """
        return (
            f"CenterOfMass(coords_key={self.coords_key}, output_key={self.output_key})"
        )
