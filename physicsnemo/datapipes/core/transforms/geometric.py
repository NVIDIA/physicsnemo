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
Geometric transforms for spatial data processing.

Provides transforms for computing signed distance fields, normals,
and applying spatial invariances (translation, scaling).
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from tensordict import TensorDict

from physicsnemo.datapipes.core.registry import register
from physicsnemo.datapipes.core.transforms.base import Transform
from physicsnemo.nn.sdf import signed_distance_field


@register()
class ComputeSDF(Transform):
    r"""
    Compute signed distance field from a mesh.

    Computes the signed distance from query points to the nearest point on
    a triangular mesh surface. Optionally returns the closest points on the
    mesh surface for each query point.

    Example:
        >>> transform = ComputeSDF(
        ...     input_keys=["volume_mesh_centers"],
        ...     output_key="sdf_nodes",
        ...     mesh_coords_key="stl_coordinates",
        ...     mesh_faces_key="stl_faces",
        ...     closest_points_key="closest_points"
        ... )
        >>> sample = Tensordict({
        ...     "volume_mesh_centers": torch.randn(10000, 3),
        ...     "stl_coordinates": torch.randn(5000, 3),
        ...     "stl_faces": torch.randint(0, 5000, (10000,))
        ... })
        >>> result = transform(sample)
        >>> print(result["sdf_nodes"].shape)
        torch.Size([10000, 1])
    """

    def __init__(
        self,
        input_keys: list[str],
        output_key: str,
        mesh_coords_key: str,
        mesh_faces_key: str,
        *,
        use_winding_number: bool = True,
        closest_points_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the SDF computation transform.

        Args:
            input_keys: List of keys containing query points to compute SDF for.
                       Each tensor should have shape :math:`(N, 3)`.
            output_key: Key to store the computed SDF values.
            mesh_coords_key: Key for mesh vertex coordinates, shape :math:`(M, 3)`.
            mesh_faces_key: Key for mesh face indices (flattened), shape :math:`(F*3,)`.
            use_winding_number: If True, use winding number for sign determination.
            closest_points_key: Optional key to store closest points on mesh.
        """
        super().__init__()
        self.input_keys = input_keys
        self.output_key = output_key
        self.mesh_coords_key = mesh_coords_key
        self.mesh_faces_key = mesh_faces_key
        self.use_winding_number = use_winding_number
        self.closest_points_key = closest_points_key

    def __call__(self, data: TensorDict) -> TensorDict:
        """Compute SDF for the sample."""
        # Get mesh data
        if self.mesh_coords_key not in data:
            raise KeyError(f"Mesh coordinates key '{self.mesh_coords_key}' not found")
        if self.mesh_faces_key not in data:
            raise KeyError(f"Mesh faces key '{self.mesh_faces_key}' not found")

        mesh_coords = data[self.mesh_coords_key]
        mesh_faces = data[self.mesh_faces_key].to(torch.int32)

        updates = {}

        # Compute SDF for each input key
        for key in self.input_keys:
            if key not in data:
                raise KeyError(f"Input key '{key}' not found")

            query_points = data[key]

            # Compute SDF and closest points
            sdf, closest_points = signed_distance_field(
                mesh_coords,
                mesh_faces,
                query_points,
                use_sign_winding_number=self.use_winding_number,
            )

            # Store SDF with output key (add suffix if multiple inputs)
            if len(self.input_keys) == 1:
                updates[self.output_key] = sdf.reshape(-1, 1)
                if self.closest_points_key is not None:
                    updates[self.closest_points_key] = closest_points
            else:
                suffix = f"_{key}"
                updates[f"{self.output_key}{suffix}"] = sdf.reshape(-1, 1)
                if self.closest_points_key is not None:
                    updates[f"{self.closest_points_key}{suffix}"] = closest_points

        return data.update(updates)

    def __repr__(self) -> str:
        return f"ComputeSDF(input_keys={self.input_keys}, output_key={self.output_key})"


@register()
class ComputeNormals(Transform):
    r"""
    Compute normal vectors from closest points.

    Computes normalized direction vectors from query points to their closest
    points on a surface. Handles zero-distance edge cases by falling back to
    center of mass direction.

    Example:
        >>> transform = ComputeNormals(
        ...     positions_key="volume_mesh_centers",
        ...     closest_points_key="closest_points",
        ...     center_of_mass_key="center_of_mass",
        ...     output_key="volume_normals"
        ... )
    """

    def __init__(
        self,
        positions_key: str,
        closest_points_key: str,
        center_of_mass_key: str,
        output_key: str,
        *,
        handle_zero_distance: bool = True,
    ) -> None:
        """
        Initialize the normal computation transform.

        Args:
            positions_key: Key for position tensor, shape :math:`(N, 3)`.
            closest_points_key: Key for closest points tensor, shape :math:`(N, 3)`.
            center_of_mass_key: Key for center of mass, shape :math:`(1, 3)` or :math:`(3,)`.
            output_key: Key to store computed normals.
            handle_zero_distance: If True, use center_of_mass fallback for zero distances.
        """
        super().__init__()
        self.positions_key = positions_key
        self.closest_points_key = closest_points_key
        self.center_of_mass_key = center_of_mass_key
        self.output_key = output_key
        self.handle_zero_distance = handle_zero_distance

    def __call__(self, data: TensorDict) -> TensorDict:
        """Compute normals for the sample."""
        positions = data[self.positions_key]
        closest_points = data[self.closest_points_key]
        center_of_mass = data[self.center_of_mass_key]

        # Ensure center_of_mass has shape (1, 3)
        if center_of_mass.ndim == 1:
            center_of_mass = center_of_mass.unsqueeze(0)

        # Compute initial normals
        normals = positions - closest_points

        if self.handle_zero_distance:
            # Handle zero-distance points (on or very close to surface)
            distance_to_closest = torch.norm(normals, dim=-1)
            null_points = distance_to_closest < 1e-6

            # For null points, use direction from center of mass
            if null_points.any():
                normals[null_points] = positions[null_points] - center_of_mass

        # Normalize
        norm = torch.norm(normals, dim=-1, keepdim=True) + 1e-6
        normals = normals / norm

        return data.update({self.output_key: normals})

    def __repr__(self) -> str:
        return (
            f"ComputeNormals(positions_key={self.positions_key}, "
            f"output_key={self.output_key})"
        )


@register()
class Translate(Transform):
    r"""
    Apply a translation by subtracting a center point.

    Subtracts a reference point (typically center of mass) from position-like
    tensors to make the representation translation invariant.

    Example:
        >>> transform = TranslationInvariance(
        ...     input_keys=["volume_mesh_centers", "surface_mesh_centers"],
        ...     center_key_or_value="center_of_mass"
        ... )
    """

    def __init__(
        self,
        input_keys: list[str],
        center_key_or_value: Union[str, torch.Tensor],
    ) -> None:
        """
        Initialize the translation invariance transform.

        Args:
            input_keys: List of position tensor keys to translate.
            center_key_or_value: Either a key name (str) for a tensor in the sample,
                                or a fixed tensor value to subtract.
        """
        super().__init__()
        self.input_keys = input_keys
        self.center_key_or_value = center_key_or_value
        self.is_key = isinstance(center_key_or_value, str)

    def __call__(self, data: TensorDict) -> TensorDict:
        """Apply translation to the sample."""
        # Get center value
        if isinstance(self.center_key_or_value, str):
            if self.center_key_or_value not in data:
                raise KeyError(f"Center key '{self.center_key_or_value}' not found")
            center = data[self.center_key_or_value]
        else:
            if not isinstance(self.center_key_or_value, torch.Tensor):
                raise TypeError(
                    f"center_key_or_value should be torch.Tensor but got {type(self.center_key_or_value)}"
                )
            center = self.center_key_or_value
            # Move to same device as data if needed
            if data.device is not None and center.device != data.device:
                center = center.to(data.device)

        # Ensure center has shape (1, 3) or (1, D)
        if center.ndim == 1:
            center = center.unsqueeze(0)

        # Apply translation to all keys
        updates = {}
        for key in self.input_keys:
            if key in data:
                updates[key] = data[key] - center

        return data.update(updates)

    def to(self, device: Union[torch.device, str]) -> "Translate":
        """Move center tensor to the specified device (if not a key reference)."""
        super().to(device)
        if not self.is_key:
            if not isinstance(self.center_key_or_value, torch.Tensor):
                raise TypeError(
                    f"center_key_or_value should be torch.Tensor but got {type(self.center_key_or_value)}"
                )
            device = torch.device(device) if isinstance(device, str) else device
            self.center_key_or_value = self.center_key_or_value.to(device)
        return self

    def __repr__(self) -> str:
        return (
            f"TranslationInvariance(input_keys={self.input_keys}, "
            f"center={self.center_key_or_value})"
        )


@register()
class ReScale(Transform):
    r"""
    Apply a scale factor by dividing by a reference scale.

    Divides position tensors by a reference scale to make the representation
    scale invariant.

    Example:
        >>> transform = ReScale(
        ...     input_keys=["volume_mesh_centers", "geometry_coordinates"],
        ...     reference_scale=torch.tensor([[1.0, 1.0, 1.0]])
        ... )
    """

    def __init__(
        self,
        input_keys: list[str],
        reference_scale: torch.Tensor,
    ) -> None:
        """
        Initialize the scale invariance transform.

        Args:
            input_keys: List of position tensor keys to scale.
            reference_scale: Tensor to divide by, shape :math:`(1, D)` or :math:`(D,)`.
        """
        super().__init__()
        self.input_keys = input_keys
        self.reference_scale = reference_scale

    def __call__(self, data: TensorDict) -> TensorDict:
        """Apply scaling to the data."""
        scale = self.reference_scale

        # Ensure scale has batch dimension
        if scale.ndim == 1:
            scale = scale.unsqueeze(0)

        # Move scale to same device as data if needed
        if data.device is not None and scale.device != data.device:
            scale = scale.to(data.device)

        # Apply scaling to all keys
        updates = {}
        for key in self.input_keys:
            if key in data:
                updates[key] = data[key] / scale

        return data.update(updates)

    def to(self, device: Union[torch.device, str]) -> "ReScale":
        """Move reference scale tensor to the specified device."""
        super().to(device)
        device = torch.device(device) if isinstance(device, str) else device
        self.reference_scale = self.reference_scale.to(device)
        return self

    def __repr__(self) -> str:
        return (
            f"ScaleInvariance(input_keys={self.input_keys}, "
            f"scale_shape={self.reference_scale.shape})"
        )
