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

from typing import Optional

import torch
from tensordict import TensorDict

from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.base import Transform
from physicsnemo.nn.functional import signed_distance_field


@register()
class ComputeSDF(Transform):
    r"""
    Compute signed distance field from a mesh.

    Computes the signed distance from query points to the nearest point on
    a triangular mesh surface. Optionally returns the closest points on the
    mesh surface for each query point.

    Parameters
    ----------
    input_keys : list[str]
        List of keys containing query points to compute SDF for.
        Each tensor should have shape :math:`(N, 3)`.
    output_key : str
        Key to store the computed SDF values.
    mesh_coords_key : str
        Key for mesh vertex coordinates, shape :math:`(M, 3)`.
    mesh_faces_key : str
        Key for mesh face indices (flattened), shape :math:`(F*3,)`.
    use_winding_number : bool, default=True
        If True, use winding number for sign determination.
    closest_points_key : str, optional
        Optional key to store closest points on mesh.

    Examples
    --------
    >>> transform = ComputeSDF(
    ...     input_keys=["volume_mesh_centers"],
    ...     output_key="sdf_nodes",
    ...     mesh_coords_key="stl_coordinates",
    ...     mesh_faces_key="stl_faces",
    ...     closest_points_key="closest_points"
    ... )
    >>> sample = TensorDict({
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

        Parameters
        ----------
        input_keys : list[str]
            List of keys containing query points to compute SDF for.
            Each tensor should have shape :math:`(N, 3)`.
        output_key : str
            Key to store the computed SDF values.
        mesh_coords_key : str
            Key for mesh vertex coordinates, shape :math:`(M, 3)`.
        mesh_faces_key : str
            Key for mesh face indices (flattened), shape :math:`(F*3,)`.
        use_winding_number : bool, default=True
            If True, use winding number for sign determination.
        closest_points_key : str, optional
            Optional key to store closest points on mesh.
        """
        super().__init__()
        self.input_keys = input_keys
        self.output_key = output_key
        self.mesh_coords_key = mesh_coords_key
        self.mesh_faces_key = mesh_faces_key
        self.use_winding_number = use_winding_number
        self.closest_points_key = closest_points_key

    def __call__(self, data: TensorDict) -> TensorDict:
        """
        Compute SDF for the sample.

        Parameters
        ----------
        data : TensorDict
            Input TensorDict containing mesh and query point data.

        Returns
        -------
        TensorDict
            TensorDict with computed SDF values added.

        Raises
        ------
        KeyError
            If mesh or query point keys are not found in the data.
        """
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
        """
        Return string representation.

        Returns
        -------
        str
            String representation of the transform.
        """
        return f"ComputeSDF(input_keys={self.input_keys}, output_key={self.output_key})"


@register()
class ComputeNormals(Transform):
    r"""
    Compute normal vectors from closest points.

    Computes normalized direction vectors from query points to their closest
    points on a surface. Handles zero-distance edge cases by falling back to
    center of mass direction.

    Parameters
    ----------
    positions_key : str
        Key for position tensor, shape :math:`(N, 3)`.
    closest_points_key : str
        Key for closest points tensor, shape :math:`(N, 3)`.
    center_of_mass_key : str
        Key for center of mass, shape :math:`(1, 3)` or :math:`(3,)`.
    output_key : str
        Key to store computed normals.
    handle_zero_distance : bool, default=True
        If True, use center_of_mass fallback for zero distances.

    Examples
    --------
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

        Parameters
        ----------
        positions_key : str
            Key for position tensor, shape :math:`(N, 3)`.
        closest_points_key : str
            Key for closest points tensor, shape :math:`(N, 3)`.
        center_of_mass_key : str
            Key for center of mass, shape :math:`(1, 3)` or :math:`(3,)`.
        output_key : str
            Key to store computed normals.
        handle_zero_distance : bool, default=True
            If True, use center_of_mass fallback for zero distances.
        """
        super().__init__()
        self.positions_key = positions_key
        self.closest_points_key = closest_points_key
        self.center_of_mass_key = center_of_mass_key
        self.output_key = output_key
        self.handle_zero_distance = handle_zero_distance

    def __call__(self, data: TensorDict) -> TensorDict:
        """
        Compute normals for the sample.

        Parameters
        ----------
        data : TensorDict
            Input TensorDict containing position and closest point data.

        Returns
        -------
        TensorDict
            TensorDict with computed normals added.
        """
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
        """
        Return string representation.

        Returns
        -------
        str
            String representation of the transform.
        """
        return (
            f"ComputeNormals(positions_key={self.positions_key}, "
            f"output_key={self.output_key})"
        )


@register()
class Translate(Transform):
    r"""
    Apply a translation by adding or subtracting a center point.

    By default, this will ADD the translation. But you can also
    use the subtract mode: this is particularly useful when composed
    with CenterOfMass: you can compute the CoM and apply a translation
    as a CoM subtraction to center the data at the origin.

    Parameters
    ----------
    input_keys : list[str]
        List of position tensor keys to translate.
    center_key_or_value : str or torch.Tensor
        Either a key name (str) for a tensor in the sample,
        or a fixed tensor value to add/subtract.
    subtract : bool, default=False
        If False (default), ADD the translation (data + center).
        If True, SUBTRACT the translation (data - center).
        Use subtract=True when centering data around a reference point
        like center of mass.

    Examples
    --------
    Add mode (default) - shift points by a fixed offset:

    >>> transform = Translate(
    ...     input_keys=["positions"],
    ...     center_key_or_value=torch.tensor([1.0, 2.0, 3.0])
    ... )
    >>> # result["positions"] = original + [1, 2, 3]

    Subtract mode - center points by subtracting center of mass:

    >>> transform = Translate(
    ...     input_keys=["volume_mesh_centers", "surface_mesh_centers"],
    ...     center_key_or_value="center_of_mass",
    ...     subtract=True
    ... )
    >>> # result["positions"] = original - center_of_mass
    """

    def __init__(
        self,
        input_keys: list[str],
        center_key_or_value: str | torch.Tensor,
        *,
        subtract: bool = False,
    ) -> None:
        """
        Initialize the translation transform.

        Parameters
        ----------
        input_keys : list[str]
            List of position tensor keys to translate.
        center_key_or_value : str or torch.Tensor
            Either a key name (str) for a tensor in the sample,
            or a fixed tensor value to add/subtract.
        subtract : bool, default=False
            If False (default), ADD the translation (data + center).
            If True, SUBTRACT the translation (data - center).
            Use subtract=True when centering data around a reference point
            like center of mass.
        """
        super().__init__()
        self.input_keys = input_keys
        self.center_key_or_value = center_key_or_value
        self.is_key = isinstance(center_key_or_value, str)
        self.subtract = subtract

    def __call__(self, data: TensorDict) -> TensorDict:
        """
        Apply translation to the sample.

        Parameters
        ----------
        data : TensorDict
            Input TensorDict containing position data.

        Returns
        -------
        TensorDict
            TensorDict with translated positions.

        Raises
        ------
        KeyError
            If the center key is not found in the data.
        TypeError
            If center_key_or_value is not a string or torch.Tensor.
        """
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
                if self.subtract:
                    updates[key] = data[key] - center
                else:
                    updates[key] = data[key] + center

        return data.update(updates)

    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str
            String representation of the transform.
        """
        mode = "subtract" if self.subtract else "add"
        return (
            f"Translate(input_keys={self.input_keys}, "
            f"center={self.center_key_or_value}, mode={mode})"
        )


@register()
class Scale(Transform):
    r"""
    Apply a scale factor by multiplying or dividing by a reference scale.

    By default, this will MULTIPLY by the scale factor. But you can also
    use the divide mode: this is particularly useful for normalizing data
    to make the representation scale invariant (e.g., dividing by a
    characteristic length scale).

    Parameters
    ----------
    input_keys : list[str]
        List of position tensor keys to scale.
    scale : torch.Tensor
        Scale factor tensor, shape :math:`(1, D)` or :math:`(D,)`.
    divide : bool, default=False
        If False (default), MULTIPLY by the scale (data * scale).
        If True, DIVIDE by the scale (data / scale).
        Use divide=True when normalizing data by a reference scale.

    Examples
    --------
    Multiply mode (default) - scale up positions by 2x:

    >>> transform = Scale(
    ...     input_keys=["positions"],
    ...     scale=torch.tensor([2.0, 2.0, 2.0])
    ... )
    >>> # result["positions"] = original * [2, 2, 2]

    Divide mode - normalize by a reference scale:

    >>> transform = Scale(
    ...     input_keys=["volume_mesh_centers", "geometry_coordinates"],
    ...     scale=torch.tensor([1.0, 1.0, 1.0]),
    ...     divide=True
    ... )
    >>> # result["positions"] = original / scale
    """

    def __init__(
        self,
        input_keys: list[str],
        scale: torch.Tensor,
        *,
        divide: bool = False,
    ) -> None:
        """
        Initialize the scale transform.

        Parameters
        ----------
        input_keys : list[str]
            List of position tensor keys to scale.
        scale : torch.Tensor
            Scale factor tensor, shape :math:`(1, D)` or :math:`(D,)`.
        divide : bool, default=False
            If False (default), MULTIPLY by the scale (data * scale).
            If True, DIVIDE by the scale (data / scale).
            Use divide=True when normalizing data by a reference scale.
        """
        super().__init__()
        self.input_keys = input_keys
        self.scale = scale
        self.divide = divide

    def __call__(self, data: TensorDict) -> TensorDict:
        """
        Apply scaling to the data.

        Parameters
        ----------
        data : TensorDict
            Input TensorDict containing position data.

        Returns
        -------
        TensorDict
            TensorDict with scaled positions.
        """
        scale = self.scale

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
                if self.divide:
                    updates[key] = data[key] / scale
                else:
                    updates[key] = data[key] * scale

        return data.update(updates)

    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str
            String representation of the transform.
        """
        mode = "divide" if self.divide else "multiply"
        return (
            f"Scale(input_keys={self.input_keys}, "
            f"scale_shape={self.scale.shape}, mode={mode})"
        )
