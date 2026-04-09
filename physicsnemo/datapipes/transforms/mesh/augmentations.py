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

"""
Random mesh augmentations (on-the-fly randomizations). Mesh -> Mesh.
"""

from __future__ import annotations

import math
from typing import Literal

import torch

from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.mesh.base import MeshTransform
from physicsnemo.mesh import DomainMesh, Mesh


@register()
class RandomScaleMesh(MeshTransform):
    r"""Random uniform scale of mesh. Scale factor is sampled per __call__."""

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.9, 1.1),
        transform_point_data: bool = False,
        transform_cell_data: bool = False,
        transform_global_data: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        """
        Parameters
        ----------
        scale_range : tuple[float, float]
            ``(low, high)`` bounds for the uniform scale factor.
        transform_point_data : bool
            If ``True``, transform point-data fields under scaling.
        transform_cell_data : bool
            If ``True``, transform cell-data fields under scaling.
        transform_global_data : bool
            If ``True``, transform global-data fields under scaling.
        generator : torch.Generator or None
            Optional random generator for reproducibility.  May reside on
            CPU even when the mesh is on GPU.
        """
        super().__init__()
        self.scale_range = scale_range
        self.transform_point_data = transform_point_data
        self.transform_cell_data = transform_cell_data
        self.transform_global_data = transform_global_data
        self._generator = generator

    def _sample_factor(self, device: torch.device) -> torch.Tensor:
        """Sample a uniform scale factor in ``[low, high]``.

        Random values are generated on the generator's device and then
        transferred to *device* asynchronously to avoid GPU sync points.

        Parameters
        ----------
        device : torch.device
            Target device for the returned tensor.

        Returns
        -------
        torch.Tensor
            Scalar (0-dim) tensor with the sampled factor.
        """
        low, high = self.scale_range
        return low + (high - low) * torch.rand(1, generator=self._generator).squeeze(
            0
        ).to(device)

    def __call__(self, mesh: Mesh) -> Mesh:
        """Apply a random uniform scale to *mesh*.

        Parameters
        ----------
        mesh : Mesh
            Input mesh.

        Returns
        -------
        Mesh
            Scaled mesh.
        """
        factor = self._sample_factor(mesh.points.device)
        return mesh.scale(
            factor,
            transform_point_data=self.transform_point_data,
            transform_cell_data=self.transform_cell_data,
            transform_global_data=self.transform_global_data,
        )

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Apply a random uniform scale to every mesh in *domain*.

        A single scale factor is sampled and applied consistently to the
        interior and all boundary meshes.

        Parameters
        ----------
        domain : DomainMesh
            Input domain mesh.

        Returns
        -------
        DomainMesh
            Scaled domain mesh.
        """
        factor = self._sample_factor(domain.interior.points.device)
        return domain.scale(
            factor,
            transform_point_data=self.transform_point_data,
            transform_cell_data=self.transform_cell_data,
            transform_global_data=self.transform_global_data,
        )

    def extra_repr(self) -> str:
        return f"scale_range={self.scale_range}"


@register()
class RandomTranslateMesh(MeshTransform):
    r"""Random translation of mesh. Offset is sampled per __call__."""

    def __init__(
        self,
        max_offset: float | tuple[float, float, float] = 0.1,
        generator: torch.Generator | None = None,
    ) -> None:
        """
        Parameters
        ----------
        max_offset : float or tuple[float, float, float]
            Maximum translation magnitude per axis.  A scalar is broadcast
            to all three spatial dimensions.
        generator : torch.Generator or None
            Optional random generator for reproducibility.  May reside on
            CPU even when the mesh is on GPU.
        """
        super().__init__()
        if isinstance(max_offset, (int, float)):
            max_offset = (max_offset, max_offset, max_offset)
        self.max_offset = max_offset
        self._generator = generator

    def _sample_offset(
        self, n_spatial_dims: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Sample a uniform translation offset in ``[-max, +max]`` per axis.

        Random values are generated on the generator's device and then
        transferred to *device* asynchronously to avoid GPU sync points.

        Parameters
        ----------
        n_spatial_dims : int
            Number of spatial dimensions (typically 2 or 3).
        device : torch.device
            Target device for the returned tensor.
        dtype : torch.dtype
            Target dtype for the returned tensor.

        Returns
        -------
        torch.Tensor
            Offset vector, shape ``(n_spatial_dims,)``.
        """
        if isinstance(self.max_offset, (int, float)):
            scales = (self.max_offset,) * n_spatial_dims
        else:
            scales = tuple(self.max_offset[i] for i in range(n_spatial_dims))
        scale_t = torch.tensor(scales, device=device, dtype=dtype)
        rand = torch.rand(n_spatial_dims, generator=self._generator).to(
            device=device, dtype=dtype
        )
        return (rand * 2 - 1) * scale_t

    def __call__(self, mesh: Mesh) -> Mesh:
        """Apply a random translation to *mesh*.

        Parameters
        ----------
        mesh : Mesh
            Input mesh.

        Returns
        -------
        Mesh
            Translated mesh.
        """
        offset = self._sample_offset(
            mesh.n_spatial_dims, mesh.points.device, mesh.points.dtype
        )
        return mesh.translate(offset)

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Apply a random translation to every mesh in *domain*.

        A single offset is sampled and applied consistently to the
        interior and all boundary meshes.

        Parameters
        ----------
        domain : DomainMesh
            Input domain mesh.

        Returns
        -------
        DomainMesh
            Translated domain mesh.
        """
        offset = self._sample_offset(
            domain.interior.n_spatial_dims,
            domain.interior.points.device,
            domain.interior.points.dtype,
        )
        return domain.translate(offset)

    def extra_repr(self) -> str:
        return f"max_offset={self.max_offset}"


@register()
class RandomRotateMesh(MeshTransform):
    r"""Random rotation of mesh. Axis and angle are sampled per __call__.

    Two modes are supported:

    * ``"axis_aligned"`` (default) – picks one of the candidate *axes*
      uniformly at random and samples an angle from *angle_range*.  This
      limits rotations to the three cardinal planes.
    * ``"uniform"`` – samples a rotation uniformly from SO(3) via random
      unit quaternions (3-D meshes only).  *axes* and *angle_range* are
      ignored in this mode.
    """

    def __init__(
        self,
        axes: list[Literal["x", "y", "z"]] | None = None,
        angle_range: tuple[float, float] = (-math.pi, math.pi),
        mode: Literal["axis_aligned", "uniform"] = "axis_aligned",
        transform_point_data: bool = False,
        transform_cell_data: bool = False,
        transform_global_data: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        """
        Parameters
        ----------
        axes : list[{"x", "y", "z"}] or None
            Candidate rotation axes.  One is chosen uniformly at random
            per call.  Defaults to ``["x", "y", "z"]``.
            Only used when ``mode="axis_aligned"``.
        angle_range : tuple[float, float]
            ``(low, high)`` bounds (radians) for the rotation angle.
            Only used when ``mode="axis_aligned"``.
        mode : {"axis_aligned", "uniform"}
            ``"axis_aligned"`` picks a random cardinal axis and angle
            each call.  ``"uniform"`` samples a rotation uniformly from
            SO(3) via random quaternions (3-D only).
        transform_point_data : bool
            If ``True``, transform point-data fields under rotation.
        transform_cell_data : bool
            If ``True``, transform cell-data fields under rotation.
        transform_global_data : bool
            If ``True``, transform global-data fields under rotation.
        generator : torch.Generator or None
            Optional random generator for reproducibility.  May reside on
            CPU even when the mesh is on GPU.
        """
        super().__init__()
        if mode not in ("axis_aligned", "uniform"):
            raise ValueError(f"mode must be 'axis_aligned' or 'uniform', got {mode!r}")
        self.axes = axes if axes is not None else ["x", "y", "z"]
        self.angle_range = angle_range
        self.mode = mode
        self.transform_point_data = transform_point_data
        self.transform_cell_data = transform_cell_data
        self.transform_global_data = transform_global_data
        self._generator = generator

        # Coefficient matrix mapping outer(q,q).flatten() (16,) -> R.flatten() (9,).
        # Derived from the standard unit-quaternion rotation formula using
        # w²+x²+y²+z² = 1 to rewrite 1-2(…) terms as sums of squared components.
        #                ww  wx  wy  wz  xw  xx  xy  xz  yw  yx  yy  yz  zw  zx  zy  zz
        self._q2r_map = torch.tensor(
            [
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                [0, 0, 0, -2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1],
                [0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
                [0, 0, -2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # axis-aligned helpers
    # ------------------------------------------------------------------

    def _sample_axis_and_angle(self, device: torch.device) -> tuple[str, torch.Tensor]:
        """Sample a random axis and rotation angle.

        The axis index is drawn on CPU (no GPU sync).  The angle is
        generated on the generator's device and transferred to *device*
        asynchronously.

        Parameters
        ----------
        device : torch.device
            Target device for the returned angle tensor.

        Returns
        -------
        axis : str
            One of ``"x"``, ``"y"``, ``"z"``.
        angle : torch.Tensor
            Scalar (0-dim) tensor with the sampled angle in radians.
        """
        axis_idx = torch.randint(len(self.axes), (1,), generator=self._generator)
        axis = self.axes[axis_idx]
        low, high = self.angle_range
        angle = low + (high - low) * torch.rand(1, generator=self._generator).squeeze(
            0
        ).to(device)
        return axis, angle

    # ------------------------------------------------------------------
    # uniform SO(3) helpers
    # ------------------------------------------------------------------

    def _quaternion_to_rotation_matrix(
        self,
        q: torch.Tensor,
    ) -> torch.Tensor:
        """Convert a unit quaternion to a 3x3 rotation matrix.

        Parameters
        ----------
        q : torch.Tensor
            Unit quaternion ``(w, x, y, z)``, shape ``(4,)``.

        Returns
        -------
        torch.Tensor
            Rotation matrix, shape ``(3, 3)``.
        """
        # 2 dispatches: outer product + matrix-vector multiply.
        return (self._q2r_map.to(q) @ torch.outer(q, q).reshape(16)).reshape(3, 3)

    def _sample_uniform_rotation(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Sample a rotation matrix uniformly from SO(3).

        Uses the random unit quaternion method: sample a 4-D isotropic
        Gaussian vector, normalize to the unit sphere, and convert to a
        rotation matrix.

        Parameters
        ----------
        device : torch.device
            Target device for the returned matrix.
        dtype : torch.dtype
            Target dtype for the returned matrix.

        Returns
        -------
        torch.Tensor
            Rotation matrix, shape ``(3, 3)``.
        """
        q = torch.randn(4, generator=self._generator)
        q = q / q.norm()
        return self._quaternion_to_rotation_matrix(q).to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # __call__ / apply_to_domain
    # ------------------------------------------------------------------

    def __call__(self, mesh: Mesh[..., 3]) -> Mesh[..., 3]:
        """Apply a random rotation to *mesh*.

        Parameters
        ----------
        mesh : Mesh
            Input mesh.

        Returns
        -------
        Mesh
            Rotated mesh.
        """
        if self.mode == "uniform":
            if mesh.n_spatial_dims != 3:
                raise ValueError(
                    f"mode='uniform' requires 3-D meshes, "
                    f"got n_spatial_dims={mesh.n_spatial_dims}"
                )
            R = self._sample_uniform_rotation(mesh.points.device, mesh.points.dtype)
            return mesh.transform(
                R,
                transform_point_data=self.transform_point_data,
                transform_cell_data=self.transform_cell_data,
                transform_global_data=self.transform_global_data,
                assume_invertible=True,
            )

        axis, angle = self._sample_axis_and_angle(mesh.points.device)
        return mesh.rotate(
            angle,
            axis=axis,
            transform_point_data=self.transform_point_data,
            transform_cell_data=self.transform_cell_data,
            transform_global_data=self.transform_global_data,
        )

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Apply a random rotation to every mesh in *domain*.

        A single rotation is sampled and applied consistently to the
        interior and all boundary meshes.

        Parameters
        ----------
        domain : DomainMesh
            Input domain mesh.

        Returns
        -------
        DomainMesh
            Rotated domain mesh.
        """
        if self.mode == "uniform":
            if domain.interior.n_spatial_dims != 3:
                raise ValueError(
                    f"mode='uniform' requires 3-D meshes, "
                    f"got n_spatial_dims={domain.interior.n_spatial_dims}"
                )
            R = self._sample_uniform_rotation(
                domain.interior.points.device, domain.interior.points.dtype
            )
            return domain.transform(
                R,
                transform_point_data=self.transform_point_data,
                transform_cell_data=self.transform_cell_data,
                transform_global_data=self.transform_global_data,
                assume_invertible=True,
            )

        axis, angle = self._sample_axis_and_angle(domain.interior.points.device)
        return domain.rotate(
            angle,
            axis=axis,
            transform_point_data=self.transform_point_data,
            transform_cell_data=self.transform_cell_data,
            transform_global_data=self.transform_global_data,
        )

    def extra_repr(self) -> str:
        if self.mode == "uniform":
            return "mode='uniform'"
        return f"axes={self.axes}, angle_range={self.angle_range}"
