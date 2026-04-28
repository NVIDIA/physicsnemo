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

Augmentation parameters are sampled from ``torch.distributions.Distribution``
objects, enabling arbitrary continuous distributions (Gaussian, Laplace,
Cauchy, etc.) while preserving ``torch.Generator``-based reproducibility
via the inverse CDF (ICDF) method.  See ``DISTRIBUTIONS.md`` in this
directory for full design documentation.
"""

from __future__ import annotations

import math
import warnings
from typing import Literal

import torch
from jaxtyping import Float

from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.mesh.base import MeshTransform
from physicsnemo.mesh import DomainMesh, Mesh


def _sample_distribution(
    distribution: torch.distributions.Distribution,
    shape: tuple[int, ...],
    generator: torch.Generator | None,
    fallback_device: torch.device | None = None,
) -> torch.Tensor:
    """Sample from a distribution using ICDF + generator for reproducibility.

    Draws ``U ~ Uniform(0, 1)`` with the provided generator, then
    transforms through ``distribution.icdf(U)``.  The generator and
    distribution parameters must already reside on the same device
    (ensured by :meth:`MeshTransform.to`).

    For distributions that do not implement ``icdf`` (e.g. Poisson),
    falls back to ``distribution.sample()`` without generator
    reproducibility.

    Parameters
    ----------
    distribution : torch.distributions.Distribution
        The target distribution to sample from.
    shape : tuple[int, ...]
        Shape of the sample to draw.
    generator : torch.Generator or None
        Random generator for reproducibility.  When provided, uniform
        samples are generated on ``generator.device``.
    fallback_device : torch.device or None
        Device used for ``torch.rand`` when *generator* is ``None``.
        Typically ``self._device`` set by :meth:`MeshTransform.to`.

    Returns
    -------
    torch.Tensor
        Sampled tensor with the requested *shape*.
    """
    if generator is not None:
        u = torch.rand(shape, generator=generator, device=generator.device)
    else:
        u = torch.rand(shape, device=fallback_device)
    try:
        return distribution.icdf(u)
    except NotImplementedError:
        warnings.warn(
            f"{type(distribution).__name__} does not implement icdf; "
            "falling back to .sample() without generator reproducibility.",
            stacklevel=3,
        )
        return distribution.sample(shape).to(device=u.device)


@register()
class RandomScaleMesh(MeshTransform):
    r"""Random scale of mesh.  Scale factor is sampled per ``__call__``.

    The scale factor is drawn from *distribution* (default
    ``Uniform(0.9, 1.1)``).  Any ``torch.distributions.Distribution``
    with an ``icdf`` method can be used; see ``DISTRIBUTIONS.md``.
    """

    def __init__(
        self,
        distribution: torch.distributions.Distribution | None = None,
        transform_point_data: bool = False,
        transform_cell_data: bool = False,
        transform_global_data: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        distribution : torch.distributions.Distribution or None
            Distribution from which the scale factor is sampled.
            Defaults to ``Uniform(0.9, 1.1)``.
        transform_point_data : bool
            If ``True``, transform point-data fields under scaling.
        transform_cell_data : bool
            If ``True``, transform cell-data fields under scaling.
        transform_global_data : bool
            If ``True``, transform global-data fields under scaling.
        """
        super().__init__()
        self._distribution = distribution or torch.distributions.Uniform(0.9, 1.1)
        self.transform_point_data = transform_point_data
        self.transform_cell_data = transform_cell_data
        self.transform_global_data = transform_global_data
        self._generator: torch.Generator | None = None

    def _sample_factor(self) -> Float[torch.Tensor, ""]:
        """Sample a scale factor from ``self._distribution``.

        Returns
        -------
        torch.Tensor
            Scalar (0-dim) tensor with the sampled factor.
        """
        return _sample_distribution(
            self._distribution, (1,), self._generator, self._device
        ).squeeze(0)

    def __call__(self, mesh: Mesh) -> Mesh:
        """Apply a random scale to *mesh*.

        Parameters
        ----------
        mesh : Mesh
            Input mesh.

        Returns
        -------
        Mesh
            Scaled mesh.
        """
        factor = self._sample_factor()
        return mesh.scale(
            factor,
            transform_point_data=self.transform_point_data,
            transform_cell_data=self.transform_cell_data,
            transform_global_data=self.transform_global_data,
        )

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Apply a random scale to every mesh in *domain*.

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
        factor = self._sample_factor()
        return domain.scale(
            factor,
            transform_point_data=self.transform_point_data,
            transform_cell_data=self.transform_cell_data,
            transform_global_data=self.transform_global_data,
        )

    def extra_repr(self) -> str:
        return f"distribution={self._distribution}"


@register()
class RandomTranslateMesh(MeshTransform):
    r"""Random translation of mesh.  Offset is sampled per ``__call__``.

    Each spatial axis is sampled independently from *distribution*
    (default ``Uniform(-0.1, 0.1)``).  Pass a batched distribution to
    control each axis separately, e.g.
    ``Uniform(tensor([-0.1, -0.2, -0.3]), tensor([0.1, 0.2, 0.3]))``.
    """

    def __init__(
        self,
        distribution: torch.distributions.Distribution | None = None,
    ) -> None:
        """
        Parameters
        ----------
        distribution : torch.distributions.Distribution or None
            Distribution from which the per-axis offsets are sampled.
            A scalar distribution produces IID samples per axis; a
            batched distribution (``batch_shape == (n_spatial_dims,)``)
            allows different parameters per axis.
            Defaults to ``Uniform(-0.1, 0.1)``.
        """
        super().__init__()
        self._distribution = distribution or torch.distributions.Uniform(-0.1, 0.1)
        self._generator: torch.Generator | None = None

    def _sample_offset(
        self, n_spatial_dims: int
    ) -> Float[torch.Tensor, " spatial_dims"]:
        """Sample a translation offset from ``self._distribution``.

        Parameters
        ----------
        n_spatial_dims : int
            Number of spatial dimensions (typically 2 or 3).

        Returns
        -------
        torch.Tensor
            Offset vector, shape ``(n_spatial_dims,)``.
        """
        return _sample_distribution(
            self._distribution, (n_spatial_dims,), self._generator, self._device
        )

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
        offset = self._sample_offset(mesh.n_spatial_dims)
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
        offset = self._sample_offset(domain.interior.n_spatial_dims)
        return domain.translate(offset)

    def extra_repr(self) -> str:
        return f"distribution={self._distribution}"


@register()
class RandomRotateMesh(MeshTransform):
    r"""Random rotation of mesh.  Axis and angle are sampled per ``__call__``.

    Two modes are supported:

    * ``"axis_aligned"`` (default) -- picks one of the candidate *axes*
      uniformly at random and samples an angle from *distribution*.
      This limits rotations to the three cardinal planes.
    * ``"uniform"`` -- samples a rotation uniformly from SO(3) via random
      unit quaternions (3-D meshes only).  *axes* and *distribution* are
      ignored in this mode.
    """

    def __init__(
        self,
        axes: list[Literal["x", "y", "z"]] | None = None,
        distribution: torch.distributions.Distribution | None = None,
        mode: Literal["axis_aligned", "uniform"] = "uniform",
        transform_point_data: bool = False,
        transform_cell_data: bool = False,
        transform_global_data: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        axes : list[{"x", "y", "z"}] or None
            Candidate rotation axes.  One is chosen uniformly at random
            per call.  Defaults to ``["x", "y", "z"]``.
            Only used when ``mode="axis_aligned"``.
        distribution : torch.distributions.Distribution or None
            Distribution from which the rotation angle (radians) is
            sampled.  Defaults to ``Uniform(-pi, pi)``.
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
        """
        super().__init__()
        if mode not in ("axis_aligned", "uniform"):
            raise ValueError(f"mode must be 'axis_aligned' or 'uniform', got {mode!r}")
        self.axes = axes if axes is not None else ["x", "y", "z"]
        self._distribution = distribution or torch.distributions.Uniform(
            -math.pi, math.pi
        )
        self.mode = mode
        self.transform_point_data = transform_point_data
        self.transform_cell_data = transform_cell_data
        self.transform_global_data = transform_global_data
        self._generator: torch.Generator | None = None

    # ------------------------------------------------------------------
    # axis-aligned helpers
    # ------------------------------------------------------------------

    def _sample_axis_and_angle(self) -> tuple[str, Float[torch.Tensor, ""]]:
        """Sample a random axis and rotation angle.

        The axis index is drawn via ``torch.randint`` with the generator.
        The angle is sampled from ``self._distribution`` via ICDF.

        Returns
        -------
        axis : str
            One of ``"x"``, ``"y"``, ``"z"``.
        angle : torch.Tensor
            Scalar (0-dim) tensor with the sampled angle in radians.
        """
        gen_device = (
            self._generator.device if self._generator is not None else self._device
        )
        axis_idx = torch.randint(
            len(self.axes), (1,), generator=self._generator, device=gen_device
        )
        axis = self.axes[axis_idx]
        angle = _sample_distribution(
            self._distribution, (1,), self._generator, self._device
        ).squeeze(0)
        return axis, angle

    # ------------------------------------------------------------------
    # uniform SO(3) helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _quaternion_to_rotation_matrix(
        q: Float[torch.Tensor, "4"],
    ) -> Float[torch.Tensor, "3 3"]:
        r"""Unit quaternion :math:`(w, \vec v)` to rotation matrix via Rodrigues' formula.

        :math:`R = (2w^2 - 1)\,I + 2\,\vec v\vec v^\top + 2w\,[\vec v]_\times`,
        where :math:`[\vec v]_\times` is the skew-symmetric cross-product matrix of
        :math:`\vec v`.

        Parameters
        ----------
        q : torch.Tensor
            Unit quaternion :math:`(w, x, y, z)`, shape :math:`(4,)`.

        Returns
        -------
        torch.Tensor
            Rotation matrix, shape :math:`(3, 3)`.
        """
        w, x, y, z = q.unbind()
        zero = torch.zeros_like(w)
        v_cross = torch.stack(
            [
                torch.stack([zero, -z, y]),
                torch.stack([z, zero, -x]),
                torch.stack([-y, x, zero]),
            ]
        )
        return (
            (2 * w * w - 1) * torch.eye(3, dtype=q.dtype, device=q.device)
            + 2 * torch.outer(q[1:], q[1:])
            + 2 * w * v_cross
        )

    def _sample_uniform_rotation(self) -> Float[torch.Tensor, "3 3"]:
        """Sample a rotation matrix uniformly from SO(3).

        Uses the random unit quaternion method: sample a 4-D isotropic
        Gaussian vector, normalize to the unit sphere, and convert to a
        rotation matrix.

        Returns
        -------
        torch.Tensor
            Rotation matrix, shape ``(3, 3)``.
        """
        gen_device = (
            self._generator.device if self._generator is not None else self._device
        )
        q = torch.randn(4, generator=self._generator, device=gen_device)
        q = q / q.norm()
        return self._quaternion_to_rotation_matrix(q)

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
        match self.mode:
            case "uniform":
                if mesh.n_spatial_dims != 3:
                    raise ValueError(
                        f"mode='uniform' requires 3-D meshes, "
                        f"got n_spatial_dims={mesh.n_spatial_dims}"
                    )
                R = self._sample_uniform_rotation()
                return mesh.transform(
                    R,
                    transform_point_data=self.transform_point_data,
                    transform_cell_data=self.transform_cell_data,
                    transform_global_data=self.transform_global_data,
                    assume_invertible=True,
                )
            case "axis_aligned":
                axis, angle = self._sample_axis_and_angle()
                return mesh.rotate(
                    angle,
                    axis=axis,
                    transform_point_data=self.transform_point_data,
                    transform_cell_data=self.transform_cell_data,
                    transform_global_data=self.transform_global_data,
                )
            case _:
                raise ValueError(
                    f"Unknown rotation mode {self.mode!r}. "
                    f"Expected 'uniform' or 'axis_aligned'."
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
        match self.mode:
            case "uniform":
                if domain.interior.n_spatial_dims != 3:
                    raise ValueError(
                        f"mode='uniform' requires 3-D meshes, "
                        f"got n_spatial_dims={domain.interior.n_spatial_dims}"
                    )
                R = self._sample_uniform_rotation()
                return domain.transform(
                    R,
                    transform_point_data=self.transform_point_data,
                    transform_cell_data=self.transform_cell_data,
                    transform_global_data=self.transform_global_data,
                    assume_invertible=True,
                )
            case "axis_aligned":
                axis, angle = self._sample_axis_and_angle()
                return domain.rotate(
                    angle,
                    axis=axis,
                    transform_point_data=self.transform_point_data,
                    transform_cell_data=self.transform_cell_data,
                    transform_global_data=self.transform_global_data,
                )
            case _:
                raise ValueError(
                    f"Unknown rotation mode {self.mode!r}. "
                    f"Expected 'uniform' or 'axis_aligned'."
                )

    def extra_repr(self) -> str:
        if self.mode == "uniform":
            return "mode='uniform'"
        return f"axes={self.axes}, distribution={self._distribution}"
