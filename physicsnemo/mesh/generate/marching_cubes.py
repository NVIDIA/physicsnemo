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

"""Isosurface extraction via the marching cubes algorithm."""

from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from jaxtyping import Float

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def marching_cubes(
    field: Float[torch.Tensor, "nx ny nz"],
    threshold: float = 0.0,
    coords: tuple[
        Float[torch.Tensor, " nx"],
        Float[torch.Tensor, " ny"],
        Float[torch.Tensor, " nz"],
    ]
    | None = None,
) -> "Mesh":
    r"""Extract an isosurface from a 3D scalar field using marching cubes.

    Given a volumetric scalar field (e.g. a signed distance field), this
    function extracts the isosurface at the specified threshold and returns
    it as a triangle :class:`~physicsnemo.mesh.Mesh`.

    When ``coords`` is provided, vertex positions are mapped from grid-index
    space into the physical coordinate system defined by the coordinate
    vectors. When ``coords`` is ``None``, vertices are returned in grid-index
    space.

    Uses `NVIDIA Warp <https://nvidia.github.io/warp/modules/runtime.html#marching-cubes>`_
    for the marching cubes implementation.

    Parameters
    ----------
    field : torch.Tensor
        A 3D scalar field with shape :math:`(N_x, N_y, N_z)`. Converted to
        float32 internally if necessary.
    threshold : float, optional
        Iso-value at which to extract the surface. Default is ``0.0``, which
        is the standard choice for signed distance fields.
    coords : tuple of 3 torch.Tensor, optional
        Physical coordinates along each grid axis, as 1D tensors of lengths
        :math:`N_x`, :math:`N_y`, :math:`N_z` respectively (e.g. from
        ``torch.linspace``). When provided, output vertices are mapped from
        grid-index space into physical space via piecewise linear
        interpolation along each axis. Both uniform and non-uniform grids
        are supported. When ``None``, vertices are in grid-index space.

    Returns
    -------
    Mesh
        A triangle mesh with ``points`` of shape :math:`(N_v, 3)` (float32)
        and ``cells`` of shape :math:`(N_f, 3)` (int64).

    Raises
    ------
    NotImplementedError
        If ``field`` is not 3-dimensional (higher/lower dimensions may be
        supported in a future release).
    ValueError
        If ``coords`` is provided but the lengths do not match the
        corresponding ``field`` dimensions.

    Notes
    -----
    This operation is **not differentiable**. The input tensor is detached
    and transferred to CPU/NumPy before being passed to Warp's marching cubes
    kernel, so gradients do not flow through this function.

    Examples
    --------
    Extract the zero-level set of a sphere SDF on a 64^3 grid in physical
    coordinates:

    >>> import torch
    >>> from physicsnemo.mesh.generate import marching_cubes
    >>> coords = torch.linspace(-1, 1, 64)
    >>> xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing="ij")
    >>> sdf = torch.sqrt(xx**2 + yy**2 + zz**2) - 0.5
    >>> sphere = marching_cubes(sdf, threshold=0.0, coords=(coords, coords, coords))
    >>> sphere.n_manifold_dims
    2
    >>> sphere.n_spatial_dims
    3
    """
    from physicsnemo.mesh.mesh import Mesh

    if field.ndim != 3:
        raise NotImplementedError(
            f"Only 3D scalar fields are currently supported, got {field.ndim}D "
            f"tensor with shape {tuple(field.shape)}"
        )

    if coords is not None:
        for dim, c in enumerate(coords):
            if c.shape[0] != field.shape[dim]:
                raise ValueError(
                    f"coords[{dim}] has length {c.shape[0]}, but field has "
                    f"size {field.shape[dim]} along dimension {dim}"
                )

    field_np = field.detach().cpu().numpy().astype(np.float32)
    field_wp = wp.array(field_np)

    mc = wp.MarchingCubes(
        nx=field_np.shape[0],
        ny=field_np.shape[1],
        nz=field_np.shape[2],
    )
    mc.surface(field=field_wp, threshold=threshold)

    points = torch.as_tensor(mc.verts.numpy(), dtype=torch.float32)  # (N_v, 3)
    cells = torch.as_tensor(
        mc.indices.numpy().reshape(-1, 3), dtype=torch.int64
    )  # (N_f, 3)

    ### Map from grid-index space to physical coordinates via piecewise linear interp
    if coords is not None:
        for dim, c in enumerate(coords):
            idx = points[:, dim]
            i = idx.long().clamp(0, c.shape[0] - 2)
            points[:, dim] = c[i] + (idx - i.float()) * (c[i + 1] - c[i])

    return Mesh(points=points, cells=cells)
