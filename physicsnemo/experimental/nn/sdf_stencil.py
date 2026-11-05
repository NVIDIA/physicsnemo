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

r"""Signed-distance stencil features of query points relative to a triangle surface."""

from __future__ import annotations

import torch
from jaxtyping import Float, Int

from physicsnemo.core.module import Module
from physicsnemo.nn.functional import signed_distance_field


class SignedDistanceStencil(Module):
    r"""Signed distances to a surface at a stencil of probes around each point, encoded by an MLP.

    For a point :math:`x` and a spacing :math:`h_j`, the signed distance :math:`\phi` to the surface (negative inside)
    is probed at the :math:`k^3` points :math:`x + h_j\,o` of a cubic stencil, :math:`o \in \{-1, \dots, 1\}^3`. Each
    value is divided by its spacing and clipped to :math:`\pm c`, so that a point in a thin member sees both of its
    faces within the finest spacings. An MLP maps the :math:`J k^3` values to ``out_features`` through an output
    projection that is initialized at zero: added to existing features, the encoding leaves them unchanged at the
    start of training.

    The distances are computed without gradient by
    :func:`~physicsnemo.nn.functional.signed_distance_field` (Warp BVH queries).

    Parameters
    ----------
    spacings : tuple[float, ...]
        Stencil spacings :math:`h_j`, in the units of the coordinates.
    out_features : int
        Number of output features.
    stencil : int, optional, default=3
        Points per axis :math:`k`; odd, at least 3.
    hidden : int, optional, default=64
        Width of the MLP.
    clip : float, optional, default=4.0
        Clip value :math:`c` of the signed distance divided by its spacing.
    use_sign_winding_number : bool, optional, default=False
        Take the sign from the generalized winding number, which tolerates holes in the surface, instead of the
        triangle normals. Either way the triangles have to face outward consistently: reversing them flips the sign.
    max_probes_per_call : int, optional, default=8388608
        Probes evaluated per call of the distance query (memory bound).

    Forward
    -------
    points : torch.Tensor
        Query points of shape :math:`(B, N, 3)`.
    vertices : torch.Tensor
        Surface vertices of shape :math:`(B, V, 3)`.
    faces : torch.Tensor
        Surface triangles of shape :math:`(B, F, 3)`, integer indices into ``vertices``.

    Outputs
    -------
    torch.Tensor
        Features of shape :math:`(B, N, \text{out_features})`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.nn import SignedDistanceStencil
    >>> layer = SignedDistanceStencil(spacings=(0.1, 0.4), out_features=8)
    >>> layer.num_probes
    54
    """

    def __init__(
        self,
        spacings: tuple[float, ...],
        out_features: int,
        stencil: int = 3,
        hidden: int = 64,
        clip: float = 4.0,
        use_sign_winding_number: bool = False,
        max_probes_per_call: int = 8 << 20,
    ) -> None:
        super().__init__()
        if not spacings or min(spacings) <= 0:
            raise ValueError("spacings must be positive")
        if stencil < 3 or stencil % 2 == 0:
            raise ValueError("stencil must be odd and at least 3")
        if out_features < 1 or hidden < 1 or clip <= 0 or max_probes_per_call < 1:
            raise ValueError("out_features, hidden, clip and max_probes_per_call must be positive")
        self.spacings = tuple(float(s) for s in spacings)
        self.stencil, self.clip = int(stencil), float(clip)
        self.use_sign_winding_number = bool(use_sign_winding_number)
        self.max_probes_per_call = int(max_probes_per_call)
        axis = torch.linspace(-1.0, 1.0, self.stencil)
        offsets = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
        spacing = torch.tensor(self.spacings)
        # (J * k^3, 3) probe offsets and the spacing of each probe
        self.register_buffer("offsets", (spacing[:, None, None] * offsets[None]).reshape(-1, 3), persistent=False)
        self.register_buffer("scales", spacing.repeat_interleave(offsets.shape[0]), persistent=False)
        self.num_probes = int(self.offsets.shape[0])
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.num_probes, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, hidden), torch.nn.GELU()
        )
        self.out = torch.nn.Linear(hidden, out_features)
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    @torch.no_grad()
    def probe_values(
        self,
        points: Float[torch.Tensor, "batch points 3"],
        vertices: Float[torch.Tensor, "batch vertices 3"],
        faces: Int[torch.Tensor, "batch faces 3"],
    ) -> Float[torch.Tensor, "batch points probes"]:
        r"""Signed distance of every probe divided by its spacing and clipped.

        Parameters
        ----------
        points : torch.Tensor
            Query points of shape :math:`(B, N, 3)`.
        vertices : torch.Tensor
            Surface vertices of shape :math:`(B, V, 3)`.
        faces : torch.Tensor
            Surface triangles of shape :math:`(B, F, 3)`.

        Returns
        -------
        torch.Tensor
            Values of shape :math:`(B, N, P)` in float32, :math:`P` = ``num_probes``, negative inside.
        """
        if not torch.compiler.is_compiling():
            if points.ndim != 3 or points.shape[-1] != 3:
                raise ValueError(f"Expected points of shape (B, N, 3), got {tuple(points.shape)}")
            if vertices.ndim != 3 or vertices.shape[-1] != 3 or vertices.shape[0] != points.shape[0]:
                raise ValueError(f"Expected vertices of shape (B, V, 3), got {tuple(vertices.shape)}")
            if faces.ndim != 3 or faces.shape[-1] != 3 or faces.shape[0] != points.shape[0]:
                raise ValueError(f"Expected faces of shape (B, F, 3), got {tuple(faces.shape)}")
        b, n, _ = points.shape
        offsets, scales = self.offsets.float(), self.scales.float()
        rows = max(1, self.max_probes_per_call // self.num_probes)
        limit = self.max_probes_per_call
        out = []
        with torch.autocast(device_type=points.device.type, enabled=False):
            for i in range(b):
                v, f, c = vertices[i].float().contiguous(), faces[i].long().contiguous(), points[i].float()
                pieces = [c.new_zeros((0, self.num_probes))]
                for s in range(0, n, rows):
                    # (rows, P, 3) probe positions -> signed distances, at most max_probes_per_call per call
                    probes = (c[s : s + rows, None] + offsets[None]).reshape(-1, 3)
                    values = []
                    for t in range(0, probes.shape[0], limit):
                        sdf = signed_distance_field(
                            v, f, probes[t : t + limit], use_sign_winding_number=self.use_sign_winding_number
                        )
                        values.append(sdf[0] if isinstance(sdf, (tuple, list)) else sdf)
                    pieces.append(torch.cat(values).reshape(-1, self.num_probes))
                out.append(torch.clamp(torch.cat(pieces) / scales, -self.clip, self.clip))
        return torch.stack(out)

    def forward(
        self,
        points: Float[torch.Tensor, "batch points 3"],
        vertices: Float[torch.Tensor, "batch vertices 3"],
        faces: Int[torch.Tensor, "batch faces 3"],
    ) -> Float[torch.Tensor, "batch points out_features"]:
        r"""Encode the signed distances of the stencil probes around every point.

        Parameters
        ----------
        points : torch.Tensor
            Query points of shape :math:`(B, N, 3)`.
        vertices : torch.Tensor
            Surface vertices of shape :math:`(B, V, 3)`.
        faces : torch.Tensor
            Surface triangles of shape :math:`(B, F, 3)`, facing outward.

        Returns
        -------
        torch.Tensor
            Features of shape :math:`(B, N, \text{out_features})`.
        """
        if not torch.compiler.is_compiling():
            if points.ndim != 3 or points.shape[-1] != 3:
                raise ValueError(f"Expected points of shape (B, N, 3), got {tuple(points.shape)}")
            if vertices.ndim != 3 or vertices.shape[-1] != 3 or vertices.shape[0] != points.shape[0]:
                raise ValueError(f"Expected vertices of shape (B, V, 3), got {tuple(vertices.shape)}")
            if faces.ndim != 3 or faces.shape[-1] != 3 or faces.shape[0] != points.shape[0]:
                raise ValueError(f"Expected faces of shape (B, F, 3), got {tuple(faces.shape)}")
        values = self.probe_values(points, vertices, faces).to(points.dtype)
        return self.out(self.mlp(values))
