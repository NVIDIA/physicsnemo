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

r"""Radius-pooled local features of query points from a cloud of candidate points."""

from __future__ import annotations

import torch
from jaxtyping import Bool, Float

from physicsnemo.core.module import Module
from physicsnemo.nn.functional import radius_search

#: Coordinate of the placeholder candidate at index 0, so that index 0 always marks an empty neighbour slot
_FAR = 1e6


class RadiusPooledFeatures(Module):
    r"""Local features from the neighbours within each radius: a shared MLP of their relative positions, max-pooled.

    For every radius :math:`r_s` the candidates are voxel-downsampled at :math:`v_s` (the first candidate of each
    occupied voxel, in index order), so that the features do not depend on the local mesh density. Up to
    ``query_cap`` of them within :math:`r_s` of a query are found by
    :func:`~physicsnemo.nn.functional.radius_search`, and :math:`K_s` of those are drawn uniformly at random: a fresh
    draw at every training step, a fixed seed in evaluation. Their positions relative to the query, divided by
    :math:`r_s`, go through a shared MLP, are max-pooled over the drawn neighbours and pass through :math:`\tanh`; a
    query without neighbours gets zeros.

    Parameters
    ----------
    radii : tuple[float, ...]
        Radii :math:`r_s`, in the units of the coordinates.
    caps : tuple[int, ...]
        Neighbours :math:`K_s` drawn per radius.
    voxel_sizes : tuple[float, ...]
        Voxel edges :math:`v_s` of the candidate downsampling per radius.
    hidden : int
        Width of the MLPs and of each output.
    query_cap : int, optional, default=128
        Neighbours found per query and radius before the draw; at least ``max(caps)``.
    seed : int, optional, default=0
        Seed of the draw in evaluation mode.

    Forward
    -------
    candidates : torch.Tensor
        Candidate points of shape :math:`(B, P, 3)`.
    queries : torch.Tensor
        Query points of shape :math:`(B, N, 3)`.

    Outputs
    -------
    tuple[torch.Tensor, ...]
        One feature tensor of shape :math:`(B, N, \text{hidden})` per radius.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.nn import RadiusPooledFeatures
    >>> layer = RadiusPooledFeatures(radii=(0.1,), caps=(4,), voxel_sizes=(0.02,), hidden=8, query_cap=8)
    >>> len(layer.mlps)
    1
    """

    def __init__(
        self,
        radii: tuple[float, ...],
        caps: tuple[int, ...],
        voxel_sizes: tuple[float, ...],
        hidden: int,
        query_cap: int = 128,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if not radii or len(caps) != len(radii) or len(voxel_sizes) != len(radii):
            raise ValueError("radii, caps and voxel_sizes need one entry per radius")
        if min(radii) <= 0 or min(voxel_sizes) <= 0 or min(caps) < 1 or hidden < 1:
            raise ValueError("radii, voxel_sizes, caps and hidden must be positive")
        if query_cap < max(caps):
            raise ValueError("query_cap must be at least max(caps)")
        self.radii = tuple(float(r) for r in radii)
        self.caps = tuple(int(c) for c in caps)
        self.voxel_sizes = tuple(float(v) for v in voxel_sizes)
        self.query_cap, self.seed = int(query_cap), int(seed)
        self.mlps = torch.nn.ModuleList(
            torch.nn.Sequential(torch.nn.Linear(3, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, hidden))
            for _ in self.radii
        )

    @staticmethod
    def _voxel_first(points: Float[torch.Tensor, "points 3"], voxel: float) -> Float[torch.Tensor, "kept 3"]:
        r"""The first point of every occupied voxel of edge ``voxel``, in index order.

        Parameters
        ----------
        points : torch.Tensor
            Points of shape :math:`(P, 3)`.
        voxel : float
            Voxel edge.

        Returns
        -------
        torch.Tensor
            Kept points of shape :math:`(P', 3)`.
        """
        if points.shape[0] == 0:
            return points
        _, inverse = torch.unique(torch.floor(points / voxel).long(), dim=0, return_inverse=True)
        first = torch.full((int(inverse.max()) + 1,), points.shape[0], device=points.device, dtype=torch.long)
        first = first.scatter_reduce(0, inverse, torch.arange(points.shape[0], device=points.device), "amin")
        return points[first.sort().values]

    @staticmethod
    def _check_points(candidates: torch.Tensor, queries: torch.Tensor) -> None:
        r"""Raise ``ValueError`` unless candidates are :math:`(B, P, 3)` and queries :math:`(B, N, 3)`."""
        if candidates.ndim != 3 or candidates.shape[-1] != 3:
            raise ValueError(f"Expected candidates of shape (B, P, 3), got {tuple(candidates.shape)}")
        if queries.ndim != 3 or queries.shape[-1] != 3 or queries.shape[0] != candidates.shape[0]:
            raise ValueError(f"Expected queries of shape (B, N, 3), got {tuple(queries.shape)}")

    @torch.no_grad()
    def neighbours(
        self,
        candidates: Float[torch.Tensor, "batch candidates 3"],
        queries: Float[torch.Tensor, "batch queries 3"],
    ) -> list[tuple[Float[torch.Tensor, "batch queries cap 3"], Bool[torch.Tensor, "batch queries cap"]]]:
        r"""The drawn neighbours of every query for each radius.

        Parameters
        ----------
        candidates : torch.Tensor
            Candidate points of shape :math:`(B, P, 3)`.
        queries : torch.Tensor
            Query points of shape :math:`(B, N, 3)`.

        Returns
        -------
        list[tuple[torch.Tensor, torch.Tensor]]
            Per radius, neighbour coordinates of shape :math:`(B, N, K_s, 3)` and their validity :math:`(B, N, K_s)`.
        """
        if not torch.compiler.is_compiling():
            self._check_points(candidates, queries)
        b, n, _ = queries.shape
        generator = None if self.training else torch.Generator(device=queries.device).manual_seed(self.seed)
        out = []
        with torch.autocast(device_type=queries.device.type, enabled=False):
            for radius, voxel, cap in zip(self.radii, self.voxel_sizes, self.caps):
                coords, valid = [], []
                for i in range(b):
                    q = queries[i].float()
                    if n == 0:  # no queries: empty outputs
                        coords.append(q.new_zeros((0, cap, 3)))
                        valid.append(torch.zeros((0, cap), dtype=torch.bool, device=q.device))
                        continue
                    cloud = self._voxel_first(candidates[i].float(), voxel)
                    cloud = torch.cat([cloud.new_full((1, 3), _FAR), cloud])  # index 0: empty-slot marker
                    idx = radius_search(cloud, q, radius, max_points=self.query_cap)  # (N, query_cap), 0 = empty
                    ok = idx > 0
                    # uniform draw of K_s among the found neighbours: random priorities, empty slots last
                    priority = torch.where(ok, torch.rand(ok.shape, device=q.device, generator=generator), -1.0)
                    pick = priority.topk(cap, dim=1).indices
                    chosen, keep = idx.gather(1, pick), priority.gather(1, pick) >= 0
                    coords.append(cloud[chosen])
                    valid.append(keep)
                out.append((torch.stack(coords), torch.stack(valid)))
        return out

    def pool(
        self,
        neighbours: list[tuple[Float[torch.Tensor, "batch queries cap 3"], Bool[torch.Tensor, "batch queries cap"]]],
        queries: Float[torch.Tensor, "batch queries 3"],
    ) -> tuple[Float[torch.Tensor, "batch queries hidden"], ...]:
        r"""Features of given neighbours: the differentiable part of :meth:`forward`.

        Splitting the draw (:meth:`neighbours`) from the pooling lets a caller recompute only the pooling under
        activation checkpointing, so that the recomputation sees the same random draw.

        Parameters
        ----------
        neighbours : list[tuple[torch.Tensor, torch.Tensor]]
            Output of :meth:`neighbours`.
        queries : torch.Tensor
            Query points of shape :math:`(B, N, 3)`.

        Returns
        -------
        tuple[torch.Tensor, ...]
            One feature tensor of shape :math:`(B, N, \text{hidden})` per radius.
        """
        if not torch.compiler.is_compiling():
            if queries.ndim != 3 or queries.shape[-1] != 3:
                raise ValueError(f"Expected queries of shape (B, N, 3), got {tuple(queries.shape)}")
            if len(neighbours) != len(self.radii):
                raise ValueError(f"Expected neighbours for {len(self.radii)} radii, got {len(neighbours)}")
            for (coords, valid), cap in zip(neighbours, self.caps):
                if coords.shape != (*queries.shape[:2], cap, 3) or valid.shape != coords.shape[:3]:
                    raise ValueError(
                        f"Expected neighbours of shape (B, N, {cap}, 3) and (B, N, {cap}), "
                        f"got {tuple(coords.shape)} and {tuple(valid.shape)}"
                    )
        features = []
        for (coords, valid), radius, mlp in zip(neighbours, self.radii, self.mlps):
            coords = coords.to(queries.dtype)
            # (B, N, K, 3) positions relative to the query, in units of the radius; empty slots at zero
            relative = torch.where(valid[..., None], (coords - queries[:, :, None]) / radius, torch.zeros_like(coords))
            h = mlp(relative).masked_fill(~valid[..., None], float("-inf")).amax(dim=2)  # (B, N, hidden)
            features.append(torch.tanh(torch.where(torch.isfinite(h), h, torch.zeros_like(h))))
        return tuple(features)

    def forward(
        self,
        candidates: Float[torch.Tensor, "batch candidates 3"],
        queries: Float[torch.Tensor, "batch queries 3"],
    ) -> tuple[Float[torch.Tensor, "batch queries hidden"], ...]:
        r"""Draw the neighbours of every query and pool their features (:meth:`neighbours`, then :meth:`pool`).

        Parameters
        ----------
        candidates : torch.Tensor
            Candidate points of shape :math:`(B, P, 3)`.
        queries : torch.Tensor
            Query points of shape :math:`(B, N, 3)`.

        Returns
        -------
        tuple[torch.Tensor, ...]
            One feature tensor of shape :math:`(B, N, \text{hidden})` per radius.
        """
        if not torch.compiler.is_compiling():
            self._check_points(candidates, queries)
        return self.pool(self.neighbours(candidates, queries), queries)
