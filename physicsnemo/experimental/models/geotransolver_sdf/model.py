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

r"""GeoTransolver with local geometry encodings at every mesh node (experimental).

A :class:`~physicsnemo.models.geotransolver.GeoTransolver` backbone in which every node of a volume mesh is a query
and sees its neighbourhood through two encodings: radius-pooled features of the neighbouring nodes
(:class:`~physicsnemo.experimental.nn.RadiusPooledFeatures`) and a stencil of signed distances to the surface
(:class:`~physicsnemo.experimental.nn.SignedDistanceStencil`). The slice aggregation of the GALE blocks scales its
regularization with the number of nodes (:class:`ScaledSliceGALE`), so that meshes of very different sizes are sliced
alike.
"""

from __future__ import annotations

import torch
from jaxtyping import Float, Int

from physicsnemo import ModelMetaData, Module
from physicsnemo.experimental.nn import RadiusPooledFeatures, SignedDistanceStencil
from physicsnemo.models.geotransolver import GeoTransolver
from physicsnemo.nn.functional import knn
from physicsnemo.nn.module.gale import GALE


class ScaledSliceGALE(GALE):
    r"""GALE attention whose slice normalization scales with the number of tokens.

    :class:`~physicsnemo.nn.module.gale.GALE` aggregates the tokens into slices with weights normalized by their sum
    plus a fixed :math:`\epsilon`. On meshes from thousands to hundreds of thousands of nodes a fixed :math:`\epsilon`
    regularizes small meshes much more than large ones; here it grows with the number of tokens :math:`N`,

    .. math::

        \epsilon_N = \epsilon \, N / N_\text{ref},

    and the slice temperature is clamped from below at ``temperature_min`` instead of 0.5. Everything else is
    :class:`~physicsnemo.nn.module.gale.GALE`.

    Parameters
    ----------
    dim : int
        Token width.
    heads : int
        Number of attention heads.
    dim_head : int
        Width per head.
    slice_num : int
        Number of physics slices.
    context_dim : int
        Width of the context for cross-attention.
    dropout : float, optional, default=0.0
        Dropout rate.
    slice_epsilon : float, optional, default=0.01
        :math:`\epsilon` at :math:`N = N_\text{ref}` tokens.
    slice_epsilon_reference : int, optional, default=4096
        Reference token count :math:`N_\text{ref}`.
    temperature_min : float, optional, default=0.3
        Lower clamp of the slice temperature.

    Forward
    -------
    x : tuple[torch.Tensor, ...]
        Input tensors, each of shape :math:`(B, N, C)`.
    context : torch.Tensor
        Context of shape :math:`(B, H, S_c, D_c)`.

    Outputs
    -------
    list[torch.Tensor]
        One output tensor per input, each of shape :math:`(B, N, C)`.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        slice_num: int,
        context_dim: int,
        dropout: float = 0.0,
        slice_epsilon: float = 0.01,
        slice_epsilon_reference: int = 4096,
        temperature_min: float = 0.3,
    ) -> None:
        super().__init__(
            dim, heads=heads, dim_head=dim_head, dropout=dropout, slice_num=slice_num, context_dim=context_dim
        )
        if not slice_epsilon > 0 or slice_epsilon_reference < 1 or not 0 < temperature_min <= 5:
            raise ValueError("slice_epsilon, slice_epsilon_reference and temperature_min (at most 5) must be positive")
        self.slice_epsilon = float(slice_epsilon)
        self.slice_epsilon_reference = int(slice_epsilon_reference)
        self.temperature_min = float(temperature_min)

    def _compute_slices_from_projections(
        self,
        slice_projections: Float[torch.Tensor, "B N H S"],
        fx: Float[torch.Tensor, "B N H D"],
    ) -> tuple[Float[torch.Tensor, "B N H S"], Float[torch.Tensor, "B H S D"]]:
        r"""Slice weights and tokens with the token-count-scaled normalization.

        Parameters
        ----------
        slice_projections : torch.Tensor
            Slice logits of shape :math:`(B, N, H, S)`.
        fx : torch.Tensor
            Token features of shape :math:`(B, N, H, D)`.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Slice weights of shape :math:`(B, N, H, S)` (before normalization) and slice tokens of shape
            :math:`(B, H, S, D)`.
        """
        temperature = torch.clamp(self.temperature, min=self.temperature_min, max=5.0).to(slice_projections.dtype)
        weights = torch.softmax(slice_projections / temperature, dim=-1).to(slice_projections.dtype)  # (B, N, H, S)
        epsilon = self.slice_epsilon * slice_projections.shape[1] / self.slice_epsilon_reference
        normed = weights / (weights.sum(1)[:, None] + epsilon)
        # (B, H, S, N) @ (B, H, N, D) -> (B, H, S, D)
        tokens = torch.matmul(normed.permute(0, 2, 3, 1), fx.permute(0, 2, 1, 3))
        return weights, tokens


def _knn(
    points: Float[torch.Tensor, "points 3"], queries: Float[torch.Tensor, "queries 3"], k: int, chunk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Exact :math:`k` nearest points of every query, nearest first.

    On CUDA one cuML call (it builds its index once, so splitting the queries would rebuild it per piece); without
    cuML, on the CPU or when cuML runs out of memory, the PyTorch implementation on ``chunk`` queries at a time.

    Parameters
    ----------
    points : torch.Tensor
        Candidate points of shape :math:`(P, 3)`.
    queries : torch.Tensor
        Query points of shape :math:`(Q, 3)`.
    k : int
        Neighbours per query, at most :math:`P`.
    chunk : int
        Queries per call of the PyTorch implementation.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Indices and Euclidean distances, each of shape :math:`(Q, k)`.
    """
    if points.is_cuda:
        try:
            return knn(points, queries, k, implementation="cuml")
        except (ImportError, KeyError, MemoryError):
            torch.cuda.empty_cache()
    found = [knn(points, queries[s : s + chunk], k, implementation="torch") for s in range(0, len(queries), chunk)]
    return torch.cat([f[0] for f in found]), torch.cat([f[1] for f in found])


def _nearest_within_radii(
    points: Float[torch.Tensor, "batch points 3"],
    queries: Float[torch.Tensor, "batch queries 3"],
    radii: tuple[float, ...],
    counts: tuple[int, ...],
    chunk: int,
) -> list[Float[torch.Tensor, "batch queries neighbor_coordinates"]]:
    r"""Coordinates of the up to :math:`k_s` nearest points within each radius :math:`r_s` of every query.

    Parameters
    ----------
    points : torch.Tensor
        Candidate points of shape :math:`(B, P, 3)`.
    queries : torch.Tensor
        Query points of shape :math:`(B, Q, 3)`.
    radii : tuple[float, ...]
        Radii :math:`r_s`.
    counts : tuple[int, ...]
        Neighbours :math:`k_s` per radius.
    chunk : int
        Queries per call of the PyTorch nearest-neighbour search (see :func:`_knn`).

    Returns
    -------
    list[torch.Tensor]
        Per radius, flattened neighbour coordinates of shape :math:`(B, Q, 3 k_s)`, nearest first, zeros for empty
        slots.
    """
    b, nq = queries.shape[:2]
    k = min(max(counts), points.shape[1])
    out: list[list[torch.Tensor]] = [[] for _ in radii]
    with torch.no_grad(), torch.autocast(device_type=queries.device.type, enabled=False):
        for i in range(b):
            p, q = points[i].float(), queries[i].float()
            idx, dist = _knn(p, q, k, chunk)  # (Q, k), nearest first
            for j, (radius, count) in enumerate(zip(radii, counts)):
                width = min(count, k)
                coords = torch.where((dist[:, :width] <= radius)[..., None], p[idx[:, :width]], 0.0)  # (Q, w, 3)
                if width < count:
                    coords = torch.cat([coords, coords.new_zeros(nq, count - width, 3)], dim=1)
                out[j].append(coords.reshape(nq, 3 * count))
    return [torch.stack(o).to(queries.dtype) for o in out]


class GeoTransolverSDF(Module):
    r"""GeoTransolver with radius-pooled node features and a signed-distance stencil at every mesh node.

    Every node of a volume mesh is a query. Its input features (coordinates first) are projected as in
    :class:`~physicsnemo.models.geotransolver.GeoTransolver` and concatenated with
    :class:`~physicsnemo.experimental.nn.RadiusPooledFeatures` of the neighbouring nodes; a
    :class:`~physicsnemo.experimental.nn.SignedDistanceStencil` of the surface is added, which tells each node the
    local member thickness. The context of the GALE blocks is built from the geometry points, each described by its
    nearest nodes within each radius, plus the global features. The blocks use :class:`ScaledSliceGALE`.

    Parameters
    ----------
    functional_dim : int, optional, default=7
        Width of the node features; the first three are the coordinates.
    out_dim : int, optional, default=4
        Output channels per node.
    geometry_dim : int, optional, default=3
        Width of the geometry points; their coordinates, so 3.
    global_dim : int, optional, default=4
        Width of the global features.
    n_layers : int, optional, default=8
        Number of GALE blocks.
    n_hidden : int, optional, default=192
        Width of the projected node features.
    n_head : int, optional, default=6
        Number of attention heads.
    mlp_ratio : int, optional, default=2
        Hidden-width ratio of the block MLPs.
    slice_num : int, optional, default=256
        Number of physics slices.
    n_hidden_local : int, optional, default=48
        Width of each local feature (per radius).
    radii : tuple[float, ...], optional, default=(0.05, 0.2)
        Radii of the local features, in the units of the coordinates.
    neighbors_in_radius : tuple[int, ...], optional, default=(8, 32)
        Nearest nodes per radius that describe a geometry point in the context.
    local_caps : tuple[int, ...], optional, default=(64, 64)
        Neighbouring nodes drawn per radius for the pooled node features.
    radius_voxels : tuple[float, ...], optional, default=(0.015, 0.06)
        Voxel edges of the node downsampling per radius.
    radius_query_cap : int, optional, default=128
        Neighbours found per node and radius before the draw.
    sdf_spacings : tuple[float, ...], optional, default=(0.005, 0.02, 0.06)
        Spacings of the signed-distance stencil.
    sdf_stencil : int, optional, default=3
        Stencil points per axis.
    sdf_hidden : int, optional, default=64
        Width of the stencil MLP.
    sdf_winding_number : bool, optional, default=False
        Sign of the stencil distances from the winding number instead of the triangle normals.
    slice_epsilon : float, optional, default=0.01
        Slice-normalization :math:`\epsilon` at ``slice_epsilon_reference`` nodes.
    slice_epsilon_reference : int, optional, default=4096
        Reference node count of the slice normalization.
    temperature_min : float, optional, default=0.3
        Lower clamp of the slice temperature.
    activation_checkpointing : bool, optional, default=False
        Recompute the blocks and the pooled features in the backward pass.
    context_chunk : int, optional, default=512
        Geometry points per call of the PyTorch nearest-neighbour search when building the context (used on the CPU,
        or on CUDA without cuML).

    Forward
    -------
    local_embedding : torch.Tensor
        Node features of shape :math:`(B, N, F)`, coordinates first.
    geometry : torch.Tensor
        Geometry points of shape :math:`(B, M, 3)`.
    global_embedding : torch.Tensor
        Global features of shape :math:`(B, 1, G)`.
    surface_vertices : torch.Tensor
        Surface vertices of shape :math:`(B, V, 3)`.
    surface_faces : torch.Tensor
        Surface triangles of shape :math:`(B, T, 3)`, facing outward.

    Outputs
    -------
    torch.Tensor
        Node outputs of shape :math:`(B, N, \text{out_dim})`.

    Notes
    -----
    The neighbour searches and the signed distances run per sample of the batch, without gradient, in float32. The
    model is meant for one sample per GPU (meshes of different sizes are not padded into a batch) and has not been
    tested with domain parallelism (:class:`~physicsnemo.domain_parallel.ShardTensor`), under which
    :class:`ScaledSliceGALE` would count the tokens of the local shard.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.geotransolver_sdf import GeoTransolverSDF
    >>> model = GeoTransolverSDF(n_layers=1, n_hidden=16, n_head=2, slice_num=4, n_hidden_local=8,
    ...                          local_caps=(4, 4), radius_query_cap=8, neighbors_in_radius=(2, 4))
    >>> sum(p.numel() for p in model.pooling.parameters())
    208
    """

    def __init__(
        self,
        functional_dim: int = 7,
        out_dim: int = 4,
        geometry_dim: int = 3,
        global_dim: int = 4,
        n_layers: int = 8,
        n_hidden: int = 192,
        n_head: int = 6,
        mlp_ratio: int = 2,
        slice_num: int = 256,
        n_hidden_local: int = 48,
        radii: tuple[float, ...] = (0.05, 0.2),
        neighbors_in_radius: tuple[int, ...] = (8, 32),
        local_caps: tuple[int, ...] = (64, 64),
        radius_voxels: tuple[float, ...] = (0.015, 0.06),
        radius_query_cap: int = 128,
        sdf_spacings: tuple[float, ...] = (0.005, 0.02, 0.06),
        sdf_stencil: int = 3,
        sdf_hidden: int = 64,
        sdf_winding_number: bool = False,
        slice_epsilon: float = 0.01,
        slice_epsilon_reference: int = 4096,
        temperature_min: float = 0.3,
        activation_checkpointing: bool = False,
        context_chunk: int = 512,
    ) -> None:
        super().__init__(meta=ModelMetaData(amp=True, auto_grad=True))
        if len(neighbors_in_radius) != len(radii) or min(neighbors_in_radius) < 1 or context_chunk < 1:
            raise ValueError("neighbors_in_radius needs one positive entry per radius, and context_chunk must be positive")
        if geometry_dim != 3:
            raise ValueError("geometry_dim must be 3: the context neighbours are found from the geometry coordinates")
        self.functional_dim, self.geometry_dim, self.global_dim = int(functional_dim), int(geometry_dim), int(global_dim)
        self.radii = tuple(float(r) for r in radii)
        self.neighbors_in_radius = tuple(int(k) for k in neighbors_in_radius)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.context_chunk = int(context_chunk)
        self.core = GeoTransolver(
            functional_dim=functional_dim,
            out_dim=out_dim,
            geometry_dim=geometry_dim,
            global_dim=global_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_head=n_head,
            mlp_ratio=mlp_ratio,
            slice_num=slice_num,
            include_local_features=True,
            radii=list(self.radii),
            neighbors_in_radius=list(self.neighbors_in_radius),
            n_hidden_local=n_hidden_local,
        )
        width = n_hidden + n_hidden_local * len(self.radii)
        context_dim = self.core.context_builder.get_context_dim()
        for block in self.core.blocks:
            block.Attn = ScaledSliceGALE(
                width,
                heads=n_head,
                dim_head=width // n_head,
                slice_num=slice_num,
                context_dim=context_dim,
                slice_epsilon=slice_epsilon,
                slice_epsilon_reference=slice_epsilon_reference,
                temperature_min=temperature_min,
            )
        self.pooling = RadiusPooledFeatures(
            self.radii, local_caps, radius_voxels, n_hidden_local, query_cap=radius_query_cap
        )
        self.stencil = SignedDistanceStencil(
            sdf_spacings, out_features=width, stencil=sdf_stencil, hidden=sdf_hidden,
            use_sign_winding_number=sdf_winding_number,
        )

    def forward(
        self,
        local_embedding: Float[torch.Tensor, "batch nodes functional_dim"],
        geometry: Float[torch.Tensor, "batch geometry_points 3"],
        global_embedding: Float[torch.Tensor, "batch 1 global_dim"],
        surface_vertices: Float[torch.Tensor, "batch vertices 3"],
        surface_faces: Int[torch.Tensor, "batch triangles 3"],
    ) -> Float[torch.Tensor, "batch nodes out_dim"]:
        r"""Predict the outputs at every node.

        Parameters
        ----------
        local_embedding : torch.Tensor
            Node features of shape :math:`(B, N, F)`, coordinates first.
        geometry : torch.Tensor
            Geometry points of shape :math:`(B, M, 3)`.
        global_embedding : torch.Tensor
            Global features of shape :math:`(B, 1, G)`.
        surface_vertices : torch.Tensor
            Surface vertices of shape :math:`(B, V, 3)`.
        surface_faces : torch.Tensor
            Surface triangles of shape :math:`(B, T, 3)`, facing outward.

        Returns
        -------
        torch.Tensor
            Node outputs of shape :math:`(B, N, \text{out_dim})`.
        """
        if not torch.compiler.is_compiling():
            b = local_embedding.shape[0]
            if local_embedding.ndim != 3 or local_embedding.shape[-1] != self.functional_dim:
                raise ValueError(
                    f"Expected local_embedding of shape (B, N, {self.functional_dim}), got {tuple(local_embedding.shape)}"
                )
            if geometry.ndim != 3 or geometry.shape[0] != b or geometry.shape[-1] != self.geometry_dim:
                raise ValueError(f"Expected geometry of shape (B, M, {self.geometry_dim}), got {tuple(geometry.shape)}")
            if global_embedding.shape != (b, 1, self.global_dim):
                raise ValueError(
                    f"Expected global_embedding of shape (B, 1, {self.global_dim}), got {tuple(global_embedding.shape)}"
                )
            if surface_vertices.ndim != 3 or surface_vertices.shape[0] != b or surface_vertices.shape[-1] != 3:
                raise ValueError(f"Expected surface_vertices of shape (B, V, 3), got {tuple(surface_vertices.shape)}")
            if surface_faces.ndim != 3 or surface_faces.shape[0] != b or surface_faces.shape[-1] != 3:
                raise ValueError(f"Expected surface_faces of shape (B, T, 3), got {tuple(surface_faces.shape)}")
        coords = local_embedding[..., :3]
        checkpointing = self.activation_checkpointing and torch.is_grad_enabled()

        # Context: each geometry point described by its nearest nodes per radius, tokenized per radius, then the
        # geometry and global tokens; (B, H, S_c, D_c) concatenated along the last axis
        builder = self.core.context_builder
        extractor = builder.local_extractors[0]
        neighbours = _nearest_within_radii(coords, geometry, self.radii, self.neighbors_in_radius, self.context_chunk)
        context = torch.cat(
            [
                *(
                    tokenizer(torch.tanh(processor.mlp(n)))
                    for processor, tokenizer, n in zip(extractor.processors, extractor.tokenizers, neighbours)
                ),
                builder.geometry_tokenizer(geometry),
                builder.global_tokenizer(global_embedding),
            ],
            dim=-1,
        )

        # Node rows: projected features, pooled neighbour features per radius, plus the signed-distance stencil
        drawn = self.pooling.neighbours(coords, coords)  # outside the checkpoint: the recomputation sees the same draw
        if checkpointing:
            pooled = torch.utils.checkpoint.checkpoint(self.pooling.pool, drawn, coords, use_reentrant=False)
        else:
            pooled = self.pooling.pool(drawn, coords)
        x = torch.cat([self.core.preprocess[0](local_embedding), *pooled], dim=-1)  # (B, N, width)
        x = x + self.stencil(coords, surface_vertices, surface_faces)

        for block in self.core.blocks:
            if checkpointing:
                x = torch.utils.checkpoint.checkpoint(block, (x,), context, use_reentrant=False)[0]
            else:
                x = block((x,), context)[0]
        return self.core.ln_mlp_out[0](x)
