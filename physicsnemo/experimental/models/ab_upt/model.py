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

r"""Independent implementation of AB-UPT from arXiv:2502.09692v4.

The implementation follows Sections 3.1--3.3 and Appendix C.3. It uses
PhysicsNeMo layers and neighbor search; no research implementation, weights,
or reference outputs were used. Explicit anchor coordinates keep sampling
outside the operator and make repeated/chunked queries refer to one field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F

from physicsnemo import ModelMetaData, Module
from physicsnemo.nn import FourierPositionalEmbedding, Mlp
from physicsnemo.nn.functional import radius_search
from physicsnemo.nn.module.rope import (
    apply_rotary_pos_emb,
    build_rope_cos_sin_1d_continuous,
)

_KV = tuple[Tensor, Tensor]
_RoPE = tuple[Tensor, Tensor]


@dataclass(frozen=True)
class ABUPTContext:
    r"""Per-forward anchor context returned by :meth:`ABUPT.encode`.

    Parameters
    ----------
    surface : tuple[tuple[torch.Tensor, torch.Tensor], ...]
        Projected keys and values for each surface layer. Each tensor has
        shape :math:`(B,H,M,D_h)`, with :math:`H` heads, :math:`M` source tokens,
        and head width :math:`D_h`. Keys already include rotary position
        encoding. The first pair represents
        geometry supernodes and is shared by the enabled branches. Later pairs
        represent that layer's source anchors (the opposite branch for cross
        attention, the same branch for self-attention).
    volume : tuple[tuple[torch.Tensor, torch.Tensor], ...]
        Corresponding volume contexts of shape :math:`(B,H,M,D_h)` per layer.
        Either branch's context is empty when that branch is disabled.
    surface_predictions : torch.Tensor or None
        Predictions at surface anchors, shape :math:`(B,M_s,C_s)`.
    volume_predictions : torch.Tensor or None
        Predictions at volume anchors, shape :math:`(B,M_v,C_v)`.
    conditioning : torch.Tensor or None
        Embedded global conditions of shape :math:`(B,1,d)`.

    Notes
    -----
    Caches retain their autograd graph. Reuse them for chunks of the same
    sample, model weights, device, precision, and training step only. They are
    not model state or persistent checkpoints. No query tokens enter a cache.
    """

    surface: tuple[_KV, ...]
    volume: tuple[_KV, ...]
    surface_predictions: Tensor | None
    volume_predictions: Tensor | None
    conditioning: Tensor | None


class _AnchorBlock(nn.Module):
    r"""AB-UPT pre-norm block with externally supplied anchor keys/values."""

    def __init__(self, width: int, heads: int, mlp_ratio: int, cross: bool = False):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width) if cross else self.norm
        self.query = nn.Linear(width, width)
        self.key_value = nn.Linear(width, 2 * width)
        self.output = nn.Linear(width, width)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = Mlp(width, mlp_ratio * width, width)

    @staticmethod
    def _rotate(features: Tensor, tables: _RoPE) -> Tensor:
        r"""Rotate ``(B,H,N,D_h)`` features using shared tables; retain tail channels."""
        width = tables[0].shape[-1]
        return torch.cat(
            (
                apply_rotary_pos_emb(features[..., :width], *tables),
                features[..., width:],
            ),
            dim=-1,
        )

    def _key_value(self, features: Tensor, tables: _RoPE) -> _KV:
        kv = self.key_value(self.context_norm(features))
        key, value = kv.unflatten(-1, (2, self.heads, self.head_dim)).unbind(-3)
        return self._rotate(key.transpose(1, 2), tables), value.transpose(1, 2)

    def forward(self, features: Tensor, tables: _RoPE, context: _KV) -> Tensor:
        query = self.query(self.norm(features)).unflatten(
            -1, (self.heads, self.head_dim)
        )
        query = self._rotate(query.transpose(1, 2), tables)
        attended = F.scaled_dot_product_attention(query, *context)
        features = features + self.output(attended.transpose(1, 2).flatten(-2))
        return features + self.mlp(self.mlp_norm(features))


class ABUPT(Module):
    r"""Anchored-Branched Universal Physics Transformer for 3D fields.

    Geometry is reduced by radius-based supernode pooling and one Transformer
    block. Surface and volume anchors then attend to geometry, exchange
    information through shared self/cross-attention blocks, and pass through
    branch-specific decoder blocks. Queries follow the same layers using only
    cached anchor keys and values (paper Eq. 9).

    Parameters
    ----------
    surface_channels : int, optional, default=4
        Surface output channels. Zero removes the entire surface branch.
    volume_channels : int, optional, default=7
        Volume output channels. Zero removes the entire volume branch.
    hidden_dim : int, optional, default=192
        Token width, divisible by ``num_heads``.
    num_heads : int, optional, default=3
        Attention heads. Each head must have at least six channels for 3D RoPE.
    num_interaction_blocks : int, optional, default=2
        Pairs of shared within-branch and cross-branch attention blocks. With
        one branch enabled, both blocks use self-attention, preserving depth.
    num_shared_layers : int, optional, default=1
        Shared within-branch blocks after the interactions.
    num_decoder_layers : int, optional, default=6
        Independently parameterized blocks per enabled output branch.
    mlp_ratio : int, optional, default=4
        Feed-forward expansion ratio.
    pooling_radius : float, optional, default=0.25
        Geometry neighborhood radius, in the supplied coordinate units.
    pooling_chunk_size : int, optional, default=4096
        Maximum messages embedded at once. Also bounds reference-point chunks
        for the PyTorch radius search backend on either CPU or CUDA.
    parameter_channels : int, optional, default=0
        Optional global conditioning channels, broadcast to both branches.
    radius_search_backend : {None, "torch", "warp"}, optional, default=None
        ``None`` selects Warp on CPU and CUDA, except for float64 coordinates,
        which use PyTorch to preserve precision. Neighbor search
        is discrete; message passing remains differentiable in coordinates.
        Warp searches in float32; the PyTorch backend preserves float64 inputs.
    pooling_message_fp32 : bool, optional, default=False
        Disable autocast for the geometry message MLP to reduce sensitivity
        to pooling chunk size. Requires float32 message weights under AMP;
        explicit float64 models retain double precision. This is an optional
        numerical variant, not a precision setting specified by the paper.
    coordinate_scale : float, optional, default=1.0
        Multiplier applied by ``forward`` to coordinates and to the configured
        ``pooling_radius``. Low-level encode/decode methods use scaled coordinates.
    parameter_scale : float, optional, default=1.0
        Divisor applied by ``forward`` to global conditioning values.
    query_chunk_size : int, optional, default=4096
        Maximum queries decoded at once using a single shared anchor context.
        Bounds individual attention operations, not the full autograd graph.
    train_on_anchors : bool, optional, default=False
        Use anchor predictions during training. Supplied queries must equal
        their anchors in shape, coordinates, and order. Evaluation decodes queries.

    Forward
    -------
    geometry : torch.Tensor
        Geometry point cloud of shape :math:`(B,N_g,3)`.
    supernodes : torch.Tensor
        Selected geometry supernode coordinates of shape :math:`(B,S,3)`.
        Normally a uniform subset of ``geometry``; chosen by the caller.
    surface_anchors : torch.Tensor or None, optional, default=None
        Fixed surface anchor coordinates of shape :math:`(B,M_s,3)`, required
        when the surface branch is enabled and otherwise omitted.
    volume_anchors : torch.Tensor or None, optional, default=None
        Fixed volume anchor coordinates of shape :math:`(B,M_v,3)`, required
        when the volume branch is enabled and otherwise omitted.
    surface_queries : torch.Tensor or None, optional, default=None
        Surface query coordinates of shape :math:`(B,Q_s,3)`. ``None`` uses
        predictions at the surface anchors without another decoder pass.
    volume_queries : torch.Tensor or None, optional, default=None
        Volume query coordinates of shape :math:`(B,Q_v,3)`. ``None`` uses
        predictions at the volume anchors.
    parameters : torch.Tensor or None, optional, default=None
        Global conditions of shape :math:`(B,P)` or :math:`(B,1,P)`, required
        exactly when ``parameter_channels > 0``.

    Outputs
    -------
    tuple[torch.Tensor or None, torch.Tensor or None]
        Surface and volume predictions, respectively, with shapes
        :math:`(B,Q_s,C_s)` and :math:`(B,Q_v,C_v)`. Disabled heads return
        ``None``. When queries are omitted, :math:`Q_s=M_s` and :math:`Q_v=M_v`.

    Notes
    -----
    Independently implemented from `the AB-UPT paper
    <https://arxiv.org/abs/2502.09692v4>`_. Defaults give the 12-block DrivAerML
    branch layout in Appendix C.3: 1 geometry cross-attention, 4 interleaved
    blocks, 1 shared self-attention, and 6 branch-specific blocks. The paper
    leaves some details unspecified; this implementation uses mean message
    aggregation, GELU MLPs, axis-wise sinusoidal/RoPE encodings, and no dropout.
    It does not claim numerical compatibility with the research code.

    Single-branch models are ablations of the paper's coupled architecture:
    they retain geometry encoding and anchor decoding, replace cross-branch
    attention with self-attention, and require no inactive-branch inputs.

    Sampling belongs to the caller. Input coordinates and the configured
    ``pooling_radius`` must use the same units. ``forward`` scales both by
    ``coordinate_scale`` and casts inputs to the model weight dtype. The positional
    frequencies span approximately 1 to 1e-4; coordinates of order 1000 exercise more frequency
    bands than unit-scale coordinates. When rescaling coordinates, rescale the
    radius as well; the default radius is in caller-supplied units.

    Queries must never be used to resample anchors between chunks.
    :meth:`encode` / :meth:`decode_queries` expose this separation, and
    :meth:`decode_vorticity` implements the paper's finite-difference curl with
    normalization corrections. Direct volume outputs are not constrained to
    equal the curl; the helper computes it separately from predicted velocity.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.ab_upt import ABUPT
    >>> model = ABUPT(hidden_dim=24, num_heads=3, num_interaction_blocks=1,
    ...               num_shared_layers=0, num_decoder_layers=1)
    >>> geometry = torch.rand(1, 20, 3)
    >>> supernodes = geometry[:, ::4]
    >>> surface = geometry[:, 1::3]
    >>> volume = torch.rand(1, 7, 3)
    >>> pressure, flow = model(geometry, supernodes, surface, volume)
    >>> pressure.shape, flow.shape
    (torch.Size([1, 7, 4]), torch.Size([1, 7, 7]))
    """

    def __init__(
        self,
        surface_channels: int = 4,
        volume_channels: int = 7,
        hidden_dim: int = 192,
        num_heads: int = 3,
        num_interaction_blocks: int = 2,
        num_shared_layers: int = 1,
        num_decoder_layers: int = 6,
        mlp_ratio: int = 4,
        pooling_radius: float = 0.25,
        pooling_chunk_size: int = 4096,
        parameter_channels: int = 0,
        radius_search_backend: Literal["torch", "warp"] | None = None,
        pooling_message_fp32: bool = False,
        coordinate_scale: float = 1.0,
        parameter_scale: float = 1.0,
        query_chunk_size: int = 4096,
        train_on_anchors: bool = False,
    ):
        super().__init__(
            meta=ModelMetaData(
                jit=False,
                cuda_graphs=False,
                amp_cpu=True,
                amp_gpu=True,
                onnx=False,
                auto_grad=True,
            )
        )
        for name, value, minimum in (
            ("surface_channels", surface_channels, 0),
            ("volume_channels", volume_channels, 0),
            ("hidden_dim", hidden_dim, 6),
            ("num_heads", num_heads, 1),
            ("num_interaction_blocks", num_interaction_blocks, 0),
            ("num_shared_layers", num_shared_layers, 0),
            ("num_decoder_layers", num_decoder_layers, 0),
            ("mlp_ratio", mlp_ratio, 1),
            ("pooling_chunk_size", pooling_chunk_size, 1),
            ("parameter_channels", parameter_channels, 0),
            ("query_chunk_size", query_chunk_size, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
            setattr(self, name, value)
        if hidden_dim % num_heads or hidden_dim // num_heads < 6:
            raise ValueError(
                "hidden_dim must be divisible by num_heads with head width >= 6"
            )
        if surface_channels + volume_channels == 0:
            raise ValueError("At least one output head must be enabled")
        if not math.isfinite(pooling_radius) or pooling_radius <= 0:
            raise ValueError("pooling_radius must be finite and positive")
        if radius_search_backend not in (None, "torch", "warp"):
            raise ValueError("radius_search_backend must be None, 'torch', or 'warp'")
        if not isinstance(pooling_message_fp32, bool):
            raise ValueError("pooling_message_fp32 must be a bool")
        self.pooling_message_fp32 = pooling_message_fp32
        for name, value in (
            ("coordinate_scale", coordinate_scale),
            ("parameter_scale", parameter_scale),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, value)
        if not isinstance(train_on_anchors, bool):
            raise ValueError("train_on_anchors must be a boolean")
        self.train_on_anchors = train_on_anchors
        self.pooling_radius = pooling_radius * coordinate_scale
        self.radius_search_backend = radius_search_backend
        pairs = hidden_dim // 6
        self.position = FourierPositionalEmbedding(
            in_dim=3,
            include_input=False,
            freqs=10000.0 ** (-torch.arange(pairs) / pairs),
        )
        rotary_pairs = (hidden_dim // num_heads) // 6
        self.register_buffer(
            "rotary_frequencies",
            10000.0 ** (-torch.arange(rotary_pairs) / rotary_pairs),
        )
        self.message = Mlp(4, hidden_dim, hidden_dim)
        self.pool_projection = Mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        self.geometry_block = _AnchorBlock(hidden_dim, num_heads, mlp_ratio)
        self.surface_embedding = (
            Mlp(hidden_dim, hidden_dim, hidden_dim) if surface_channels else None
        )
        self.volume_embedding = (
            Mlp(hidden_dim, hidden_dim, hidden_dim) if volume_channels else None
        )
        self.conditioning = (
            nn.Linear(parameter_channels, hidden_dim) if parameter_channels else None
        )
        self.geometry_attention = _AnchorBlock(
            hidden_dim, num_heads, mlp_ratio, cross=True
        )
        # Alternate within-branch and cross-branch blocks, then shared self
        # attention. Single-branch models use self-attention throughout.
        self.cross_branch = (
            False,
            bool(surface_channels and volume_channels),
        ) * num_interaction_blocks + (False,) * num_shared_layers
        self.shared_blocks = nn.ModuleList(
            [
                _AnchorBlock(hidden_dim, num_heads, mlp_ratio, cross=cross)
                for cross in self.cross_branch
            ]
        )
        self.surface_decoder = nn.ModuleList(
            [
                _AnchorBlock(hidden_dim, num_heads, mlp_ratio)
                for _ in range(num_decoder_layers if surface_channels else 0)
            ]
        )
        self.volume_decoder = nn.ModuleList(
            [
                _AnchorBlock(hidden_dim, num_heads, mlp_ratio)
                for _ in range(num_decoder_layers if volume_channels else 0)
            ]
        )
        self.surface_head = (
            nn.Linear(hidden_dim, surface_channels) if surface_channels else None
        )
        self.volume_head = (
            nn.Linear(hidden_dim, volume_channels) if volume_channels else None
        )

    def _embed_positions(self, positions: Tensor) -> Tensor:
        r"""Interleave shared Fourier features per axis and pad to the token width."""
        dtype = torch.float64 if positions.dtype == torch.float64 else torch.float32
        encoded = self.position(positions.to(dtype))
        encoded = encoded.unflatten(-1, (3, 2, self.position.num_bands))
        encoded = encoded.transpose(-1, -2).flatten(-3)
        if self.hidden_dim % 6:
            encoded = F.pad(encoded, (0, self.hidden_dim % 6))
        return encoded.to(positions.dtype)

    def _rope_tables(self, positions: Tensor) -> _RoPE:
        r"""Build shared axial tables of shape ``(B,1,N,6 * pairs)`` once per point set."""
        cos, sin = build_rope_cos_sin_1d_continuous(positions, self.rotary_frequencies)
        return cos.flatten(-2).unsqueeze(1), sin.flatten(-2).unsqueeze(1)

    def _encode_geometry(
        self, geometry: Tensor, supernodes: Tensor, tables: _RoPE
    ) -> Tensor:
        r"""Encode geometry with the caller's supernode rotation tables."""
        features = self._pool(geometry, supernodes)
        return self.geometry_block(
            features, tables, self.geometry_block._key_value(features, tables)
        )

    @staticmethod
    def _validate_points(
        points: Tensor, name: str, reference: Tensor | None = None, empty: bool = False
    ) -> None:
        if (
            points.ndim != 3
            or points.shape[-1] != 3
            or points.shape[0] == 0
            or (points.shape[1] == 0 and not empty)
        ):
            raise ValueError(
                f"Expected tensor of shape (B, N, 3) for {name} but got tensor of shape {tuple(points.shape)}"
            )
        if not points.is_floating_point():
            raise ValueError(f"{name} must be floating point")
        if reference is not None and (
            points.shape[0] != reference.shape[0]
            or points.device != reference.device
            or points.dtype != reference.dtype
        ):
            raise ValueError(
                f"{name} must match the reference batch size, device, and dtype"
            )

    def _pool(self, geometry: Tensor, supernodes: Tensor) -> Tensor:
        r"""Pool ``(B,N_g,3)`` geometry at ``(B,S,3)`` supernodes into ``(B,S,d)``.

        Empty neighborhoods contribute zero local messages before projection.
        """
        # Query/reference tiling bounds the PyTorch backend's distance matrix.
        # Warp uses a spatial grid and can search the complete cloud directly.
        backend = self.radius_search_backend or (
            "torch" if geometry.dtype == torch.float64 else "warp"
        )
        batch, count, _ = supernodes.shape
        # Counts and message sums must not accumulate in fp16/bf16: large
        # neighborhoods can overflow fp16 counts or round away bf16 increments.
        # Retain double precision when explicitly requested by the caller.
        accumulation_dtype = (
            torch.float64 if geometry.dtype == torch.float64 else torch.float32
        )
        if self.pooling_message_fp32 and any(
            p.dtype != accumulation_dtype for p in self.message.parameters()
        ):
            raise ValueError(
                "pooling_message_fp32 requires float32 message weights "
                "(float64 for float64 coordinates); use autocast instead of "
                "converting model weights to float16 or bfloat16"
            )
        search_dtype = accumulation_dtype if backend == "torch" else torch.float32
        pooled = torch.zeros(
            batch * count,
            self.hidden_dim,
            device=geometry.device,
            dtype=accumulation_dtype,
        )
        counts = torch.zeros(
            batch * count, 1, device=geometry.device, dtype=accumulation_dtype
        )
        query_step = count if backend == "warp" else min(128, count)
        point_step = geometry.shape[1] if backend == "warp" else self.pooling_chunk_size
        for q_start in range(0, count, query_step):
            for p_start in range(0, geometry.shape[1], point_step):
                with torch.no_grad():
                    edges = radius_search(
                        geometry[:, p_start : p_start + point_step]
                        .detach()
                        .to(search_dtype)
                        .contiguous(),
                        supernodes[:, q_start : q_start + query_step]
                        .detach()
                        .to(search_dtype)
                        .contiguous(),
                        self.pooling_radius,
                        implementation=backend,
                    )
                for chunk in edges.split(self.pooling_chunk_size, dim=1):
                    b, q, p = chunk.long().unbind(0)
                    q, p = q + q_start, p + p_start
                    relative = geometry[b, p] - supernodes[b, q]
                    message_input = torch.cat(
                        (relative, relative.norm(dim=-1, keepdim=True)), dim=-1
                    )
                    if self.pooling_message_fp32:
                        with torch.autocast(geometry.device.type, enabled=False):
                            message = self.message(message_input.to(accumulation_dtype))
                    else:
                        message = self.message(message_input)
                    ids = b * count + q
                    pooled.index_add_(0, ids, message.to(pooled.dtype))
                    counts.index_add_(0, ids, counts.new_ones(ids.shape[0], 1))
        # Empty neighborhoods contribute zero local information; absolute
        # supernode coordinates still identify their geometry tokens.
        pooled = (
            (pooled / counts.clamp_min(1))
            .reshape(batch, count, self.hidden_dim)
            .to(geometry.dtype)
        )
        return self.pool_projection(
            torch.cat((pooled, self._embed_positions(supernodes)), dim=-1)
        )

    def encode_geometry(
        self,
        geometry: Float[Tensor, "B Ng 3"],
        supernodes: Float[Tensor, "B S 3"],
    ) -> Float[Tensor, "B S D"]:
        r"""Pool local geometry and exchange information between supernodes.

        Parameters
        ----------
        geometry : torch.Tensor
            Geometry coordinates of shape :math:`(B,N_g,3)`.
        supernodes : torch.Tensor
            Supernode coordinates of shape :math:`(B,S,3)`.

        Returns
        -------
        torch.Tensor
            Encoded geometry of shape :math:`(B,S,d)`.
        """
        if not torch.compiler.is_compiling():
            self._validate_points(geometry, "geometry")
            self._validate_points(supernodes, "supernodes", geometry)
        return self._encode_geometry(
            geometry, supernodes, self._rope_tables(supernodes)
        )

    def encode(
        self,
        geometry: Float[Tensor, "B Ng 3"],
        supernodes: Float[Tensor, "B S 3"],
        surface_anchors: Float[Tensor, "B Ms 3"] | None = None,
        volume_anchors: Float[Tensor, "B Mv 3"] | None = None,
        parameters: Float[Tensor, "B P"] | None = None,
    ) -> ABUPTContext:
        r"""Build a reusable field context from geometry and fixed anchors.

        Parameters
        ----------
        geometry : torch.Tensor
            Geometry coordinates of shape :math:`(B,N_g,3)`.
        supernodes : torch.Tensor
            Geometry supernodes of shape :math:`(B,S,3)`.
        surface_anchors : torch.Tensor or None, optional, default=None
            Surface anchors of shape :math:`(B,M_s,3)`. Omit for volume-only.
        volume_anchors : torch.Tensor or None, optional, default=None
            Volume anchors of shape :math:`(B,M_v,3)`. Omit for surface-only.
        parameters : torch.Tensor or None, optional, default=None
            Global conditions of shape :math:`(B,P)`.

        Returns
        -------
        ABUPTContext
            Layer contexts and anchor predictions, retaining their autograd
            graph for training. In joint mode, streams interact synchronously.
        """
        if not torch.compiler.is_compiling():
            self._validate_points(geometry, "geometry")
            self._validate_points(supernodes, "supernodes", geometry)
            for name, points, channels in (
                ("surface_anchors", surface_anchors, self.surface_channels),
                ("volume_anchors", volume_anchors, self.volume_channels),
            ):
                if channels:
                    if points is None:
                        raise ValueError(f"{name} are required for the enabled branch")
                    self._validate_points(points, name, geometry)
                elif points is not None:
                    raise ValueError(f"{name} supplied for a disabled branch")
            if self.parameter_channels:
                if parameters is None:
                    raise ValueError(
                        "parameters are required when parameter_channels > 0"
                    )
                if parameters.shape != (
                    geometry.shape[0],
                    self.parameter_channels,
                ):
                    raise ValueError(
                        f"Expected tensor of shape (B, {self.parameter_channels}) for parameters but got tensor of shape {tuple(parameters.shape)}"
                    )
                if (
                    parameters.device != geometry.device
                    or parameters.dtype != geometry.dtype
                ):
                    raise ValueError("parameters must match geometry device and dtype")
            elif parameters is not None:
                raise ValueError("parameters requires parameter_channels > 0")
        geometry_tables = self._rope_tables(supernodes)
        surface_tables = (
            self._rope_tables(surface_anchors) if self.surface_channels else None
        )
        volume_tables = (
            self._rope_tables(volume_anchors) if self.volume_channels else None
        )
        geometry_features = self._encode_geometry(geometry, supernodes, geometry_tables)
        geometry_kv = self.geometry_attention._key_value(
            geometry_features, geometry_tables
        )
        conditioning = (
            self.conditioning(parameters).unsqueeze(1)
            if self.conditioning is not None
            else None
        )
        surface = volume = None
        surface_context, volume_context = [], []
        if self.surface_embedding is not None:
            surface = self.surface_embedding(self._embed_positions(surface_anchors))
            if conditioning is not None:
                surface = surface + conditioning
            surface = self.geometry_attention(surface, surface_tables, geometry_kv)
            surface_context.append(geometry_kv)
        if self.volume_embedding is not None:
            volume = self.volume_embedding(self._embed_positions(volume_anchors))
            if conditioning is not None:
                volume = volume + conditioning
            volume = self.geometry_attention(volume, volume_tables, geometry_kv)
            volume_context.append(geometry_kv)
        for block, cross in zip(self.shared_blocks, self.cross_branch):
            surface_kv = (
                block._key_value(surface, surface_tables)
                if surface is not None
                else None
            )
            volume_kv = (
                block._key_value(volume, volume_tables) if volume is not None else None
            )
            if cross:
                surface_kv, volume_kv = volume_kv, surface_kv
            # Form both contexts before updating either branch (Eq. 8).
            if surface is not None:
                surface_context.append(surface_kv)
                surface = block(surface, surface_tables, surface_kv)
            if volume is not None:
                volume_context.append(volume_kv)
                volume = block(volume, volume_tables, volume_kv)
        for block in self.surface_decoder:
            kv = block._key_value(surface, surface_tables)
            surface_context.append(kv)
            surface = block(surface, surface_tables, kv)
        for block in self.volume_decoder:
            kv = block._key_value(volume, volume_tables)
            volume_context.append(kv)
            volume = block(volume, volume_tables, kv)
        return ABUPTContext(
            tuple(surface_context),
            tuple(volume_context),
            self.surface_head(surface) if self.surface_head is not None else None,
            self.volume_head(volume) if self.volume_head is not None else None,
            conditioning,
        )

    def decode_queries(
        self,
        queries: Float[Tensor, "B Q 3"],
        context: ABUPTContext,
        field: Literal["surface", "volume"] = "surface",
    ) -> Float[Tensor, "B Q C"]:
        r"""Evaluate independent query coordinates using one fixed context.

        Parameters
        ----------
        queries : torch.Tensor
            Coordinates of shape :math:`(B,Q,3)`. Empty query sets are allowed.
        context : ABUPTContext
            Context from :meth:`encode` with the same batch size and device.
        field : {"surface", "volume"}, optional, default="surface"
            Enabled output branch to evaluate.

        Returns
        -------
        torch.Tensor
            Predictions of shape :math:`(B,Q,C)` independent of other queries.
        """
        if field not in ("surface", "volume"):
            raise ValueError("field must be 'surface' or 'volume'")
        head = self.surface_head if field == "surface" else self.volume_head
        if head is None:
            raise ValueError(f"{field} output head is disabled")
        embedding = (
            self.surface_embedding if field == "surface" else self.volume_embedding
        )
        decoder = self.surface_decoder if field == "surface" else self.volume_decoder
        blocks = (self.geometry_attention, *self.shared_blocks, *decoder)
        if not torch.compiler.is_compiling():
            self._validate_points(queries, "queries", empty=True)
            if not isinstance(context, ABUPTContext):
                raise ValueError("context must be an ABUPTContext returned by encode")
        contexts = context.surface if field == "surface" else context.volume
        if not torch.compiler.is_compiling():
            if len(contexts) != len(blocks):
                raise ValueError(
                    f"Expected {len(blocks)} layer contexts but got {len(contexts)}"
                )
            for kv in contexts:
                if not isinstance(kv, tuple) or len(kv) != 2:
                    raise ValueError("Each layer context must be a (key, value) pair")
                for tensor in kv:
                    if (
                        tensor.ndim != 4
                        or tensor.shape[0] != queries.shape[0]
                        or tensor.shape[1] != self.num_heads
                        or tensor.shape[2] == 0
                        or tensor.shape[3] != self.hidden_dim // self.num_heads
                    ):
                        raise ValueError(
                            f"Expected tensor of shape (B, {self.num_heads}, M, {self.hidden_dim // self.num_heads}) but got tensor of shape {tuple(tensor.shape)}"
                        )
                    if tensor.device != queries.device:
                        raise ValueError(
                            "queries and context must be on the same device"
                        )
                if kv[0].shape != kv[1].shape or kv[0].dtype != kv[1].dtype:
                    raise ValueError("Context key/value shapes and dtypes must match")
            if context.conditioning is not None and context.conditioning.shape != (
                queries.shape[0],
                1,
                self.hidden_dim,
            ):
                raise ValueError(
                    f"Expected tensor of shape (B, 1, {self.hidden_dim}) for conditioning but got tensor of shape {tuple(context.conditioning.shape)}"
                )
        features = embedding(self._embed_positions(queries))
        if context.conditioning is not None:
            features = features + context.conditioning
        tables = self._rope_tables(queries)
        for block, kv in zip(blocks, contexts):
            features = block(features, tables, kv)
        return head(features)

    def decode_vorticity(
        self,
        queries: Float[Tensor, "B Q 3"],
        context: ABUPTContext,
        step: float = 1.0e-3,
        velocity_channels: tuple[int, int, int] = (1, 2, 3),
        coordinate_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        velocity_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Float[Tensor, "B Q 3"]:
        r"""Approximate physical vorticity with the curl in paper Eq. 11.

        Parameters
        ----------
        queries : torch.Tensor
            Network-space coordinates of shape :math:`(B,Q,3)`.
        context : ABUPTContext
            Fixed context shared by all six offset evaluations.
        step : float, optional, default=1.0e-3
            Central-difference step in network coordinate units.
        velocity_channels : tuple[int, int, int], optional, default=(1, 2, 3)
            Volume output indices for x/y/z velocity, respectively.
        coordinate_scale : tuple[float, float, float], optional, default=(1.0, 1.0, 1.0)
            Diagonal scaling from physical to network coordinates.
        velocity_scale : tuple[float, float, float], optional, default=(1.0, 1.0, 1.0)
            Diagonal scaling from network to physical velocity. Additive
            normalization offsets cancel when differentiating.

        Returns
        -------
        torch.Tensor
            Physical vorticity of shape :math:`(B,Q,3)`. Central differences
            approximate the continuous curl; their error depends on ``step``
            and precision. Offset evaluations disable autocast and use the
            model's float32 or float64 weight dtype, including cached keys,
            values, and conditioning. Explicit half-precision weights are not
            supported. Gradients propagate to the model and context.
        """
        if not math.isfinite(step) or step <= 0:
            raise ValueError("step must be finite and positive")
        if (
            len(velocity_channels) != 3
            or len(set(velocity_channels)) != 3
            or any(
                not isinstance(c, int) or not 0 <= c < self.volume_channels
                for c in velocity_channels
            )
        ):
            raise ValueError(
                "velocity_channels must contain three distinct valid volume channels"
            )
        for name, scale in (
            ("coordinate_scale", coordinate_scale),
            ("velocity_scale", velocity_scale),
        ):
            if len(scale) != 3 or any(not math.isfinite(s) or s <= 0 for s in scale):
                raise ValueError(f"{name} must contain three finite positive values")
        if not torch.compiler.is_compiling():
            self._validate_points(queries, "queries", empty=True)
        dtype = self.volume_head.weight.dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError(
                "decode_vorticity requires float32 or float64 model weights"
            )
        queries = queries.to(dtype)
        context = replace(
            context,
            volume=tuple((k.to(dtype), v.to(dtype)) for k, v in context.volume),
            conditioning=(
                context.conditioning.to(dtype)
                if context.conditioning is not None
                else None
            ),
        )
        derivatives = []
        with torch.autocast(queries.device.type, enabled=False):
            for axis in range(3):
                offset = queries.new_zeros(3)
                offset[axis] = step
                plus = self.decode_queries(queries + offset, context, field="volume")
                minus = self.decode_queries(queries - offset, context, field="volume")
                derivative = (
                    plus[..., list(velocity_channels)]
                    - minus[..., list(velocity_channels)]
                ) / (2 * step)
                derivatives.append(
                    derivative
                    * queries.new_tensor(velocity_scale)
                    * coordinate_scale[axis]
                )
        dx, dy, dz = derivatives
        return torch.stack(
            (dy[..., 2] - dz[..., 1], dz[..., 0] - dx[..., 2], dx[..., 1] - dy[..., 0]),
            dim=-1,
        )

    def get_extra_state(self):
        """Persist numerical settings that are not represented by weight shapes."""
        return {
            "coordinate_scale": self.coordinate_scale,
            "parameter_scale": self.parameter_scale,
            "train_on_anchors": self.train_on_anchors,
            "pooling_radius": self.pooling_radius,
            "num_heads": self.num_heads,
            "cross_branch": list(self.cross_branch),
        }

    def set_extra_state(self, state):
        """Reject a checkpoint loaded with different numerical settings."""
        if state != self.get_extra_state():
            raise ValueError(
                "AB-UPT checkpoint settings do not match the current model"
            )

    def forward(
        self,
        geometry: Float[Tensor, "B Ng 3"],
        supernodes: Float[Tensor, "B S 3"],
        surface_anchors: Float[Tensor, "B Ms 3"] | None = None,
        volume_anchors: Float[Tensor, "B Mv 3"] | None = None,
        surface_queries: Float[Tensor, "B Qs 3"] | None = None,
        volume_queries: Float[Tensor, "B Qv 3"] | None = None,
        parameters: Float[Tensor, "B P"] | Float[Tensor, "B 1 P"] | None = None,
    ) -> tuple[Float[Tensor, "B Qs Cs"] | None, Float[Tensor, "B Qv Cv"] | None]:
        r"""Scale inputs, encode once, and predict at anchors or chunked queries.

        Disabled branches return ``None``. Low-level :meth:`encode` and
        :meth:`decode_queries` accept coordinates in the scaled model frame.
        """
        if not torch.compiler.is_compiling():
            self._validate_points(geometry, "geometry")
            for name, queries, anchors, channels in (
                ("surface", surface_queries, surface_anchors, self.surface_channels),
                ("volume", volume_queries, volume_anchors, self.volume_channels),
            ):
                if queries is not None:
                    if not channels:
                        raise ValueError(
                            f"{name}_queries supplied for a disabled output head"
                        )
                    self._validate_points(queries, f"{name}_queries", empty=True)
                    if (
                        queries.shape[0] != geometry.shape[0]
                        or queries.device != geometry.device
                    ):
                        raise ValueError(
                            f"{name}_queries must match the geometry batch size and device"
                        )
                    if (
                        self.training
                        and self.train_on_anchors
                        and (
                            anchors is None
                            or queries.shape != anchors.shape
                            or not torch.equal(queries, anchors)
                        )
                    ):
                        raise ValueError(
                            f"Anchor-supervised training requires {name}_queries to "
                            f"equal {name}_anchors in shape, coordinates, and order"
                        )
        dtype = next(self.parameters()).dtype

        def scale(points):
            return (
                None
                if points is None
                else points.to(dtype=dtype) * self.coordinate_scale
            )

        if parameters is not None:
            if parameters.ndim == 3 and parameters.shape[1] == 1:
                parameters = parameters[:, 0]
            parameters = parameters.to(dtype=dtype) / self.parameter_scale
        context = self.encode(
            scale(geometry),
            scale(supernodes),
            scale(surface_anchors),
            scale(volume_anchors),
            parameters,
        )
        predictions = []
        for field, queries in (
            ("surface", surface_queries),
            ("volume", volume_queries),
        ):
            if queries is None or (self.training and self.train_on_anchors):
                prediction = getattr(context, f"{field}_predictions")
            else:
                prediction = torch.cat(
                    [
                        self.decode_queries(scale(chunk), context, field)
                        for chunk in queries.split(self.query_chunk_size, dim=1)
                    ],
                    dim=1,
                )
            predictions.append(prediction)
        return predictions[0], predictions[1]
