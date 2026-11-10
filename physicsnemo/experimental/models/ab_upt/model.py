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
from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F

from physicsnemo import ModelMetaData, Module
from physicsnemo.nn import Mlp
from physicsnemo.nn.functional import radius_search

_KV = tuple[Tensor, Tensor]


@dataclass(frozen=True)
class ABUPTContext:
    r"""Per-forward anchor context returned by :meth:`ABUPT.encode`.

    Parameters
    ----------
    surface : tuple[tuple[torch.Tensor, torch.Tensor], ...]
        Projected keys and values for each surface layer. Each tensor has
        shape :math:`(B,H,M,D)`; keys already include rotary position encoding.
    volume : tuple[tuple[torch.Tensor, torch.Tensor], ...]
        Corresponding volume contexts of shape :math:`(B,H,M,D)` per layer.
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


class _Coordinates(nn.Module):
    r"""Axis-wise sinusoidal encoding used by this AB-UPT implementation."""

    def __init__(self, width: int):
        super().__init__()
        self.width = width
        pairs = math.ceil(width / 6)
        self.register_buffer("frequencies", 10000.0 ** (-torch.arange(pairs) / pairs))

    def forward(self, positions: Tensor) -> Tensor:
        # Preserve phase accuracy under autocast, including large coordinates.
        dtype = torch.float64 if positions.dtype == torch.float64 else torch.float32
        phase = positions.to(dtype).unsqueeze(-1) * self.frequencies.to(dtype)
        return (
            torch.stack((phase.sin(), phase.cos()), dim=-1)
            .flatten(-3)[..., : self.width]
            .to(positions.dtype)
        )


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
        pairs = self.head_dim // 6
        self.register_buffer("frequencies", 10000.0 ** (-torch.arange(pairs) / pairs))

    def _rotate(self, features: Tensor, positions: Tensor) -> Tensor:
        # Allocate equal rotary pairs to x/y/z; leave remaining head channels
        # unchanged when the head width is not divisible by six.
        pairs = self.frequencies.numel()
        rotary_width = 6 * pairs
        dtype = (
            torch.float64
            if torch.float64 in (positions.dtype, features.dtype)
            else torch.float32
        )
        phase = positions.to(dtype).unsqueeze(1).unsqueeze(-1) * self.frequencies.to(
            dtype
        )
        # Keep the rotation arithmetic, not just its phases, in fp32 (paper
        # Appendix A.6). Round once at the end to retain the attention dtype.
        cosine, sine = phase.cos(), phase.sin()
        rotated = features[..., :rotary_width].to(dtype).unflatten(-1, (3, pairs, 2))
        even, odd = rotated.unbind(-1)
        rotated = torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine), -1
        )
        return torch.cat(
            (rotated.flatten(-3).to(features.dtype), features[..., rotary_width:]),
            dim=-1,
        )

    def _key_value(self, features: Tensor, positions: Tensor) -> _KV:
        kv = self.key_value(self.context_norm(features))
        key, value = kv.unflatten(-1, (2, self.heads, self.head_dim)).unbind(-3)
        return self._rotate(key.transpose(1, 2), positions), value.transpose(1, 2)

    def forward(self, features: Tensor, positions: Tensor, context: _KV) -> Tensor:
        query = self.query(self.norm(features)).unflatten(
            -1, (self.heads, self.head_dim)
        )
        query = self._rotate(query.transpose(1, 2), positions)
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
        for the CPU radius search fallback.
    parameter_channels : int, optional, default=0
        Optional global conditioning channels, broadcast to both branches.
    radius_search_backend : {None, "torch", "warp"}, optional, default=None
        ``None`` selects Warp on CUDA and PyTorch otherwise. Neighbor search
        is discrete; message passing remains differentiable in coordinates.
        Warp searches in float32; the PyTorch backend preserves float64 inputs.
    pooling_message_fp32 : bool, optional, default=False
        Disable autocast for the geometry message MLP to reduce sensitivity
        to pooling chunk size. Requires float32 message weights under AMP;
        explicit float64 models retain double precision. This is an optional
        numerical variant, not a precision setting specified by the paper.

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
        Global conditions of shape :math:`(B,P)`, required exactly when
        ``parameter_channels > 0``.

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

    Sampling and normalization belong to the caller. Queries must never be
    used to resample anchors between chunks. ``encode`` / ``decode_queries``
    expose this separation, and ``decode_vorticity`` implements the paper's
    finite-difference curl with normalization corrections. Direct volume
    output channels are unconstrained unless that helper is explicitly used.

    Examples
    --------
    >>> model = ABUPT(hidden_dim=24, num_heads=3, num_interaction_blocks=1,
    ...               num_shared_layers=0, num_decoder_layers=1)
    >>> geometry = torch.rand(1, 20, 3)
    >>> surface = geometry[:, :5]
    >>> volume = torch.rand(1, 7, 3)
    >>> pressure, flow = model(geometry, surface, surface, volume)
    >>> pressure.shape, flow.shape
    (torch.Size([1, 5, 4]), torch.Size([1, 7, 7]))
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
    ):
        super().__init__(
            meta=ModelMetaData(
                name="ABUPT",
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
        self.pooling_radius = pooling_radius
        self.radius_search_backend = radius_search_backend
        self.position = _Coordinates(hidden_dim)
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
        # Query/reference tiling bounds the CPU fallback's distance matrix.
        # Warp uses a spatial grid and can search the complete cloud directly.
        backend = self.radius_search_backend or (
            "warp" if geometry.is_cuda else "torch"
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
            torch.cat((pooled, self.position(supernodes)), dim=-1)
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
        features = self._pool(geometry, supernodes)
        return self.geometry_block(
            features, supernodes, self.geometry_block._key_value(features, supernodes)
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
                if parameters is None or parameters.shape != (
                    geometry.shape[0],
                    self.parameter_channels,
                ):
                    raise ValueError(
                        f"Expected tensor of shape (B, {self.parameter_channels}) for parameters but got tensor of shape {None if parameters is None else tuple(parameters.shape)}"
                    )
                if (
                    parameters.device != geometry.device
                    or parameters.dtype != geometry.dtype
                ):
                    raise ValueError("parameters must match geometry device and dtype")
            elif parameters is not None:
                raise ValueError("parameters requires parameter_channels > 0")
        geometry_features = self.encode_geometry(geometry, supernodes)
        geometry_kv = self.geometry_attention._key_value(geometry_features, supernodes)
        conditioning = (
            self.conditioning(parameters).unsqueeze(1)
            if self.conditioning is not None
            else None
        )
        surface = volume = None
        surface_context, volume_context = [], []
        if self.surface_embedding is not None:
            surface = self.surface_embedding(self.position(surface_anchors))
            if conditioning is not None:
                surface = surface + conditioning
            surface = self.geometry_attention(surface, surface_anchors, geometry_kv)
            surface_context.append(geometry_kv)
        if self.volume_embedding is not None:
            volume = self.volume_embedding(self.position(volume_anchors))
            if conditioning is not None:
                volume = volume + conditioning
            volume = self.geometry_attention(volume, volume_anchors, geometry_kv)
            volume_context.append(geometry_kv)
        for block, cross in zip(self.shared_blocks, self.cross_branch):
            surface_kv = (
                block._key_value(surface, surface_anchors)
                if surface is not None
                else None
            )
            volume_kv = (
                block._key_value(volume, volume_anchors) if volume is not None else None
            )
            if cross:
                surface_kv, volume_kv = volume_kv, surface_kv
            # Form both contexts before updating either branch (Eq. 8).
            if surface is not None:
                surface_context.append(surface_kv)
                surface = block(surface, surface_anchors, surface_kv)
            if volume is not None:
                volume_context.append(volume_kv)
                volume = block(volume, volume_anchors, volume_kv)
        for block in self.surface_decoder:
            kv = block._key_value(surface, surface_anchors)
            surface_context.append(kv)
            surface = block(surface, surface_anchors, kv)
        for block in self.volume_decoder:
            kv = block._key_value(volume, volume_anchors)
            volume_context.append(kv)
            volume = block(volume, volume_anchors, kv)
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
        features = embedding(self.position(queries))
        if context.conditioning is not None:
            features = features + context.conditioning
        for block, kv in zip(blocks, contexts):
            features = block(features, queries, kv)
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
            and precision. Gradients propagate to the model and context.
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
        derivatives = []
        for axis in range(3):
            offset = queries.new_zeros(3)
            offset[axis] = step
            plus = self.decode_queries(queries + offset, context, field="volume")
            minus = self.decode_queries(queries - offset, context, field="volume")
            derivative = (
                plus[..., list(velocity_channels)] - minus[..., list(velocity_channels)]
            ) / (2 * step)
            derivatives.append(
                derivative * queries.new_tensor(velocity_scale) * coordinate_scale[axis]
            )
        dx, dy, dz = derivatives
        return torch.stack(
            (dy[..., 2] - dz[..., 1], dz[..., 0] - dx[..., 2], dx[..., 1] - dy[..., 0]),
            dim=-1,
        )

    def forward(
        self,
        geometry: Float[Tensor, "B Ng 3"],
        supernodes: Float[Tensor, "B S 3"],
        surface_anchors: Float[Tensor, "B Ms 3"] | None = None,
        volume_anchors: Float[Tensor, "B Mv 3"] | None = None,
        surface_queries: Float[Tensor, "B Qs 3"] | None = None,
        volume_queries: Float[Tensor, "B Qv 3"] | None = None,
        parameters: Float[Tensor, "B P"] | None = None,
    ) -> tuple[Float[Tensor, "B Qs Cs"] | None, Float[Tensor, "B Qv Cv"] | None]:
        r"""Encode geometry/anchors once and predict at anchors or queries."""
        if not torch.compiler.is_compiling():
            self._validate_points(geometry, "geometry")
            for name, queries, channels in (
                ("surface_queries", surface_queries, self.surface_channels),
                ("volume_queries", volume_queries, self.volume_channels),
            ):
                if queries is not None:
                    if not channels:
                        raise ValueError(f"{name} supplied for a disabled output head")
                    self._validate_points(queries, name, geometry, empty=True)
        context = self.encode(
            geometry, supernodes, surface_anchors, volume_anchors, parameters
        )
        surface = (
            context.surface_predictions
            if surface_queries is None
            else self.decode_queries(surface_queries, context, "surface")
        )
        volume = (
            context.volume_predictions
            if volume_queries is None
            else self.decode_queries(volume_queries, context, "volume")
        )
        return surface, volume
