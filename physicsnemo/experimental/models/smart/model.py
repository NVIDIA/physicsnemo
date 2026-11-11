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

r"""Independent implementation of the SMART paper, arXiv:2601.18707v2.

Geometry encoding and field decoding can be used independently.
No research implementation is used as a dependency.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F

from physicsnemo import ModelMetaData, Module
from physicsnemo.nn import FCLayer, Mlp


class _ModulatedPosition(nn.Module):
    r"""Embed each scalar coordinate independently, then project concatenated axes."""

    def __init__(self, width: int):
        super().__init__()
        # Even axis widths accommodate the paper's d=128 and d=256 settings.
        axis_width = 2 * math.ceil(width / 6)
        self.modulation = Mlp(1, axis_width, 2 * axis_width)
        nn.init.zeros_(self.modulation.layers[-1].weight)
        nn.init.zeros_(self.modulation.layers[-1].bias)
        exponents = 2 * torch.div(torch.arange(axis_width), 2, rounding_mode="floor")
        self.register_buffer("frequencies", 10000.0 ** (-exponents / axis_width))
        self.register_buffer("even", torch.arange(axis_width) % 2 == 0)
        self.project = FCLayer(3 * axis_width, width)

    def forward(self, coordinates: Tensor) -> Tensor:
        # Keep phases in fp32 under autocast; trigonometry amplifies fp16 rounding.
        with torch.autocast(device_type=coordinates.device.type, enabled=False):
            values = coordinates.to(self.frequencies.dtype).unsqueeze(-1)
            scale, shift = self.modulation(values).chunk(2, dim=-1)
            phase = values * self.frequencies * (1 + scale) + shift
            embedded = torch.where(self.even, phase.sin(), phase.cos())
        # Preserve fp32 residual streams under autocast. Repeated bf16 residual
        # additions can round away small learned updates in deep decoders.
        return self.project(embedded.flatten(-2)).to(embedded.dtype)


class _CrossAttention(nn.Module):
    r"""Pre-normalized residual cross-attention without query-to-query mixing."""

    def __init__(
        self,
        width: int,
        heads: int,
        qk_normalization: bool = False,
        qk_temperature_max: float | None = None,
    ):
        super().__init__()
        self.qk_normalization = qk_normalization
        self.qk_temperature_max = qk_temperature_max
        if qk_temperature_max is not None:
            if (
                not qk_normalization
                or not math.isfinite(qk_temperature_max)
                or qk_temperature_max <= 1
            ):
                raise ValueError(
                    "qk_temperature_max requires Q/K normalization and a finite value > 1"
                )
            # max * sigmoid(logit) starts at exactly unit scale while remaining
            # bounded. This restores learnable sharpness without unbounded Q/K.
            self.temperature_logit = nn.Parameter(
                torch.full((heads,), -math.log(qk_temperature_max - 1))
            )
        else:
            self.register_parameter("temperature_logit", None)
        self.query_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        # Retain MultiheadAttention projection names for checkpoint compatibility.
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)

    def forward(self, queries: Tensor, context: Tensor) -> Tensor:
        normalized = self.context_norm(context)
        query = self.query_norm(queries)
        update = self._projected_attention(query, normalized)
        return queries + update

    def _projected_attention(self, queries: Tensor, context: Tensor) -> Tensor:
        r"""Reuse projections with optional Q/K normalization.

        Reuse the MultiheadAttention projections and checkpoint parameter names.
        Normalization is in float32. An optional bounded per-head temperature
        restores learnable attention sharpness. Both normalization and temperature
        are explicit extensions to the paper and are opt-in architecture choices.
        """
        attention = self.attention
        batch, count, width = queries.shape
        heads = attention.num_heads
        head_dim = width // heads
        weight, bias = attention.in_proj_weight, attention.in_proj_bias
        query = F.linear(queries, weight[:width], bias[:width])
        key, value = F.linear(context, weight[width:], bias[width:]).chunk(2, dim=-1)
        query = query.reshape(batch, count, heads, head_dim).transpose(1, 2)
        key = key.reshape(batch, -1, heads, head_dim).transpose(1, 2)
        value = value.reshape(batch, -1, heads, head_dim).transpose(1, 2)
        if self.qk_normalization:
            with torch.autocast(queries.device.type, enabled=False):
                query = F.rms_norm(query.float(), (head_dim,), eps=1e-6).to(value.dtype)
                key = F.rms_norm(key.float(), (head_dim,), eps=1e-6).to(value.dtype)
        if self.temperature_logit is not None:
            temperature = self.qk_temperature_max * self.temperature_logit.sigmoid()
            query = query * temperature.to(query.dtype).view(1, heads, 1, 1)
        result = F.scaled_dot_product_attention(query, key, value)
        result = result.transpose(1, 2).reshape(batch, count, width)
        return attention.out_proj(result)


class _Refinement(nn.Module):
    r"""Latent/query refinement with simulation modulation inside the MLP."""

    def __init__(
        self,
        width: int,
        heads: int,
        expansion: int,
        parameters: int,
        qk_normalization: bool = False,
        qk_temperature_max: float | None = None,
    ):
        super().__init__()
        self.cross_attention = _CrossAttention(
            width,
            heads,
            qk_normalization,
            qk_temperature_max,
        )
        self.norm = nn.LayerNorm(width)
        self.hidden = FCLayer(width, width * expansion, activation_fn=nn.GELU())
        self.output = FCLayer(width * expansion, width)
        self.modulation = (
            Mlp(parameters, width, 2 * width * expansion) if parameters else None
        )
        if self.modulation is not None:
            nn.init.zeros_(self.modulation.layers[-1].weight)
            nn.init.zeros_(self.modulation.layers[-1].bias)

    def forward(
        self,
        queries: Tensor,
        context: Tensor,
        parameters: Tensor | None,
    ) -> Tensor:
        queries = self.cross_attention(queries, context)
        hidden = self.hidden(self.norm(queries))
        if self.modulation is not None:
            scale, shift = self.modulation(parameters).unsqueeze(1).chunk(2, dim=-1)
            hidden = hidden * (1 + scale) + shift
        return queries + self.output(hidden)


class SMART(Module):
    r"""Mesh-free steady-field surrogate derived from the SMART paper.

    Encodes geometry once into one latent set per depth and decodes each spatial
    query independently. The surface and volume branches share the decoder and
    use separate output heads. See `SMART <https://arxiv.org/abs/2601.18707>`_.

    Parameters
    ----------
    surface_channels : int, optional, default=1
        Surface outputs. Zero disables this head.
    volume_channels : int, optional, default=3
        Volume outputs. Zero disables this head.
    parameter_channels : int, optional, default=0
        Number of scalar simulation conditions per geometry.
    hidden_dim : int, optional, default=256
        Latent feature width, divisible by ``num_heads``.
    num_heads : int, optional, default=8
        Attention heads. An implementation choice not specified in the paper.
    num_layers : int, optional, default=6
        Matched encoder and decoder depth.
    num_latents : int, optional, default=2048
        Maximum coarse geometry size. Small inputs use all available points.
    geometry_subsample : int, optional, default=65536
        Maximum geometry points attended to at each depth.
    mlp_ratio : int, optional, default=4
        Hidden expansion of the modulated MLPs.
    share_weights : bool, optional, default=True
        Share each encoder refinement with its corresponding decoder refinement.
    eval_seed : int, optional, default=0
        Fixed geometry sampling seed during evaluation. Training uses the active
        RNG or the generator supplied to :meth:`encode_geometry`.
    qk_normalization : bool, optional, default=False
        Normalize projected queries and keys by their per-head RMS in float32,
        with unit gain unless ``qk_temperature_max`` enables a bounded learned
        gain. Bounds attention logits against weight growth.
        This changes the attention function; train a new model when enabling it.
        Disabled preserves the original paper-derived implementation.
    qk_temperature_max : float or None, optional, default=None
        Optional upper bound (> 1) on a learned per-head multiplier of normalized
        attention logits, initialized at 1. Requires ``qk_normalization=True``.
        This bounded-temperature extension is not specified by the paper.
    surface_query_feature_channels, volume_query_feature_channels : int, optional, default=0
        Optional geometric feature dimensions, projected into each query's initial
        embedding. For CFD-derived features use 3 surface-normal components and
        4 volume components (surface distance and nearest-surface direction).
        This feature extension is not specified by the paper. Disable unused
        output heads by setting their output channel count to zero; features for
        disabled heads must also be zero. No unused branch parameters are created.

    Forward
    -------
    geometry : torch.Tensor
        Geometry-only point cloud of shape :math:`(B,N,3)`, for example CFD
        surface mesh coordinates. Keep field queries separate from this cloud.
    surface_queries : torch.Tensor or None, optional, default=None
        Surface coordinates of shape :math:`(B,M_s,3)`.
    volume_queries : torch.Tensor or None, optional, default=None
        Volume coordinates of shape :math:`(B,M_v,3)`.
    parameters : torch.Tensor or None, optional, default=None
        Simulation conditions of shape :math:`(B,P)`; required if :math:`P>0`.
    geometry_generator : torch.Generator or None, optional, default=None
        Sampling RNG on the geometry device. Overrides ``eval_seed`` and keeps
        geometry sampling independent of other random operations.
    surface_query_features, volume_query_features : torch.Tensor or None, optional, default=None
        Aligned geometric features of shape :math:`(B,M,C)`, required when the
        corresponding feature-channel count is nonzero. Projections start at zero
        so features initially leave the coordinate-only predictions unchanged.
        Their physical scaling is a preprocessing contract; the model does not
        normalize them.

    Outputs
    -------
    tuple[torch.Tensor or None, torch.Tensor or None]
        Surface and volume predictions of shape :math:`(B,M_s,C_s)` and
        :math:`(B,M_v,C_v)`. An omitted query set produces ``None``.

    Notes
    -----
    Mixed precision follows the caller's :func:`torch.autocast` context.
    Coordinates must be finite. Apply the same coordinate normalization to geometry
    and field queries before calling the model. The positional embedding does not
    normalize coordinates internally. Padding is not supported. Use fixed-size
    sampled batches or batch size one for ragged clouds. Sampling is without
    replacement. Evaluation is repeatable for a fixed ordered cloud.
    Random-key sampling selects different subsets from earlier randperm-based
    versions for the same seed. A permutation can change the selected subset.
    Permutation invariance holds when all geometry points are retained, up to
    floating-point rounding.

    :meth:`decode_queries` can be compiled independently and evaluated in chunks
    against cached latents. Cache validity depends on geometry, parameters,
    weights, dtype, and device. Re-encode after any of these change.

    Examples
    --------
    >>> model = SMART(hidden_dim=24, num_heads=4, num_layers=2,
    ...               num_latents=8, geometry_subsample=16).eval()
    >>> geometry = torch.randn(2, 32, 3)
    >>> pressure, velocity = model(geometry, torch.randn(2, 12, 3),
    ...                            torch.randn(2, 20, 3))
    >>> pressure.shape, velocity.shape
    (torch.Size([2, 12, 1]), torch.Size([2, 20, 3]))
    """

    __model_checkpoint_version__ = "0.2.0"
    __supported_model_checkpoint_version__ = {
        "0.1.0": "Loading SMART 0.1.0 with caller-controlled autocast."
    }

    @classmethod
    def _backward_compat_arg_mapper(cls, version: str, args: dict) -> dict:
        """Load earlier checkpoints only when they used caller-controlled precision."""
        args = super()._backward_compat_arg_mapper(version, args).copy()
        if args.pop("attention_precision", None) is not None:
            raise ValueError(
                "Checkpoints with an attention_precision override require the "
                "earlier SMART implementation. New models use caller-controlled autocast."
            )
        args.pop("attention_precision_scope", None)
        return args

    def __init__(
        self,
        surface_channels: int = 1,
        volume_channels: int = 3,
        parameter_channels: int = 0,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        num_latents: int = 2048,
        geometry_subsample: int = 65536,
        mlp_ratio: int = 4,
        share_weights: bool = True,
        eval_seed: int = 0,
        qk_normalization: bool = False,
        qk_temperature_max: float | None = None,
        surface_query_feature_channels: int = 0,
        volume_query_feature_channels: int = 0,
    ):
        super().__init__(meta=ModelMetaData(amp=True, auto_grad=True))
        for name, value in (
            ("hidden_dim", hidden_dim),
            ("num_heads", num_heads),
            ("num_layers", num_layers),
            ("num_latents", num_latents),
            ("geometry_subsample", geometry_subsample),
            ("mlp_ratio", mlp_ratio),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if (
            any(
                isinstance(c, bool) or not isinstance(c, int) or c < 0
                for c in (surface_channels, volume_channels, parameter_channels)
            )
            or surface_channels + volume_channels == 0
        ):
            raise ValueError(
                "Channel counts must be nonnegative with at least one output"
            )
        for field, channels, outputs in (
            ("surface", surface_query_feature_channels, surface_channels),
            ("volume", volume_query_feature_channels, volume_channels),
        ):
            if (
                isinstance(channels, bool)
                or not isinstance(channels, int)
                or channels < 0
            ):
                raise ValueError(
                    f"{field}_query_feature_channels must be a nonnegative integer"
                )
            if channels and not outputs:
                raise ValueError(f"Cannot configure features for disabled {field} head")
        if isinstance(eval_seed, bool) or not isinstance(eval_seed, int):
            raise ValueError("eval_seed must be an integer")
        if not -(2**63) <= eval_seed < 2**64:
            raise ValueError("eval_seed must be between -2**63 and 2**64 - 1")
        self.surface_query_feature_channels = surface_query_feature_channels
        self.volume_query_feature_channels = volume_query_feature_channels
        self.qk_temperature_max = qk_temperature_max
        self.surface_channels = surface_channels
        self.volume_channels = volume_channels
        self.parameter_channels = parameter_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_latents = num_latents
        self.geometry_subsample = geometry_subsample
        self.mlp_ratio = mlp_ratio
        self.share_weights = share_weights
        self.eval_seed = eval_seed
        self.qk_normalization = qk_normalization
        self.position = _ModulatedPosition(hidden_dim)
        self.geometry_attention = nn.ModuleList(
            _CrossAttention(
                hidden_dim,
                num_heads,
                qk_normalization,
                qk_temperature_max,
            )
            for _ in range(num_layers)
        )
        self.decoder = nn.ModuleList(
            _Refinement(
                hidden_dim,
                num_heads,
                mlp_ratio,
                parameter_channels,
                qk_normalization,
                qk_temperature_max,
            )
            for _ in range(num_layers)
        )
        # E^(L) is unused: the decoder consumes E_G^(L), so omit a dead final
        # encoder refinement. Shared layers remain used by the decoder.
        self.encoder_refinement = nn.ModuleList(
            self.decoder[i]
            if share_weights
            else _Refinement(
                hidden_dim,
                num_heads,
                mlp_ratio,
                parameter_channels,
                qk_normalization,
                qk_temperature_max,
            )
            for i in range(num_layers - 1)
        )
        self.surface_head = (
            Mlp(hidden_dim, hidden_dim, surface_channels) if surface_channels else None
        )
        self.volume_head = (
            Mlp(hidden_dim, hidden_dim, volume_channels) if volume_channels else None
        )
        # Geometry-only query features are explicit extensions to SMART. They
        # must never contain target CFD fields. Absent modules preserve legacy
        # state dictionaries and the default model's parameter count exactly.
        self.surface_query_projection = (
            nn.Linear(surface_query_feature_channels, hidden_dim, bias=False)
            if surface_query_feature_channels
            else None
        )
        self.volume_query_projection = (
            nn.Linear(volume_query_feature_channels, hidden_dim, bias=False)
            if volume_query_feature_channels
            else None
        )
        # Zero starts make feature and coordinate-only arms identical at step 0;
        # the feature projection can learn from the first backward pass. These
        # modules are built last to preserve shared initialization for a fixed seed.
        for projection in (self.surface_query_projection, self.volume_query_projection):
            if projection is not None:
                nn.init.zeros_(projection.weight)

    def _validate_parameters(
        self, parameters: Tensor | None, batch: int, device: torch.device
    ) -> None:
        if self.parameter_channels:
            if parameters is None or parameters.shape != (
                batch,
                self.parameter_channels,
            ):
                shape = None if parameters is None else tuple(parameters.shape)
                raise ValueError(
                    f"Expected tensor of shape (B, {self.parameter_channels}) but got tensor of shape {shape}"
                )
            if not parameters.is_floating_point() or parameters.device != device:
                raise ValueError(
                    "parameters must have a floating-point dtype on the input device"
                )
        elif parameters is not None:
            raise ValueError("parameters must be None when parameter_channels is zero")

    @staticmethod
    def _validate_coordinates(coordinates: Tensor, batch: int | None = None) -> None:
        if (
            coordinates.ndim != 3
            or coordinates.shape[-1] != 3
            or (batch is not None and coordinates.shape[0] != batch)
        ):
            raise ValueError(
                f"Expected tensor of shape (B, N, 3) but got tensor of shape {tuple(coordinates.shape)}"
            )
        if not coordinates.is_floating_point():
            raise ValueError("Coordinates must have a floating-point dtype")

    @staticmethod
    def _sample(
        points: Tensor, count: int, generator: torch.Generator | None
    ) -> Tensor:
        count = min(count, points.shape[1])
        if count == points.shape[1]:
            return points
        # Select random keys without constructing a full permutation per cloud.
        # Keep keys in fp32 even when the default dtype is a reduced precision.
        indices = torch.rand(
            points.shape[:2],
            device=points.device,
            dtype=torch.float32,
            generator=generator,
        ).topk(count, dim=1, sorted=False).indices
        return points.gather(1, indices.unsqueeze(-1).expand(-1, -1, 3))

    def encode_geometry(
        self,
        geometry: Float[Tensor, "batch geometry_points 3"],
        parameters: Float[Tensor, "batch parameter_channels"] | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[Float[Tensor, "batch latents hidden_dim"], ...]:
        r"""Build intermediate geometry latents without observing spatial queries.

        Parameters
        ----------
        geometry : torch.Tensor
            Point cloud of shape :math:`(B,N,3)`, with :math:`N>0`.
        parameters : torch.Tensor or None, optional, default=None
            Conditions of shape :math:`(B,P)`.
        generator : torch.Generator or None, optional, default=None
            Sampling RNG on the geometry device. Overrides ``eval_seed``.

        Returns
        -------
        tuple[torch.Tensor, ...]
            One :math:`(B,\min(N,K),D)` context per layer. Gradients are retained.
        """
        if not torch.compiler.is_compiling():
            self._validate_coordinates(geometry)
            if geometry.shape[0] == 0 or geometry.shape[1] == 0:
                raise ValueError("Geometry batch and point count must be nonempty")
            self._validate_parameters(parameters, geometry.shape[0], geometry.device)
        if generator is None and not self.training:
            generator = torch.Generator(device=geometry.device).manual_seed(
                self.eval_seed
            )
        state = self.position(self._sample(geometry, self.num_latents, generator))
        contexts = []
        for layer, attention in enumerate(self.geometry_attention):
            # Embed only selected geometry points, avoiding O(N*D) dense storage.
            sampled = self.position(
                self._sample(geometry, self.geometry_subsample, generator)
            )
            context = attention(state, sampled)
            contexts.append(context)
            if layer < self.num_layers - 1:
                state = self.encoder_refinement[layer](
                    state,
                    context,
                    parameters,
                )
        return tuple(contexts)

    def decode_queries(
        self,
        queries: Float[Tensor, "batch query_points 3"],
        contexts: tuple[Float[Tensor, "batch latents hidden_dim"], ...],
        parameters: Float[Tensor, "batch parameter_channels"] | None = None,
        *,
        field: Literal["surface", "volume"] = "surface",
        query_features: Float[Tensor, "batch query_points feature_channels"]
        | None = None,
    ) -> Float[Tensor, "batch query_points output_channels"]:
        r"""Evaluate independent queries against cached geometry contexts.

        Parameters
        ----------
        queries : torch.Tensor
            Coordinates of shape :math:`(B,M,3)`; :math:`M=0` is supported.
        contexts : tuple[torch.Tensor, ...]
            The ``num_layers`` contexts returned by :meth:`encode_geometry`.
        parameters : torch.Tensor or None, optional, default=None
            The same :math:`(B,P)` conditions used to encode the geometry.
        field : {"surface", "volume"}, optional, default="surface"
            Output head to use.
        query_features : torch.Tensor or None, optional, default=None
            Aligned geometric features of shape :math:`(B,M,C)`. When evaluating
            chunks, slice these with the same indices as ``queries``.

        Returns
        -------
        torch.Tensor
            Predictions of shape :math:`(B,M,C)` for the selected head.
        """
        if field not in ("surface", "volume"):
            raise ValueError("field must be 'surface' or 'volume'")
        head = self.surface_head if field == "surface" else self.volume_head
        if head is None:
            raise ValueError(f"The {field} head is disabled")
        if not torch.compiler.is_compiling():
            self._validate_coordinates(queries)
            self._validate_parameters(parameters, queries.shape[0], queries.device)
            if len(contexts) != self.num_layers:
                raise ValueError(f"Expected {self.num_layers} context tensors")
            for context in contexts:
                if (
                    context.ndim != 3
                    or context.shape[0] != queries.shape[0]
                    or context.shape[1] == 0
                    or context.shape[-1] != self.hidden_dim
                ):
                    raise ValueError(
                        f"Expected tensor of shape (B, K, {self.hidden_dim}) but got tensor of shape {tuple(context.shape)}"
                    )
                if not context.is_floating_point() or context.device != queries.device:
                    raise ValueError(
                        "Contexts must have a floating-point dtype on the query device"
                    )
        projection = getattr(self, f"{field}_query_projection")
        feature_channels = getattr(self, f"{field}_query_feature_channels")
        if not torch.compiler.is_compiling():
            if projection is None and query_features is not None:
                raise ValueError(f"The {field} query features are disabled")
            if projection is not None:
                actual_shape = (
                    None if query_features is None else tuple(query_features.shape)
                )
                if actual_shape != (*queries.shape[:2], feature_channels):
                    raise ValueError(
                        f"Expected tensor of shape (B, M, {feature_channels}) "
                        f"but got tensor of shape {actual_shape} for {field} query_features"
                    )
                if (
                    not query_features.is_floating_point()
                    or query_features.device != queries.device
                ):
                    raise ValueError(
                        f"The {field} query_features must have a floating-point dtype "
                        "on the query device"
                    )
        state = self.position(queries)
        if projection is not None:
            state = state + projection(query_features.to(state.dtype)).to(state.dtype)
        for refinement, context in zip(self.decoder, contexts):
            state = refinement(state, context, parameters)
        return head(state)

    def forward(
        self,
        geometry: Float[Tensor, "batch geometry_points 3"],
        surface_queries: Float[Tensor, "batch surface_points 3"] | None = None,
        volume_queries: Float[Tensor, "batch volume_points 3"] | None = None,
        parameters: Float[Tensor, "batch parameter_channels"] | None = None,
        *,
        geometry_generator: torch.Generator | None = None,
        surface_query_features: Float[
            Tensor, "batch surface_points surface_query_feature_channels"
        ]
        | None = None,
        volume_query_features: Float[
            Tensor, "batch volume_points volume_query_feature_channels"
        ]
        | None = None,
    ) -> tuple[
        Float[Tensor, "batch surface_points surface_channels"] | None,
        Float[Tensor, "batch volume_points volume_channels"] | None,
    ]:
        r"""Encode once; an explicit geometry RNG leaves other randomness untouched."""
        if surface_queries is None and volume_queries is None:
            raise ValueError("At least one query set is required")
        if surface_queries is None and surface_query_features is not None:
            raise ValueError("surface_query_features requires surface_queries")
        if volume_queries is None and volume_query_features is not None:
            raise ValueError("volume_query_features requires volume_queries")
        if not torch.compiler.is_compiling():
            self._validate_coordinates(geometry)
            for field, queries in (
                ("surface", surface_queries),
                ("volume", volume_queries),
            ):
                if queries is not None:
                    self._validate_coordinates(queries, geometry.shape[0])
                    if queries.device != geometry.device:
                        raise ValueError("Queries must be on the geometry device")
                    if not getattr(self, f"{field}_channels"):
                        raise ValueError(f"The {field} head is disabled")
        contexts = self.encode_geometry(
            geometry, parameters, generator=geometry_generator
        )
        surface = (
            None
            if surface_queries is None
            else self.decode_queries(
                surface_queries,
                contexts,
                parameters,
                field="surface",
                query_features=surface_query_features,
            )
        )
        volume = (
            None
            if volume_queries is None
            else self.decode_queries(
                volume_queries,
                contexts,
                parameters,
                field="volume",
                query_features=volume_query_features,
            )
        )
        return surface, volume
