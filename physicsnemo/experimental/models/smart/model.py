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
    query independently. Each instance predicts either surface or volume fields.
    The selected output head determines the field channels. See `SMART <https://arxiv.org/abs/2601.18707>`_.

    Parameters
    ----------
    field : {"surface", "volume"}, optional, default="surface"
        Field predicted by this model. Only this output head is constructed.
    out_dim : int, optional, default=1
        Number of output channels.
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
    query_feature_channels : int, optional, default=0
        Aligned geometric feature channels. For example, surface normals or
        volume distance/direction features. Their projection starts at zero.
        This feature extension is not specified by the paper.
    eval_chunk_size : int, optional, default=8192
        Maximum number of queries decoded at once during evaluation.
    coordinate_scale : tuple of three positive floats, optional, default=(1, 1, 1)
        Axis scales applied to both point clouds after subtracting ``frame_center``.
    center_on_geometry : bool, optional, default=False
        When ``frame_center`` is omitted, center both clouds on the geometry's
        bounding-box midpoint before scaling.

    Forward
    -------
    geometry, queries : torch.Tensor
        Separate geometry and query clouds of shape :math:`(B,N,3)` and :math:`(B,M,3)`.
    parameters : torch.Tensor or None, optional, default=None
        Conditions of shape :math:`(B,P)` or :math:`(B,1,P)`.
    query_features : torch.Tensor or None, optional, default=None
        Aligned features of shape :math:`(B,M,C)` in the caller's chosen units.
    frame_center : torch.Tensor or None, optional, default=None
        Origin of shape :math:`(B,3)` or :math:`(B,1,3)`. Overrides
        ``center_on_geometry``. When omitted, use the geometry's bounding-box
        midpoint if ``center_on_geometry=True``, otherwise zero.

    Outputs
    -------
    torch.Tensor
        Field predictions of shape :math:`(B,M,\mathrm{out\_dim})`.

    Notes
    -----
    Mixed precision follows the caller's :func:`torch.autocast` context.
    Coordinates must be finite. ``forward`` normalizes both clouds together.
    Low-level ``encode_geometry`` and ``decode_queries`` expect coordinates
    already in the model frame. Padding is not supported. Use fixed-size
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
    >>> pressure = model(geometry, torch.randn(2, 12, 3))
    >>> pressure.shape
    torch.Size([2, 12, 1])
    """

    __model_checkpoint_version__ = "0.3.0"

    def __init__(
        self,
        field: str = "surface",
        out_dim: int = 1,
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
        query_feature_channels: int = 0,
        eval_chunk_size: int = 8192,
        coordinate_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        center_on_geometry: bool = False,
    ):
        super().__init__(meta=ModelMetaData(amp=True, auto_grad=True))
        if field not in ("surface", "volume"):
            raise ValueError("field must be surface or volume")
        for name, value in (("out_dim", out_dim), ("eval_chunk_size", eval_chunk_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if len(coordinate_scale) != 3 or any(
            isinstance(x, bool) or not math.isfinite(x) or x <= 0
            for x in coordinate_scale
        ):
            raise ValueError(
                "coordinate_scale must contain three positive finite values"
            )
        self.field = field
        self.out_dim = out_dim
        self.query_feature_channels = query_feature_channels
        self.eval_chunk_size = eval_chunk_size
        self.coordinate_scale = tuple(coordinate_scale)
        self.center_on_geometry = center_on_geometry
        self._args["__args__"]["coordinate_scale"] = self.coordinate_scale
        surface_channels = out_dim if field == "surface" else 0
        volume_channels = out_dim if field == "volume" else 0
        surface_query_feature_channels = (
            query_feature_channels if field == "surface" else 0
        )
        volume_query_feature_channels = (
            query_feature_channels if field == "volume" else 0
        )
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
        for name, channels in (
            ("parameter_channels", parameter_channels),
            ("query_feature_channels", query_feature_channels),
        ):
            if (
                isinstance(channels, bool)
                or not isinstance(channels, int)
                or channels < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer")
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
        # must never contain target CFD fields. Only the selected field creates
        # an output head and optional feature projection.
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
        indices = (
            torch.rand(
                points.shape[:2],
                device=points.device,
                dtype=torch.float32,
                generator=generator,
            )
            .topk(count, dim=1, sorted=False)
            .indices
        )
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
        query_features : torch.Tensor or None, optional, default=None
            Aligned geometric features of shape :math:`(B,M,C)`. When evaluating
            chunks, slice these with the same indices as ``queries``.

        Returns
        -------
        torch.Tensor
            Predictions of shape :math:`(B,M,C)` for the selected head.
        """
        field = self.field
        head = self.surface_head if field == "surface" else self.volume_head
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
        queries: Float[Tensor, "batch query_points 3"],
        parameters: Float[Tensor, "batch ... parameter_channels"] | None = None,
        *,
        query_features: Float[Tensor, "batch query_points feature_channels"]
        | None = None,
        frame_center: Float[Tensor, "..."] | None = None,
    ) -> Float[Tensor, "batch query_points out_dim"]:
        r"""Encode geometry once and predict the configured field at query points.

        Training samples geometry using the active PyTorch RNG. Evaluation uses
        ``eval_seed`` to keep geometry sampling fixed across query chunks.

        Parameters
        ----------
        geometry, queries : torch.Tensor
            Separate point clouds of shape :math:`(B,N,3)` and :math:`(B,M,3)`.
        parameters : torch.Tensor or None, optional, default=None
            Conditions of shape :math:`(B,P)` or :math:`(B,1,P)`.
        query_features : torch.Tensor or None, optional, default=None
            Aligned features of shape :math:`(B,M,C)`.
        frame_center : torch.Tensor or None, optional, default=None
            Coordinate origin of shape :math:`(B,3)` or :math:`(B,1,3)`.
            A three-element tensor broadcasts one origin across the batch.
            Overrides ``center_on_geometry``. When omitted, use the geometry's
            bounding-box midpoint if ``center_on_geometry=True``, otherwise zero.
            Both clouds are divided by ``coordinate_scale``.

        Returns
        -------
        torch.Tensor
            Predictions of shape :math:`(B,M,\mathrm{out\_dim})`. Evaluation
            decodes at most ``eval_chunk_size`` queries at once.
        """
        if not torch.compiler.is_compiling():
            self._validate_coordinates(geometry)
        batch = geometry.shape[0]
        if not torch.compiler.is_compiling():
            self._validate_coordinates(queries, batch)
            if self.query_feature_channels:
                if (
                    query_features is None
                    or query_features.shape
                    != (*queries.shape[:2], self.query_feature_channels)
                    or not query_features.is_floating_point()
                    or query_features.device != queries.device
                ):
                    actual_shape = (
                        None if query_features is None else tuple(query_features.shape)
                    )
                    raise ValueError(
                        f"Expected tensor of shape (B, M, {self.query_feature_channels}) "
                        f"but got tensor of shape {actual_shape} for query_features. "
                        f"Expected floating-point values on {queries.device}"
                    )
            elif query_features is not None:
                raise ValueError("query_features are disabled")
            if queries.device != geometry.device:
                raise ValueError("Queries must be on the geometry device")
            if frame_center is not None and (
                frame_center.shape
                not in ((3,), (1, 3), (1, 1, 3), (batch, 3), (batch, 1, 3))
                or not frame_center.is_floating_point()
                or frame_center.device != geometry.device
                or not torch.isfinite(frame_center).all()
            ):
                raise ValueError(
                    "Expected tensor of shape (B, 3) or (B, 1, 3), including B=1, or (3,) "
                    f"but got tensor of shape {tuple(frame_center.shape)} for frame_center. "
                    f"Expected finite floating-point coordinates on {geometry.device}"
                )
            if parameters is not None and parameters.shape not in (
                (batch, self.parameter_channels),
                (batch, 1, self.parameter_channels),
            ):
                raise ValueError(
                    f"Expected tensor of shape (B, {self.parameter_channels}) or "
                    f"(B, 1, {self.parameter_channels}) but got tensor of shape "
                    f"{tuple(parameters.shape)} for parameters"
                )
        if parameters is not None:
            parameters = parameters.reshape(batch, self.parameter_channels)
        if not torch.compiler.is_compiling():
            self._validate_parameters(parameters, batch, geometry.device)
        if frame_center is None and self.center_on_geometry:
            frame_center = (geometry.amin(dim=1) + geometry.amax(dim=1)) * 0.5
        if frame_center is not None:
            center = frame_center.reshape(-1, 1, 3)
            geometry = geometry - center
            queries = queries - center
        scale = geometry.new_tensor(self.coordinate_scale)
        geometry = geometry / scale
        queries = queries / scale
        contexts = self.encode_geometry(geometry, parameters)
        if self.training or queries.shape[1] == 0:
            return self.decode_queries(
                queries, contexts, parameters, query_features=query_features
            )
        return torch.cat(
            [
                self.decode_queries(
                    queries[:, start : start + self.eval_chunk_size],
                    contexts,
                    parameters,
                    query_features=query_features[
                        :, start : start + self.eval_chunk_size
                    ]
                    if query_features is not None
                    else None,
                )
                for start in range(0, queries.shape[1], self.eval_chunk_size)
            ],
            dim=1,
        )
