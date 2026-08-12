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

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float
from torch import Tensor

from physicsnemo.core.module import Module
from physicsnemo.experimental.models.geotransolver.context_projector import (
    ContextProjector,
)
from physicsnemo.nn.module.embedding_layers import PositionalEmbedding

# Numerical floor added to a standard deviation before dividing by it.
_STD_EPS: float = 1e-8
# ln(2*pi), reused by the Gaussian-mixture negative log-likelihoods.
_LOG_2PI: float = math.log(2.0 * math.pi)
# Default soft-clamp bounds and sharpness for log_sigma in every GMMHead; both
# are overridable per run via the GMMHead / ParticleGeoTransolver arguments.
_LOG_SIGMA_CLAMP: tuple[float, float] = (-10.0, 2.0)
_LOG_SIGMA_SMOOTH_BETA: float = 1.0


def _make_mlp(
    in_dim: int, out_dim: int, hidden_dim: int | None = None
) -> nn.Sequential:
    """Two-layer GELU MLP. ``hidden_dim`` defaults to ``max(in_dim, out_dim)``."""
    if hidden_dim is None:
        hidden_dim = max(in_dim, out_dim)
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


def fourier_features_xyz(
    xyz: Float[Tensor, "*B 3"],
    num_freqs: int,
    base: float = 2.0,
) -> Float[Tensor, "*B C"]:
    """Lift 3-D coordinates into a richer Fourier feature space.

    Use this to give a network a stronger spatial basis than the raw
    ``(x, y, z)`` coordinates. Each axis is expanded with a bank of
    sine/cosine features at geometrically increasing frequencies, which
    lets downstream layers resolve fine spatial structure they would
    otherwise miss. The raw coordinates are kept alongside the added
    features.

    Parameters
    ----------
    xyz : Tensor of shape ``(..., 3)``
        Coordinates to lift (any leading batch dimensions).
    num_freqs : int
        Number of frequencies per axis. More frequencies resolve finer
        spatial detail at the cost of a wider output.
    base : float
        Geometric ratio between successive frequencies.

    Returns
    -------
    Tensor of shape ``(..., 3 + 2 * num_freqs * 3)``
        The raw coordinates followed by the Fourier features.
    """
    freqs = base ** torch.arange(num_freqs, dtype=xyz.dtype, device=xyz.device)
    angles = math.pi * xyz.unsqueeze(-1) * freqs  # (..., 3, num_freqs)
    sin_part = angles.sin().flatten(start_dim=-2)  # (..., 3 * num_freqs)
    cos_part = angles.cos().flatten(start_dim=-2)
    return torch.cat([xyz, sin_part, cos_part], dim=-1)


class LogConcatTimeEmbedding(Module):
    """Embed a time-related scalar that spans many orders of magnitude.

    Use this for any time-like quantity (the current simulation time, a
    per-particle delay, or the predicted next-event delay) whose values
    range from near zero to very large. A single embedding tends to blur
    the many small values together; this one combines a linear-scale
    embedding (which separates the large, well-populated values) with a
    log-scale embedding (which keeps the crowded near-zero values
    distinguishable), so the model resolves the whole range well.

    Pass the same z-scored value the rest of the model consumes. The
    embedding stores the dataset statistics it needs to also view the
    quantity on a log scale, and those statistics travel with the
    checkpoint so training and inference stay consistent.

    Parameters
    ----------
    num_channels : int
        Total output width. Split evenly across the two scales, so it
        must be even.
    max_positions_lin : int
        Frequency range of the linear-scale embedding.
    max_positions_log : int
        Frequency range of the log-scale embedding.
    prescale_lin : float
        Multiplier applied to the value before the linear-scale
        embedding.
    prescale_log : float
        Multiplier applied to the value before the log-scale embedding.
    delay_mean, delay_std : float
        Normalization statistics of the quantity on its linear scale
        (the ``"delay"`` entry of ``stats.json``).
    log_delay_mean, log_delay_std : float
        Normalization statistics of the quantity on its log scale (the
        ``"log_delay"`` entry of ``stats.json``).
    log_eps : float
        Small floor added inside the log so exact-zero values stay
        finite. Matches the value used when computing the statistics.

    Forward
    -------
    z_lin : Tensor of shape ``(B,)``
        The z-scored time scalar.

    Outputs
    -------
    Tensor of shape ``(B, num_channels)``
        The embedding of the input scalar.

    Examples
    --------
    >>> emb = LogConcatTimeEmbedding(
    ...     num_channels=64, max_positions_lin=1000, max_positions_log=1000,
    ...     prescale_lin=100.0, prescale_log=100.0,
    ...     delay_mean=69.0, delay_std=254.0,
    ...     log_delay_mean=-1.54, log_delay_std=4.65,
    ... )
    >>> z_lin = torch.linspace(-0.27, 5.0, 4)
    >>> emb(z_lin).shape
    torch.Size([4, 64])
    """

    delay_mean: Tensor
    delay_std: Tensor
    log_delay_mean: Tensor
    log_delay_std: Tensor

    def __init__(
        self,
        num_channels: int,
        max_positions_lin: int,
        max_positions_log: int,
        prescale_lin: float,
        prescale_log: float,
        delay_mean: float,
        delay_std: float,
        log_delay_mean: float,
        log_delay_std: float,
        log_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if num_channels % 2 != 0:
            raise ValueError(
                f"num_channels={num_channels} must be even (split evenly across "
                "the linear and log branches)."
            )
        channels_per_branch = num_channels // 2
        self.embed_lin = PositionalEmbedding(
            num_channels=channels_per_branch,
            max_positions=max_positions_lin,
            learnable=True,
        )
        self.embed_log = PositionalEmbedding(
            num_channels=channels_per_branch,
            max_positions=max_positions_log,
            learnable=True,
        )
        self.prescale_lin = float(prescale_lin)
        self.prescale_log = float(prescale_log)
        self.log_eps = float(log_eps)
        # Normalization stats kept as persistent buffers so they travel with
        # the model checkpoint and stay consistent across training and inference.
        self.register_buffer(
            "delay_mean", torch.tensor(float(delay_mean), dtype=torch.float32)
        )
        self.register_buffer(
            "delay_std", torch.tensor(float(delay_std), dtype=torch.float32)
        )
        self.register_buffer(
            "log_delay_mean",
            torch.tensor(float(log_delay_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "log_delay_std",
            torch.tensor(float(log_delay_std), dtype=torch.float32),
        )

    def _to_z_log(self, z_lin: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
        raw = z_lin * self.delay_std + self.delay_mean
        raw = raw.clamp(min=0.0)
        return (torch.log(raw + self.log_eps) - self.log_delay_mean) / (
            self.log_delay_std + _STD_EPS
        )

    def forward(self, z_lin: Float[Tensor, " B"]) -> Float[Tensor, "B C"]:
        z_log = self._to_z_log(z_lin)
        emb_lin = self.embed_lin(z_lin * self.prescale_lin)
        emb_log = self.embed_log(z_log * self.prescale_log)
        return torch.cat([emb_lin, emb_log], dim=-1)


class MaskedFlareAttention(Module):
    """Attention over a per-slot stream that masks absent slots to enforce
    causality.

    Use this where some rows of the per-slot stream are absent (padding,
    or particles that do not exist yet) and must not leak into the
    output. A boolean ``kv_mask`` selects the slots that may be attended
    to, enforcing causality: every output depends only on the real
    slots, never on the masked ones. Inputs where every slot is masked
    are handled gracefully, so callers never have to special-case them.

    The block can optionally fold in a cross-attention path against a
    persistent context (a global summary built elsewhere). Enable it by
    constructing with ``context_dim > 0`` and passing a ``context`` to
    :meth:`forward`.

    Parameters
    ----------
    dim : int
        Per-token hidden width of the input/output stream.
    num_heads : int
        Number of attention heads. ``dim`` must be divisible by this.
    num_flare_global_queries : int
        Capacity of the attention bottleneck. Larger values let the
        block carry more information per token, at the cost of more
        parameters.
    context_dim : int
        Channel width of the optional cross-attention context. Set to 0
        to run self-attention only; the block then rejects a non-None
        ``context``.

    Forward
    -------
    x : Tensor of shape ``(B, P, dim)``
        Per-slot input stream.
    context : Tensor of shape ``(B, num_heads, S, context_dim)`` or None
        Persistent cross-attention context, or ``None`` for
        self-attention only. Must be ``None`` iff ``context_dim == 0``.
    kv_mask : Tensor of shape ``(B, P)`` (bool)
        ``True`` for slots that may be attended to, ``False`` for slots
        to ignore.

    Outputs
    -------
    Tensor of shape ``(B, P, dim)``
        Updated per-slot stream.

    Examples
    --------
    >>> attn = MaskedFlareAttention(dim=64, num_heads=4, num_flare_global_queries=8, context_dim=16)
    >>> x = torch.randn(2, 10, 64)
    >>> context = torch.randn(2, 4, 6, 16)
    >>> kv_mask = torch.tensor([[True] * 8 + [False] * 2] * 2)
    >>> y = attn(x, context, kv_mask)
    >>> y.shape
    torch.Size([2, 10, 64])
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_flare_global_queries: int,
        context_dim: int,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.context_dim = context_dim

        self.in_project_x = nn.Linear(dim, dim)
        self.self_k = nn.Linear(self.head_dim, self.head_dim)
        self.self_v = nn.Linear(self.head_dim, self.head_dim)
        self.q_global = nn.Parameter(
            torch.randn(1, num_heads, num_flare_global_queries, self.head_dim)
        )
        # Always-on (k, v) pair so the compress softmax has at least one
        # active column when every per-slot row is masked out.
        self.always_on_k = nn.Parameter(
            torch.randn(1, num_heads, 1, self.head_dim) * 0.02
        )
        self.always_on_v = nn.Parameter(
            torch.randn(1, num_heads, 1, self.head_dim) * 0.02
        )

        if context_dim > 0:
            self.cross_q = nn.Linear(self.head_dim, self.head_dim)
            self.cross_k = nn.Linear(context_dim, self.head_dim)
            self.cross_v = nn.Linear(context_dim, self.head_dim)
            self.state_mixing = nn.Parameter(torch.tensor(0.0))

        self.out_linear = nn.Linear(dim, dim)

    def forward(
        self,
        x: Float[Tensor, "B P C"],
        context: Float[Tensor, "B H S Dc"] | None,
        kv_mask: Bool[Tensor, "B P"],
    ) -> Float[Tensor, "B P C"]:
        if not torch.compiler.is_compiling():
            if (context is None) != (self.context_dim == 0):
                raise ValueError(
                    "Inconsistent cross-attention configuration: "
                    f"context_dim={self.context_dim} but context is "
                    f"{'None' if context is None else 'provided'}. "
                    "Pass a context iff the block was constructed with "
                    "context_dim > 0."
                )

        B = x.shape[0]
        x_mid = self.in_project_x(x)
        x_mid = rearrange(
            x_mid, "b p (h d) -> b h p d", h=self.num_heads, d=self.head_dim
        )
        k = self.self_k(x_mid)  # (B, H, P, head_dim)
        v = self.self_v(x_mid)  # (B, H, P, head_dim)

        # Prepend the always-on key/value pair so the masked-softmax cannot
        # see an all-False column.
        always_on_k = self.always_on_k.expand(B, -1, -1, -1)
        always_on_v = self.always_on_v.expand(B, -1, -1, -1)
        k_aug = torch.cat([always_on_k, k], dim=2)  # (B, H, 1+P, head_dim)
        v_aug = torch.cat([always_on_v, v], dim=2)

        always_on_col = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        compress_mask = torch.cat([always_on_col, kv_mask], dim=1)  # (B, 1+P)
        compress_mask = compress_mask[:, None, None, :]  # broadcasts over (H, M)

        G = self.q_global.expand(B, -1, -1, -1)  # (B, H, M, head_dim)
        z = F.scaled_dot_product_attention(G, k_aug, v_aug, attn_mask=compress_mask)
        # Decompress: per-slot keys query the compressed summary. SDPA is
        # per-query independent so no mask is required here.
        h_self = F.scaled_dot_product_attention(k, G, z)  # (B, H, P, head_dim)

        if context is not None:
            q_c = self.cross_q(x_mid)
            k_c = self.cross_k(context)
            v_c = self.cross_v(context)
            h_cross = F.scaled_dot_product_attention(q_c, k_c, v_c)
            mixing = torch.sigmoid(self.state_mixing)
            h_attn = mixing * h_self + (1.0 - mixing) * h_cross
        else:
            h_attn = h_self

        h_attn = rearrange(h_attn, "b h p d -> b p (h d)")
        return self.out_linear(h_attn)


class ParticleGALEBlock(Module):
    """Causal transformer block for a per-slot stream, conditioned on time and context.

    Use this as the building block of the model's tower. Each block
    refines the per-slot token stream while being steered by (a) a
    global conditioning vector (a time embedding) and (b) an optional
    shared geometric context. It is causal: a boolean ``kv_mask`` keeps
    absent slots from influencing any output, so a prediction depends
    only on the real particles. The block starts as a no-op at
    initialization and learns how strongly to modulate the stream as
    training proceeds, which keeps early training stable.

    Parameters
    ----------
    dim : int
        Per-token hidden width of the input/output stream.
    num_heads : int
        Number of attention heads. ``dim`` must be divisible by this.
    num_flare_global_queries : int
        Attention-bottleneck capacity (see :class:`MaskedFlareAttention`).
    context_dim : int
        Width of the shared cross-attention context. Set to 0 to run
        with self-attention only.
    cond_dim : int
        Width of the conditioning vector (e.g. a time embedding) that
        steers the block.
    mlp_ratio : int
        Feed-forward expansion ratio; the hidden width is
        ``mlp_ratio * dim``.

    Forward
    -------
    x : Tensor of shape ``(B, P, dim)``
        Per-slot input stream.
    c_t : Tensor of shape ``(B, cond_dim)``
        Conditioning vector that steers this block.
    context : Tensor of shape ``(B, num_heads, S, context_dim)`` or None
        Shared cross-attention context. Required iff the block was
        constructed with ``context_dim > 0``.
    kv_mask : Tensor of shape ``(B, P)`` (bool)
        ``True`` for real slots, ``False`` for absent slots that must
        not influence the output.

    Outputs
    -------
    Tensor of shape ``(B, P, dim)``
        Updated per-slot stream.

    Examples
    --------
    >>> block = ParticleGALEBlock(
    ...     dim=64, num_heads=4, num_flare_global_queries=8,
    ...     context_dim=16, cond_dim=32,
    ... )
    >>> x = torch.randn(2, 10, 64)
    >>> c_t = torch.randn(2, 32)
    >>> context = torch.randn(2, 4, 6, 16)
    >>> kv_mask = torch.ones(2, 10, dtype=torch.bool)
    >>> y = block(x, c_t, context, kv_mask)
    >>> y.shape
    torch.Size([2, 10, 64])
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_flare_global_queries: int,
        context_dim: int,
        cond_dim: int,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.attn = MaskedFlareAttention(
            dim=dim,
            num_heads=num_heads,
            num_flare_global_queries=num_flare_global_queries,
            context_dim=context_dim,
        )
        ffn_hidden = mlp_ratio * dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, dim),
        )
        # AdaLN-Zero conditioning MLP: emits six chunks of width `dim` to
        # modulate the two sublayers. Zero-initialized so the block starts
        # as identity (per AdaLN-Zero / DiT).
        adaLN_linear = nn.Linear(cond_dim, 6 * dim, bias=True)
        nn.init.zeros_(adaLN_linear.weight)
        nn.init.zeros_(adaLN_linear.bias)
        self.adaLN = nn.Sequential(nn.SiLU(), adaLN_linear)

    def forward(
        self,
        x: Float[Tensor, "B P C"],
        c_t: Float[Tensor, "B Cc"],
        context: Float[Tensor, "B H S Dc"] | None,
        kv_mask: Bool[Tensor, "B P"],
    ) -> Float[Tensor, "B P C"]:
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaLN(c_t).chunk(6, dim=-1)
        gamma1, beta1, alpha1 = (v.unsqueeze(1) for v in (gamma1, beta1, alpha1))
        gamma2, beta2, alpha2 = (v.unsqueeze(1) for v in (gamma2, beta2, alpha2))

        x_norm = self.ln_1(x) * (1.0 + gamma1) + beta1
        x = x + alpha1 * self.attn(x_norm, context, kv_mask)

        x_norm = self.ln_2(x) * (1.0 + gamma2) + beta2
        x = x + alpha2 * self.mlp(x_norm)
        return x


class MaskedContextProjector(ContextProjector):
    """Summarize a token stream into a few context tokens, ignoring absent tokens.

    A context projector compresses an arbitrarily large set of input
    tokens (here: particles, mesh points, and a time token) into a small,
    fixed number of "context" tokens that the rest of the model can
    attend to cheaply. This variant adds causality: a boolean
    ``kv_mask`` excludes absent tokens (padding, or particles that do
    not exist yet) so they never influence the summary.

    Build the context once per forward pass and share it across the
    attention blocks.

    Parameters
    ----------
    dim : int
        Channel width of the input tokens.
    heads : int
        Number of attention heads the context is produced for.
    dim_head : int
        Channel width of each produced context token.
    slice_num : int
        Number of context tokens to compress the stream into.

    Forward
    -------
    x : Tensor of shape ``(B, N, C)``
        Input token stream.
    kv_mask : Tensor of shape ``(B, N)`` (bool)
        ``True`` for tokens that may contribute to the summary,
        ``False`` for absent tokens to exclude.

    Outputs
    -------
    Tensor of shape ``(B, H, S, D)``
        The context tokens (``H`` heads, ``S`` tokens of width ``D``).

    Examples
    --------
    >>> proj = MaskedContextProjector(dim=64, heads=4, dim_head=16, slice_num=8)
    >>> x = torch.randn(2, 20, 64)
    >>> kv_mask = torch.tensor([[True] * 15 + [False] * 5] * 2)
    >>> ctx = proj(x, kv_mask=kv_mask)
    >>> ctx.shape
    torch.Size([2, 4, 8, 16])
    """

    def forward(
        self,
        x: Float[Tensor, "B N C"],
        *,
        kv_mask: Bool[Tensor, "B N"] | None = None,
    ) -> Float[Tensor, "B H S D"]:
        if not torch.compiler.is_compiling():
            if x.ndim != 3:
                raise ValueError(
                    f"Expected 3-D input (B, N, C), got {x.ndim}-D {tuple(x.shape)}."
                )
        projection = self.project_input_onto_slices(x)
        if isinstance(projection, tuple):
            projected_x, feature_projection = projection
        else:
            projected_x = projection
            feature_projection = projection
        slice_projections = self.in_project_slice(projected_x)  # (B, N, H, S)

        clamped_temp = torch.clamp(self.temperature, min=0.5, max=5).to(
            slice_projections.dtype
        )
        slice_weights = F.softmax(
            slice_projections / clamped_temp, dim=-1
        )  # (B, N, H, S)

        # Zero out absent tokens (broadcast over heads and slices).
        if kv_mask is not None:
            slice_weights = slice_weights * kv_mask[:, :, None, None].to(
                slice_weights.dtype
            )

        slice_norm = slice_weights.sum(1)  # (B, H, S)
        normed_weights = slice_weights / (slice_norm[:, None, :, :] + 1e-2)
        slice_tokens = torch.einsum(
            "bnhs,bnhd->bhsd", normed_weights, feature_projection
        )  # (B, H, S, D)

        if self.output_dropout is not None:
            slice_tokens = self.output_dropout(slice_tokens)
        return slice_tokens


def _smooth_clamp(
    x: Float[Tensor, "*shape"],
    lo: float,
    hi: float,
    beta: float = 1.0,
) -> Float[Tensor, "*shape"]:
    """Bound a tensor to ``[lo, hi]`` while still passing a gradient at the bounds.

    Use this instead of a hard clamp when an output must stay in a range
    but should keep learning when it drifts past the limits: a hard
    clamp has zero gradient once saturated, whereas this version still
    nudges the value back toward the range.

    Parameters
    ----------
    x : Tensor
        Input tensor of any shape.
    lo, hi : float
        Lower and upper bounds the output saturates toward.
    beta : float
        Sharpness of the saturation. Larger values track a hard clamp
        more closely.

    Returns
    -------
    Tensor
        Same shape as ``x``, with values approximately in ``[lo, hi]``.
    """
    return lo + F.softplus(x - lo, beta=beta) - F.softplus(x - hi, beta=beta)


class GMMHead(Module):
    """Head that emits ``(logits, mu, log_sigma)`` for a Gaussian mixture.

    Use this whenever a model needs to parameterize a Gaussian-mixture
    distribution over a vector (``output_dim`` dimensions, with
    ``output_dim = 1`` for a scalar quantity) from a single conditioning
    vector. The head is a single linear layer that emits all mixture
    parameters at once; ``log_sigma`` is clamped to a stability range so
    the consuming NLL/sampler functions stay well-conditioned. The
    outputs always carry an explicit ``output_dim`` axis, so the same
    NLL/sampling utilities handle the scalar and vector cases uniformly.

    Parameters
    ----------
    input_dim : int
        Width of the conditioning vector ``h``.
    num_components : int
        Number of mixture components.
    output_dim : int
        Dimensionality of the modeled variable (use ``1`` for a scalar).
    log_sigma_clamp : tuple[float, float]
        Lower and upper bounds applied to ``log_sigma`` for numerical
        stability. Ignored when ``log_sigma_clamp_type == "none"``.
    log_sigma_smooth_beta : float
        Sharpness of the ``"smooth"`` clamp; larger values track a hard
        clamp more closely. Only used when
        ``log_sigma_clamp_type == "smooth"``.
    log_sigma_clamp_type : Literal["smooth", "hard", "none"]
        How ``log_sigma`` is bounded. ``"smooth"`` (default) keeps a
        learning signal even past the bounds; ``"hard"`` clamps with no
        gradient once saturated; ``"none"`` leaves it unbounded.

    Forward
    -------
    h : Tensor of shape ``(B, input_dim)``
        Conditioning vector.

    Outputs
    -------
    logits : Tensor of shape ``(B, num_components)``
        Unnormalized mixture weights; pass through ``log_softmax`` to
        get mixture log-probabilities.
    mu : Tensor of shape ``(B, num_components, output_dim)``
        Per-component means.
    log_sigma : Tensor of shape ``(B, num_components, output_dim)``
        Per-component log-standard-deviations, clamped to a stable range
        for downstream NLL / sampling use.
    log_sigma_pre : Tensor of same shape as ``log_sigma``
        The un-clamped linear-projection output, kept differentiable so
        callers can apply an L2 regularizer that pulls the projection
        back toward zero whenever it drifts into the smooth clamp's
        saturated region.

    Examples
    --------
    >>> head = GMMHead(input_dim=64, num_components=4, output_dim=1)
    >>> h = torch.randn(2, 64)
    >>> logits, mu, log_sigma, log_sigma_pre = head(h)
    >>> logits.shape, mu.shape, log_sigma.shape, log_sigma_pre.shape
    (torch.Size([2, 4]), torch.Size([2, 4, 1]), torch.Size([2, 4, 1]), torch.Size([2, 4, 1]))
    """

    def __init__(
        self,
        input_dim: int,
        num_components: int,
        output_dim: int,
        log_sigma_clamp: tuple[float, float] = _LOG_SIGMA_CLAMP,
        log_sigma_smooth_beta: float = _LOG_SIGMA_SMOOTH_BETA,
        log_sigma_clamp_type: Literal["smooth", "hard", "none"] = "smooth",
    ) -> None:
        super().__init__()
        self.num_components = num_components
        self.output_dim = output_dim
        self.log_sigma_clamp = (float(log_sigma_clamp[0]), float(log_sigma_clamp[1]))
        self.log_sigma_smooth_beta = float(log_sigma_smooth_beta)
        self.log_sigma_clamp_type = log_sigma_clamp_type
        self.proj = nn.Linear(input_dim, num_components * (1 + 2 * output_dim))

    def forward(
        self, h: Float[Tensor, "B C"]
    ) -> tuple[
        Float[Tensor, "B G"],
        Float[Tensor, "B G D"],
        Float[Tensor, "B G D"],
        Float[Tensor, "B G D"],
    ]:
        out = self.proj(h)  # (B, G * (1 + 2*D))
        G, D = self.num_components, self.output_dim
        logits = out[:, :G]
        mu = out[:, G : G + G * D].view(-1, G, D)
        log_sigma_pre = out[:, G + G * D :].view(-1, G, D)
        if self.log_sigma_clamp_type == "smooth":
            log_sigma = _smooth_clamp(
                log_sigma_pre,
                self.log_sigma_clamp[0],
                self.log_sigma_clamp[1],
                self.log_sigma_smooth_beta,
            )
        elif self.log_sigma_clamp_type == "hard":
            log_sigma = torch.clamp(
                log_sigma_pre,
                self.log_sigma_clamp[0],
                self.log_sigma_clamp[1],
            )
        else:  # "none"
            log_sigma = log_sigma_pre
        return logits, mu, log_sigma, log_sigma_pre


def diagonal_gmm_nll(
    target: Float[Tensor, "B D"],
    logits: Float[Tensor, "B G"],
    mu: Float[Tensor, "B G D"],
    log_sigma: Float[Tensor, "B G D"],
) -> Float[Tensor, " B"]:
    """Per-sample NLL of a target under a diagonal Gaussian mixture in ``R^D``.

    Use this as the training loss for a model that emits a diagonal
    Gaussian-mixture distribution over a vector quantity. Each component
    has a per-dimension mean and standard deviation, with no
    cross-dimension correlation.

    Parameters
    ----------
    target : Tensor of shape ``(B, D)``
        Per-sample targets.
    logits : Tensor of shape ``(B, G)``
        Mixture logits.
    mu : Tensor of shape ``(B, G, D)``
        Per-component, per-dimension means.
    log_sigma : Tensor of shape ``(B, G, D)``
        Per-component, per-dimension log-standard-deviations.

    Returns
    -------
    Tensor of shape ``(B,)``
        Per-sample negative log-likelihood.
    """
    log_pi = F.log_softmax(logits, dim=-1)
    sigma = log_sigma.exp()
    diff = target.unsqueeze(-2) - mu  # (B, G, D)
    log_comp_per_d = -0.5 * (diff / sigma).pow(2) - log_sigma - 0.5 * _LOG_2PI
    log_comp = log_comp_per_d.sum(dim=-1)  # (B, G)
    log_p = torch.logsumexp(log_pi + log_comp, dim=-1)  # (B,)
    return -log_p


def diagonal_gmm_mean_std(
    logits: Float[Tensor, "B G"],
    mu: Float[Tensor, "B G D"],
    log_sigma: Float[Tensor, "B G D"],
) -> tuple[Float[Tensor, "B D"], Float[Tensor, "B D"]]:
    """Closed-form per-dimension mean and standard deviation of a diagonal Gaussian mixture.

    Returns the per-dimension marginal mean and standard deviation of
    the mixture in ``R^D`` defined by ``(logits, mu, log_sigma)``. Use
    this to report point summaries (with uncertainty) of a distribution
    predicted by a :class:`GMMHead`, or to plot error bands around
    inferred values.

    Parameters
    ----------
    logits : Tensor of shape ``(B, G)``
    mu : Tensor of shape ``(B, G, D)``
    log_sigma : Tensor of shape ``(B, G, D)``

    Returns
    -------
    mean : Tensor of shape ``(B, D)``
    std : Tensor of shape ``(B, D)``
    """
    pi = F.softmax(logits, dim=-1)  # (B, G)
    sigma = log_sigma.exp()
    mix_mean = (pi.unsqueeze(-1) * mu).sum(dim=-2)  # (B, D)
    second_moment = (pi.unsqueeze(-1) * (sigma.pow(2) + mu.pow(2))).sum(dim=-2)
    mix_var = (second_moment - mix_mean.pow(2)).clamp(min=0.0)
    return mix_mean, mix_var.sqrt()


def sample_diagonal_gmm(
    logits: Float[Tensor, "B G"],
    mu: Float[Tensor, "B G D"],
    log_sigma: Float[Tensor, "B G D"],
    generator: torch.Generator | None = None,
    eps: float = 1e-6,
) -> Float[Tensor, "B D"]:
    """Draw one D-dim sample per batch element from a diagonal Gaussian mixture.

    Use this at inference time to sample from the diagonal GMM distribution
    produced by :class:`GMMHead` (typically the particle-features head).
    Sampling is closed-form: pick a component via the Gumbel-max trick on
    the mixture logits, then sample its diagonal Gaussian directly.

    Parameters
    ----------
    logits : Tensor of shape ``(B, G)``
        Mixture logits.
    mu : Tensor of shape ``(B, G, D)``
        Per-component, per-dimension means.
    log_sigma : Tensor of shape ``(B, G, D)``
        Per-component, per-dimension log-standard-deviations.
    generator : torch.Generator, optional
        Random generator for reproducible draws.
    eps : float
        Small floor used in the Gumbel-max trick to keep the log/log
        chain well-defined.

    Returns
    -------
    Tensor of shape ``(B, D)``
        One sample per batch element.
    """
    log_pi = F.log_softmax(logits, dim=-1)
    cat_u = torch.rand(logits.shape, generator=generator, device=logits.device)
    gumbel = -torch.log(-torch.log(cat_u.clamp(min=eps, max=1 - eps)))
    cat_g = (log_pi + gumbel).argmax(dim=-1)  # (B,)
    idx = cat_g[:, None, None].expand(-1, 1, mu.shape[-1])
    mu_g = mu.gather(1, idx).squeeze(1)  # (B, D)
    sigma_g = log_sigma.exp().gather(1, idx).squeeze(1)  # (B, D)
    noise = torch.randn(mu_g.shape, generator=generator, device=mu_g.device)
    return mu_g + sigma_g * noise


def particle_gmm_loss(
    delay_target: Float[Tensor, " B"],
    particle_features_target: Float[Tensor, "B D"],
    delay_params: tuple[
        Float[Tensor, "B G"],
        Float[Tensor, "B G 1"],
        Float[Tensor, "B G 1"],
        Float[Tensor, "B G 1"],
    ],
    particle_features_params: tuple[
        Float[Tensor, "B G"],
        Float[Tensor, "B G D"],
        Float[Tensor, "B G D"],
        Float[Tensor, "B G D"],
    ],
    lambda_particle_features: float = 1.0,
    lambda_log_sigma: float = 0.0,
) -> tuple[Float[Tensor, ""], dict[str, float]]:
    """Combined NLL loss for the (delay, particle features) two-head model.

    Aggregates the mixture NLL for the next-event delay and for the new
    particle features into a single scalar loss, with an explicit weight
    on the particle-features term and an optional L2 regularizer on each
    head's pre-clamp ``log_sigma`` output. The regularizer keeps the
    un-bounded linear projection from drifting deep into the clamp's
    saturated band (where the NLL gradient is faint); set
    ``lambda_log_sigma = 0.0`` to disable it. The function also returns
    the three averaged components separately so the caller can log them.

    Parameters
    ----------
    delay_target : Tensor of shape ``(B,)``
        Ground-truth normalized delay.
    particle_features_target : Tensor of shape ``(B, D)``
        Ground-truth normalized particle features.
    delay_params : 4-tuple
        ``(logits, mu, log_sigma, log_sigma_pre)`` from the delay head
        (a 1-component-dimension mixture, i.e. ``D = 1``).
        ``log_sigma_pre`` is the un-clamped output used for the L2
        regularizer; the other three drive the NLL.
    particle_features_params : 4-tuple
        ``(logits, mu, log_sigma, log_sigma_pre)`` from the
        particle-features head.
    lambda_particle_features : float
        Weight on the particle-features NLL.
    lambda_log_sigma : float
        Weight on the L2 regularizer applied to the concatenation of
        both heads' pre-clamp ``log_sigma`` outputs. Small positive
        values (e.g. ``1e-2``) keep the un-bounded projection close to
        the differentiable band without dominating the data-fit terms.

    Returns
    -------
    loss : Tensor of shape ``()``
        Total batch-averaged loss.
    parts : dict
        ``{"delay": float, "particle_features": float, "log_sigma_reg":
        float}`` averaged per-component values, for logging.
    """
    delay_logits, delay_mu, delay_log_sigma, delay_log_sigma_pre = delay_params
    pf_logits, pf_mu, pf_log_sigma, pf_log_sigma_pre = particle_features_params
    # The delay is the D == 1 case of the same diagonal-mixture NLL.
    loss_delay = diagonal_gmm_nll(
        delay_target.unsqueeze(-1), delay_logits, delay_mu, delay_log_sigma
    ).mean()
    loss_pf = diagonal_gmm_nll(
        particle_features_target, pf_logits, pf_mu, pf_log_sigma
    ).mean()
    reg_log_sigma = delay_log_sigma_pre.pow(2).mean() + pf_log_sigma_pre.pow(2).mean()
    total = (
        loss_delay
        + lambda_particle_features * loss_pf
        + lambda_log_sigma * reg_log_sigma
    )
    parts = {
        "delay": float(loss_delay.detach().item()),
        "particle_features": float(loss_pf.detach().item()),
        "log_sigma_reg": float(reg_log_sigma.detach().item()),
    }
    return total, parts


class ParticleGeoTransolver(Module):
    """Probabilistic surrogate for a kinetic Monte Carlo solver, one event at a time.

    A kinetic Monte Carlo (KMC) solver evolves a system as a sequence of
    discrete events, each creating a new particle after a stochastic
    delay. This model emulates that solver: from the current state it
    predicts the next event probabilistically, and rolling it out
    autoregressively reproduces whole trajectories (and, across many
    rollouts, their uncertainty).

    The state at each step is (a) the particles created so far, each with
    spatial coordinates, a configurable set of scalar features, and the
    delay since the previous event; (b) a static background mesh of
    points carrying the fields the events unfold in (the boundary or
    initial conditions); and (c) the current simulation time. The model
    produces two probabilistic heads:

    - :meth:`predict_delay` gives a distribution over the delay until the
      next event.
    - :meth:`predict_particle_features` gives a distribution over the new
      particle's coordinates and scalar features, conditioned on the
      realized delay.

    The model is causal: the particle list is padded to a fixed
    ``num_particles_max``, and absent slots are masked out everywhere so
    a prediction depends only on the particles that already exist. The
    next-event prediction is read from the first free slot after the
    last real particle.

    Train it with teacher forcing (feed the ground-truth delay to the
    particle-features head); at inference, sample the delay head and feed
    that sample back to roll the trajectory forward.

    Parameters
    ----------
    num_particles_max : int
        Maximum number of particles per step; inputs are padded to this.
    num_particle_features : int
        Number of per-particle input columns. The first three are the
        spatial coordinates, the last is the inter-event delay, and any
        in-between columns are scalar features. The particle-features
        head predicts everything except the delay, i.e.
        ``num_particle_features - 1`` outputs.
    num_mesh_features : int
        Number of per-mesh-point input columns: the first three are
        spatial coordinates, the rest are scalar fields.
    num_hidden : int
        Hidden width of the model. Must be divisible by ``num_heads``.
    num_heads : int
        Number of attention heads in every block.
    num_blocks : int
        Depth of the model (number of transformer blocks).
    num_flare_global_queries : int
        Capacity of each block's attention bottleneck. Larger values let
        the model carry more information per token at more cost.
    num_context_slices : int
        Number of tokens the shared geometric context is summarized
        into (built once per step, shared across blocks). More tokens
        give a richer context at more cost.
    time_embed_channels : int
        Width of the embedding applied to each time-related scalar.
    token_type_embed_dim : int
        Width of the learnable tags that mark particle / mesh / time
        tokens so the context build can tell the three streams apart.
    context_token_dim : int
        Width of the tokens fed into the shared context.
    mlp_ratio : int
        Feed-forward expansion ratio inside each block.
    num_gmm_components : int
        Number of mixture components in each output head; more
        components model more complex distributions.
    dt_conditioning_embed_dim : int
        Width of the embedding of the realized delay that conditions the
        particle-features head.
    delay_mean, delay_std : float
        Normalization statistics for every time-related scalar (the
        ``"delay"`` entry of ``stats.json``). Stored with the checkpoint.
    log_delay_mean, log_delay_std : float
        Log-scale normalization statistics (the ``"log_delay"`` entry of
        ``stats.json``). Stored with the checkpoint.
    log_sigma_clamp : tuple[float, float]
        Bounds applied to the predicted log-standard-deviation in both
        heads, for numerical stability.
    log_sigma_smooth_beta : float
        Sharpness of the smooth clamp (see ``log_sigma_clamp_type``).
    delay_head_type : Literal["gmm"]
        Predictive distribution of the delay head. Only ``"gmm"`` is
        supported: a Gaussian mixture whose samples are kept positive at
        inference time by rejection sampling.
    log_sigma_clamp_type : Literal["smooth", "hard", "none"]
        How the predicted log-standard-deviation is bounded in both
        heads. ``"smooth"`` (default) keeps a learning signal past the
        bounds; ``"hard"`` clamps with no gradient once saturated;
        ``"none"`` leaves it unbounded.
    mesh_xyz_embedding_type : Literal["fourier", "raw"]
        How the mesh coordinates enter the model. ``"fourier"``
        (default) adds a Fourier feature lift that helps the model
        resolve fine mesh geometry; ``"raw"`` uses the coordinates as-is.
    time_delay_embedding_type : Literal["log_concat", "raw"]
        How time-related scalars are embedded. ``"log_concat"``
        (default) combines a linear-scale and a log-scale embedding so
        values spanning many orders of magnitude stay distinguishable;
        ``"raw"`` uses a single linear-scale embedding.
    time_max_positions_lin, time_max_positions_log : int
        Frequency range of the linear-scale and log-scale time
        embeddings.
    time_prescale_lin, time_prescale_log : float
        Multipliers applied to the time scalars before their linear-scale
        and log-scale embeddings.
    mesh_fourier_num_freqs : int
        Number of Fourier frequencies in the mesh-coordinate lift (used
        when ``mesh_xyz_embedding_type == "fourier"``).
    mesh_fourier_base : float
        Geometric ratio between successive mesh-coordinate frequencies.

    All forward inputs are expected to be already normalized, and every
    quantity is passed under its own argument (coordinates, scalar
    features, and delay separately) so the model never has to slice a
    packed feature tensor by position.

    Forward
    -------
    particle_coords : Tensor of shape ``(B, P, 3)``
        Per-particle spatial coordinates.
    particle_features : Tensor of shape ``(B, P, num_particle_features - 4)``
        Per-particle scalar features (the named features, in config
        order).
    delay : Tensor of shape ``(B, P)``
        Per-particle inter-event delay.
    particle_state : Tensor of shape ``(B, P)``
        Float-valued state vector, ``1.0`` for real particles and
        ``0.0`` for padded slots.
    mesh_coords : Tensor of shape ``(B, N, 3)``
        Per-mesh-point spatial coordinates.
    mesh_features : Tensor of shape ``(B, N, num_mesh_features - 3)``
        Per-mesh-point scalar fields (the named fields, in config order).
    t_n : Tensor of shape ``(B,)``
        Current simulation time.

    Outputs
    -------
    Tensor of shape ``(B, num_hidden)``
        Ghost-slot hidden vector ``h_g``, to be passed to
        :meth:`predict_delay` and :meth:`predict_particle_features`.

    Examples
    --------
    >>> model = ParticleGeoTransolver(num_particles_max=16, num_hidden=64, num_heads=4, num_blocks=2)
    >>> coords = torch.randn(2, 16, 3)
    >>> features = torch.randn(2, 16, 1)  # num_particle_features - 4
    >>> delay = torch.randn(2, 16)
    >>> state = torch.tensor([[1.0] * 5 + [0.0] * 11] * 2)
    >>> mesh_coords = torch.randn(2, 20, 3)
    >>> mesh_features = torch.randn(2, 20, 2)  # num_mesh_features - 3
    >>> t_n = torch.tensor([0.3, 0.7])
    >>> h_g = model(coords, features, delay, state, mesh_coords, mesh_features, t_n)
    >>> h_g.shape
    torch.Size([2, 64])
    >>> delay_logits, delay_mu, delay_log_sigma, _ = model.predict_delay(h_g)
    >>> delay_logits.shape
    torch.Size([2, 4])
    """

    def __init__(
        self,
        num_particles_max: int,
        num_particle_features: int = 5,
        num_mesh_features: int = 5,
        num_hidden: int = 128,
        num_heads: int = 4,
        num_blocks: int = 4,
        num_flare_global_queries: int = 32,
        num_context_slices: int = 32,
        time_embed_channels: int = 64,
        token_type_embed_dim: int = 8,
        context_token_dim: int = 64,
        mlp_ratio: int = 4,
        num_gmm_components: int = 4,
        dt_conditioning_embed_dim: int = 32,
        delay_mean: float = 0.0,
        delay_std: float = 1.0,
        log_delay_mean: float = 0.0,
        log_delay_std: float = 1.0,
        log_sigma_clamp: tuple[float, float] = _LOG_SIGMA_CLAMP,
        log_sigma_smooth_beta: float = _LOG_SIGMA_SMOOTH_BETA,
        log_sigma_clamp_type: Literal["smooth", "hard", "none"] = "smooth",
        delay_head_type: Literal["gmm"] = "gmm",
        mesh_xyz_embedding_type: Literal["fourier", "raw"] = "fourier",
        time_delay_embedding_type: Literal["log_concat", "raw"] = "log_concat",
        time_max_positions_lin: int = 1000,
        time_max_positions_log: int = 1000,
        time_prescale_lin: float = 100.0,
        time_prescale_log: float = 100.0,
        mesh_fourier_num_freqs: int = 6,
        mesh_fourier_base: float = 2.0,
    ) -> None:
        super().__init__()
        self.delay_head_type = delay_head_type
        self.mesh_xyz_embedding_type = mesh_xyz_embedding_type
        self.time_delay_embedding_type = time_delay_embedding_type
        if num_hidden % num_heads != 0:
            raise ValueError(
                f"num_hidden={num_hidden} must be divisible by num_heads={num_heads}."
            )
        if num_particle_features < 4:
            raise ValueError(
                f"num_particle_features={num_particle_features} must be >= 4 "
                "(x, y, z, ..., delay)."
            )
        if num_mesh_features < 3:
            raise ValueError(
                f"num_mesh_features={num_mesh_features} must be >= 3 (x, y, z)."
            )
        self.num_particles_max = num_particles_max
        self.num_particle_features = num_particle_features
        self.num_mesh_features = num_mesh_features
        self.num_hidden = num_hidden
        self._mesh_fourier_num_freqs = mesh_fourier_num_freqs
        self._mesh_fourier_base = mesh_fourier_base

        # The particle-features head emits everything except the delay.
        num_particle_features_out = num_particle_features - 1

        # Two time embeddings (one for the absolute simulation time ``t_n``,
        # one for the per-particle / output delay scalar). With "log_concat"
        # we use the dual linear+log branch wrapper, which expands the lower
        # end of the distribution that would otherwise be crushed when delays
        # span many orders of magnitude. With "raw" we use a plain learnable
        # Fourier embedding on the linearly-normalized scalar; the pre-scale
        # is stored as ``self._embed_prescale`` and applied at call sites (it
        # is 1.0 in the "log_concat" path since that wrapper handles its own
        # pre-scaling internally).
        if time_delay_embedding_type == "log_concat":
            self.time_embed: nn.Module = LogConcatTimeEmbedding(
                num_channels=time_embed_channels,
                max_positions_lin=time_max_positions_lin,
                max_positions_log=time_max_positions_log,
                prescale_lin=time_prescale_lin,
                prescale_log=time_prescale_log,
                delay_mean=delay_mean,
                delay_std=delay_std,
                log_delay_mean=log_delay_mean,
                log_delay_std=log_delay_std,
            )
            self.delay_embed: nn.Module = LogConcatTimeEmbedding(
                num_channels=time_embed_channels,
                max_positions_lin=time_max_positions_lin,
                max_positions_log=time_max_positions_log,
                prescale_lin=time_prescale_lin,
                prescale_log=time_prescale_log,
                delay_mean=delay_mean,
                delay_std=delay_std,
                log_delay_mean=log_delay_mean,
                log_delay_std=log_delay_std,
            )
            self._embed_prescale = 1.0
        else:
            self.time_embed = PositionalEmbedding(
                num_channels=time_embed_channels,
                max_positions=time_max_positions_lin,
                learnable=True,
            )
            self.delay_embed = PositionalEmbedding(
                num_channels=time_embed_channels,
                max_positions=time_max_positions_lin,
                learnable=True,
            )
            self._embed_prescale = float(time_prescale_lin)

        # AdaLN conditioning vector c_t = MLP(time_embed(t_n)).
        self.t_cond_mlp = _make_mlp(time_embed_channels, num_hidden)

        # Type embeddings (concat-style; one row per token kind).
        self.e_exist = nn.Parameter(torch.randn(token_type_embed_dim) * 0.02)
        self.e_mesh = nn.Parameter(torch.randn(token_type_embed_dim) * 0.02)
        self.e_time = nn.Parameter(torch.randn(token_type_embed_dim) * 0.02)

        # Local input projection: spatial + non-delay scalars + embedded delay.
        # The trailing delay scalar is replaced by its time embedding before
        # projection.
        num_non_delay_scalars = num_particle_features - 4  # exclude xyz and delay
        local_input_dim = 3 + num_non_delay_scalars + time_embed_channels
        self.input_proj = _make_mlp(local_input_dim, num_hidden)
        # Learnable null token replacing absent slots' projected features.
        self.null_token = nn.Parameter(torch.randn(1, 1, num_hidden) * 0.02)

        # Typed-token MLPs feeding the masked context projector.
        particle_tok_in_dim = (
            3 + num_non_delay_scalars + time_embed_channels + token_type_embed_dim
        )
        # Mesh tokens always include the raw input features and the type
        # embedding; ``mesh_xyz_embedding_type == "fourier"`` adds a per-axis
        # Fourier lift of the xyz columns (``2 * num_freqs * 3`` extra
        # channels), ``"raw"`` skips the lift entirely.
        if mesh_xyz_embedding_type == "fourier":
            mesh_fourier_dim = 2 * self._mesh_fourier_num_freqs * 3
        else:
            mesh_fourier_dim = 0
        mesh_tok_in_dim = num_mesh_features + mesh_fourier_dim + token_type_embed_dim
        time_tok_in_dim = time_embed_channels + token_type_embed_dim
        self.mlp_particle_tok = _make_mlp(particle_tok_in_dim, context_token_dim)
        self.mlp_mesh_tok = _make_mlp(mesh_tok_in_dim, context_token_dim)
        self.mlp_time_tok = _make_mlp(time_tok_in_dim, context_token_dim)

        # Masked context projector emits (B, num_heads, num_context_slices, head_dim).
        head_dim = num_hidden // num_heads
        self.context_projector = MaskedContextProjector(
            dim=context_token_dim,
            heads=num_heads,
            dim_head=head_dim,
            slice_num=num_context_slices,
            use_te=False,
        )

        # Block tower: every block's cross-attention reads from the shared
        # context emitted by the projector above (per-head slice tokens).
        self.blocks = nn.ModuleList(
            [
                ParticleGALEBlock(
                    dim=num_hidden,
                    num_heads=num_heads,
                    num_flare_global_queries=num_flare_global_queries,
                    context_dim=head_dim,
                    cond_dim=num_hidden,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_ln = nn.LayerNorm(num_hidden)

        # Heads. The particle-features head reads h_g concatenated with the
        # delay-conditioning embedding xi(Delta_t). Both heads share the same
        # log_sigma clamp configuration.
        self.delay_head = GMMHead(
            input_dim=num_hidden,
            num_components=num_gmm_components,
            output_dim=1,
            log_sigma_clamp=log_sigma_clamp,
            log_sigma_smooth_beta=log_sigma_smooth_beta,
            log_sigma_clamp_type=log_sigma_clamp_type,
        )
        self.dt_cond_mlp = _make_mlp(time_embed_channels, dt_conditioning_embed_dim)
        self.particle_features_head = GMMHead(
            input_dim=num_hidden + dt_conditioning_embed_dim,
            num_components=num_gmm_components,
            output_dim=num_particle_features_out,
            log_sigma_clamp=log_sigma_clamp,
            log_sigma_smooth_beta=log_sigma_smooth_beta,
            log_sigma_clamp_type=log_sigma_clamp_type,
        )

    def _build_context(
        self,
        particle_coords: Float[Tensor, "B P 3"],
        particle_features: Float[Tensor, "B P F_p"],
        delay: Float[Tensor, "B P"],
        particle_state: Float[Tensor, "B P"],
        mesh_coords: Float[Tensor, "B N 3"],
        mesh_features: Float[Tensor, "B N F_m"],
        t_emb: Float[Tensor, "B Et"],
    ) -> Float[Tensor, "B H S D"]:
        """Build the persistent context tensor consumed by every block."""
        B, P, _ = particle_coords.shape
        N = mesh_coords.shape[1]

        # Embed the per-particle delay scalar (see __init__ for the role of
        # self._embed_prescale across the two embedding modes).
        delay_emb = self.delay_embed(delay.reshape(-1) * self._embed_prescale).view(
            B, P, -1
        )
        particle_input = torch.cat(
            [
                particle_coords,
                particle_features,
                delay_emb,
                self.e_exist.expand(B, P, -1),
            ],
            dim=-1,
        )
        x_particle = self.mlp_particle_tok(particle_input)  # (B, P, context_token_dim)

        # Mesh tokens. With ``mesh_xyz_embedding_type == "fourier"`` the mesh
        # coordinates are first lifted with a per-axis Fourier basis (the scalar
        # fields are kept alongside); with ``"raw"`` the coordinates enter as-is.
        if self.mesh_xyz_embedding_type == "fourier":
            mesh_coords = fourier_features_xyz(
                mesh_coords,
                num_freqs=self._mesh_fourier_num_freqs,
                base=self._mesh_fourier_base,
            )
        mesh_input = torch.cat(
            [mesh_coords, mesh_features, self.e_mesh.expand(B, N, -1)], dim=-1
        )
        x_mesh = self.mlp_mesh_tok(mesh_input)  # (B, N, context_token_dim)

        time_input = torch.cat([t_emb, self.e_time.expand(B, -1)], dim=-1).unsqueeze(1)
        x_time = self.mlp_time_tok(time_input)  # (B, 1, context_token_dim)

        tokens = torch.cat([x_particle, x_mesh, x_time], dim=1)  # (B, P+N+1, ...)
        kv_mask = torch.cat(
            [
                particle_state > 0.5,
                torch.ones(B, N + 1, dtype=torch.bool, device=tokens.device),
            ],
            dim=1,
        )
        return self.context_projector(tokens, kv_mask=kv_mask)

    def forward(
        self,
        particle_coords: Float[Tensor, "B P 3"],
        particle_features: Float[Tensor, "B P F_p"],
        delay: Float[Tensor, "B P"],
        particle_state: Float[Tensor, "B P"],
        mesh_coords: Float[Tensor, "B N 3"],
        mesh_features: Float[Tensor, "B N F_m"],
        t_n: Float[Tensor, " B"],
    ) -> Float[Tensor, "B C"]:
        if not torch.compiler.is_compiling():
            n_particle_scalars = self.num_particle_features - 4
            if particle_features.shape[-1] != n_particle_scalars:
                raise ValueError(
                    f"particle_features has {particle_features.shape[-1]} scalar "
                    f"features in its last dim, but the model expects "
                    f"{n_particle_scalars} (num_particle_features - 4)."
                )
            n_mesh_fields = self.num_mesh_features - 3
            if mesh_features.shape[-1] != n_mesh_fields:
                raise ValueError(
                    f"mesh_features has {mesh_features.shape[-1]} fields in its "
                    f"last dim, but the model expects {n_mesh_fields} "
                    "(num_mesh_features - 3)."
                )

        B, P, _ = particle_coords.shape

        # Pre-scaled time embeddings (the log_concat wrapper carries its own
        # internal pre-scale, so self._embed_prescale is 1.0 in that path).
        t_emb = self.time_embed(t_n * self._embed_prescale)  # (B, Et)
        c_t = self.t_cond_mlp(t_emb)  # (B, num_hidden)

        # Local stream: embed delay, project, then null-substitute absent slots.
        delay_emb = self.delay_embed(delay.reshape(-1) * self._embed_prescale).view(
            B, P, -1
        )
        local_input = torch.cat([particle_coords, particle_features, delay_emb], dim=-1)
        x_proj = self.input_proj(local_input)  # (B, P, num_hidden)
        state = particle_state.unsqueeze(-1) > 0.5
        x = torch.where(state, x_proj, self.null_token.expand(B, P, -1))

        # Persistent context (shared across blocks).
        context = self._build_context(
            particle_coords,
            particle_features,
            delay,
            particle_state,
            mesh_coords,
            mesh_features,
            t_emb,
        )

        # Block tower; every block's FLARE compress respects the mask.
        kv_mask_particles = particle_state > 0.5
        for block in self.blocks:
            x = block(x, c_t, context, kv_mask_particles)

        x = self.final_ln(x)

        # Read out the ghost slot: index = number of real particles per batch element.
        ghost_idx = particle_state.sum(dim=-1).clamp(max=P - 1).long()  # (B,)
        h_g = x[torch.arange(B, device=x.device), ghost_idx]  # (B, num_hidden)
        return h_g

    def predict_delay(
        self, h_g: Float[Tensor, "B C"]
    ) -> tuple[
        Float[Tensor, "B G"],
        Float[Tensor, "B G 1"],
        Float[Tensor, "B G 1"],
        Float[Tensor, "B G 1"],
    ]:
        """Return GMM parameters ``(logits, mu, log_sigma, log_sigma_pre)`` for ``Delta_t``.

        The delay is a scalar, so ``mu``/``log_sigma`` carry a trailing
        dimension of size 1, matching the particle-features head and
        letting the same mixture utilities handle both. The last element
        is the un-clamped ``log_sigma`` used by the L2 penalty in
        :func:`particle_gmm_loss`; inference paths only need the first
        three.
        """
        return self.delay_head(h_g)

    def predict_particle_features(
        self,
        h_g: Float[Tensor, "B C"],
        delay_normalized: Float[Tensor, " B"],
    ) -> tuple[
        Float[Tensor, "B G"],
        Float[Tensor, "B G D"],
        Float[Tensor, "B G D"],
        Float[Tensor, "B G D"],
    ]:
        """Return diagonal-GMM parameters ``(logits, mu, log_sigma, log_sigma_pre)`` for the new particle features given ``Delta_t``.

        The last element mirrors :meth:`predict_delay` -- the un-clamped
        ``log_sigma`` linear-projection output, exposed for the L2
        regularizer inside :func:`particle_gmm_loss`.
        """
        delay_emb = self.delay_embed(delay_normalized * self._embed_prescale)
        xi = self.dt_cond_mlp(delay_emb)
        return self.particle_features_head(torch.cat([h_g, xi], dim=-1))
