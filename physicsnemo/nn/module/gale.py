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

"""GALE (Geometry-Aware Latent Embeddings) attention layer and transformer block.

This module provides the GALE attention mechanism and GALEBlock transformer block,
which extend the Transolver physics attention with cross-attention capabilities for
geometry and global context embeddings.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch.distributed.tensor.placement_types import Replicate

import physicsnemo  # noqa: F401 for docs
from physicsnemo.core.version_check import OptionalImport

from .concrete_dropout import ConcreteDropout
from .flare_attention import (
    _flare_encode,
    _flare_encode_te,
    _flare_self_attention,
    _flare_self_attention_te,
)
from .mlp_layers import Mlp
from .physics_attention import (
    PhysicsAttentionIrregularMesh,
    PhysicsAttentionStructuredMesh2D,
    PhysicsAttentionStructuredMesh3D,
    _project_input,
)

te = OptionalImport("transformer_engine.pytorch")


def _mix_self_and_cross(
    self_attn: torch.Tensor,
    cross_attn: torch.Tensor,
    mode: str,
    state_mixing: nn.Parameter | None = None,
    concat_project: nn.Module | None = None,
) -> torch.Tensor:
    r"""Blend self-attention and cross-attention outputs.

    Parameters
    ----------
    self_attn : torch.Tensor
        Self-attention output.
    cross_attn : torch.Tensor
        Cross-attention output (same shape as ``self_attn``).
    mode : str
        ``"weighted"`` for sigmoid-gated sum, ``"concat_project"`` for
        concatenation followed by a learned projection.
    state_mixing : nn.Parameter or None
        Learnable scalar for ``"weighted"`` mode.
    concat_project : nn.Module or None
        Projection module for ``"concat_project"`` mode.

    Returns
    -------
    torch.Tensor
        Blended output, same shape as inputs.
    """
    match mode:
        case "weighted":
            w = torch.sigmoid(state_mixing)
            return w * self_attn + (1 - w) * cross_attn
        case "concat_project":
            return concat_project(torch.cat([self_attn, cross_attn], dim=-1))
        case _:
            raise ValueError(f"Invalid state_mixing_mode: {mode!r}")


def _gale_compute_slice_attention_cross(
    module: nn.Module,
    slice_tokens: list[Float[torch.Tensor, "batch heads slices dim"]],
    context: Float[torch.Tensor, "batch heads context_slices context_dim"],
) -> list[Float[torch.Tensor, "batch heads slices dim"]]:
    r"""Shared cross-attention between slice tokens and context.

    Used by :class:`GALE` and :class:`_GALEStructuredForwardMixin` so the
    cross-attention implementation lives in one place. Projects queries from
    concatenated slice tokens, keys and values from context; runs Transformer
    Engine or SDPA attention; splits the result back to one tensor per input.

    Parameters
    ----------
    module : nn.Module
        Module with ``cross_q``, ``cross_k``, ``cross_v``, ``use_te``,
        ``heads``, ``dim_head``, and (if ``use_te``) ``attn_fn``.
    slice_tokens : list[torch.Tensor]
        One tensor per input, each of shape :math:`(B, H, S, D)`.
    context : torch.Tensor
        Context tensor of shape :math:`(B, H, S_c, D_c)`.

    Returns
    -------
    list[torch.Tensor]
        One cross-attention output per element of ``slice_tokens``, each
        of shape :math:`(B, H, S, D)`.
    """
    q_input = torch.cat(slice_tokens, dim=-2)

    # Slice tokens and context are reductions over the (possibly sharded)
    # token axis: distributed inputs arrive as unreduced Partial sums, and
    # everything from here on (projection bias, softmax) is nonlinear in
    # them. Resolve to Replicate before projecting. Duck-typed because nn
    # cannot import domain_parallel.
    if hasattr(q_input, "redistribute"):
        q_input = q_input.redistribute(placements=[Replicate()])
    if hasattr(context, "redistribute"):
        context = context.redistribute(placements=[Replicate()])

    q = module.cross_q(q_input)
    k = module.cross_k(context)
    v = module.cross_v(context)
    if module.use_te:
        q = rearrange(q, "b h s d -> b s h d")
        k = rearrange(k, "b h s d -> b s h d")
        v = rearrange(v, "b h s d -> b s h d")
        cross_attention = module.attn_fn(q, k, v)
        cross_attention = rearrange(
            cross_attention,
            "b s (h d) -> b h s d",
            h=module.heads,
            d=module.dim_head,
        )
    else:
        cross_attention = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=False
        )
    cross_attention = torch.split(cross_attention, slice_tokens[0].shape[-2], dim=-2)
    return list(cross_attention)


def _gale_forward_impl(
    module: nn.Module,
    x: tuple[Float[torch.Tensor, "batch tokens channels"], ...],
    context: Float[torch.Tensor, "batch heads context_slices context_dim"] | None,
) -> list[Float[torch.Tensor, "batch tokens channels"]]:
    r"""Single implementation of the GALE forward pipeline.

    Shared by :class:`GALE` and :class:`_GALEStructuredForwardMixin`. Steps:
    validate inputs; project onto slices; compute slice weights and tokens;
    apply self-attention on slices; optionally cross-attend to context and
    mix with ``state_mixing``; project attention outputs back to token space.

    Parameters
    ----------
    module : nn.Module
        GALE-like module with ``project_input_onto_slices``,
        ``in_project_slice``, ``_compute_slices_from_projections``,
        ``_compute_slice_attention_te``, ``_compute_slice_attention_sdpa``,
        ``compute_slice_attention_cross``, ``_project_attention_outputs``,
        plus attributes ``use_te``, ``plus``, ``state_mixing_mode``, and
        ``state_mixing`` (if weighted) or ``concat_project`` (if concat).
    x : tuple[torch.Tensor, ...]
        Input tensors, each of shape :math:`(B, N, C)`; must be non-empty.
    context : torch.Tensor or None
        Optional context of shape :math:`(B, H, S_c, D_c)` for cross-attention.
        If ``None``, only self-attention is applied.

    Returns
    -------
    list[torch.Tensor]
        One output tensor per input, each of shape :math:`(B, N, C)`.

    Raises
    ------
    ValueError
        If ``x`` is empty or any element is not 3D.
    """
    if not torch.compiler.is_compiling():
        if len(x) == 0:
            raise ValueError("Expected non-empty tuple of input tensors")
        for i, tensor in enumerate(x):
            if tensor.ndim != 3:
                raise ValueError(
                    f"Expected 3D input tensor (B, N, C) at index {i}, "
                    f"got {tensor.ndim}D tensor with shape {tuple(tensor.shape)}"
                )
    if module.plus:
        x_mid = [module.project_input_onto_slices(_x) for _x in x]
        fx_mid = [_x_mid for _x_mid in x_mid]
    else:
        x_mid, fx_mid = zip(*[module.project_input_onto_slices(_x) for _x in x])
    slice_projections = [module.in_project_slice(_x_mid) for _x_mid in x_mid]
    slice_weights, slice_tokens = zip(
        *[
            module._compute_slices_from_projections(proj, _fx_mid)
            for proj, _fx_mid in zip(slice_projections, fx_mid)
        ]
    )
    if module.use_te:
        self_slice_token = [
            module._compute_slice_attention_te(_slice_token)
            for _slice_token in slice_tokens
        ]
    else:
        self_slice_token = [
            module._compute_slice_attention_sdpa(_slice_token)
            for _slice_token in slice_tokens
        ]
    if context is not None:
        cross_slice_token = [
            module.compute_slice_attention_cross([_slice_token], context)[0]
            for _slice_token in slice_tokens
        ]
        out_slice_token = [
            _mix_self_and_cross(
                sst,
                cst,
                module.state_mixing_mode,
                state_mixing=getattr(module, "state_mixing", None),
                concat_project=getattr(module, "concat_project", None),
            )
            for sst, cst in zip(self_slice_token, cross_slice_token)
        ]
    else:
        # Use only self-attention when no context is provided
        out_slice_token = self_slice_token
    outputs = [
        module._project_attention_outputs(ost, sw)
        for ost, sw in zip(out_slice_token, slice_weights)
    ]
    return outputs


class GALE(PhysicsAttentionIrregularMesh):
    r"""Geometry-Aware Latent Embeddings (GALE) attention layer.

    This is an extension of the Transolver PhysicsAttention mechanism to support
    cross-attention with a context vector, built from geometry and global embeddings.
    GALE combines self-attention on learned physical state slices with cross-attention
    to geometry-aware context, using a learnable mixing weight to blend the two.

    Parameters
    ----------
    dim : int
        Input dimension of the features.
    heads : int, optional
        Number of attention heads. Default is 8.
    dim_head : int, optional
        Dimension of each attention head. Default is 64.
    dropout : float, optional
        Dropout rate. Default is 0.0.
    slice_num : int, optional
        Number of learned physical state slices. Default is 64.
    use_te : bool, optional
        Whether to use Transformer Engine backend when available. Default is False.
    plus : bool, optional
        Whether to use Transolver++ features. Default is False.
    context_dim : int, optional
        Dimension of the context vector for cross-attention. Default is 0.
    concrete_dropout : bool, optional
        Whether to use ConcreteDropout instead of standard dropout. Default is False.
    state_mixing_mode : str, optional
        How to blend self-attention and cross-attention outputs. ``"weighted"`` uses
        a learnable sigmoid-gated weighted sum. ``"concat_project"``
        concatenates the two along the head dimension and projects back with a
        linear layer. Default is ``"weighted"``.

    Forward
    -------
    x : tuple[torch.Tensor, ...]
        Tuple of input tensors, each of shape :math:`(B, N, C)` where :math:`B` is
        batch size, :math:`N` is number of tokens, and :math:`C` is number of channels.
    context : tuple[torch.Tensor, ...] | None, optional
        Context tensor for cross-attention of shape :math:`(B, H, S_c, D_c)` where
        :math:`H` is number of heads, :math:`S_c` is number of context slices, and
        :math:`D_c` is context dimension. If ``None``, only self-attention is applied.
        Default is ``None``.

    Outputs
    -------
    list[torch.Tensor]
        List of output tensors, each of shape :math:`(B, N, C)`, same shape as inputs.

    Notes
    -----
    The mixing between self-attention and cross-attention is controlled by a learnable
    parameter ``state_mixing`` which is passed through a sigmoid function to ensure
    the mixing weight stays in :math:`[0, 1]`.

    See Also
    --------
    :class:`physicsnemo.models.transolver.Physics_Attention.PhysicsAttentionIrregularMesh` : Base physics attention class.
    :class:`GALEBlock` : Transformer block using GALE attention.

    Examples
    --------
    >>> import torch
    >>> gale = GALE(dim=256, heads=8, dim_head=32, context_dim=32, use_te=False)
    >>> x = (torch.randn(2, 100, 256),)  # Single input tensor in tuple
    >>> context = torch.randn(2, 8, 64, 32)  # Context for cross-attention
    >>> outputs = gale(x, context)
    >>> len(outputs)
    1
    >>> outputs[0].shape
    torch.Size([2, 100, 256])
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 64,
        use_te: bool = False,
        plus: bool = False,
        context_dim: int = 0,
        concrete_dropout: bool = False,
        state_mixing_mode: str = "weighted",
    ) -> None:
        super().__init__(dim, heads, dim_head, dropout, slice_num, use_te, plus)
        _gale_cross_init(self, dim_head, context_dim, use_te, state_mixing_mode)

        # Replace inherited out_dropout with ConcreteDropout when enabled
        if concrete_dropout:
            self.out_dropout = ConcreteDropout(
                in_features=dim,
                init_p=max(dropout, 0.05),
            )

    def compute_slice_attention_cross(
        self,
        slice_tokens: list[Float[torch.Tensor, "batch heads slices dim"]],
        context: Float[torch.Tensor, "batch heads context_slices context_dim"],
    ) -> list[Float[torch.Tensor, "batch heads slices dim"]]:
        r"""Compute cross-attention between slice tokens and context.

        Parameters
        ----------
        slice_tokens : list[torch.Tensor]
            List of slice token tensors, each of shape :math:`(B, H, S, D)` where
            :math:`B` is batch size, :math:`H` is number of heads, :math:`S` is
            number of slices, and :math:`D` is head dimension.
        context : torch.Tensor
            Context tensor of shape :math:`(B, H, S_c, D_c)` where :math:`S_c` is
            number of context slices and :math:`D_c` is context dimension.

        Returns
        -------
        list[torch.Tensor]
            List of cross-attention outputs, each of shape :math:`(B, H, S, D)`.
        """
        return _gale_compute_slice_attention_cross(self, slice_tokens, context)

    def forward(
        self,
        x: tuple[Float[torch.Tensor, "batch tokens channels"], ...],
        context: Float[torch.Tensor, "batch heads context_slices context_dim"]
        | None = None,
    ) -> list[Float[torch.Tensor, "batch tokens channels"]]:
        r"""Forward pass of the GALE module.

        Applies physics-aware self-attention combined with optional cross-attention
        to geometry and global context.

        Parameters
        ----------
        x : tuple[torch.Tensor, ...]
            Tuple of input tensors, each of shape :math:`(B, N, C)` where :math:`B`
            is batch size, :math:`N` is number of tokens, and :math:`C` is number
            of channels.
        context : torch.Tensor | None, optional
            Context tensor for cross-attention of shape :math:`(B, H, S_c, D_c)`
            where :math:`H` is number of heads, :math:`S_c` is number of context
            slices, and :math:`D_c` is context dimension. If ``None``, only
            self-attention is applied. Default is ``None``.

        Returns
        -------
        list[torch.Tensor]
            List of output tensors, each of shape :math:`(B, N, C)``, same shape
            as inputs.
        """
        return _gale_forward_impl(self, x, context)


def _gale_cross_init(
    self: nn.Module,
    dim_head: int,
    context_dim: int,
    use_te: bool,
    state_mixing_mode: str = "weighted",
) -> None:
    # Match GALE: TE linear only when TE is installed (GALEBlock already errors if use_te without TE)
    linear_layer = te.Linear if (use_te and te.available) else nn.Linear
    self.cross_q = linear_layer(dim_head, dim_head)
    self.cross_k = linear_layer(context_dim, dim_head)
    self.cross_v = linear_layer(context_dim, dim_head)

    self.state_mixing_mode = state_mixing_mode

    match state_mixing_mode:
        case "weighted":
            # Learnable mixing weight between self and cross attention
            # Initialize near 0.0 since sigmoid(0) = 0.5, giving balanced initial mixing
            self.state_mixing = nn.Parameter(torch.tensor(0.0))
        case "concat_project":
            # Concatenate self and cross attention and project back to dim_head
            self.concat_project = nn.Sequential(
                linear_layer(2 * dim_head, dim_head),
                nn.GELU(),
            )
        case _:
            raise ValueError(
                f"Invalid state_mixing_mode: {state_mixing_mode!r}. "
                f"Expected 'weighted' or 'concat_project'."
            )


class _GALEStructuredForwardMixin:
    """Shared cross-attention and forward for structured GALE (2D/3D conv projection)."""

    def compute_slice_attention_cross(
        self,
        slice_tokens: list[Float[torch.Tensor, "batch heads slices dim"]],
        context: Float[torch.Tensor, "batch heads context_slices context_dim"],
    ) -> list[Float[torch.Tensor, "batch heads slices dim"]]:
        return _gale_compute_slice_attention_cross(self, slice_tokens, context)

    def forward(
        self,
        x: tuple[Float[torch.Tensor, "batch tokens channels"], ...],
        context: Float[torch.Tensor, "batch heads context_slices context_dim"]
        | None = None,
    ) -> list[Float[torch.Tensor, "batch tokens channels"]]:
        return _gale_forward_impl(self, x, context)


class GALEStructuredMesh2D(
    _GALEStructuredForwardMixin, PhysicsAttentionStructuredMesh2D
):
    r"""GALE with Conv2d slice projection for 2D structured grids (see :class:`GALE`)."""

    def __init__(
        self,
        dim: int,
        spatial_shape: tuple[int, int],
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 64,
        kernel: int = 3,
        use_te: bool = False,
        plus: bool = False,
        context_dim: int = 0,
        state_mixing_mode: str = "weighted",
    ) -> None:
        super().__init__(
            dim,
            spatial_shape,
            heads,
            dim_head,
            dropout,
            slice_num,
            kernel,
            use_te,
            plus,
        )
        _gale_cross_init(self, dim_head, context_dim, use_te, state_mixing_mode)


class GALEStructuredMesh3D(
    _GALEStructuredForwardMixin, PhysicsAttentionStructuredMesh3D
):
    r"""GALE with Conv3d slice projection for 3D structured grids (see :class:`GALE`)."""

    def __init__(
        self,
        dim: int,
        spatial_shape: tuple[int, int, int],
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 64,
        kernel: int = 3,
        use_te: bool = False,
        plus: bool = False,
        context_dim: int = 0,
        state_mixing_mode: str = "weighted",
    ) -> None:
        super().__init__(
            dim,
            spatial_shape,
            heads,
            dim_head,
            dropout,
            slice_num,
            kernel,
            use_te,
            plus,
        )
        _gale_cross_init(self, dim_head, context_dim, use_te, state_mixing_mode)


class GALE_FA(nn.Module):
    r"""GALE_FA: Geometry-Aware Latent Embeddings with FLARE self-Attention attention layer.

    Adopted:

    - FLARE attention: Fast Low-rank Attention Routing Engine
        paper: https://arxiv.org/abs/2508.12594
    - GeoTransolver context:
        paper: https://arxiv.org/abs/2512.20399

    GALE_FA is an alternative to the GALE attention mechanism of the GeoTransolver.
    It supports cross-attention with a context vector, built from geometry and global embeddings.
    GALE_FA combines FLARE self-attention on learned physical state slices with cross-attention
    to geometry-aware context, using a learnable mixing weight to blend the two.

    Two independent axes control the context read: ``context_placement``
    moves the cross-attention queries from the point features to the FLARE
    latent tokens, and ``context_source_dims`` switches the blending from
    the single-projection state mixing to a per-source gated read. Any
    combination of the two is valid.

    Parameters
    ----------
    dim : int
        Input dimension of the features.
    heads : int, optional
        Number of attention heads. Default is 8.
    dim_head : int, optional
        Dimension of each attention head. Default is 64.
    dropout : float, optional
        Dropout rate. Default is 0.0.
    n_global_queries : int, optional
        Number of learned global queries. Default is 64.
    use_te : bool, optional
        Whether to use Transformer Engine backend when available. Default is False.
    context_dim : int, optional
        Dimension of the context vector for cross-attention. Default is 0.
    concrete_dropout : bool, optional
        Whether to use learned concrete dropout instead of standard dropout.
        Default is ``False``.
    state_mixing_mode : str, optional
        How to blend self-attention and cross-attention outputs.  ``"weighted"`` uses
        a learnable sigmoid-gated weighted sum. ``"concat_project"``
        concatenates the two along the head dimension and projects back with a
        linear layer. Only used when ``context_source_dims`` is ``None``.
        Default is ``"weighted"``.
    context_placement : {"points", "latents"}, optional
        Where the context cross-attention queries come from. The
        ``"points"`` placement reads the context from the :math:`N` point
        features. The ``"latents"`` placement reads it from the
        ``n_global_queries`` FLARE latent tokens, between the encode and
        decode attention passes, and blends at the latent level before
        decoding; this reduces the context-attention cost by a factor
        ``n_global_queries`` :math:`/ N`. Default is ``"points"``.
    context_source_dims : tuple[int, ...] | None, optional
        Channel widths of the sources concatenated in the context (e.g.
        ball-query scales, geometry, and global embeddings); must sum to
        ``context_dim``. When provided, all sources share one attention
        score matrix but each source applies its own value projection to
        its channel slice, and a learned per-source, per-channel softmax
        gate blends the reads with the self stream (see Notes). Passing
        ``None`` keeps the single value projection over the full context
        and the ``state_mixing_mode`` blend. Default is ``None``.

    Forward
    -------
    x : tuple[torch.Tensor, ...]
        Tuple of input tensors, each of shape :math:`(B, N, C)` where :math:`B` is
        batch size, :math:`N` is number of tokens, and :math:`C` is number of channels.
    context : tuple[torch.Tensor, ...] | None, optional
        Context tensor for cross-attention of shape :math:`(B, H, S_c, D_c)` where
        :math:`H` is number of heads, :math:`S_c` is number of context slices, and
        :math:`D_c` is context dimension. If ``None``, only self-attention is applied.
        Default is ``None``.

    Outputs
    -------
    list[torch.Tensor]
        List of output tensors, each of shape :math:`(B, N, C)`, same shape as inputs.

    Notes
    -----
    With ``context_source_dims=None``, a learnable parameter
    ``state_mixing`` controls the mixing between self-attention and
    cross-attention; a sigmoid keeps the mixing weight in :math:`[0, 1]`.

    With ``context_source_dims`` set, the layer splits the context
    :math:`C = [C_1, \dots, C_S]` channel-wise into its sources and blends
    the self stream :math:`z` with the per-source reads through a
    channel-wise softmax gate:

    .. math::

        \alpha = \operatorname{softmax}(\eta, \text{dim}=0)
        \in \mathbb{R}^{(1 + S) \times D},
        \qquad
        \tilde{z} = \alpha_0 \odot z + \sum_{s=1}^{S} \alpha_s \odot
        \operatorname{softmax}\left(Q K^\top \cdot \text{scale}\right)
        V_s(C_s),

    where the logits :math:`\eta` are zero-initialized so the blend starts
    uniform over the streams. The per-source value decomposition adds no
    expressivity over a single value projection (the gate weights commute
    with the attention row mixing), but it changes the parameterization,
    the per-source weighting, and the initialization; unlike a residual
    addition, the gate can also attenuate the self stream.

    See Also
    --------
    :class:`GALE` : Original GeoTransolver GALE attention class.
    :class:`GALEBlock` : Transformer block that calls GALE or GALE_FA attention.

    Examples
    --------
    >>> import torch
    >>> gale_fa = GALE_FA(dim=256, heads=8, dim_head=32, context_dim=32)
    >>> x = (torch.randn(2, 100, 256),)  # Single input tensor in tuple
    >>> context = torch.randn(2, 8, 64, 32)  # Context for cross-attention
    >>> outputs = gale_fa(x, context)
    >>> len(outputs)
    1
    >>> outputs[0].shape
    torch.Size([2, 100, 256])

    With the context read at the latent bottleneck and the gated per-source
    blend (the context concatenates a 24- and an 8-channel source):

    >>> gale_fa = GALE_FA(
    ...     dim=256,
    ...     heads=8,
    ...     dim_head=32,
    ...     context_dim=32,
    ...     context_placement="latents",
    ...     context_source_dims=(24, 8),
    ... )
    >>> outputs = gale_fa(x, context)
    >>> outputs[0].shape
    torch.Size([2, 100, 256])
    """

    def __init__(
        self,
        dim,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        n_global_queries: int = 64,
        use_te: bool = False,
        context_dim: int = 0,
        concrete_dropout: bool = False,
        state_mixing_mode: str = "weighted",
        context_placement: Literal["points", "latents"] = "points",
        context_source_dims: tuple[int, ...] | None = None,
    ):
        # With use_te, linear projections and attention run on Transformer
        # Engine; otherwise on PyTorch. A missing TE install raises with an
        # install hint on first use, so no extra guard is needed here.
        super().__init__()
        self.use_te = use_te
        self.heads = heads
        self.dim_head = dim_head
        self.scale = 1.0
        # It is recommended by the FLARE authors to use self.scale = 1 if self.dim_head <= 8 else (self.dim_head ** -0.5)
        # but we use self.scale = 1.0 because the recommended scaling is not tested yet.
        inner_dim = dim_head * heads

        # Bind the placement and blending paths once; forward has no
        # per-call dispatch on either option.
        self.context_placement = context_placement
        match context_placement:
            case "points":
                self._attend = self._attend_points
            case "latents":
                self._attend = self._attend_latents
            case _:
                raise ValueError(
                    f"Invalid context_placement: {context_placement!r}. "
                    f"Expected 'points' or 'latents'."
                )

        if context_source_dims is None:
            self.context_source_dims = None
            self._project_context_values = self._project_context_values_single
            self._blend_streams = self._blend_streams_state_mixing
        else:
            self.context_source_dims = tuple(context_source_dims)
            if len(self.context_source_dims) == 0 or any(
                w <= 0 for w in self.context_source_dims
            ):
                raise ValueError(
                    f"context_source_dims must be a non-empty tuple of "
                    f"positive channel widths, got {context_source_dims!r}"
                )
            if sum(self.context_source_dims) != context_dim:
                raise ValueError(
                    f"context_source_dims {context_source_dims!r} must sum "
                    f"to context_dim ({context_dim})"
                )
            self._project_context_values = self._project_context_values_per_source
            self._blend_streams = self._blend_streams_source_gate

        linear_layer = te.Linear if self.use_te else nn.Linear

        # Global queries for FLARE self-attention
        self.q_global = nn.Parameter(torch.randn(1, heads, n_global_queries, dim_head))

        # Linear projections for self-attention
        self.in_project_x = linear_layer(dim, inner_dim)
        self.self_k = linear_layer(dim_head, dim_head)
        self.self_v = linear_layer(dim_head, dim_head)

        # FLARE's self-attention passes and the cross-attention all have
        # differing q/kv lengths, so TE runs them as cross-attention (BSHD).
        # Keep dropout in out_dropout so TE and PyTorch use the same dropout site.
        if self.use_te:
            self.attn_fn = te.DotProductAttention(
                num_attention_heads=self.heads,
                kv_channels=self.dim_head,
                attention_dropout=0.0,
                attn_mask_type="no_mask",
                attention_type="cross",
                qkv_format="bshd",
                softmax_scale=self.scale,
            )

        if context_dim > 0:
            if self.context_source_dims is None:
                _gale_cross_init(self, dim_head, context_dim, use_te, state_mixing_mode)
            else:
                # Match _gale_cross_init: TE linear only when TE is available
                cross_linear = te.Linear if (use_te and te.available) else nn.Linear
                self.cross_q = cross_linear(dim_head, dim_head)
                self.cross_k = cross_linear(context_dim, dim_head)
                self.cross_v_sources = nn.ModuleList(
                    [cross_linear(w, dim_head) for w in self.context_source_dims]
                )
                # Zero init: the softmax gate starts as a uniform blend over
                # the self stream and the per-source reads
                self.source_gate_logits = nn.Parameter(
                    torch.zeros(1 + len(self.context_source_dims), dim_head)
                )

        # Linear projection for output
        self.out_linear = linear_layer(inner_dim, dim)
        if concrete_dropout:
            self.out_dropout = ConcreteDropout(
                in_features=dim,
                init_p=max(dropout, 0.05),
            )
        else:
            self.out_dropout = nn.Dropout(dropout)

    def _project_context_values_single(
        self,
        context: Float[torch.Tensor, "batch heads context_slices context_dim"],
    ) -> list[Float[torch.Tensor, "batch heads context_slices dim"]]:
        r"""Project the full context through the single value projection."""
        return [self.cross_v(context)]

    def _project_context_values_per_source(
        self,
        context: Float[torch.Tensor, "batch heads context_slices context_dim"],
    ) -> list[Float[torch.Tensor, "batch heads context_slices dim"]]:
        r"""Project each context source through its own value projection."""
        chunks = torch.split(context, self.context_source_dims, dim=-1)
        return [proj(chunk) for proj, chunk in zip(self.cross_v_sources, chunks)]

    def _blend_streams_state_mixing(
        self,
        stream: Float[torch.Tensor, "batch heads tokens dim"]
        | Float[torch.Tensor, "batch tokens heads dim"],
        reads: list[
            Float[torch.Tensor, "batch heads tokens dim"]
            | Float[torch.Tensor, "batch tokens heads dim"]
        ],
    ) -> (
        Float[torch.Tensor, "batch heads tokens dim"]
        | Float[torch.Tensor, "batch tokens heads dim"]
    ):
        r"""Blend the self stream with the single context read via state mixing."""
        return _mix_self_and_cross(
            stream,
            reads[0],
            self.state_mixing_mode,
            state_mixing=getattr(self, "state_mixing", None),
            concat_project=getattr(self, "concat_project", None),
        )

    def _blend_streams_source_gate(
        self,
        stream: Float[torch.Tensor, "batch heads tokens dim"]
        | Float[torch.Tensor, "batch tokens heads dim"],
        reads: list[
            Float[torch.Tensor, "batch heads tokens dim"]
            | Float[torch.Tensor, "batch tokens heads dim"]
        ],
    ) -> (
        Float[torch.Tensor, "batch heads tokens dim"]
        | Float[torch.Tensor, "batch tokens heads dim"]
    ):
        r"""Blend the self stream with the per-source reads via the softmax gate.

        Channel-wise softmax over the stacked streams; the blend treats both
        backend layouts identically because it only mixes the last (channel)
        axis.
        """
        alpha = torch.softmax(self.source_gate_logits.to(dtype=stream.dtype), dim=0)
        stacked = torch.stack([stream, *reads], dim=-2)  # (..., 1 + S, D)
        return (alpha * stacked).sum(dim=-2)  # (..., D)

    def _attend_points(
        self,
        x_mid: list[Float[torch.Tensor, "batch heads tokens dim"]],
        context: Float[torch.Tensor, "batch heads context_slices context_dim"] | None,
    ) -> list[Float[torch.Tensor, "batch heads tokens dim"]]:
        r"""FLARE self-attention with the context read from the point features."""
        # FLARE self-attention per input
        if self.use_te:
            self_attention = [
                _flare_self_attention_te(
                    _x_mid,
                    self.q_global,
                    self.self_k,
                    self.self_v,
                    self.attn_fn,
                    self.heads,
                )
                for _x_mid in x_mid
            ]
        else:
            self_attention = [
                _flare_self_attention(
                    _x_mid,
                    self.q_global,
                    self.self_k,
                    self.self_v,
                    self.scale,
                )
                for _x_mid in x_mid
            ]

        # Cross-attention with context, blended per point
        if context is not None:
            values = self._project_context_values(context)
            if self.use_te:
                # TE cross-attention: reshape (B, H, S, D) -> bshd, run through
                # the shared DotProductAttention, then back to (B, H, N, D).
                k = rearrange(self.cross_k(context), "b h s d -> b s h d")
                values = [rearrange(_v, "b h s d -> b s h d") for _v in values]
                q = [
                    rearrange(self.cross_q(_x_mid), "b h n d -> b n h d")
                    for _x_mid in x_mid
                ]
                reads = [
                    [
                        rearrange(
                            self.attn_fn(_q, k, _v),
                            "b n (h d) -> b h n d",
                            h=self.heads,
                        )
                        for _v in values
                    ]
                    for _q in q
                ]
            else:
                q = [self.cross_q(_x_mid) for _x_mid in x_mid]
                k = self.cross_k(context)
                reads = [
                    [
                        F.scaled_dot_product_attention(_q, k, _v, scale=self.scale)
                        for _v in values
                    ]
                    for _q in q
                ]
            outputs = [
                self._blend_streams(sa, _reads)
                for sa, _reads in zip(self_attention, reads)
            ]
        else:
            outputs = self_attention
        return outputs

    def _attend_latents(
        self,
        x_mid: list[Float[torch.Tensor, "batch heads tokens dim"]],
        context: Float[torch.Tensor, "batch heads context_slices context_dim"] | None,
    ) -> list[Float[torch.Tensor, "batch heads tokens dim"]]:
        r"""FLARE attention with the context read from the latent tokens.

        Runs the FLARE encode, blends the latent tokens with context reads
        whose queries come from the latents, then decodes the blended
        latents back to the point tokens. The per-point context read of
        ``_attend_points`` is never executed on this path.
        """
        # FLARE encode per input: latent tokens gather the point tokens
        if self.use_te:
            encoded = [
                _flare_encode_te(
                    _x_mid,
                    self.q_global,
                    self.self_k,
                    self.self_v,
                    self.attn_fn,
                    self.heads,
                )
                for _x_mid in x_mid
            ]
        else:
            encoded = [
                _flare_encode(
                    _x_mid,
                    self.q_global,
                    self.self_k,
                    self.self_v,
                    self.scale,
                )
                for _x_mid in x_mid
            ]

        # Context read and blend at the latent bottleneck
        if context is not None:
            values = self._project_context_values(context)
            blended = []
            if self.use_te:
                # Latents are already in the bshd layout; only the context
                # projections need reshaping around the DotProductAttention.
                k_ctx = rearrange(self.cross_k(context), "b h s d -> b s h d")
                values = [rearrange(_v, "b h s d -> b s h d") for _v in values]
                for k, G, z in encoded:
                    q = self.cross_q(z)  # (B, S, H, D)
                    reads = [
                        rearrange(
                            self.attn_fn(q, k_ctx, _v),
                            "b s (h d) -> b s h d",
                            h=self.heads,
                        )
                        for _v in values
                    ]
                    blended.append((k, G, self._blend_streams(z, reads)))
            else:
                k_ctx = self.cross_k(context)
                for k, G, z in encoded:
                    q = self.cross_q(z)  # (B, H, S, D)
                    reads = [
                        F.scaled_dot_product_attention(q, k_ctx, _v, scale=self.scale)
                        for _v in values
                    ]
                    blended.append((k, G, self._blend_streams(z, reads)))
            encoded = blended

        # FLARE decode per input: point tokens read the blended latent tokens
        if self.use_te:
            return [
                rearrange(self.attn_fn(k, G, z), "b n (h d) -> b h n d", h=self.heads)
                for k, G, z in encoded
            ]
        return [
            F.scaled_dot_product_attention(k, G, z, scale=self.scale)
            for k, G, z in encoded
        ]

    def forward(
        self,
        x: tuple[Float[torch.Tensor, "batch tokens channels"], ...],
        context: Float[torch.Tensor, "batch heads context_slices context_dim"]
        | None = None,
    ) -> list[Float[torch.Tensor, "batch tokens channels"]]:
        r"""Forward pass of the GALE_FA module.

        Applies GALE_FA attention to the input features.

        Parameters
        ----------
        x : tuple[torch.Tensor, ...]
            Tuple of input tensors, each of shape :math:`(B, N, C)` where :math:`B`
            is batch size, :math:`N` is number of tokens, and :math:`C` is number
            of channels.
        context : torch.Tensor | None, optional
            Context tensor for cross-attention of shape :math:`(B, H, S_c, D_c)`
            where :math:`H` is number of heads, :math:`S_c` is number of context
            slices, and :math:`D_c` is context dimension. If ``None``, the
            layer applies only self-attention. Default is ``None``.

        Returns
        -------
        list[torch.Tensor]
            List of output tensors, each of shape :math:`(B, N, C)`, same shape
            as inputs.
        """
        # Input projection: (B, N, C) -> (B, N, H, D) -> (B, H, N, D)
        x_mid = [
            _project_input(
                _x,
                self.in_project_x,
                self.heads,
                self.dim_head,
                "B N (H D) -> B N H D",
            ).permute(0, 2, 1, 3)
            for _x in x
        ]

        # Self-attention and context read; construction binds the placement
        outputs = self._attend(x_mid, context)

        # Back to token layout: (B, H, N, D) -> (B, N, H, D)
        outputs = [_y.permute(0, 2, 1, 3) for _y in outputs]
        outputs = [rearrange(_out, "b n h d -> b n (h d)") for _out in outputs]
        outputs = [self.out_linear(_out) for _out in outputs]
        return [self.out_dropout(_out) for _out in outputs]


class GALEBlock(nn.Module):
    r"""Transformer encoder block using GALE attention.

    This block replaces standard self-attention with the GALE (Geometry-Aware Latent
    Embeddings) attention mechanism, which combines physics-aware self-attention with
    cross-attention to geometry and global context.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    hidden_dim : int
        Hidden dimension of the transformer.
    dropout : float
        Dropout rate.
    act : str, optional
        Activation function name. Default is ``"gelu"``.
    mlp_ratio : int, optional
        Ratio of MLP hidden dimension to ``hidden_dim``. Default is 4.
    last_layer : bool, optional
        Whether this is the last layer in the model. Default is ``False``.
    out_dim : int, optional
        Output dimension (only used if ``last_layer=True``). Default is 1.
    slice_num : int, optional
        Number of learned physical state slices. Default is 32.
    use_te : bool, optional
        Whether to use Transformer Engine backend. Default is ``False``.
    plus : bool, optional
        Whether to use Transolver++ features. Default is ``False``.
    context_dim : int, optional
        Dimension of the context vector for cross-attention. Default is 0.
    spatial_shape : tuple[int, ...] | None, optional
        If ``None``, uses irregular-mesh GALE. Length-2 tuple enables 2D Conv2d
        projection; length-3 tuple enables 3D Conv3d projection (flattened
        :math:`N = H \times W` or :math:`H \times W \times D`). Default is ``None``.
    attention_type : str, optional
        Attention backend to use. ``"GALE"`` uses the standard physics-aware
        slice attention; ``"GALE_FA"`` uses flash-attention variant.
        Default is ``"GALE"``.
    state_mixing_mode : str, optional
        How to blend self-attention and cross-attention outputs. ``"weighted"`` uses
        a learnable sigmoid-gated weighted sum. ``"concat_project"``
        concatenates the two along the head dimension and projects back with a
        linear layer. Default is ``"weighted"``.
    context_placement : {"points", "latents"}, optional
        Forwarded to :class:`GALE_FA`: where its context cross-attention
        queries come from, either the point features or the FLARE latent
        tokens. Requires ``attention_type="GALE_FA"``. Default is
        ``"points"``.
    context_source_dims : tuple[int, ...] | None, optional
        Forwarded to :class:`GALE_FA`: channel widths of the context sources
        for the per-source gated blend; must sum to ``context_dim``. Requires
        ``attention_type="GALE_FA"``. The validated combination is
        ``context_placement="latents"`` with ``context_source_dims`` set.
        Default is ``None``.

    Forward
    -------
    fx : tuple[torch.Tensor, ...]
        Tuple of input tensors, each of shape :math:`(B, N, C)` where :math:`B` is
        batch size, :math:`N` is number of tokens, and :math:`C` is hidden dimension.
    global_context : tuple[torch.Tensor, ...]
        Global context tensor for cross-attention of shape :math:`(B, H, S_c, D_c)`
        where :math:`H` is number of heads, :math:`S_c` is number of context slices,
        and :math:`D_c` is context dimension.

    Outputs
    -------
    list[torch.Tensor]
        List of output tensors, each of shape :math:`(B, N, C)`, same shape as inputs.

    Notes
    -----
    The block applies layer normalization before the attention operation and uses
    residual connections after both the attention and MLP layers.

    See Also
    --------
    :class:`GALE` : The attention mechanism used in this block.
    :class:`physicsnemo.models.geotransolver.GeoTransolver` : Main model using GALEBlock.

    Examples
    --------
    >>> import torch
    >>> block = GALEBlock(num_heads=8, hidden_dim=256, dropout=0.1, context_dim=32, use_te=False)
    >>> fx = (torch.randn(2, 100, 256),)  # Single input tensor in tuple
    >>> context = torch.randn(2, 8, 64, 32)  # Global context
    >>> outputs = block(fx, context)
    >>> len(outputs)
    1
    >>> outputs[0].shape
    torch.Size([2, 100, 256])
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act: str = "gelu",
        mlp_ratio: int = 4,
        last_layer: bool = False,
        out_dim: int = 1,
        slice_num: int = 32,
        use_te: bool = False,
        plus: bool = False,
        context_dim: int = 0,
        spatial_shape: tuple[int, ...] | None = None,
        attention_type: str = "GALE",
        concrete_dropout: bool = False,
        state_mixing_mode: str = "weighted",
        context_placement: Literal["points", "latents"] = "points",
        context_source_dims: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()

        self.last_layer = last_layer

        # Layer normalization before attention
        if use_te:
            self.ln_1 = te.LayerNorm(hidden_dim)
        else:
            self.ln_1 = nn.LayerNorm(hidden_dim)

        dim_head = hidden_dim // num_heads
        # First match on attention backend, then on spatial shape
        match attention_type:
            case "GALE":
                if context_placement != "points" or context_source_dims is not None:
                    raise ValueError(
                        f"context_placement and context_source_dims require "
                        f"attention_type='GALE_FA'; got attention_type='GALE' "
                        f"with context_placement={context_placement!r} and "
                        f"context_source_dims={context_source_dims!r}"
                    )
                if spatial_shape is None:
                    self.Attn = GALE(
                        hidden_dim,
                        heads=num_heads,
                        dim_head=dim_head,
                        dropout=dropout,
                        slice_num=slice_num,
                        use_te=use_te,
                        plus=plus,
                        context_dim=context_dim,
                        concrete_dropout=concrete_dropout,
                        state_mixing_mode=state_mixing_mode,
                    )
                elif len(spatial_shape) == 2:
                    self.Attn = GALEStructuredMesh2D(
                        hidden_dim,
                        spatial_shape=(int(spatial_shape[0]), int(spatial_shape[1])),
                        heads=num_heads,
                        dim_head=dim_head,
                        dropout=dropout,
                        slice_num=slice_num,
                        use_te=use_te,
                        plus=plus,
                        context_dim=context_dim,
                        state_mixing_mode=state_mixing_mode,
                    )
                elif len(spatial_shape) == 3:
                    self.Attn = GALEStructuredMesh3D(
                        hidden_dim,
                        spatial_shape=(
                            int(spatial_shape[0]),
                            int(spatial_shape[1]),
                            int(spatial_shape[2]),
                        ),
                        heads=num_heads,
                        dim_head=dim_head,
                        dropout=dropout,
                        slice_num=slice_num,
                        use_te=use_te,
                        plus=plus,
                        context_dim=context_dim,
                        state_mixing_mode=state_mixing_mode,
                    )
                else:
                    raise ValueError(
                        f"spatial_shape must be None, length-2, or length-3; got {spatial_shape!r}"
                    )
            case "GALE_FA":
                self.Attn = GALE_FA(
                    hidden_dim,
                    heads=num_heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    n_global_queries=slice_num,
                    use_te=use_te,
                    context_dim=context_dim,
                    concrete_dropout=concrete_dropout,
                    state_mixing_mode=state_mixing_mode,
                    context_placement=context_placement,
                    context_source_dims=context_source_dims,
                )
            case _:
                raise ValueError(
                    f"Invalid attention type: {attention_type}. "
                    f"Expected 'GALE' or 'GALE_FA'."
                )

        # Feed-forward network with layer normalization
        if use_te:
            self.ln_mlp1 = te.LayerNormMLP(
                hidden_size=hidden_dim,
                ffn_hidden_size=hidden_dim * mlp_ratio,
            )
        else:
            self.ln_mlp1 = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                Mlp(
                    in_features=hidden_dim,
                    hidden_features=hidden_dim * mlp_ratio,
                    out_features=hidden_dim,
                    act_layer=act,
                    use_te=False,
                ),
            )

        # Concrete dropout after attention and FFN residuals
        if concrete_dropout:
            self.attn_dropout = ConcreteDropout(
                in_features=hidden_dim,
                init_p=max(dropout, 0.05),
            )
            self.ffn_dropout = ConcreteDropout(
                in_features=hidden_dim,
                init_p=max(dropout, 0.05),
            )
        else:
            self.attn_dropout = None
            self.ffn_dropout = None

    def forward(
        self,
        fx: tuple[Float[torch.Tensor, "batch tokens hidden_dim"], ...],
        global_context: Float[torch.Tensor, "batch heads context_slices context_dim"],
    ) -> list[Float[torch.Tensor, "batch tokens hidden_dim"]]:
        r"""Forward pass of the GALE block.

        Parameters
        ----------
        fx : tuple[torch.Tensor, ...]
            Tuple of input tensors, each of shape :math:`(B, N, C)` where :math:`B`
            is batch size, :math:`N` is number of tokens, and :math:`C` is hidden
            dimension.
        global_context : torch.Tensor
            Global context tensor for cross-attention of shape :math:`(B, H, S_c, D_c)`
            where :math:`H` is number of heads, :math:`S_c` is number of context slices,
            and :math:`D_c` is context dimension.

        Returns
        -------
        list[torch.Tensor]
            List of output tensors, each of shape :math:`(B, N, C)`, same shape as inputs.
        """
        ### Input validation
        if not torch.compiler.is_compiling():
            if len(fx) == 0:
                raise ValueError("Expected non-empty tuple of input tensors")
            for i, tensor in enumerate(fx):
                if tensor.ndim != 3:
                    raise ValueError(
                        f"Expected 3D input tensor (B, N, C) at index {i}, "
                        f"got {tensor.ndim}D tensor with shape {tuple(tensor.shape)}"
                    )

        # Apply pre-normalization to all inputs
        normed_inputs = [self.ln_1(_fx) for _fx in fx]

        # Apply GALE attention with cross-attention to global context
        attn = self.Attn(tuple(normed_inputs), global_context)

        # Residual connection after attention
        fx_out = [attn[i] + fx[i] for i in range(len(fx))]

        # Concrete dropout after attention residual
        if self.attn_dropout is not None:
            fx_out = [self.attn_dropout(_fx) for _fx in fx_out]

        # Feed-forward network with residual connection
        fx_out = [self.ln_mlp1(_fx) + _fx for _fx in fx_out]

        # Concrete dropout after FFN residual
        if self.ffn_dropout is not None:
            fx_out = [self.ffn_dropout(_fx) for _fx in fx_out]

        return fx_out
