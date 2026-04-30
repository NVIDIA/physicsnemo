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

This module provides the GALE attention mechanism and GALE_block transformer block,
which extend the Transolver physics attention with cross-attention capabilities for
geometry and global context embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float

import physicsnemo  # noqa: F401 for docs
from physicsnemo.core.version_check import check_version_spec, OptionalImport
from physicsnemo.nn import Mlp
from physicsnemo.nn.module.physics_attention import (
    PhysicsAttentionIrregularMesh,
    PhysicsAttentionStructuredMesh2D,
    PhysicsAttentionStructuredMesh3D,
)

from physicsnemo.experimental.models.geotransolver.gale_fa import GALE_FA
from physicsnemo.nn import ConcreteDropout

# Check optional dependency availability
TE_AVAILABLE = check_version_spec("transformer_engine", "0.1.0", hard_fail=False)
te = OptionalImport("transformer_engine.pytorch", "0.1.0")


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
    cross_attention = torch.split(
        cross_attention, slice_tokens[0].shape[-2], dim=-2
    )
    return list(cross_attention)


def _gale_forward_impl(
    module: nn.Module,
    x: tuple[Float[torch.Tensor, "batch tokens channels"], ...],
    context: Float[torch.Tensor, "batch heads context_slices context_dim"]
    | None,
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
        x_mid, fx_mid = zip(
            *[module.project_input_onto_slices(_x) for _x in x]
        )
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
        # Blend self-attention and cross-attention
        match module.state_mixing_mode:
            case "weighted":
                mixing_weight = torch.sigmoid(module.state_mixing)
                out_slice_token = [
                    mixing_weight * sst + (1 - mixing_weight) * cst
                    for sst, cst in zip(self_slice_token, cross_slice_token)
                ]
            case "concat_project":
                out_slice_token = [
                    module.concat_project(torch.cat([sst, cst], dim=-1))
                    for sst, cst in zip(self_slice_token, cross_slice_token)
                ]
            case _:
                raise ValueError(
                    f"Invalid state_mixing_mode: {module.state_mixing_mode!r}"
                )
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
        Whether to use Transformer Engine backend when available. Default is True.
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
    :class:`GALE_block` : Transformer block using GALE attention.

    Examples
    --------
    >>> import torch
    >>> gale = GALE(dim=256, heads=8, dim_head=32, context_dim=32)
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
        use_te: bool = True,
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
        return _gale_compute_slice_attention_cross(
            self, slice_tokens, context
        )

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
    # Match GALE: TE linear only when TE is installed (GALE_block already errors if use_te without TE)
    linear_layer = te.Linear if (use_te and TE_AVAILABLE) else nn.Linear
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
        return _gale_compute_slice_attention_cross(
            self, slice_tokens, context
        )

    def forward(
        self,
        x: tuple[Float[torch.Tensor, "batch tokens channels"], ...],
        context: Float[torch.Tensor, "batch heads context_slices context_dim"]
        | None = None,
    ) -> list[Float[torch.Tensor, "batch tokens channels"]]:
        return _gale_forward_impl(self, x, context)


class GALEStructuredMesh2D(_GALEStructuredForwardMixin, PhysicsAttentionStructuredMesh2D):
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
        use_te: bool = True,
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


class GALEStructuredMesh3D(_GALEStructuredForwardMixin, PhysicsAttentionStructuredMesh3D):
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
        use_te: bool = True,
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


class GALE_block(nn.Module):
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
        Whether to use Transformer Engine backend. Default is ``True``.
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
    :class:`physicsnemo.experimental.models.geotransolver.GeoTransolver` : Main model using GALE_block.

    Examples
    --------
    >>> import torch
    >>> block = GALE_block(num_heads=8, hidden_dim=256, dropout=0.1, context_dim=32)
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
        use_te: bool = True,
        plus: bool = False,
        context_dim: int = 0,
        spatial_shape: tuple[int, ...] | None = None,
        attention_type: str = "GALE",
        concrete_dropout: bool = False,
        state_mixing_mode: str = "weighted",
    ) -> None:
        super().__init__()

        if use_te and not TE_AVAILABLE:
            raise ImportError(
                "Transformer Engine is not installed. "
                "Please install it with: pip install transformer-engine>=0.1.0"
            )

        self.last_layer = last_layer

        # Layer normalization before attention
        if use_te:
            self.ln_1 = te.LayerNorm(hidden_dim)
        else:
            self.ln_1 = nn.LayerNorm(hidden_dim)

        dim_head = hidden_dim // num_heads
        # First match on attention backend, then on spatial shape
        match attention_type:
            case 'GALE':
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
            case 'GALE_FA':
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