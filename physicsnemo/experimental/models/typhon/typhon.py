# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange

import physicsnemo  # noqa: F401 for docs
from physicsnemo.utils.version_check import check_min_version
from physicsnemo.models.transolver.Physics_Attention import (
    PhysicsAttentionIrregularMesh,
    gumbel_softmax,
)
from physicsnemo.models.transolver.transolver import MLP

from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module

# Check optional dependency availability
TE_AVAILABLE = check_min_version("transformer-engine", "0.1.0", hard_fail=False)
if TE_AVAILABLE:
    import transformer_engine.pytorch as te

ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


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

    Notes
    -----
    The mixing between self-attention and cross-attention is controlled by a learnable
    parameter ``state_mixing`` which is passed through a sigmoid function to ensure
    the mixing weight stays in \([0, 1]\).

    See Also
    --------
    :class:`physicsnemo.models.transolver.Physics_Attention.PhysicsAttentionIrregularMesh` : Base physics attention class.
    :class:`GALE_block` : Transformer block using GALE attention.
    """

    def __init__(
        self,
        dim,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 64,
        use_te: bool = True,
        plus: bool = False,
        context_dim: int = 0,
    ):
        super().__init__(dim, heads, dim_head, dropout, slice_num, use_te, plus)

        linear_layer = te.Linear if self.use_te else nn.Linear

        # We have additional parameters, here:
        self.cross_q = linear_layer(dim_head, dim_head)
        self.cross_k = linear_layer(context_dim, dim_head)
        self.cross_v = linear_layer(context_dim, dim_head)

        # This is the learnable mixing weight between self and cross attention.
        # We start near 0.0 since it is passed through a sigmoid to keep the
        # mixing weight between 0 and 1.
        self.state_mixing = nn.Parameter(torch.tensor(0.0))

    def compute_slice_attention_cross(
        self, slice_tokens: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        r"""Compute cross-attention between slice tokens and context.

        Parameters
        ----------
        slice_tokens : torch.Tensor
            Slice tokens of shape \((B, H, N, D)\) where \(B\) is batch size, \(H\) is number of heads, \(N\) is number of slices, and \(D\) is head dimension.
        context : torch.Tensor
            Context tensor of shape \((B, H, N_c, D_c)\) where \(N_c\) is number of context slices and \(D_c\) is context dimension.

        Returns
        -------
        torch.Tensor
            Cross-attention output of shape \((B, H, N, D)\).
        """

        # Project the slice and context tokens:
        q = self.cross_q(slice_tokens)
        k = self.cross_k(context)
        v = self.cross_v(context)

        # Compute the attention:
        if self.use_te:
            cross_attention = self.attn_fn(q, k, v)
        else:
            cross_attention = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=False
            )

        return cross_attention

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""Forward pass of the GALE module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape \((B, N, C)\) where \(B\) is batch size, \(N\) is number of tokens, and \(C\) is number of channels.
        context : torch.Tensor, optional
            Context tensor for cross-attention of shape \((B, H, S_c, D_c)\) where \(H\) is number of heads, \(S_c\) is number of context slices, and \(D_c\) is context dimension. If None, only self-attention is applied. Default is None.

        Returns
        -------
        torch.Tensor
            Output tensor of shape \((B, N, C)\), same shape as input.
        """

        # Project the inputs onto learned spaces:
        if self.plus:
            x_mid = self.project_input_onto_slices(x)
            # In transolver ++, fx_mid is gone.
            # x_mid is used to compute the projections instead:
            fx_mid = x_mid
        else:
            x_mid, fx_mid = self.project_input_onto_slices(x)

        # Perform the linear projection of learned latent space onto slices:
        slice_projections = self.in_project_slice(x_mid)

        # Slice projections has shape [B, N_head, N_tokens, Head_dim], but head_dim may have changed!

        # Use the slice projections and learned spaces to compute the slices, and their weights:
        slice_weights, slice_tokens = self.compute_slices_from_projections(
            slice_projections, fx_mid
        )
        # slice_weights has shape [Batch, N_heads, N_tokens, Slice_num]
        # slice_tokens has shape  [Batch, N_heads, N_tokens, head_dim]

        # Apply attention to the slice tokens
        if self.use_te:
            self_slice_token = self.compute_slice_attention_te(slice_tokens)
        else:
            self_slice_token = self.compute_slice_attention_sdpa(slice_tokens)

        # HERE, we are differing: apply cross-attention with physical states:
        if context is not None:
            cross_slice_token = self.compute_slice_attention_cross(
                slice_tokens, context
            )

            # Apply learnable mixing:
            mixing_weight = torch.sigmoid(self.state_mixing)
            out_slice_token = (
                mixing_weight * self_slice_token
                + (1 - mixing_weight) * cross_slice_token
            )

        else:
            # Just keep self attention:
            out_slice_token = self_slice_token

        # Shape unchanged

        # Deslice:
        outputs = self.project_attention_outputs(out_slice_token, slice_weights)

        # Outputs now has the same shape as the original input x

        return outputs


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
        Activation function name. Default is "gelu".
    mlp_ratio : int, optional
        Ratio of MLP hidden dimension to ``hidden_dim``. Default is 4.
    last_layer : bool, optional
        Whether this is the last layer in the model. Default is False.
    out_dim : int, optional
        Output dimension (only used if ``last_layer=True``). Default is 1.
    slice_num : int, optional
        Number of learned physical state slices. Default is 32.
    use_te : bool, optional
        Whether to use Transformer Engine backend. Default is True.
    plus : bool, optional
        Whether to use Transolver++ features. Default is False.
    context_dim : int, optional
        Dimension of the context vector for cross-attention. Default is 0.

    Notes
    -----
    The block applies layer normalization before the attention operation and uses
    residual connections after both the attention and MLP layers.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
        use_te=True,
        plus: bool = False,
        context_dim: int = 0,
    ):
        super().__init__()

        if use_te and not TE_AVAILABLE:
            raise ImportError(
                "Transformer Engine is not installed. Please install it with: pip install transformer-engine>=0.1.0"
            )

        self.last_layer = last_layer
        if use_te:
            self.ln_1 = te.LayerNorm(hidden_dim)
        else:
            self.ln_1 = nn.LayerNorm(hidden_dim)

        self.Attn = GALE(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
            use_te=use_te,
            plus=plus,
            context_dim=context_dim,
        )

        if use_te:
            self.ln_mlp1 = te.LayerNormMLP(
                hidden_size=hidden_dim,
                ffn_hidden_size=hidden_dim * mlp_ratio,
            )
        else:
            self.ln_mlp1 = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                MLP(
                    hidden_dim,
                    hidden_dim * mlp_ratio,
                    hidden_dim,
                    n_layers=0,
                    res=False,
                    act=act,
                    use_te=False,
                ),
            )

    def forward(self, fx: torch.Tensor, global_context: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the GALE block.

        Parameters
        ----------
        fx : torch.Tensor
            Input tensor of shape \((B, N, C)\) where \(B\) is batch size, \(N\) is number of tokens, and \(C\) is hidden dimension.
        global_context : torch.Tensor
            Global context tensor for cross-attention of shape \((B, H, S_c, D_c)\) where \(H\) is number of heads, \(S_c\) is number of context slices, and \(D_c\) is context dimension.

        Returns
        -------
        torch.Tensor
            Output tensor of shape \((B, N, C)\), same shape as input.
        """
        fx = self.Attn(self.ln_1(fx), global_context) + fx
        fx = self.ln_mlp1(fx) + fx

        return fx


@dataclass
class TyphonMetaData(ModelMetaData):
    """
    Data class for storing essential meta data needed for the Typhon model.
    """

    name: str = "Typhon"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = True
    # Inference
    onnx_cpu: bool = False  # No FFT op on CPU
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class ContextProjector(nn.Module):
    r"""Projects context features onto physical state space.

    This context projector is conceptually similar to half of a GALE attention layer.
    It projects context values (geometry or global embeddings) onto a learned physical
    state space, but unlike a full attention layer, it never projects back to the
    original space. The projected features are used as context in all GALE blocks
    of the Typhon model.

    Parameters
    ----------
    dim : int
        Input dimension of the context features.
    heads : int, optional
        Number of projection heads. Default is 8.
    dim_head : int, optional
        Dimension of each projection head. Default is 64.
    dropout : float, optional
        Dropout rate. Default is 0.0.
    slice_num : int, optional
        Number of learned physical state slices. Default is 64.
    use_te : bool, optional
        Whether to use Transformer Engine backend when available. Default is True.
    plus : bool, optional
        Whether to use Transolver++ features. Default is False.

    Notes
    -----
    The global features are reused in all blocks of the model, so the learned
    projections must capture globally useful features rather than layer-specific ones.

    See Also
    --------
    :class:`GALE` : Full GALE attention layer that uses these projected context features.
    :class:`Typhon` : Main model that uses ContextProjector for geometry and global embeddings.
    """

    def __init__(
        self,
        dim,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 64,
        use_te: bool = True,
        plus: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.plus = plus
        self.scale = dim_head**-0.5
        self.use_te = use_te

        # Keep below here:
        if use_te:
            self.in_project_x = te.Linear(dim, inner_dim)
            if not plus:
                self.in_project_fx = te.Linear(dim, inner_dim)
        else:
            self.in_project_x = nn.Linear(dim, inner_dim)
            if not plus:
                self.in_project_fx = nn.Linear(dim, inner_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        if plus:
            linear_layer = te.Linear if self.use_te else nn.Linear
            self.proj_temperature = torch.nn.Sequential(
                linear_layer(self.dim_head, slice_num),
                nn.GELU(),
                linear_layer(slice_num, 1),
                nn.GELU(),
            )

        if self.use_te:
            self.in_project_slice = te.Linear(dim_head, slice_num)
        else:
            self.in_project_slice = nn.Linear(dim_head, slice_num)

    def project_input_onto_slices(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r"""Project the input onto the slice space.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape \((B, N, C)\) where \(B\) is batch size, \(N\) is number of tokens, and \(C\) is number of channels.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            If ``plus=True``, returns single tensor ``x_mid`` of shape \((B, H, N, D)\) where \(H\) is number of heads and \(D\) is head dimension. If ``plus=False``, returns tuple ``(x_mid, fx_mid)`` both of shape \((B, H, N, D)\).
        """
        x_mid = rearrange(
            self.in_project_x(x), "B N (h d) -> B h N d", h=self.heads, d=self.dim_head
        )
        if self.plus:
            return x_mid
        else:
            fx_mid = rearrange(
                self.in_project_fx(x),
                "B N (h d) -> B h N d",
                h=self.heads,
                d=self.dim_head,
            )

            return x_mid, fx_mid

    def compute_slices_from_projections(
        self, slice_projections: torch.Tensor, fx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Compute slice weights and slice tokens from input projections and latent features.

        Parameters
        ----------
        slice_projections : torch.Tensor
            Projected input tensor of shape \((B, N, H, S)\) where \(B\) is batch size, \(H\) is number of heads, \(N\) is number of tokens, and \(S\) is number of slices, representing the projection of each token onto each slice for each attention head.
        fx : torch.Tensor
            Latent feature tensor of shape \((B, N, H, D)\) where \(D\) is head dimension, representing the learned states to be aggregated by the slice weights.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - ``slice_weights``: Tensor of shape \((B, N, H, S)\), representing the normalized weights for each slice per token and head.
            - ``slice_token``: Tensor of shape \((B, H, S, D)\), representing the aggregated latent features for each slice, head, and batch.

        Notes
        -----
        The function computes a temperature-scaled softmax over the slice projections to obtain
        slice weights, then aggregates the latent features for each slice using these weights.
        The aggregated features are normalized by the sum of weights for numerical stability.
        """

        # Project the latent space vectors on to the weight computation space,
        # and compute a temperature adjusted softmax.

        if self.plus:
            temperature = self.temperature + self.proj_temperature(fx)
            clamped_temp = torch.clamp(temperature, min=0.01).to(
                slice_projections.dtype
            )
            slice_weights = gumbel_softmax(
                slice_projections, clamped_temp
            )  # [Batch, N_heads, N_tokens, Slice_num]

        else:
            clamped_temp = torch.clamp(self.temperature, min=0.5, max=5).to(
                slice_projections.dtype
            )
            slice_weights = nn.functional.softmax(
                slice_projections / clamped_temp, dim=-1
            )  # [Batch, N_heads, N_tokens, Slice_num]

        # Cast to the computation type (since the parameter is probably fp32)
        slice_weights = slice_weights.to(slice_projections.dtype)

        # This does the projection of the latent space fx by the weights:

        # Computing the slice tokens is a matmul followed by a normalization.
        # It can, unfortunately, overflow in reduced precision, so normalize first:
        slice_norm = slice_weights.sum(2)  # [Batch, N_heads, Slice_num]
        normed_weights = slice_weights / (slice_norm[:, :, None, :] + 1e-2)
        slice_token = torch.matmul(normed_weights.transpose(2, 3), fx)

        # Return the original weights, not the normed weights:
        return slice_weights, slice_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Reduced forward pass projecting inputs to physical state slices.

        This performs a partial physics attention operation: it projects the input onto
        learned physical state slices but does not project back to the original space.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape \((B, N, C)\) where \(B\) is batch size, \(N\) is number of tokens, and \(C\) is number of channels.

        Returns
        -------
        torch.Tensor
            Slice tokens of shape \((B, H, S, D)\) where \(H\) is number of heads, \(S\) is number of slices, and \(D\) is head dimension.
        """

        # All of this is derived from the PhysicsAttention Layer

        # Project the inputs onto learned spaces:
        if self.plus:
            x_mid = self.project_input_onto_slices(x)
            # In transolver ++, fx_mid is gone.
            # x_mid is used to compute the projections instead:
            fx_mid = x_mid
        else:
            x_mid, fx_mid = self.project_input_onto_slices(x)

        # Perform the linear projection of learned latent space onto slices:
        slice_projections = self.in_project_slice(x_mid)

        # Slice projections has shape [B, N_head, N_tokens, Head_dim], but head_dim may have changed!

        # Use the slice projections and learned spaces to compute the slices, and their weights:
        _, slice_tokens = self.compute_slices_from_projections(
            slice_projections, fx_mid
        )
        # _ has shape [Batch, N_heads, N_tokens, Slice_num]
        # slice_tokens has shape  [Batch, N_heads, N_tokens, head_dim]

        return slice_tokens


class Typhon(Module):
    r"""Typhon: Geometry-Aware Physics Attention Transformer.

    Typhon is an adaptation of the Transolver architecture, replacing standard attention
    with GALE (Geometry-Aware Latent Embeddings) attention. GALE combines physics-aware
    self-attention on learned state slices with cross-attention to geometry and global
    context embeddings.

    The model projects geometry and global features onto physical state spaces, which are
    then used as context in all transformer blocks. This design enables the model to
    incorporate geometric structure and global information throughout the forward pass.

    Parameters
    ----------
    functional_dim : int
        Dimension of the input values (local embeddings), not including global embeddings or geometry features. Input will be projected to ``n_hidden`` before processing.
    out_dim : int
        Dimension of the output of the model.
    geometry_dim : int, optional
        Pointwise dimension of the geometry input features. If provided, geometry features will be projected onto physical states and used as context in all GALE layers. Default is None.
    global_dim : int, optional
        Dimension of the global embedding features. If provided, global features will be projected onto physical states and used as context in all GALE layers. Default is None.
    n_layers : int, optional
        Number of GALE layers in the model. Default is 4.
    n_hidden : int, optional
        Hidden dimension of the transformer. Default is 256.
    dropout : float, optional
        Dropout rate applied across the GALE layers. Default is 0.0.
    n_head : int, optional
        Number of attention heads in each GALE layer. Must evenly divide ``n_hidden`` to yield an integer head dimension. Default is 8.
    act : str, optional
        Activation function name. Default is "gelu".
    mlp_ratio : int, optional
        Ratio of MLP hidden dimension to ``n_hidden``. Default is 4.
    slice_num : int, optional
        Number of learned physical state slices in the GALE layers, representing the number of learned states each layer should project inputs onto. Default is 32.
    use_te : bool, optional
        Whether to use Transformer Engine backend when available. Default is True.
    time_input : bool, optional
        Whether to include time embeddings. Default is False.
    plus : bool, optional
        Whether to use Transolver++ features in the GALE layers. Default is False.

    Raises
    ------
    ValueError
        If ``n_hidden`` is not evenly divisible by ``n_head``.


    Forward
    ----------
    local_embedding : torch.Tensor
        Local embedding of the input data of shape \((B, N, C)\) where \(B\) is batch size, \(N\) is number of nodes/tokens, and \(C\) is ``functional_dim``. Output will have the same \((B, N)\) shape but with ``out_dim`` channels.
    global_embedding : torch.Tensor, optional
        Global embedding of the input data of shape \((B, N_g, C_g)\) where \(N_g\) is number of global tokens and \(C_g\) is ``global_dim``. If None, global context is not used. Default is None.
    geometry : torch.Tensor, optional
        Geometry features of the input data of shape \((B, N, C_{geo})\) where \(C_{geo}\) is ``geometry_dim``. If None, geometry context is not used. Default is None.
    time : torch.Tensor, optional
        Time embedding (currently not implemented). Default is None.

    Returns
    -------
    torch.Tensor
        Output tensor of shape \((B, N, C_{out})\) where \(C_{out}\) is ``out_dim``.

    Notes
    -----
    Typhon currently supports unstructured mesh input only. Enhancements for image-based
    and voxel-based inputs may be available in the future.

    For more details on Transolver, see:
    - https://arxiv.org/pdf/2402.02366
    - https://arxiv.org/pdf/2502.02414

    See Also
    --------
    :class:`GALE` : The attention mechanism used in Typhon.
    :class:`GALE_block` : Transformer block using GALE attention.
    :class:`ContextProjector` : Projects context features onto physical states.

    Examples
    --------
    Basic usage with local embeddings only:

    >>> import torch
    >>> import physicsnemo
    >>> model = physicsnemo.models.Typhon(
    ...     functional_dim=64,
    ...     out_dim=3,
    ...     n_hidden=256,
    ...     n_layers=4
    ... )
    >>> local_emb = torch.randn(2, 1000, 64)  # (batch, nodes, features)
    >>> output = model(local_emb)
    >>> output.shape
    torch.Size([2, 1000, 3])

    Usage with geometry and global context:

    >>> model = physicsnemo.models.Typhon(
    ...     functional_dim=64,
    ...     out_dim=3,
    ...     geometry_dim=3,
    ...     global_dim=16,
    ...     n_hidden=256,
    ...     n_layers=4
    ... )
    >>> local_emb = torch.randn(2, 1000, 64)
    >>> geometry = torch.randn(2, 1000, 3)  # (batch, nodes, spatial_dim)
    >>> global_emb = torch.randn(2, 1, 16)  # (batch, 1, global_features)
    >>> output = model(local_emb, global_embedding=global_emb, geometry=geometry)
    >>> output.shape
    torch.Size([2, 1000, 3])
    """

    def __init__(
        self,
        functional_dim: int,
        out_dim: int,
        geometry_dim: int | None = None,
        global_dim: int | None = None,
        n_layers: int = 4,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 8,
        act: str = "gelu",
        mlp_ratio: int = 4,
        slice_num: int = 32,
        use_te: bool = True,
        time_input: bool = False,
        plus: bool = False,
    ) -> None:
        super().__init__(meta=TyphonMetaData())
        self.__name__ = "Typhon"

        self.use_te = use_te
        # Check that the hidden dimension and head dimensions are compatible:
        if not n_hidden % n_head == 0:
            raise ValueError(
                f"Typhon requires n_hidden % n_head == 0, but instead got {n_hidden % n_head}"
            )

        # These are to project geometry embeddings and global embeddings onto
        # a physical state space:
        context_dim = 0
        if geometry_dim is not None:
            self.geometry_tokenizer = ContextProjector(
                geometry_dim,
                n_head,
                n_hidden // n_head,
                dropout,
                slice_num,
                use_te,
                plus,
            )
            context_dim += n_hidden // n_head
        if global_dim is not None:
            self.global_tokenizer = ContextProjector(
                global_dim, n_head, n_hidden // n_head, dropout, slice_num, use_te, plus
            )
            context_dim += n_hidden // n_head

        # This MLP is the initial projection onto the hidden space
        self.preprocess = MLP(
            functional_dim,
            n_hidden * 2,
            n_hidden,
            n_layers=0,
            res=False,
            act=act,
            use_te=use_te,
        )

        self.n_hidden = n_hidden

        self.blocks = nn.ModuleList(
            [
                GALE_block(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    slice_num=slice_num,
                    last_layer=(_ == n_layers - 1),
                    use_te=use_te,
                    plus=plus,
                    context_dim=context_dim,
                )
                for _ in range(n_layers)
            ]
        )

        if use_te:
            self.ln_mlp_out = te.LayerNormLinear(
                in_features=n_hidden, out_features=out_dim
            )
        else:
            self.ln_mlp_out = nn.Sequential(
                nn.LayerNorm(n_hidden),
                nn.Linear(n_hidden, out_dim),
            )

        self.time_input = time_input
        if time_input:
            self.time_fc = nn.Sequential(
                nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden)
            )

    def forward(
        self,
        local_embedding: torch.Tensor,
        global_embedding: torch.Tensor | None = None,
        geometry: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Forward pass of the Typhon model.

        The model constructs global context embeddings from geometry and global features by
        projecting them onto physical state spaces. These context embeddings are then used
        in all GALE blocks via cross-attention, allowing geometric and global information to
        guide the learned physical state dynamics.

        """

        # First, construct the global context vectors:
        global_context_input = []

        if geometry is not None:
            geometry_states = self.geometry_tokenizer(geometry)
            global_context_input.append(geometry_states)

        if global_embedding is not None:
            global_states = self.global_tokenizer(global_embedding)
            global_context_input.append(global_states)

        # Construct the embedding states:
        if len(global_context_input) > 0:
            embedding_states = torch.cat(global_context_input, dim=-1)

        # Project the inputs to the hidden dimension:
        x = self.preprocess(local_embedding)

        for block in self.blocks:
            x = block(x, embedding_states)

        # Now, pass the data through the model:
        x = self.ln_mlp_out(x)

        return x
