# ignore_header_test
# ruff: noqa: E402
""""""

"""
Transolver model. This code was modified from, https://github.com/thuml/Transolver

The following license is provided from their source,

MIT License

Copyright (c) 2024 THUML @ Tsinghua University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import transformer_engine.pytorch as te  # noqa: F401
from einops import rearrange
from torch.autograd.profiler import record_function
from torch.distributed.tensor.placement_types import Replicate

from physicsnemo.distributed import ShardTensor


class PhysicsAttentionBase(nn.Module, ABC):
    """
    Base class for all physics attention modules.

    Implements key functionality that is common across domains:
    - Slice weighting and computation
    - Attention among slices
    - Deslicing
    - Output Projection

    Each subclass must implement it's own methods for projecting input domain tokens onto the slice space.

    Deliberately, there are not default values for any of the parameters.  It's assumed you will
    assign them in the subclass.

    """

    def __init__(self, dim, heads, dim_head, dropout, slice_num, use_te):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads

        self.scale = dim_head**-0.5

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, 1, heads, 1]) * 0.5)
        self.use_te = use_te

        if self.use_te:
            self.in_project_slice = te.Linear(dim_head, slice_num)
        else:
            self.in_project_slice = nn.Linear(dim_head, slice_num)

        for l_i in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l_i.weight)  # use a principled initialization
        if not use_te:
            self.qkv_project = nn.Linear(dim_head, 3 * dim_head, bias=False)
        else:
            # These are used in the transformer engine pass function:
            self.qkv_project = te.Linear(dim_head, 3 * dim_head, bias=False)
            self.attn_fn = te.DotProductAttention(
                num_attention_heads=self.heads,
                kv_channels=self.dim_head,
                attention_dropout=dropout,
                qkv_format="bshd",
                softmax_scale=self.scale,
            )

        if self.use_te:
            self.out_linear = te.Linear(inner_dim, dim)
        else:
            self.out_linear = nn.Linear(inner_dim, dim)

        self.out_dropout = nn.Dropout(dropout)

    @abstractmethod
    def project_input_onto_slices(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Project the input onto the slice space.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def compute_slices_from_projections(
        self, slice_projections: torch.Tensor, fx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute slice weights and slice tokens from input projections and latent features.

        In a domain-parallel setting, this function will do an implicit allreduce.
        When we sum over the slice_weights over a sharded dimension
        and use the output, it will resolve Partial->Replicated placement (aka
        allreduce) implicitly.

        Args:
            slice_projections (torch.Tensor):
                The projected input tensor of shape [Batch, N_tokens, N_heads, Slice_num],
                representing the projection of each token onto each slice for each attention head.
            fx (torch.Tensor):
                The latent feature tensor of shape [Batch, N_tokens, N_heads, Head_dim],
                representing the learned states to be aggregated by the slice weights.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - slice_weights: Tensor of shape [Batch, N_tokens, N_heads, Slice_num],
                representing the normalized weights for each slice per token and head.
                - slice_token: Tensor of shape [Batch, N_heads, Slice_num, Head_dim],
                representing the aggregated latent features for each slice, head, and batch.

        Notes:
            - The function first computes a temperature-scaled softmax over the slice projections to obtain slice weights.
            - It then aggregates the latent features (fx) for each slice using these weights.
            - The aggregated features are normalized by the sum of weights for numerical stability.
        """

        with record_function("compute_slices_from_projections"):
            # Project the latent space vectors on to the weight computation space,
            # and compute a temperature adjusted softmax.
            clamped_temp = torch.clamp(self.temperature, min=0.5, max=5).to(
                slice_projections.dtype
            )

            slice_weights = nn.functional.softmax(
                slice_projections / clamped_temp, dim=-1
            )  # [Batch, N_tokens, N_heads, Slice_num]

            # Cast to the computation type (since the parameter is probably fp32)
            slice_weights = slice_weights.to(slice_projections.dtype)

            # This does the projection of the latent space fx by the weights:

            # Computing the slice tokens is a matmul followed by a normalization.
            # It can, unfortunately, overflow in reduced precision, so normalize first:
            slice_norm = slice_weights.sum(1)  # [Batch, N_heads, Slice_num]
            # Sharded note: slice_norm will be a partial sum at this point.
            # That's because the we're summing over the tokens, which are distributed
            normed_weights = slice_weights / (slice_norm[:, None, :, :] + 1e-2)
            # Normed weights has shape
            # (batch, n_tokens, n_heads, slice_num)

            # Sharded note: normed_weights will resolve the partial slice_norm
            # and the output normed_weights will be sharded.
            # fx has shape (Batch, n_tokens, n_heads, head_dim)
            # This matmul needs to contract over the tokens
            # This should produce an output with shape
            # [Batch, N_heads, Slice_num, Head_dim]

            # Like the weight norm, this sum is a **partial** sum since we are summing
            # over the tokens

            slice_token = torch.matmul(
                normed_weights.permute(0, 2, 3, 1), fx.permute(0, 2, 1, 3)
            )

            # Return the original weights, not the normed weights:

            return slice_weights, slice_token

    def compute_slice_attention_te(self, slice_tokens: torch.Tensor) -> torch.Tensor:
        """
        TE implementation of slice attention
        """

        qkv = self.qkv_project(slice_tokens)
        qkv = rearrange(qkv, " b h s (t d) -> t b s h d", t=3, d=self.dim_head)
        q_slice_token, k_slice_token, v_slice_token = qkv.unbind(0)

        out_slice_token2 = self.attn_fn(q_slice_token, k_slice_token, v_slice_token)
        out_slice_token2 = rearrange(
            out_slice_token2, "b s (h d) -> b h s d", h=self.heads, d=self.dim_head
        )

        return out_slice_token2

    def compute_slice_attention_sdpa(self, slice_tokens: torch.Tensor) -> torch.Tensor:
        """
        Torch SDPA implementation of slice attention

        Args:
            slice_tokens (torch.Tensor):
                The slice tokens tensor of shape [Batch, N_heads, Slice_num, Head_dim].

        Returns:
            torch.Tensor:
                The output tensor of shape [Batch, N_heads, Slice_num, Head_dim].
        """
        with record_function("compute_slice_attention_sdpa"):
            # In this case we're using ShardTensor, ensure slice_token is *replicated*

            qkv = self.qkv_project(slice_tokens)

            qkv = rearrange(qkv, " b h s (t d) -> b h s t d", t=3, d=self.dim_head)

            if isinstance(qkv, ShardTensor):
                # This will be a differentiable allreduce
                qkv = qkv.redistribute(placements=[Replicate()])

            q_slice_token, k_slice_token, v_slice_token = qkv.unbind(3)

            out_slice_token = torch.nn.functional.scaled_dot_product_attention(
                q_slice_token, k_slice_token, v_slice_token, is_causal=False
            )

            return out_slice_token

    def project_attention_outputs(
        self, out_slice_token: torch.Tensor, slice_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Project the attended slice tokens back onto the original token space.

        Note that in the distributed case, this will have a replicated and
        sharded inputs.  Slice tokens will be replicated, and slice weights will be sharded.

        Args:
            out_slice_token (torch.Tensor):
                The output tensor from the attention mechanism over slices,
                of shape [Batch, N_heads, Slice_num, Head_dim].
            slice_weights (torch.Tensor):
                The slice weights tensor of shape [Batch, N_tokens, N_heads, Slice_num],
                representing the contribution of each slice to each token.

        Returns:
            torch.Tensor:
                The reconstructed output tensor of shape [Batch, N_tokens, N_heads * Head_dim],
                representing the attended features for each token, with all heads concatenated.

        Notes:
            - The function projects the attended slice tokens back to the token space using the slice weights.
            - The output is reshaped to concatenate all attention heads for each token.
        """
        with record_function("project_attention_outputs"):
            # Slice weights has shape (Batch, n_tokens, n_heads, slice_num)
            # Out slice tokens has shape (Batch, n_heads, slice_num, head_dim)
            # The output of this function needs to have shape
            # (Batch, n_tokens, n_channels) == (Batch, n_tokens, n_heads * head_dim)
            # Note that tokens may be sharded, in which case slice_weights
            # is a sharded tensor and out_slice_token is a replicated tensor

            out_x = torch.einsum("bths,bhsd->bthd", slice_weights, out_slice_token)

            # Condense the last two dimensions:
            out_x = rearrange(out_x, "b t h d -> b t (h d)")

            out_x = self.out_linear(out_x)
            return self.out_dropout(out_x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Physics Attention module.

        Input x should have shape of [Batch, N_tokens, N_Channels] ([B, N, C])
        """

        with record_function("forward"):
            # Project the inputs onto learned spaces:
            x_mid, fx_mid = self.project_input_onto_slices(x)

            # Perform the linear projection of learned latent space onto slices:

            slice_projections = self.in_project_slice(x_mid)

            # Slice projections has shape [B, N_tokens, N_head, Head_dim], but head_dim may have changed!

            # Use the slice projections and learned spaces to compute the slices, and their weights:
            slice_weights, slice_tokens = self.compute_slices_from_projections(
                slice_projections, fx_mid
            )
            # slice_weights has shape [Batch, N_tokens, N_heads, Slice_num]
            # slice_tokens has shape  [Batch, N_tokens, N_heads, head_dim]

            # Apply attention to the slice tokens
            if self.use_te:
                out_slice_token = self.compute_slice_attention_te(slice_tokens)
            else:
                out_slice_token = self.compute_slice_attention_sdpa(slice_tokens)

            # Shape unchanged

            # Deslice:
            outputs = self.project_attention_outputs(out_slice_token, slice_weights)

            # Outputs now has the same shape as the original input x

            return outputs


class PhysicsAttentionIrregularMesh(PhysicsAttentionBase):
    """
    Specialization of PhysicsAttention to Irregular Meshes
    """

    def __init__(
        self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64, use_te=True
    ):
        super().__init__(dim, heads, dim_head, dropout, slice_num, use_te)
        inner_dim = dim_head * heads
        if use_te:
            self.in_project_x = te.Linear(dim, inner_dim)
            self.in_project_fx = te.Linear(dim, inner_dim)
        else:
            self.in_project_x = nn.Linear(dim, inner_dim)
            self.in_project_fx = nn.Linear(dim, inner_dim)

    def project_input_onto_slices(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Project the input onto the slice space.

        Args:
            x (torch.Tensor): The input tensor of shape [Batch, N_tokens, N_Channels]

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The projected x and fx tensors of shape [Batch, N_tokens, N_Channels], [Batch, N_tokens, N_heads, Head_dim]

        """
        fx = self.in_project_fx(x)
        fx_mid = rearrange(fx, "B N (h d) -> B N h d", h=self.heads, d=self.dim_head)

        x_mid = rearrange(
            self.in_project_x(x), "B N (h d) -> B N h d", h=self.heads, d=self.dim_head
        )

        return x_mid, fx_mid


class PhysicsAttentionStructuredMesh2D(PhysicsAttentionBase):
    """
    Specialization for 2d image-like meshes

    Only implements the projection onto the slice space.
    """

    def __init__(
        self,
        dim,
        spatial_shape,
        heads=8,
        dim_head=64,
        dropout=0.0,
        slice_num=64,
        kernel=3,
        use_te=True,
    ):  # kernel=3):
        super().__init__(dim, heads, dim_head, dropout, slice_num, use_te)

        inner_dim = dim_head * heads
        self.H = spatial_shape[0]
        self.W = spatial_shape[1]

        self.in_project_x = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)

    def project_input_onto_slices(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        # Rearrange the input tokens back to an image shape:
        b = x.shape[0]
        c = x.shape[-1]

        x = x.view(b, self.H, self.W, c)
        x = x.permute(0, 3, 1, 2)

        # Apply the projections, here they are convolutions in 2D:
        input_projected_fx = self.in_project_fx(x)
        input_projected_x = self.in_project_x(x)

        # Next, re-reshape the projections into token-like shapes:
        input_projected_fx = rearrange(
            input_projected_fx,
            "b (n_heads head_dim) h w -> b (h w) n_heads head_dim",
            head_dim=self.dim_head,
            n_heads=self.heads,
        )
        input_projected_x = rearrange(
            input_projected_x,
            "b (n_heads head_dim) h w -> b (h w) n_heads head_dim",
            head_dim=self.dim_head,
            n_heads=self.heads,
        )

        # Return the projections:
        return input_projected_x, input_projected_fx


class PhysicsAttentionStructuredMesh3D(PhysicsAttentionBase):
    """
    Specialization for 3D-image like meshes

    Only implements the projection onto the slice space.
    """

    def __init__(
        self,
        dim,
        spatial_shape,
        heads=8,
        dim_head=64,
        dropout=0.0,
        slice_num=32,
        kernel=3,
        use_te=True,
    ):
        super().__init__(dim, heads, dim_head, dropout, slice_num, use_te)

        inner_dim = dim_head * heads
        self.H = spatial_shape[0]
        self.W = spatial_shape[1]
        self.D = spatial_shape[2]

        self.in_project_x = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)

    def project_input_onto_slices(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Project the input onto the slice space.

        Input tensor has shape [Batch, N_tokens, N_Channels]
        """

        b = x.shape[0]
        c = x.shape[-1]

        # x = rearrange(x, "b (h w d) c -> b c h w d", h=self.H, w=self.W, d=self.D)
        x = x.view(b, self.H, self.W, self.D, c)
        x = x.permute(0, 4, 1, 2, 3)

        # Apply the projections, here they are convolutions:
        input_projected_fx = self.in_project_fx(x)
        input_projected_x = self.in_project_x(x)

        # Next, re-reshape the projections into token-like shapes:
        input_projected_fx = rearrange(
            input_projected_fx,
            "b (n_heads head_dim) h w d-> b (h w d) n_heads head_dim",
            head_dim=self.dim_head,
            n_heads=self.heads,
        )
        input_projected_x = rearrange(
            input_projected_x,
            "b (n_heads head_dim) h w d -> b (h w d) n_heads head_dim",
            head_dim=self.dim_head,
            n_heads=self.heads,
        )

        return input_projected_x, input_projected_fx
