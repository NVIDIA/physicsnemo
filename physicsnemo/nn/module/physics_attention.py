# ignore_header_test
# ruff: noqa: E402

r"""
Physics attention modules for the Transolver model.

This module provides physics-informed attention mechanisms that project inputs
onto learned physics slices before applying attention. These attention variants
support irregular meshes, 2D structured grids, and 3D volumetric data.

This code was modified from https://github.com/thuml/Transolver

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

import importlib
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float
from torch.autograd.profiler import record_function
from torch.distributed.tensor.placement_types import Replicate

from physicsnemo.core.version_check import check_version_spec
from physicsnemo.nn import gumbel_softmax

# Note: We use duck typing to check for ShardTensor instead of importing it
# directly to avoid circular imports (domain_parallel imports from nn).
# ShardTensor has a `redistribute` method that we check for.

TE_AVAILABLE = check_version_spec("transformer_engine", hard_fail=False)

if TE_AVAILABLE:
    te = importlib.import_module("transformer_engine.pytorch")
else:
    te = None


class PhysicsAttentionBase(nn.Module, ABC):
    r"""
    Base class for physics attention modules.

    This class implements the core physics attention mechanism that projects
    inputs onto learned physics-informed slices before applying attention.
    Subclasses implement domain-specific input projections.

    The physics attention mechanism consists of:

    1. Project inputs onto learned slice space
    2. Compute slice weights via temperature-scaled softmax
    3. Aggregate features for each slice
    4. Apply attention among slices
    5. Project attended features back to original space

    Parameters
    ----------
    dim : int
        Input feature dimension.
    heads : int
        Number of attention heads.
    dim_head : int
        Dimension per attention head.
    dropout : float
        Dropout rate.
    slice_num : int
        Number of physics slices.
    use_te : bool
        Whether to use transformer engine.
    plus : bool
        Whether to use Transolver++ variant.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, N, C)` where :math:`B` is batch size,
        :math:`N` is number of tokens, and :math:`C` is feature dimension.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, N, C)`.

    See Also
    --------
    This is an abstract base class. Use one of the concrete implementations:

    - :class:`PhysicsAttentionIrregularMesh` for unstructured mesh data
    - :class:`PhysicsAttentionStructuredMesh2D` for 2D image-like data
    - :class:`PhysicsAttentionStructuredMesh3D` for 3D volumetric data
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float,
        slice_num: int,
        use_te: bool,
        plus: bool,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.plus = plus
        self.scale = dim_head**-0.5
        self.use_te = use_te

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # Learnable temperature parameter for slice weighting
        self.temperature = nn.Parameter(torch.ones([1, 1, heads, 1]) * 0.5)

        if plus:
            # Transolver++ uses learned temperature projection
            linear_layer = te.Linear if self.use_te else nn.Linear
            self.proj_temperature = torch.nn.Sequential(
                linear_layer(self.dim_head, slice_num),
                nn.GELU(),
                linear_layer(slice_num, 1),
                nn.GELU(),
            )

        # Projection from head dimension to slice space
        if self.use_te:
            self.in_project_slice = te.Linear(dim_head, slice_num)
        else:
            self.in_project_slice = nn.Linear(dim_head, slice_num)

        # Initialize with orthogonal weights for better slice diversity
        for l_i in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l_i.weight)

        # QKV projection for slice attention
        if not use_te:
            self.qkv_project = nn.Linear(dim_head, 3 * dim_head, bias=False)
        else:
            self.qkv_project = te.Linear(dim_head, 3 * dim_head, bias=False)
            self.attn_fn = te.DotProductAttention(
                num_attention_heads=self.heads,
                kv_channels=self.dim_head,
                attention_dropout=dropout,
                qkv_format="bshd",
                softmax_scale=self.scale,
            )

        # Output projection
        if self.use_te:
            self.out_linear = te.Linear(inner_dim, dim)
        else:
            self.out_linear = nn.Linear(inner_dim, dim)

        self.out_dropout = nn.Dropout(dropout)

    @abstractmethod
    def project_input_onto_slices(
        self, x: Float[torch.Tensor, "B N C"]
    ) -> (
        Float[torch.Tensor, "B N H D"]
        | tuple[
            Float[torch.Tensor, "B N H D"],
            Float[torch.Tensor, "B N H D"],
        ]
    ):
        r"""
        Project input tensor onto the slice space.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(B, N, C)`.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            For Transolver++: single projected tensor of shape
            :math:`(B, N, H, D)` where :math:`H` is number of attention heads
            and :math:`D` is dimension per head.
            For standard Transolver: tuple of (x_mid, fx_mid) both of shape
            :math:`(B, N, H, D)`.
        """
        ...

    def _compute_slices_from_projections(
        self,
        slice_projections: Float[torch.Tensor, "B N H S"],
        fx: Float[torch.Tensor, "B N H D"],
    ) -> tuple[
        Float[torch.Tensor, "B N H S"],
        Float[torch.Tensor, "B H S D"],
    ]:
        r"""
        Compute slice weights and slice tokens from input projections.

        This method computes soft assignments of tokens to physics slices using
        temperature-scaled softmax, then aggregates features for each slice.

        In domain-parallel settings, this performs an implicit allreduce when
        summing over the sharded token dimension.

        Parameters
        ----------
        slice_projections : torch.Tensor
            Projected input of shape :math:`(B, N, H, S)` where :math:`H` is
            number of attention heads and :math:`S` is number of physics slices.
        fx : torch.Tensor
            Latent features of shape :math:`(B, N, H, D)` where :math:`D` is
            dimension per head.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - ``slice_weights``: Shape :math:`(B, N, H, S)`, normalized weights
              for each slice per token.
            - ``slice_token``: Shape :math:`(B, H, S, D)`, aggregated features
              per slice.
        """
        # Compute temperature-scaled softmax over slices
        if self.plus:
            # Transolver++ uses learned per-token temperature
            temperature = self.temperature + self.proj_temperature(fx)
            clamped_temp = torch.clamp(temperature, min=0.01).to(
                slice_projections.dtype
            )
            slice_weights = gumbel_softmax(
                slice_projections, clamped_temp
            )  # (B, N, H, S)
        else:
            # Standard Transolver uses global temperature
            clamped_temp = torch.clamp(self.temperature, min=0.5, max=5).to(
                slice_projections.dtype
            )
            slice_weights = nn.functional.softmax(
                slice_projections / clamped_temp, dim=-1
            )  # (B, N, H, S)

        # Cast to the computation type (since the parameter is probably fp32)
        slice_weights = slice_weights.to(slice_projections.dtype)

        # This does the projection of the latent space fx by the weights:

        # Computing the slice tokens is a matmul followed by a normalization.
        # It can, unfortunately, overflow in reduced precision, so normalize first:
        slice_norm = slice_weights.sum(1) + 1e-2  # (B, H, S)
        # Sharded note: slice_norm will be a partial sum at this point.
        # That's because the we're summing over the tokens, which are distributed
        normed_weights = slice_weights / (slice_norm[:, None, :, :])
        # Normed weights has shape (B, N, H, S)

        # Sharded note: normed_weights will resolve the partial slice_norm
        # and the output normed_weights will be sharded.
        # fx has shape (B, N, H, D)
        # This matmul needs to contract over the tokens
        # This should produce an output with shape (B, H, S, D)

        # Like the weight norm, this sum is a **partial** sum since we are summing
        # over the tokens

        # Aggregate features: (B, N, H, S)^T @ (B, N, H, D) -> (B, H, S, D)
        slice_token = torch.matmul(
            normed_weights.permute(0, 2, 3, 1), fx.permute(0, 2, 1, 3)
        )

        # Return the original weights, not the normed weights:

        return slice_weights, slice_token

    def _compute_slice_attention_te(
        self, slice_tokens: Float[torch.Tensor, "B H S D"]
    ) -> Float[torch.Tensor, "B H S D"]:
        r"""
        Compute attention among slices using Transformer Engine.

        Parameters
        ----------
        slice_tokens : torch.Tensor
            Slice features of shape :math:`(B, H, S, D)`.

        Returns
        -------
        torch.Tensor
            Attended slice features of shape :math:`(B, H, S, D)`.
        """
        # Project to Q, K, V
        qkv = self.qkv_project(slice_tokens)
        qkv = rearrange(qkv, "b h s (t d) -> t b s h d", t=3, d=self.dim_head)
        q_slice_token, k_slice_token, v_slice_token = qkv.unbind(0)

        # Apply TE attention
        out_slice_token = self.attn_fn(q_slice_token, k_slice_token, v_slice_token)
        out_slice_token = rearrange(
            out_slice_token, "b s (h d) -> b h s d", h=self.heads, d=self.dim_head
        )

        return out_slice_token

    def _compute_slice_attention_sdpa(
        self, slice_tokens: Float[torch.Tensor, "B H S D"]
    ) -> Float[torch.Tensor, "B H S D"]:
        r"""
        Compute attention among slices using PyTorch SDPA.

        Parameters
        ----------
        slice_tokens : torch.Tensor
            Slice features of shape :math:`(B, H, S, D)`.

        Returns
        -------
        torch.Tensor
            Attended slice features of shape :math:`(B, H, S, D)`.
        """
        with record_function("_compute_slice_attention_sdpa"):
            # In this case we're using ShardTensor, ensure slice_token is *replicated*

            qkv = self.qkv_project(slice_tokens)

            qkv = rearrange(qkv, " b h s (t d) -> b h s t d", t=3, d=self.dim_head)

            # Use duck typing to check for ShardTensor to avoid circular import
            # (domain_parallel imports from nn, so nn cannot import from domain_parallel)
            if hasattr(qkv, "redistribute"):
                # This will be a differentiable allreduce
                qkv = qkv.redistribute(placements=[Replicate()])

            q_slice_token, k_slice_token, v_slice_token = qkv.unbind(3)

            # Apply scaled dot-product attention
            out_slice_token = torch.nn.functional.scaled_dot_product_attention(
                q_slice_token, k_slice_token, v_slice_token, is_causal=False
            )

            return out_slice_token

    def _project_attention_outputs(
        self,
        out_slice_token: Float[torch.Tensor, "B H S D"],
        slice_weights: Float[torch.Tensor, "B N H S"],
    ) -> Float[torch.Tensor, "B N C"]:
        r"""
        Project attended slice features back to token space.

        In distributed settings, ``out_slice_token`` is replicated while
        ``slice_weights`` may be sharded over tokens.

        Parameters
        ----------
        out_slice_token : torch.Tensor
            Attended slice features of shape :math:`(B, H, S, D)`.
        slice_weights : torch.Tensor
            Slice weights of shape :math:`(B, N, H, S)`.

        Returns
        -------
        torch.Tensor
            Output features of shape :math:`(B, N, C)` where :math:`C = H \cdot D`.
        """
        with record_function("_project_attention_outputs"):
            # Weighted combination: (B, N, H, S) @ (B, H, S, D) -> (B, N, H, D)
            out_x = torch.einsum("bths,bhsd->bthd", slice_weights, out_slice_token)

            # Concatenate heads: (B, N, H, D) -> (B, N, C) where C = H*D
            out_x = rearrange(out_x, "b t h d -> b t (h d)")

            # Output projection with dropout
            out_x = self.out_linear(out_x)
            return self.out_dropout(out_x)

    def forward(self, x: Float[torch.Tensor, "B N C"]) -> Float[torch.Tensor, "B N C"]:
        r"""
        Forward pass of physics attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(B, N, C)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape :math:`(B, N, C)`.
        """
        # Input validation (skip during torch.compile)
        if not torch.compiler.is_compiling():
            if x.ndim != 3:
                raise ValueError(
                    f"Expected 3D input tensor (B, N, C), "
                    f"got {x.ndim}D tensor with shape {tuple(x.shape)}"
                )

            if x.shape[-1] != self.dim:
                raise ValueError(
                    f"Expected input feature dimension {self.dim}, got {x.shape[-1]}"
                )

        # Project inputs onto learned spaces
        projected = self.project_input_onto_slices(x)
        if self.plus:
            x_mid = projected
            fx_mid = x_mid  # Transolver++ reuses x_mid
        else:
            x_mid, fx_mid = projected  # type: ignore[misc]

        # Project onto slice space
        slice_projections = self.in_project_slice(x_mid)
        # slice_projections: (B, N, H, S)

        # Compute slice weights and aggregate features per slice
        slice_weights, slice_tokens = self._compute_slices_from_projections(
            slice_projections, fx_mid
        )
        # slice_weights: (B, N, H, S)
        # slice_tokens: (B, H, S, D)

        # Apply attention among slices
        if self.use_te:
            out_slice_token = self._compute_slice_attention_te(slice_tokens)
        else:
            out_slice_token = self._compute_slice_attention_sdpa(slice_tokens)
        # out_slice_token: (B, H, S, D)

        # Project back to token space
        outputs = self._project_attention_outputs(out_slice_token, slice_weights)
        # outputs: (B, N, C)

        return outputs


class PhysicsAttentionIrregularMesh(PhysicsAttentionBase):
    r"""
    Physics attention for irregular/unstructured mesh data.

    Uses linear projections to map input tokens to the slice space, suitable
    for meshes with arbitrary connectivity.

    Parameters
    ----------
    dim : int
        Input feature dimension.
    heads : int, optional, default=8
        Number of attention heads.
    dim_head : int, optional, default=64
        Dimension per attention head.
    dropout : float, optional, default=0.0
        Dropout rate.
    slice_num : int, optional, default=64
        Number of physics slices.
    use_te : bool, optional, default=True
        Whether to use transformer engine.
    plus : bool, optional, default=False
        Whether to use Transolver++ variant.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, N, C)` where :math:`B` is batch size,
        :math:`N` is number of tokens, and :math:`C` is feature dimension.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, N, C)`.

    Examples
    --------
    >>> import torch
    >>> attn = PhysicsAttentionIrregularMesh(dim=128, heads=4, dim_head=32, dropout=0.0, slice_num=16, use_te=False)
    >>> x = torch.randn(2, 1000, 128)
    >>> out = attn(x)
    >>> out.shape
    torch.Size([2, 1000, 128])
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
    ):
        super().__init__(dim, heads, dim_head, dropout, slice_num, use_te, plus)
        inner_dim = dim_head * heads

        # Linear projections for irregular mesh data
        if use_te:
            self.in_project_x = te.Linear(dim, inner_dim)
            if not plus:
                self.in_project_fx = te.Linear(dim, inner_dim)
        else:
            self.in_project_x = nn.Linear(dim, inner_dim)
            if not plus:
                self.in_project_fx = nn.Linear(dim, inner_dim)

    def project_input_onto_slices(
        self, x: Float[torch.Tensor, "B N C"]
    ) -> (
        Float[torch.Tensor, "B N H D"]
        | tuple[
            Float[torch.Tensor, "B N H D"],
            Float[torch.Tensor, "B N H D"],
        ]
    ):
        r"""
        Project input onto slice space using linear layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(B, N, C)`.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Projected tensors of shape :math:`(B, N, H, D)` where :math:`H` is
            number of attention heads and :math:`D` is dimension per head.
        """
        # Project and reshape to multi-head format
        x_mid = rearrange(
            self.in_project_x(x), "B N (H D) -> B N H D", H=self.heads, D=self.dim_head
        )

        if self.plus:
            return x_mid
        else:
            fx_mid = rearrange(
                self.in_project_fx(x),
                "B N (H D) -> B N H D",
                H=self.heads,
                D=self.dim_head,
            )
            return x_mid, fx_mid


class PhysicsAttentionStructuredMesh2D(PhysicsAttentionBase):
    r"""
    Physics attention for 2D structured/image-like data.

    Uses 2D convolutions to project inputs, leveraging spatial locality in
    structured grids.

    Parameters
    ----------
    dim : int
        Input feature dimension.
    spatial_shape : tuple[int, int]
        Spatial dimensions (height, width) of the input.
    heads : int, optional, default=8
        Number of attention heads.
    dim_head : int, optional, default=64
        Dimension per attention head.
    dropout : float, optional, default=0.0
        Dropout rate.
    slice_num : int, optional, default=64
        Number of physics slices.
    kernel : int, optional, default=3
        Convolution kernel size.
    use_te : bool, optional, default=True
        Whether to use transformer engine.
    plus : bool, optional, default=False
        Whether to use Transolver++ variant.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, N, C)` where :math:`B` is batch size,
        :math:`N` is number of tokens (flattened spatial: height times width),
        and :math:`C` is feature dimension.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, N, C)`.

    Examples
    --------
    >>> import torch
    >>> attn = PhysicsAttentionStructuredMesh2D(
    ...     dim=128,
    ...     spatial_shape=(32, 32),
    ...     heads=4,
    ...     dim_head=32,
    ...     dropout=0.0,
    ...     slice_num=16,
    ...     use_te=False,
    ... )
    >>> x = torch.randn(2, 32*32, 128)
    >>> out = attn(x)
    >>> out.shape
    torch.Size([2, 1024, 128])
    """

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
    ):
        super().__init__(dim, heads, dim_head, dropout, slice_num, use_te, plus)

        inner_dim = dim_head * heads
        self.H = spatial_shape[0]
        self.W = spatial_shape[1]

        # 2D convolution projections
        self.in_project_x = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        if not plus:
            self.in_project_fx = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)

    def project_input_onto_slices(
        self, x: Float[torch.Tensor, "B N C"]
    ) -> (
        Float[torch.Tensor, "B N H D"]
        | tuple[
            Float[torch.Tensor, "B N H D"],
            Float[torch.Tensor, "B N H D"],
        ]
    ):
        r"""
        Project input onto slice space using 2D convolutions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(B, N, C)` where :math:`N` is the
            flattened spatial dimension (height times width).

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Projected tensors of shape :math:`(B, N, H, D)` where :math:`H` is
            number of attention heads and :math:`D` is dimension per head.
        """
        B = x.shape[0]
        C = x.shape[-1]

        # Reshape from flattened to 2D spatial format: (B, N, C) -> (B, C, H_s, W_s)
        x = x.view(B, self.H, self.W, C)
        x = x.permute(0, 3, 1, 2)

        # Apply 2D convolution and reshape to multi-head format
        input_projected_x = self.in_project_x(x)
        input_projected_x = rearrange(
            input_projected_x,
            "B (H D) h w -> B (h w) H D",
            D=self.dim_head,
            H=self.heads,
        )

        if self.plus:
            return input_projected_x
        else:
            input_projected_fx = self.in_project_fx(x)
            input_projected_fx = rearrange(
                input_projected_fx,
                "B (H D) h w -> B (h w) H D",
                D=self.dim_head,
                H=self.heads,
            )
            return input_projected_x, input_projected_fx


class PhysicsAttentionStructuredMesh3D(PhysicsAttentionBase):
    r"""
    Physics attention for 3D structured/volumetric data.

    Uses 3D convolutions to project inputs, suitable for voxel-based
    representations.

    Parameters
    ----------
    dim : int
        Input feature dimension.
    spatial_shape : tuple[int, int, int]
        Spatial dimensions (height, width, depth) of the input.
    heads : int, optional, default=8
        Number of attention heads.
    dim_head : int, optional, default=64
        Dimension per attention head.
    dropout : float, optional, default=0.0
        Dropout rate.
    slice_num : int, optional, default=32
        Number of physics slices.
    kernel : int, optional, default=3
        Convolution kernel size.
    use_te : bool, optional, default=True
        Whether to use transformer engine.
    plus : bool, optional, default=False
        Whether to use Transolver++ variant.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, N, C)` where :math:`B` is batch size,
        :math:`N` is number of tokens (flattened spatial: height times width
        times depth), and :math:`C` is feature dimension.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, N, C)`.

    Examples
    --------
    >>> import torch
    >>> attn = PhysicsAttentionStructuredMesh3D(
    ...     dim=64,
    ...     spatial_shape=(16, 16, 16),
    ...     heads=4,
    ...     dim_head=16,
    ...     dropout=0.0,
    ...     slice_num=8,
    ...     use_te=False,
    ... )
    >>> x = torch.randn(2, 16*16*16, 64)
    >>> out = attn(x)
    >>> out.shape
    torch.Size([2, 4096, 64])
    """

    def __init__(
        self,
        dim: int,
        spatial_shape: tuple[int, int, int],
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 32,
        kernel: int = 3,
        use_te: bool = True,
        plus: bool = False,
    ):
        super().__init__(dim, heads, dim_head, dropout, slice_num, use_te, plus)

        inner_dim = dim_head * heads
        self.H = spatial_shape[0]
        self.W = spatial_shape[1]
        self.D = spatial_shape[2]

        # 3D convolution projections
        self.in_project_x = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        if not plus:
            self.in_project_fx = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)

    def project_input_onto_slices(
        self, x: Float[torch.Tensor, "B N C"]
    ) -> (
        Float[torch.Tensor, "B N H D"]
        | tuple[
            Float[torch.Tensor, "B N H D"],
            Float[torch.Tensor, "B N H D"],
        ]
    ):
        r"""
        Project input onto slice space using 3D convolutions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(B, N, C)` where :math:`N` is the
            flattened spatial dimension (height times width times depth).

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Projected tensors of shape :math:`(B, N, H, D)` where :math:`H` is
            number of attention heads and :math:`D` is dimension per head.
        """
        B = x.shape[0]
        C = x.shape[-1]

        # Reshape from flattened to 3D spatial format: (B, N, C) -> (B, C, H_s, W_s, D_s)
        x = x.view(B, self.H, self.W, self.D, C)
        x = x.permute(0, 4, 1, 2, 3)

        # Apply 3D convolution and reshape to multi-head format
        input_projected_x = self.in_project_x(x)
        input_projected_x = rearrange(
            input_projected_x,
            "B (H D) height width depth -> B (height width depth) H D",
            D=self.dim_head,
            H=self.heads,
        )

        if self.plus:
            return input_projected_x
        else:
            input_projected_fx = self.in_project_fx(x)
            input_projected_fx = rearrange(
                input_projected_fx,
                "B (H D) height width depth -> B (height width depth) H D",
                D=self.dim_head,
                H=self.heads,
            )
            return input_projected_x, input_projected_fx
