# ignore_header_test
# ruff: noqa: E402
""""""

r"""
Transolver model embedding utilities.

This code was modified from, https://github.com/thuml/Transolver

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

import math

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float


class RotaryEmbedding(nn.Module):
    r"""
    Rotary Position Embedding (RoPE).

    Implements rotary position embeddings that encode positional information
    by rotating query and key vectors in attention mechanisms.

    For more details, see: `RoFormer paper <https://arxiv.org/abs/2104.09864>`_

    Parameters
    ----------
    dim : int
        Embedding dimension (must be even).
    min_freq : float, optional, default=0.5
        Minimum frequency for the sinusoidal embeddings.
    scale : float, optional, default=1.0
        Scaling factor for the input coordinates.

    Forward
    -------
    coordinates : torch.Tensor
        Coordinate values of shape :math:`(B, N)` where :math:`B` is batch size
        and :math:`N` is sequence length.
    device : torch.device
        Device to place the output tensor on.

    Outputs
    -------
    torch.Tensor
        Rotary frequencies of shape :math:`(B, N, D)` where :math:`D` is the
        embedding dimension.

    Examples
    --------
    >>> import torch
    >>> rope = RotaryEmbedding(dim=64)
    >>> coords = torch.linspace(0, 1, 100).unsqueeze(0)  # (1, 100)
    >>> freqs = rope(coords, device="cpu")
    >>> freqs.shape
    torch.Size([1, 100, 64])
    """

    def __init__(self, dim: int, min_freq: float = 1 / 2, scale: float = 1.0):
        super().__init__()
        # Compute inverse frequencies for sinusoidal embeddings
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer("inv_freq", inv_freq)

    def forward(
        self,
        coordinates: Float[torch.Tensor, "batch seq"],
        device: torch.device,
    ) -> Float[torch.Tensor, "batch seq dim"]:
        r"""
        Compute rotary frequencies for given coordinates.

        Parameters
        ----------
        coordinates : torch.Tensor
            Coordinate values of shape :math:`(B, N)`.
        device : torch.device
            Target device for output tensor.

        Returns
        -------
        torch.Tensor
            Rotary frequencies of shape :math:`(B, N, D)`.
        """
        # Scale coordinates
        inv_freq: torch.Tensor = self.inv_freq  # type: ignore[assignment]
        t = coordinates.to(device).type_as(inv_freq)
        t = t * (self.scale / self.min_freq)

        # Compute frequencies via outer product
        freqs = torch.einsum("... i , j -> ... i j", t, self.inv_freq)  # (B, N, D//2)

        # Concatenate to get full dimension
        return torch.cat((freqs, freqs), dim=-1)  # (B, N, D)


def rotate_half(x: Float[torch.Tensor, "... dim"]) -> Float[torch.Tensor, "... dim"]:
    r"""
    Rotate the last dimension by splitting and swapping halves.

    This is a helper function for applying rotary position embeddings.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape :math:`(*, D)` where the last dimension will be
        split and rotated.

    Returns
    -------
    torch.Tensor
        Rotated tensor of the same shape as input.
    """
    # Split into two halves and swap with negation
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    t: Float[torch.Tensor, "... dim"],
    freqs: Float[torch.Tensor, "... dim"],
) -> Float[torch.Tensor, "... dim"]:
    r"""
    Apply rotary position embeddings to input tensor.

    Parameters
    ----------
    t : torch.Tensor
        Input tensor of shape :math:`(*, D)`.
    freqs : torch.Tensor
        Rotary frequencies of shape :math:`(*, D)` from
        :class:`RotaryEmbedding`.

    Returns
    -------
    torch.Tensor
        Tensor with rotary embeddings applied, same shape as input.
    """
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


def apply_2d_rotary_pos_emb(
    t: Float[torch.Tensor, "batch heads seq dim"],
    freqs_x: Float[torch.Tensor, "batch seq dim_half"],
    freqs_y: Float[torch.Tensor, "batch seq dim_half"],
) -> Float[torch.Tensor, "batch heads seq dim"]:
    r"""
    Apply 2D rotary position embeddings for spatial data.

    Splits the input tensor in half along the feature dimension and applies
    separate rotary embeddings for x and y coordinates.

    Parameters
    ----------
    t : torch.Tensor
        Input tensor of shape :math:`(B, H, N, D)` where :math:`H` is heads.
    freqs_x : torch.Tensor
        X-coordinate frequencies of shape :math:`(B, N, D/2)`.
    freqs_y : torch.Tensor
        Y-coordinate frequencies of shape :math:`(B, N, D/2)`.

    Returns
    -------
    torch.Tensor
        Tensor with 2D rotary embeddings applied, shape :math:`(B, H, N, D)`.
    """
    d = t.shape[-1]

    # Split input into x and y halves
    t_x, t_y = t[..., : d // 2], t[..., d // 2 :]

    # Apply rotary embeddings separately and concatenate
    return torch.cat(
        (apply_rotary_pos_emb(t_x, freqs_x), apply_rotary_pos_emb(t_y, freqs_y)), dim=-1
    )


class PositionalEncoding(nn.Module):
    r"""
    Sinusoidal positional encoding.

    Implements fixed sinusoidal positional encodings as described in
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Parameters
    ----------
    d_model : int
        Model dimension (embedding size).
    dropout : float
        Dropout probability applied after adding positional encoding.
    max_len : int, optional, default=177241
        Maximum sequence length supported (default is 421*421).

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, N, D)` where :math:`N` is sequence
        length and :math:`D` is ``d_model``.

    Outputs
    -------
    torch.Tensor
        Input with positional encoding added, shape :math:`(B, N, D)`.

    Examples
    --------
    >>> import torch
    >>> pe = PositionalEncoding(d_model=128, dropout=0.1)
    >>> x = torch.randn(2, 100, 128)
    >>> out = pe(x)
    >>> out.shape
    torch.Size([2, 100, 128])
    """

    def __init__(self, d_model: int, dropout: float, max_len: int = 421 * 421):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings in log space for numerical stability
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(
        self, x: Float[torch.Tensor, "batch seq dim"]
    ) -> Float[torch.Tensor, "batch seq dim"]:
        r"""
        Add positional encoding to input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(B, N, D)`.

        Returns
        -------
        torch.Tensor
            Input with positional encoding added, shape :math:`(B, N, D)`.
        """
        # Add positional encoding (no gradient needed for PE)
        pe: torch.Tensor = self.pe  # type: ignore[assignment]
        x = x + pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def timestep_embedding(
    timesteps: Float[torch.Tensor, " batch"],
    dim: int,
    max_period: int = 10000,
    repeat_only: bool = False,
) -> Float[torch.Tensor, "batch dim"]:
    r"""
    Create sinusoidal timestep embeddings.

    Generates embeddings for diffusion model timesteps using sinusoidal
    functions, similar to transformer positional encodings.

    Parameters
    ----------
    timesteps : torch.Tensor
        1-D tensor of :math:`N` timestep indices, one per batch element.
        These may be fractional values.
    dim : int
        Dimension of the output embeddings.
    max_period : int, optional, default=10000
        Controls the minimum frequency of the embeddings.
    repeat_only : bool, optional, default=False
        Currently unused, kept for API compatibility.

    Returns
    -------
    torch.Tensor
        Positional embeddings of shape :math:`(N, D)` where :math:`D` is
        ``dim``.

    Examples
    --------
    >>> import torch
    >>> timesteps = torch.tensor([0.0, 0.5, 1.0])
    >>> emb = timestep_embedding(timesteps, dim=64)
    >>> emb.shape
    torch.Size([3, 64])
    """
    half = dim // 2

    # Compute frequencies
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)

    # Compute embeddings via outer product
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    # Handle odd dimensions by padding with zeros
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding
