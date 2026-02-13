# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import torch
from jaxtyping import Bool, Float, Int

from physicsnemo.core.module import Module


def _compute_row_major_strides(shape: tuple[int, ...]) -> list[int]:
    strides = []
    stride = 1
    for size in reversed(shape):
        strides.insert(0, stride)
        stride *= size
    return strides


def scatter_mean(
    x: Float[torch.Tensor, "N C"],
    index: Int[torch.Tensor, "N D"],
    shape: tuple[int, ...],
    fill_value: float = float("nan"),
) -> tuple[Float[torch.Tensor, "*shape C"], Bool[torch.Tensor, "*shape"]]:
    r"""
    Scatter-mean values onto a multi-dimensional grid.

    Parameters
    ----------
    x : torch.Tensor
        Input values to aggregate, shape :math:`(N, C)`.
    index : torch.Tensor
        Integer indices of shape :math:`(N, D)` giving the :math:`D` grid
        coordinates for each element.
    shape : tuple[int, ...]
        :math:`D`-tuple specifying the output grid shape for the indexed dimensions.
    fill_value : float, optional
        Value to fill empty grid cells with. Defaults to NaN.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        - ``aggregated``: mean-aggregated values of shape :math:`(*shape, C)`.
        - ``present``: boolean mask of shape :math:`(*shape)` indicating which cells received data.

    Examples
    --------
    Scatter 4 feature vectors of dimension 2 onto a ``(batch=2, pixel=3)`` grid.

    >>> import torch
    >>> x = torch.tensor([[1.0, 10.0],
    ...                    [3.0, 30.0],
    ...                    [5.0, 50.0],
    ...                    [7.0, 70.0]])           # (N=4, C=2)
    >>> index = torch.tensor([[0, 1],
    ...                        [0, 1],
    ...                        [1, 0],
    ...                        [1, 2]])            # (N=4, D=2) -> (batch, pixel)
    >>> agg, present = scatter_mean(x, index, shape=(2, 3))
    >>> agg.shape                                  # (*shape, C) = (2, 3, 2)
    torch.Size([2, 3, 2])
    >>> agg[0]                                     # batch 0: only pixel 1 has data
    tensor([[nan, nan],
            [ 2., 20.],                            # pixel 1 is average of rows 0 & 1
            [nan, nan]])
    >>> agg[1]                                     # batch 1: pixels 0 and 2 have data
    tensor([[ 5., 50.],
            [nan, nan],
            [ 7., 70.]])
    >>> present                                    # (*shape) = (2, 3)
    tensor([[False,  True, False],                 # pixel 1 is present in batch 0
            [ True, False,  True]])                # pixels 0 and 2 are present in batch 1
    """
    strides = _compute_row_major_strides(shape)
    # manually implement the dot product since matmul doesn't support long tensors on cuda
    # avoids RuntimeError: "addmv_impl_cuda" not implemented for 'Long'
    grid_indices_flat = (index * torch.tensor(strides, device=index.device)).sum(dim=-1)
    grid_size = math.prod(shape)

    device = x.device
    dtype = x.dtype
    c = x.shape[1]

    # Initialize grid with fill_value
    values_mean = torch.full(
        (grid_size, c), fill_value, device=device, dtype=dtype
    )

    # Use scatter_reduce with mean, expanding indices to match value dimensions
    grid_indices_flat_expanded = grid_indices_flat.unsqueeze(-1).expand(
        -1, c
    )
    values_mean.scatter_reduce_(
        0, grid_indices_flat_expanded, x, reduce="mean", include_self=False
    )

    # Compute present mask (cells that are not fill_value)
    if math.isnan(fill_value):
        present = ~torch.isnan(values_mean[:, 0])
    else:
        present = values_mean[:, 0] != fill_value

    # Reshape
    aggregated = values_mean.view(*shape, c)
    present = present.view(shape)

    return aggregated, present


class ScatterAggregator(Module):
    r"""
    Scatter-aggregate sparse data onto a dense grid with a learned projection.

    Each spatial pixel has ``nbuckets`` independent slots. Sparse input
    elements are scatter-mean aggregated into their respective
    (pixel, bucket) cell, producing a :math:`(B, N_{pix}, N_{bucket}, C_{in})`
    grid. Unobserved cells are filled with zeros, a per-bucket observability
    mask is concatenated, and a pointwise MLP fuses the bucket features into
    a :math:`C_{out}`-dimensional output per pixel.

    Parameters
    ----------
    in_dim : int
        Input feature dimension per element.
    out_dim : int
        Output feature dimension per pixel.
    nbuckets : int
        Number of buckets (independent feature slots) per pixel. Aggregation
        occurs independently for each bucket before the MLP fuses them.

    Forward
    -------
    x : torch.Tensor
        Input values to aggregate, shape :math:`(N, C_{in})`.
    batch_idx : torch.Tensor
        Batch index per element, shape :math:`(N,)`.
    pix : torch.Tensor
        Pixel index per element, shape :math:`(N,)`.
    bucket_id : torch.Tensor
        Bucket index per element, shape :math:`(N,)`.
    nbatch : int
        Number of samples in the batch.
    npix : int
        Number of pixels in the spatial grid.

    Outputs
    -------
    torch.Tensor
        Per-pixel aggregated features, shape :math:`(B, N_{pix}, C_{out})`.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        nbuckets: int,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nbuckets = nbuckets

        proj_in = self.nbuckets * in_dim + self.nbuckets  # aggregated features + observability mask
        proj_out = out_dim * 2
        self.bucket_mixing_mlp = torch.nn.Sequential(
            torch.nn.Linear(proj_in, proj_out),
            torch.nn.LayerNorm(proj_out),
            torch.nn.SiLU(),
            torch.nn.Linear(proj_out, out_dim),
        )

    def forward(
        self,
        x: Float[torch.Tensor, "N c_in"],
        batch_idx: Int[torch.Tensor, "N"],
        pix: Int[torch.Tensor, "N"],
        bucket_id: Int[torch.Tensor, "N"],
        nbatch: int,
        npix: int,
    ) -> Float[torch.Tensor, "nbatch npix c_out"]:
        grid_indices = torch.stack([batch_idx, pix, bucket_id], dim=-1)

        aggregated, has_obs = scatter_mean(
            x=x,
            index=grid_indices,
            shape=(nbatch, npix, self.nbuckets),
        )  # (nbatch, npix, nbuckets, in_dim), (nbatch, npix, nbuckets)

        # Reshape and fill unobserved with zeros (scatter_mean fills empty cells with NaN)
        nbatch, npix, nbuckets, in_dim = aggregated.shape
        aggregated = aggregated.view(nbatch, npix, nbuckets * in_dim)
        aggregated = torch.nan_to_num(aggregated, nan=0.0)

        # Concatenate observability mask and project through MLP
        mlp_input = torch.cat([aggregated, has_obs.float()], dim=-1)
        return self.bucket_mixing_mlp(mlp_input)
