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
from physicsnemo.core.module import Module

def _compute_row_major_strides(shape: tuple[int, ...]) -> list[int]:
    strides = []
    stride = 1
    for size in reversed(shape):
        strides.insert(0, stride)
        stride *= size
    return strides


def scatter_mean(
    tensor: torch.Tensor,
    index: torch.Tensor,
    shape: tuple[int, ...],
    fill_value: float = float("nan"),
) -> torch.Tensor:
    r"""
    Scatter-mean values onto a multi-dimensional grid.

    Parameters
    ----------
    tensor : torch.Tensor
        Observation features of shape :math:`(N, C)`.
    index : torch.Tensor
        Integer indices of shape :math:`(N, D)`; each row gives the :math:`D`
        grid coordinates for one observation.
    shape : tuple[int, ...]
        :math:`D`-tuple specifying the output grid shape for the indexed dimensions.
    fill_value : float, optional
        Value to fill unobserved grid cells with. Defaults to NaN.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        - ``aggregated``: mean-aggregated values of shape :math:`(*shape, C)`.
        - ``present``: boolean mask of shape :math:`(*shape)` indicating which cells have values.
    """
    strides = _compute_row_major_strides(shape)
    # manually implement the dot product since matmul doesn't support long tensors on cuda
    # avoids RuntimeError: "addmv_impl_cuda" not implemented for 'Long'
    grid_indices_flat = (index * torch.tensor(strides, device=index.device)).sum(dim=-1)
    grid_size = math.prod(shape)

    device = tensor.device
    dtype = tensor.dtype
    embedding_dim = tensor.shape[1]

    # Initialize with fill_value (typically NaN)
    values_mean = torch.full(
        (grid_size, embedding_dim), fill_value, device=device, dtype=dtype
    )

    # Use scatter_reduce with mean, expanding indices to match tensor dimensions
    grid_indices_flat_expanded = grid_indices_flat.unsqueeze(-1).expand(
        -1, embedding_dim
    )
    values_mean.scatter_reduce_(
        0, grid_indices_flat_expanded, tensor, reduce="mean", include_self=False
    )

    # Compute present mask (cells that are not fill_value)
    if math.isnan(fill_value):
        present = ~torch.isnan(values_mean[:, 0])
    else:
        present = values_mean[:, 0] != fill_value

    # Reshape
    aggregated = values_mean.view(*shape, embedding_dim)
    present = present.view(shape)

    return aggregated, present



class ScatterAggregator(Module):
    r"""
    Scatter-aggregate sparse observations onto a dense grid with a learned projection.

    Aggregates sparse observations into a :math:`(B, N_{pix}, N_{bucket}, C_{in})` grid
    using scatter-mean, fills unobserved cells with zeros, then applies a pointwise MLP
    to project all bucket features into a single per-pixel output vector.

    Parameters
    ----------
    in_dim : int
        Input feature dimension per observation.
    out_dim : int
        Output feature dimension per pixel.
    nchannel : int
        Number of channels per platform.
    nplatform : int
        Number of platforms.
    npix : int
        Number of spatial pixels in the target grid.

    Forward
    -------
    obs_features : torch.Tensor
        Observation features, shape :math:`(N_{obs}, C_{in})`.
    batch_idx : torch.Tensor
        Batch index per observation, shape :math:`(N_{obs},)`.
    pix : torch.Tensor
        HEALPix pixel index per observation (NEST ordering), shape :math:`(N_{obs},)`.
    bucket_id : torch.Tensor
        Bucket index per observation, shape :math:`(N_{obs},)`.
    nbatch : int
        Number of samples in the batch.

    Outputs
    -------
    torch.Tensor
        Per-pixel aggregated features, shape :math:`(B, N_{pix}, C_{out})` in NEST ordering.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        nchannel: int,
        nplatform: int,
        npix: int,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nchannel = nchannel
        self.nplatform = nplatform
        self.npix = npix
        self.nbuckets = nchannel * nplatform

        proj_in = self.nbuckets * in_dim + self.nbuckets  # features + bucket coverage
        proj_out = out_dim * 2
        self.bucket_mixing_mlp = torch.nn.Sequential(
            torch.nn.Linear(proj_in, proj_out),
            torch.nn.LayerNorm(proj_out),
            torch.nn.SiLU(),
            torch.nn.Linear(proj_out, out_dim),
        )

    def forward(
        self,
        obs_features: torch.Tensor,
        batch_idx: torch.Tensor,
        pix: torch.Tensor,
        bucket_id: torch.Tensor,
        nbatch: int,
    ) -> torch.Tensor:
        grid_indices = torch.stack([batch_idx, pix, bucket_id], dim=-1)

        aggregated, has_obs = scatter_mean(
            tensor=obs_features,
            index=grid_indices,
            shape=(nbatch, self.npix, self.nbuckets),
        )  # (nbatch, npix, nbuckets, in_dim), (nbatch, npix, nbuckets)

        # Reshape and fill unobserved with zeros (scatter_mean fills empty cells with NaN)
        nbatch, npix, nbuckets, in_dim = aggregated.shape
        aggregated = aggregated.view(nbatch, npix, nbuckets * in_dim)
        aggregated = torch.nan_to_num(aggregated, nan=0.0)

        # Concatenate bucket coverage info and project through MLP
        mlp_input = torch.cat([aggregated, has_obs.float()], dim=-1)
        return self.bucket_mixing_mlp(mlp_input)
