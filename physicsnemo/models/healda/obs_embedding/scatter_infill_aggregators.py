# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Scatter aggregation for observation embedding. Use scatter-reduce to aggregate observations onto spatial grids."""

import torch

from ..scatter_mean import scatter_mean


class ScatterAggregator(torch.nn.Module):
    """Dense-bucket scatter aggregation (all batches together, all buckets).

    Pipeline:
        1. Aggregate observations onto spatial grid using scatter_mean
        2. Fill unobserved values with zeros
        3. Mix across all buckets using MLP

    Args:
        in_dim: Input token dimension
        out_dim: Output dimension after projection
        nchannel: Max number of channels
        nplatform: Max number of platforms
        npix: Number of spatial pixels in output grid
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
        """
        Aggregate observations to spatial grid.

        Args:
            obs_features: (nobs, in_dim) tokenized observations
            batch_idx: (nobs,) batch index for each observation
            pix: (nobs,) spatial pixel index for each observation
            bucket_id: (nobs,) bucket ID (platform * nchannel + channel) for each observation
            nbatch: Number of batch elements

        Returns:
            (nbatch, npix, out_dim) aggregated and projected spatial grid
        """
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
