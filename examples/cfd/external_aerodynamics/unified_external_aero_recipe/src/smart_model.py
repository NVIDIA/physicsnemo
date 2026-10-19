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

"""Single-field SMART adapter for the unified recipe's tensor contract."""

from __future__ import annotations

import math

import torch
from jaxtyping import Float, Int
from torch import Tensor

from physicsnemo.experimental.models.smart import SMART


class SMARTField(SMART):
    r"""Expose one SMART head through the unified recipe's tensor contract.

    Geometry remains a separate full CFD boundary cloud. Only the initial
    latents and per-layer subsamples enter the SMART encoder. Query features
    use SMART's additive projections. The recipe provides normalization,
    losses, optimizer, metrics, and physical-unit export.

    Parameters
    ----------
    field : {"surface", "volume"}
        Enabled output head.
    out_dim : int
        Number of target channels in dataset declaration order.
    query_feature_channels : int
        Number of normal or SDF/direction channels; zero disables features.
    eval_chunk_size : int, optional, default=8192
        Maximum queries decoded at once during evaluation.
    parameter_channels : int, optional, default=3
        Number of global simulation conditions, normally freestream velocity.
    parameter_scale : float, optional, default=1.0
        Fixed positive divisor for simulation conditions. The DrivAerML
        templates use 30 m/s for freestream velocity. Physical metadata is
        preserved for nondimensionalization and force integration.
    hidden_dim, num_heads, num_layers, num_latents, geometry_subsample, mlp_ratio : int, optional
        Architecture settings with the same defaults as
        :class:`~physicsnemo.experimental.models.smart.SMART`.
    share_weights : bool, optional, default=True
        Share corresponding encoder and decoder refinements.
    eval_seed : int, optional, default=0
        Fixed geometry sampling seed during evaluation.
    qk_normalization : bool, optional, default=False
        Normalize projected queries and keys by their per-head RMS.
    qk_temperature_max : float or None, optional, default=None
        Optional bound on learned normalized-attention temperature.

    Forward
    -------
    geometry : torch.Tensor
        Full CFD geometry of shape :math:`(1,N_g,3)`.
    queries : torch.Tensor
        Field-query coordinates of shape :math:`(1,N_q,3)`.
    query_features : torch.Tensor or None
        Aligned geometric features of shape :math:`(1,N_q,C)`.
    parameters : torch.Tensor or None
        Global conditions of shape :math:`(1,1,P)` or :math:`(1,P)`.
    geometry_seed : torch.Tensor
        One integer sampling seed, normally of shape :math:`(1,1,1)`.

    Outputs
    -------
    torch.Tensor
        Predictions of shape :math:`(1,N_q,\mathrm{out\_dim})`.

    Examples
    --------
    >>> model = SMARTField("surface", 4, 3, hidden_dim=24, num_heads=4,
    ...                    num_layers=2, num_latents=8, geometry_subsample=16)
    >>> prediction = model(torch.randn(1, 32, 3), torch.randn(1, 12, 3),
    ...                    torch.randn(1, 12, 3), torch.randn(1, 1, 3),
    ...                    torch.tensor([[[42]]]))
    >>> prediction.shape
    torch.Size([1, 12, 4])
    """

    def __init__(
        self,
        field: str,
        out_dim: int,
        query_feature_channels: int,
        eval_chunk_size: int = 8192,
        parameter_channels: int = 3,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        num_latents: int = 2048,
        geometry_subsample: int = 65536,
        mlp_ratio: int = 4,
        share_weights: bool = True,
        eval_seed: int = 0,
        qk_normalization: bool = False,
        qk_temperature_max: float | None = None,
        parameter_scale: float = 1.0,
    ):
        if field not in ("surface", "volume"):
            raise ValueError("field must be surface or volume")
        if (
            isinstance(eval_chunk_size, bool)
            or not isinstance(eval_chunk_size, int)
            or eval_chunk_size <= 0
        ):
            raise ValueError("eval_chunk_size must be a positive integer")
        if (
            isinstance(parameter_scale, bool)
            or not isinstance(parameter_scale, (int, float))
            or not math.isfinite(parameter_scale)
            or parameter_scale <= 0
        ):
            raise ValueError("parameter_scale must be finite and positive")
        super().__init__(
            surface_channels=out_dim if field == "surface" else 0,
            volume_channels=out_dim if field == "volume" else 0,
            surface_query_feature_channels=query_feature_channels
            if field == "surface"
            else 0,
            volume_query_feature_channels=query_feature_channels
            if field == "volume"
            else 0,
            parameter_channels=parameter_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_latents=num_latents,
            geometry_subsample=geometry_subsample,
            mlp_ratio=mlp_ratio,
            share_weights=share_weights,
            eval_seed=eval_seed,
            qk_normalization=qk_normalization,
            qk_temperature_max=qk_temperature_max,
        )
        self.field = field
        self.eval_chunk_size = eval_chunk_size
        self.parameter_scale = float(parameter_scale)

    def forward(
        self,
        geometry: Float[Tensor, "1 geometry_points 3"],
        queries: Float[Tensor, "1 query_points 3"],
        query_features: Float[Tensor, "1 query_points feature_channels"] | None,
        parameters: Float[Tensor, "1 ... parameter_channels"] | None,
        geometry_seed: Int[Tensor, "..."],
    ) -> Float[Tensor, "1 query_points out_dim"]:
        r"""Predict aligned field queries with independent geometry randomness.

        A reader-derived case/epoch seed isolates geometry draws from query
        sampling, augmentation and global model RNG. Evaluation instead uses
        the core model's fixed ``eval_seed``, including across query chunks.
        """
        if not torch.compiler.is_compiling():
            self._validate_coordinates(geometry, batch=1)
            self._validate_coordinates(queries, batch=1)
            if queries.device != geometry.device:
                raise ValueError("Queries must be on the geometry device")
            if geometry_seed.numel() != 1 or geometry_seed.dtype not in (
                torch.int32,
                torch.int64,
            ):
                raise ValueError("geometry_seed must contain one int32 or int64 seed")
            channels = getattr(self, f"{self.field}_query_feature_channels")
            if channels:
                if (
                    query_features is None
                    or query_features.shape != (*queries.shape[:2], channels)
                    or not query_features.is_floating_point()
                    or query_features.device != queries.device
                ):
                    raise ValueError(
                        "Query features must have aligned (1, N, C) shapes "
                        "and floating dtype on the query device"
                    )
            elif query_features is not None:
                raise ValueError(
                    "query_features must be None when features are disabled"
                )
            if self.parameter_channels and (
                parameters is None
                or parameters.shape
                not in ((1, self.parameter_channels), (1, 1, self.parameter_channels))
            ):
                raise ValueError("Expected parameters of shape (1, P) or (1, 1, P)")
        if self.parameter_channels:
            parameters = parameters.reshape(1, self.parameter_channels)
        if not torch.compiler.is_compiling():
            self._validate_parameters(parameters, 1, geometry.device)
        if parameters is not None:
            parameters = parameters / self.parameter_scale
        generator = None
        if self.training:
            generator = torch.Generator(device=geometry.device).manual_seed(
                int(geometry_seed.item())
            )
        contexts = self.encode_geometry(geometry, parameters, generator=generator)
        if self.training or queries.shape[1] == 0:
            return self.decode_queries(
                queries,
                contexts,
                parameters,
                field=self.field,
                query_features=query_features,
            )
        return torch.cat(
            [
                self.decode_queries(
                    queries[:, start : start + self.eval_chunk_size],
                    contexts,
                    parameters,
                    field=self.field,
                    query_features=(
                        query_features[:, start : start + self.eval_chunk_size]
                        if query_features is not None
                        else None
                    ),
                )
                for start in range(0, queries.shape[1], self.eval_chunk_size)
            ],
            dim=1,
        )
