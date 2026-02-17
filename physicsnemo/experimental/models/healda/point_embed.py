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
"""Multi-sensor observation embedding for HealDA."""

import math

import torch
from jaxtyping import Float, Int

from physicsnemo.core.module import Module

from .scatter_aggregator import ScatterAggregator


def _offsets_to_batch_idx(offsets: Int[torch.Tensor, "batch time"]) -> Int[torch.Tensor, "nobs"]:
    r"""Map each observation to its flattened :math:`(B, T)` window index.

    Given cumulative exclusive-end offsets of shape :math:`(B, T)`, return a
    1-D tensor of length ``nobs`` where entry ``i`` is the flat window index
    (in ``[0, B*T)``) that observation ``i`` belongs to.

    Examples
    --------
    >>> import torch
    >>> offsets = torch.tensor([[3, 5], [7, 10]])  # (B=2, T=2)
    >>> _offsets_to_batch_idx(offsets)
    tensor([0, 0, 0, 1, 1, 2, 2, 3, 3, 3])
    """
    bt_size = offsets.numel()

    # Convert cumulative-end offsets to per-window counts: [3, 5, 7, 10] -> [3, 2, 2, 3]
    offsets_flat = offsets.flatten()
    offsets_with_zero = torch.cat(
        [torch.tensor([0], device=offsets.device, dtype=offsets.dtype), offsets_flat]
    )
    counts = offsets_with_zero.diff()

    # Repeat each window index by its count
    window_indices = torch.arange(bt_size, dtype=torch.long, device=offsets.device)
    return window_indices.repeat_interleave(counts)


@torch.compiler.disable
def _split_by_sensor(
    obs: Float[torch.Tensor, "nobs"],
    float_metadata: Float[torch.Tensor, "nobs meta_dim"],
    pix: Int[torch.Tensor, "nobs"],
    local_channel: Int[torch.Tensor, "nobs"],
    local_platform: Int[torch.Tensor, "nobs"],
    obs_type: Int[torch.Tensor, "nobs"],
    offsets: Int[torch.Tensor, "sensors batch time"],
) -> list[tuple[torch.Tensor, ...]]:
    """Split flattened observation tensors into per-sensor slices using ``offsets``.

    Returns a list of length ``S`` (number of sensors). Each element is a tuple
    ``(obs, float_metadata, pix, local_channel, local_platform, obs_type, offsets)``
    containing the sliced tensors and ``(B, T)`` offsets for that sensor.
    """
    if offsets.ndim != 3:
        raise ValueError(f"offsets must have shape (S, B, T), got {offsets.shape}")
    nobs = obs.shape[0]
    for name, tensor in (
        ("float_metadata", float_metadata),
        ("pix", pix),
        ("local_channel", local_channel),
        ("local_platform", local_platform),
        ("obs_type", obs_type),
    ):
        if tensor.shape[0] != nobs:
            raise ValueError(
                f"{name} must have leading dimension {nobs}, got {tensor.shape}"
            )

    nsensors = offsets.shape[0]
    out: list[tuple[torch.Tensor, ...]] = []
    total_obs = obs.shape[0]

    prev_end = 0
    for sensor_idx in range(nsensors):
        start = prev_end
        # This sensor ends at its last (batch, time) window
        end = offsets[sensor_idx, -1, -1].item()
        prev_end = end

        # Normalize per-sensor offsets so they start from 0
        sensor_offsets = offsets[sensor_idx] - start

        if not (0 <= start <= total_obs and start <= end <= total_obs):
            raise ValueError(
                f"Invalid offsets for sensor index {sensor_idx}: start={start}, end={end}, "
                f"total_obs={total_obs}."
            )
        length = end - start

        def _narrow_first_dim(x: torch.Tensor) -> torch.Tensor:
            return torch.narrow(x, 0, start, length)

        out.append(
            (
                _narrow_first_dim(obs),
                _narrow_first_dim(float_metadata),
                _narrow_first_dim(pix),
                _narrow_first_dim(local_channel),
                _narrow_first_dim(local_platform),
                _narrow_first_dim(obs_type),
                sensor_offsets,
            )
        )

    return out



class ObsTokenizer(Module):
    r"""Tokenizes individual observations into feature vectors by combining 
    measurements along with their metadata, using learnable embedding tables and an MLP projection.

    Parameters
    ----------
    meta_dim : int
        Dimension of float metadata features.
    out_dim : int
        Output token dimension.
    n_embed : int, optional, default=1024
        Size of observation type embedding table.
    embed_dim : int, optional, default=4
        Dimension of observation type embeddings.

    Forward
    -------
    obs : torch.Tensor
        Observation values with shape :math:`(N_{obs},)`.
    float_metadata : torch.Tensor
        Float metadata with shape :math:`(N_{obs}, M_{float})`.
    obs_type : torch.Tensor
        Observation type ids with shape :math:`(N_{obs},)`.

    Outputs
    -------
    torch.Tensor
        Tokenized observation features of shape :math:`(N_{obs}, D_{out})`.
    """

    def __init__(
        self,
        meta_dim: int,
        out_dim: int,
        n_embed: int = 1024,
        embed_dim: int = 4,
    ):
        super().__init__()

        self.embed_table = torch.nn.Embedding(n_embed, embed_dim)

        mlp_in_dim = (
            1           # obs measurement
            + meta_dim  # float metadata
            + embed_dim # learned embedding
        )
        mlp_out_dim = out_dim - 1
        hidden_dim = out_dim * 2 if out_dim <= 32 else out_dim

        self.meta_mlp = torch.nn.Sequential(
            torch.nn.Linear(mlp_in_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, mlp_out_dim),
        )

    def forward(
        self,
        obs: Float[torch.Tensor, "nobs"],
        float_metadata: Float[torch.Tensor, "nobs meta_dim"],
        obs_type: Int[torch.Tensor, "nobs"],
    ) -> Float[torch.Tensor, "nobs out_dim"]:
        if not torch.compiler.is_compiling():
            if obs.ndim != 1:
                raise ValueError(
                    f"Expected obs of shape (nobs,), "
                    f"got {obs.ndim}D tensor with shape {tuple(obs.shape)}"
                )
            nobs = obs.shape[0]
            if float_metadata.ndim != 2:
                raise ValueError(
                    f"Expected float_metadata of shape (nobs, meta_dim), "
                    f"got {float_metadata.ndim}D tensor with shape {tuple(float_metadata.shape)}"
                )
            if float_metadata.shape[0] != nobs:
                raise ValueError(
                    f"Expected float_metadata with {nobs} rows (matching obs), "
                    f"got tensor with shape {tuple(float_metadata.shape)}"
                )
            if obs_type.ndim != 1 or obs_type.shape[0] != nobs:
                raise ValueError(
                    f"Expected obs_type of shape ({nobs},) matching obs, "
                    f"got tensor with shape {tuple(obs_type.shape)}"
                )

        embed_vec = self.embed_table(obs_type)

        x_in = torch.cat(
            [
                obs.unsqueeze(-1),
                float_metadata,
                embed_vec,
            ],
            dim=-1,
        )
        mlp_out = self.meta_mlp(x_in)
        encoded = torch.cat([obs.unsqueeze(-1), mlp_out], dim=-1)
        return encoded


class UniformFusion(Module):
    r"""Averages sensor embeddings with :math:`1/\sqrt{N}` scaling to preserve variance.


    Parameters
    ----------
    fusion_dim : int
        Feature dimension of per-sensor embeddings and the fused output.

    Forward
    -------
    sensor_embeddings : torch.Tensor
        Sensor embeddings of shape :math:`(N_{sensors}, B, T, N_{pix}, D)`.

    Outputs
    -------
    torch.Tensor
        Fused embedding of shape :math:`(B, T, N_{pix}, D)`.
    """

    def __init__(self, fusion_dim: int):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.norm = torch.nn.LayerNorm(self.fusion_dim)

    def forward(
        self,
        sensor_embeddings: Float[torch.Tensor, "num_sensors batch time npix fusion_dim"],
    ) -> Float[torch.Tensor, "batch time npix fusion_dim"]:
        if not torch.compiler.is_compiling():
            if sensor_embeddings.ndim != 5:
                raise ValueError(
                    f"Expected sensor_embeddings of shape (num_sensors, batch, time, npix, fusion_dim), "
                    f"got {sensor_embeddings.ndim}D tensor with shape {tuple(sensor_embeddings.shape)}"
                )
            if sensor_embeddings.shape[-1] != self.fusion_dim:
                raise ValueError(
                    f"Expected fusion_dim={self.fusion_dim} in last dimension, "
                    f"got {sensor_embeddings.shape[-1]} in tensor with shape {tuple(sensor_embeddings.shape)}"
                )

        num_sensors = sensor_embeddings.shape[0]
        sensor_embeddings = self.norm(sensor_embeddings)
        return sensor_embeddings.sum(dim=0) / math.sqrt(num_sensors)


class SensorEmbedder(Module):
    r"""Embeds observations from a single sensor onto a spatial grid.

    Each observation is first tokenized into a feature vector by
    :class:`ObsTokenizer`, then scatter-aggregated onto the spatial grid
    via :class:`ScatterAggregator` and projected to the output dimension.

    Parameters
    ----------
    nplatform : int
        Number of platforms for this sensor.
    nchannel : int
        Number of channels for this sensor.
    sensor_embed_dim : int, optional, default=32
        Internal feature dimension for tokenized observations.
    output_dim : int, optional, default=512
        Final output dimension per pixel.
    meta_dim : int, optional, default=28
        Dimension of float metadata features, consumed by :class:`ObsTokenizer`.
    n_embed : int, optional, default=1024
        Size of observation type embedding table.
    embed_dim : int, optional, default=4
        Dimension of observation type embeddings.
    gradient_checkpointing : bool, optional, default=False
        If ``True``, applies gradient checkpointing to reduce memory usage.

    Forward
    -------
    obs : torch.Tensor
        Observation values for a single sensor with shape :math:`(N_{obs},)`.
    float_metadata : torch.Tensor
        Float metadata with shape :math:`(N_{obs}, M_{float})`.
    pix : torch.Tensor
        Pixel index tensor with shape :math:`(N_{obs},)`.
    local_channel : torch.Tensor
        Local channel tensor with shape :math:`(N_{obs},)`.
    local_platform : torch.Tensor
        Local platform tensor with shape :math:`(N_{obs},)`.
    obs_type : torch.Tensor
        Observation type tensor with shape :math:`(N_{obs},)`.
    offsets : torch.Tensor
        Cumulative exclusive-end offsets with shape :math:`(B, T)` for this sensor.
        Indicate the end of each batch/time window.
    npix : int
        Number of pixels in the spatial grid.

    Outputs
    -------
    torch.Tensor
        Sensor embedding grid with shape :math:`(B, T, N_{pix}, D_{out})`.
    """

    def __init__(
        self,
        *,
        nplatform: int,
        nchannel: int,
        sensor_embed_dim: int = 32,
        output_dim: int = 512,
        meta_dim: int = 28,
        n_embed: int = 1024,
        embed_dim: int = 4,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.sensor_embed_dim = sensor_embed_dim
        self.output_dim = output_dim
        self.gradient_checkpointing = gradient_checkpointing
        self.nchannel = nchannel
        self.nplatform = nplatform

        self.obs_tokenizer = ObsTokenizer(
            meta_dim=meta_dim,
            out_dim=sensor_embed_dim,
            n_embed=n_embed,
            embed_dim=embed_dim,
        )

        self.scatter_infill_aggregator = ScatterAggregator(
            in_dim=sensor_embed_dim,
            out_dim=output_dim,
            nbuckets=nchannel * nplatform,
        )

    def aggregate(
        self,
        embedded_obs: Float[torch.Tensor, "nobs embed_dim"],
        pix: Int[torch.Tensor, "nobs"],
        local_channel: Int[torch.Tensor, "nobs"],
        local_platform: Int[torch.Tensor, "nobs"],
        batch_idx: Int[torch.Tensor, "nobs"],
        nbatch: int,
        npix: int,
    ) -> Float[torch.Tensor, "nbatch npix out_dim"]:
        """Aggregate tokenized observations to a spatial grid."""
        if not torch.compiler.is_compiling():
            if embedded_obs.ndim != 2:
                raise ValueError(
                    f"Expected embedded_obs of shape (nobs, embed_dim), "
                    f"got {embedded_obs.ndim}D tensor with shape {tuple(embedded_obs.shape)}"
                )
            if embedded_obs.shape[1] != self.sensor_embed_dim:
                raise ValueError(
                    f"Expected embed_dim={self.sensor_embed_dim} in embedded_obs, "
                    f"got {embedded_obs.shape[1]} in tensor with shape {tuple(embedded_obs.shape)}"
                )
            nobs = embedded_obs.shape[0]
            for name, tensor in (
                ("pix", pix),
                ("local_channel", local_channel),
                ("local_platform", local_platform),
                ("batch_idx", batch_idx),
            ):
                if tensor.ndim != 1 or tensor.shape[0] != nobs:
                    raise ValueError(
                        f"Expected {name} of shape ({nobs},) matching embedded_obs, "
                        f"got tensor with shape {tuple(tensor.shape)}"
                    )

        # Build combined bucket ID
        bucket_id = local_platform * self.nchannel + local_channel
        return self.scatter_infill_aggregator(
            x=embedded_obs,
            batch_idx=batch_idx,
            pix=pix,
            bucket_id=bucket_id,
            nbatch=nbatch,
            npix=npix,
        )

    def _forward(
        self,
        obs: Float[torch.Tensor, "nobs"],
        float_metadata: Float[torch.Tensor, "nobs meta_dim"],
        pix: Int[torch.Tensor, "nobs"],
        local_channel: Int[torch.Tensor, "nobs"],
        local_platform: Int[torch.Tensor, "nobs"],
        obs_type: Int[torch.Tensor, "nobs"],
        offsets: Int[torch.Tensor, "batch time"],
        npix: int,
    ) -> Float[torch.Tensor, "batch time npix out_dim"]:
        batch_idx = _offsets_to_batch_idx(offsets)
        batch_dims = offsets.shape  # (B, T)
        nbatch = offsets.numel()

        embedded_obs = self.obs_tokenizer(obs, float_metadata, obs_type)

        # Aggregator handles empty batches internally to keep all parameters in the computation graph
        output = self.aggregate(
            embedded_obs,
            pix,
            local_channel,
            local_platform,
            batch_idx,
            nbatch,
            npix,
        )  # (nbatch, npix, output_dim)
        output = output.view(*batch_dims, npix, self.output_dim)

        return output

    def forward(
        self,
        obs: Float[torch.Tensor, "nobs"],
        float_metadata: Float[torch.Tensor, "nobs meta_dim"],
        pix: Int[torch.Tensor, "nobs"],
        local_channel: Int[torch.Tensor, "nobs"],
        local_platform: Int[torch.Tensor, "nobs"],
        obs_type: Int[torch.Tensor, "nobs"],
        offsets: Int[torch.Tensor, "batch time"],
        npix: int,
    ) -> Float[torch.Tensor, "batch time npix out_dim"]:
        if not torch.compiler.is_compiling():
            if obs.ndim != 1:
                raise ValueError(
                    f"Expected obs of shape (nobs,), "
                    f"got {obs.ndim}D tensor with shape {tuple(obs.shape)}"
                )
            nobs = obs.shape[0]
            if float_metadata.ndim != 2:
                raise ValueError(
                    f"Expected float_metadata of shape (nobs, meta_dim), "
                    f"got {float_metadata.ndim}D tensor with shape {tuple(float_metadata.shape)}"
                )
            if float_metadata.shape[0] != nobs:
                raise ValueError(
                    f"Expected float_metadata with {nobs} rows (matching obs), "
                    f"got {float_metadata.shape[0]} in tensor with shape {tuple(float_metadata.shape)}"
                )
            for name, tensor in (
                ("pix", pix),
                ("local_channel", local_channel),
                ("local_platform", local_platform),
                ("obs_type", obs_type),
            ):
                if tensor.ndim != 1 or tensor.shape[0] != nobs:
                    raise ValueError(
                        f"Expected {name} of shape ({nobs},) matching obs, "
                        f"got tensor with shape {tuple(tensor.shape)}"
                    )
            if offsets.ndim != 2:
                raise ValueError(
                    f"Expected offsets of shape (batch, time), "
                    f"got {offsets.ndim}D tensor with shape {tuple(offsets.shape)}"
                )

        if self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                self._forward,
                obs,
                float_metadata,
                pix,
                local_channel,
                local_platform,
                obs_type,
                offsets,
                npix,
                use_reentrant=False,
            )
        else:
            return self._forward(
                obs,
                float_metadata,
                pix,
                local_channel,
                local_platform,
                obs_type,
                offsets,
                npix,
            )


class MultiSensorObsEmbedder(Module):
    r"""Multi-sensor observation embedding onto a spatial grid.

    Embeds observations from multiple sensor types into a unified representation
    on a spatial grid by applying per-sensor embedders and fusing the results.

    Parameters
    ----------
    nchannel_per_sensor : list[int]
        Number of channels for each sensor, in sensor order.
    nplatform_per_sensor : list[int]
        Number of platforms for each sensor, in sensor order.
    sensor_names : list[str], optional
        Human-readable names for each sensor, in sensor order. Used as keys
        in the internal embedders ``ModuleDict`` so the names appear in ``print(model)``.
        Defaults to ``["sensor_0", "sensor_1", ...]``.
    embed_dim : int, optional, default=32
        Tokenization dimension used by :class:`ObsTokenizer` for each sensor.
    meta_dim : int, optional, default=28
        Dimension of float point metadata features, consumed by :class:`ObsTokenizer`.
    fusion_dim : int, optional, default=512
        Output channel dimension after sensor fusion.
    gradient_checkpointing : bool, optional, default=False
        If ``True``, wraps each per-sensor forward pass with gradient
        checkpointing to trade compute for memory during training.
    torch_compile : bool, optional, default=False
        If ``True``, applies ``torch.compile`` to the forward method.

    Forward
    -------
    obs : torch.Tensor
        Flattened observation values with shape :math:`(N_{obs},)`.
    float_metadata : torch.Tensor
        Flattened float metadata with shape :math:`(N_{obs}, M_{float})`.
    pix : torch.Tensor
        Flattened pixel indices of each observation with shape :math:`(N_{obs},)`.
    local_channel : torch.Tensor
        Flattened local channel ids of each observation with shape :math:`(N_{obs},)`.
    local_platform : torch.Tensor
        Flattened local platform ids of each observation with shape :math:`(N_{obs},)`.
    obs_type : torch.Tensor
        Flattened observation type ids with shape :math:`(N_{obs},)`.
    offsets : torch.Tensor
        Cumulative exclusive-end row offsets into flattened
        observation tensors with shape :math:`(S, B, T)`, where the
        sensor order matches the initialization order.

        ``offsets[s, b, t]`` is the exclusive end row index for block ``(s, b, t)``
        under ``sensor -> batch -> time`` ordering (time changes fastest).
        So each sensor's rows are contiguous; within each sensor, each batch's
        rows are contiguous; and within each batch, each time window is contiguous.
    npix : int
        Number of pixels in the spatial grid.

    Outputs
    -------
    torch.Tensor
        Embedded observations of shape :math:`(B, D, T, N_{pix})`,
        where :math:`B` is batch size, :math:`D` is fusion dimension,
        :math:`T` is time windows, and :math:`N_{pix}` is number of grid pixels.
    """

    def __init__(
        self,
        nchannel_per_sensor: list[int],
        nplatform_per_sensor: list[int],
        sensor_names: list[str] | None = None,
        embed_dim: int = 32,
        meta_dim: int = 28,
        fusion_dim: int = 512,
        gradient_checkpointing: bool = False,
        torch_compile: bool = False,
    ):
        super().__init__()

        num_sensors = len(nchannel_per_sensor)
        if len(nplatform_per_sensor) != num_sensors:
            raise ValueError(
                f"nchannel_per_sensor and nplatform_per_sensor must have the same "
                f"length, got {len(nchannel_per_sensor)} and {len(nplatform_per_sensor)}"
            )

        if sensor_names is None:
            sensor_names = [f"sensor_{i}" for i in range(num_sensors)]
        elif len(set(sensor_names)) != num_sensors:
            raise ValueError(
                f"sensor_names must be unique and match length of nchannel_per_sensor, "
                f"got {len(set(sensor_names))} unique names and {num_sensors} sensors"
            )

        self.nchannel_per_sensor = list(nchannel_per_sensor)
        self.nplatform_per_sensor = list(nplatform_per_sensor)
        self.sensor_names = list(sensor_names)
        self.fusion_dim = fusion_dim

        # Separate embedders for each sensor, in sensor order.
        self.embedders = torch.nn.ModuleDict(
            {
                name: SensorEmbedder(
                    sensor_embed_dim=embed_dim,
                    meta_dim=meta_dim,
                    output_dim=self.fusion_dim,
                    nchannel=nchannel,
                    nplatform=nplatform,
                    gradient_checkpointing=gradient_checkpointing,
                )
                for name, nchannel, nplatform in zip(
                    sensor_names, nchannel_per_sensor, nplatform_per_sensor
                )
            }
        )

        self.sensor_fusion = UniformFusion(fusion_dim=self.fusion_dim)
        self.output_norm = torch.nn.LayerNorm(self.fusion_dim)
        if torch_compile:
            # use dynamic as each sample has variable observation count
            self.forward = torch.compile(self.forward, dynamic=True)

    def forward(
        self,
        obs: Float[torch.Tensor, "nobs"],
        float_metadata: Float[torch.Tensor, "nobs meta_dim"],
        pix: Int[torch.Tensor, "nobs"],
        local_channel: Int[torch.Tensor, "nobs"],
        local_platform: Int[torch.Tensor, "nobs"],
        obs_type: Int[torch.Tensor, "nobs"],
        offsets: Int[torch.Tensor, "sensors batch time"],
        npix: int,
    ) -> Float[torch.Tensor, "batch fusion_dim time npix"]:
        if not torch.compiler.is_compiling():
            if obs.ndim != 1:
                raise ValueError(
                    f"Expected obs of shape (nobs,), "
                    f"got {obs.ndim}D tensor with shape {tuple(obs.shape)}"
                )
            nobs = obs.shape[0]
            if float_metadata.ndim != 2:
                raise ValueError(
                    f"Expected float_metadata of shape (nobs, meta_dim), "
                    f"got {float_metadata.ndim}D tensor with shape {tuple(float_metadata.shape)}"
                )
            if float_metadata.shape[0] != nobs:
                raise ValueError(
                    f"Expected float_metadata with {nobs} rows (matching obs), "
                    f"got {float_metadata.shape[0]} in tensor with shape {tuple(float_metadata.shape)}"
                )
            for name, tensor in (
                ("pix", pix),
                ("local_channel", local_channel),
                ("local_platform", local_platform),
                ("obs_type", obs_type),
            ):
                if tensor.ndim != 1 or tensor.shape[0] != nobs:
                    raise ValueError(
                        f"Expected {name} of shape ({nobs},) matching obs, "
                        f"got tensor with shape {tuple(tensor.shape)}"
                    )
            num_sensors = len(self.embedders)
            if offsets.ndim != 3:
                raise ValueError(
                    f"Expected offsets of shape (sensors, batch, time), "
                    f"got {offsets.ndim}D tensor with shape {tuple(offsets.shape)}"
                )
            if offsets.shape[0] != num_sensors:
                raise ValueError(
                    f"Expected sensors={num_sensors} in offsets (matching nchannel_per_sensor), "
                    f"got {offsets.shape[0]} in tensor with shape {tuple(offsets.shape)}"
                )

        # Embed each sensor's observations separately
        obs_by_sensor = _split_by_sensor(
            obs=obs,
            float_metadata=float_metadata,
            pix=pix,
            local_channel=local_channel,
            local_platform=local_platform,
            obs_type=obs_type,
            offsets=offsets,
        )
        sensor_embeddings = []

        for sensor_obs, embedder in zip(obs_by_sensor, self.embedders.values()):
            (
                sensor_obs_values,
                sensor_float_metadata,
                sensor_pix,
                sensor_local_channel,
                sensor_local_platform,
                sensor_obs_type,
                sensor_offsets,
            ) = sensor_obs
            output = embedder(
                obs=sensor_obs_values,
                float_metadata=sensor_float_metadata,
                pix=sensor_pix,
                local_channel=sensor_local_channel,
                local_platform=sensor_local_platform,
                obs_type=sensor_obs_type,
                offsets=sensor_offsets,
                npix=npix,
            )  # (b, t, npix, c)
            sensor_embeddings.append(output)

        sensor_embeddings = torch.stack(
            sensor_embeddings, dim=0
        )

        # Fuse sensors
        out = self.sensor_fusion(sensor_embeddings)  # (b, t, npix, c)

        out = self.output_norm(out)
        out = out.permute(0, 3, 1, 2)

        return out
