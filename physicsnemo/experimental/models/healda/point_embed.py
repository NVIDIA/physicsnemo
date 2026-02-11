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

import importlib
import logging
import math
from typing import Any

import torch

from physicsnemo.core.module import Module
from physicsnemo.core.version_check import check_version_spec

from .scatter_aggregator import ScatterAggregator

HEALPIXPAD_AVAILABLE = check_version_spec("earth2grid", "0.1.0", hard_fail=False)

if HEALPIXPAD_AVAILABLE:
    _healpix_mod = importlib.import_module("earth2grid.healpix")
    hpx_grid = _healpix_mod.Grid
    HEALPIX_PAD_XY = _healpix_mod.HEALPIX_PAD_XY
    HEALPIX_NEST = _healpix_mod.NEST
else:
    HEALPIX_PAD_XY = None
    HEALPIX_NEST = None

    def hpx_grid(*args, **kwargs):
        """Dummy symbol for missing earth2grid backend."""
        raise ImportError(
            (
                "earth2grid is not installed, cannot use it as a backend for HEALPix padding.\n"
                "Install earth2grid from https://github.com/NVlabs/earth2grid.git to enable the accelerated path.\n"
                "pip install --no-build-isolation https://github.com/NVlabs/earth2grid/archive/main.tar.gz"
            )
        )


def _prod(shape):
    out = 1
    for s in shape:
        out *= s
    return out


def _offsets_to_batch_idx(offsets: torch.Tensor) -> torch.Tensor:
    r"""Convert 3D cumulative-end offsets to flattened :math:`(B, T)` batch indices."""
    S, B, T = offsets.shape
    bt_size = B * T

    offsets_flat = offsets.flatten()
    offsets_with_zero = torch.cat(
        [torch.tensor([0], device=offsets.device, dtype=offsets.dtype), offsets_flat]
    )
    sizes = offsets_with_zero.diff()

    window_indices = torch.arange(
        sizes.shape[0], dtype=torch.long, device=offsets.device
    )
    bt_indices = window_indices % bt_size
    return bt_indices.repeat_interleave(sizes)


@torch.compiler.disable
def _split_by_sensor(
    obs: torch.Tensor,
    float_metadata: torch.Tensor,
    pix: torch.Tensor,
    local_channel: torch.Tensor,
    local_platform: torch.Tensor,
    obs_type: torch.Tensor,
    offsets: torch.Tensor,
    expected_num_sensors: int,
) -> list[tuple[torch.Tensor, ...]]:
    """Split flattened observation tensors into per-sensor slices from ``offsets``."""
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
    if nsensors != expected_num_sensors:
        raise ValueError(
            f"offsets first dim ({nsensors}) must match expected_num_sensors ({expected_num_sensors})"
        )

    out: list[tuple[torch.Tensor, ...]] = []
    total_obs = obs.shape[0]

    for sensor_idx in range(nsensors):
        end = offsets[sensor_idx, -1, -1].item()
        start = 0 if sensor_idx == 0 else offsets[sensor_idx - 1, -1, -1].item()
        sensor_offsets = offsets[sensor_idx : sensor_idx + 1] - start

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


logger = logging.getLogger(__name__)


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
        obs: torch.Tensor,
        float_metadata: torch.Tensor,
        obs_type: torch.Tensor,
    ) -> torch.Tensor:
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
    fusion_dim : int, optional, default=256
        Dimension of the fused embedding.

    Forward
    -------
    sensor_embeddings : torch.Tensor
        Sensor embeddings of shape :math:`(N_{sensors}, *, D)`.

    Outputs
    -------
    torch.Tensor
        Fused embedding of shape :math:`(*, D)`.
    """

    def __init__(self, fusion_dim: int = 256):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.norm = torch.nn.LayerNorm(self.fusion_dim)

    def forward(
        self, sensor_embeddings: torch.Tensor
    ) -> torch.Tensor:
        num_sensors = sensor_embeddings.shape[0]
        sensor_embeddings = self.norm(sensor_embeddings)
        return sensor_embeddings.sum(dim=0) / math.sqrt(num_sensors)


class SensorEmbedder(Module):
    r"""Embeds observations from a single sensor onto the HEALPix spatial grid.

    Pipeline:
      1. Per-observation tokenization via :class:`ObsTokenizer`
      2. Scatter aggregation onto HEALPix grid via :class:`ScatterAggregator`
      3. Final projection to output dimension

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
    hpx_level : int, optional, default=6
        Model HEALPix grid level to aggregate features to.
    n_embed : int, optional, default=1024
        Size of observation type embedding table.
    embed_dim : int, optional, default=4
        Dimension of observation type embeddings.
    use_checkpoint : bool, optional, default=False
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
        3D offsets with shape :math:`(1, B, T)` indicating end of each batch/time window.
    hpx_level : int
        HEALPix level used by ``pix``.

    Outputs
    -------
    torch.Tensor
        Sensor embedding grid with shape :math:`(B, T, N_{pix}, D_{out})`
        in NEST order.
    """

    def __init__(
        self,
        *,
        nplatform: int,
        nchannel: int,
        sensor_embed_dim: int = 32,
        output_dim: int = 512,
        meta_dim: int = 28,
        hpx_level: int = 6,
        n_embed: int = 1024,
        embed_dim: int = 4,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.sensor_embed_dim = sensor_embed_dim
        self.output_dim = output_dim
        self.hpx_level = hpx_level
        self.npix = 12 * 4**hpx_level
        self.use_checkpoint = use_checkpoint
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
            nchannel=nchannel,
            nplatform=nplatform,
            npix=self.npix,
        )

    def aggregate(
        self,
        embedded_obs: torch.Tensor,
        pix: torch.Tensor,
        local_channel: torch.Tensor,
        local_platform: torch.Tensor,
        hpx_level: int,
        batch_idx: torch.Tensor,
        nbatch: int,
    ) -> torch.Tensor:
        """Aggregate observations to spatial grid and project to output dimension."""
        # Convert observation pixels to aggregator grid resolution
        aggregation_pix = pix // int(4.0 ** (hpx_level - self.hpx_level))

        # Build combined bucket ID
        bucket_id = local_platform * self.nchannel + local_channel
        return self.scatter_infill_aggregator(
            obs_features=embedded_obs,
            batch_idx=batch_idx,
            pix=aggregation_pix,
            bucket_id=bucket_id,
            nbatch=nbatch,
        )

    def _forward(
        self,
        obs: torch.Tensor,
        float_metadata: torch.Tensor,
        pix: torch.Tensor,
        local_channel: torch.Tensor,
        local_platform: torch.Tensor,
        obs_type: torch.Tensor,
        offsets: torch.Tensor,
        hpx_level: int,
    ) -> torch.Tensor:
        batch_idx = _offsets_to_batch_idx(offsets)
        batch_dims = offsets.shape[-2:]  # (S, B, T) -> (B, T)
        nbatch = _prod(batch_dims)

        embedded_obs = self.obs_tokenizer(obs, float_metadata, obs_type)

        # Aggregator handles empty batches internally to keep all parameters in the computation graph
        output = self.aggregate(
            embedded_obs,
            pix,
            local_channel,
            local_platform,
            hpx_level,
            batch_idx,
            nbatch,
        )  # NEST (nbatch, npix, output_dim)
        output = output.view(*batch_dims, self.npix, self.output_dim)

        return output

    def forward(
        self,
        obs: torch.Tensor,
        float_metadata: torch.Tensor,
        pix: torch.Tensor,
        local_channel: torch.Tensor,
        local_platform: torch.Tensor,
        obs_type: torch.Tensor,
        offsets: torch.Tensor,
        hpx_level: int,
    ) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward,
                obs,
                float_metadata,
                pix,
                local_channel,
                local_platform,
                obs_type,
                offsets,
                hpx_level,
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
                hpx_level,
            )


class MultiSensorObsEmbedding(Module):
    r"""Multi-sensor observation embedding onto a HEALPix grid.

    Embeds observations from multiple sensor types into a unified representation
    by applying per-sensor embedders and fusing the results.

    Parameters
    ----------
    sensor_configs : list[dict[str, Any]]
        Ordered per-sensor configs, one dict per sensor. Required keys:

        - ``name``: sensor name (bookkeeping, unused).
        - ``nchannel``: number of sensor channels.
        - ``nplatform``: number of sensor platforms.

        The list order must match the loaded observations: row ``i`` in ``offsets`` must
        correspond to ``sensor_configs[i]``.

        Example:
        ``[{"name": "atms", "nchannel": 22, "nplatform": 2}]``
    hpx_level : int
        HEALPix grid level for all sensors.
    embed_dim : int, optional, default=32
        Tokenization dimension used by :class:`ObsTokenizer` for each sensor.
    meta_dim : int, optional, default=28
        Dimension of float point metadata features, consumed by :class:`ObsTokenizer`.
    fusion_dim : int, optional, default=512
        Output channel dimension after sensor fusion.
    use_checkpoint : bool, optional, default=False
        If ``True``, applies gradient checkpointing to reduce memory usage.
    compile : bool, optional, default=False
        If ``True``, compiles the forward function for improved performance.

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
        observation tensors with shape :math:`(S, B, T)`, aligned to
        ``sensor_configs``.

        ``offsets[s, b, t]`` is the exclusive end row index for block ``(s, b, t)``
        under ``sensor -> batch -> time`` ordering (time changes fastest).
        So each sensor's rows are contiguous; within each sensor, each batch's
        rows are contiguous; and within each batch, each time window is contiguous.
    hpx_level : int
        HEALPix level used by input ``pix``.

    Outputs
    -------
    torch.Tensor
        Embedded observations of shape :math:`(B, D, T, N_{pix})` in HEALPIX_PAD_XY
        pixel order, where :math:`B` is batch size, :math:`D` is fusion dimension,
        :math:`T` is time steps, and :math:`N_{pix}` is number of HEALPix pixels.
    """

    def __init__(
        self,
        sensor_configs: list[dict[str, Any]],
        hpx_level: int,
        embed_dim: int = 32,
        meta_dim: int = 28,
        fusion_dim: int = 512,
        use_checkpoint: bool = False,
        compile: bool = False,
    ):
        super().__init__()

        self._validate_sensor_configs(sensor_configs)
        self.sensor_configs = sensor_configs
        self.sensor_names = [config["name"] for config in self.sensor_configs]
        self.fusion_dim = fusion_dim
        self.hpx_level = hpx_level
        self.npix = 12 * 4**hpx_level

        # Aggregate onto NEST order grid
        self.grid = hpx_grid(hpx_level, pixel_order=HEALPIX_NEST)

        # Separate embedders for each sensor, in config order.
        self.embedders = torch.nn.ModuleList(
            [
                SensorEmbedder(
                    sensor_embed_dim=embed_dim,
                    meta_dim=meta_dim,
                    output_dim=self.fusion_dim,
                    hpx_level=self.hpx_level,
                    nchannel=config["nchannel"],
                    nplatform=config["nplatform"],
                    use_checkpoint=use_checkpoint,
                )
                for config in self.sensor_configs
            ]
        )

        self.sensor_fusion = UniformFusion(fusion_dim=self.fusion_dim)
        self.output_norm = torch.nn.LayerNorm(self.fusion_dim)
        if compile:
            self.forward = torch.compile(self.forward, dynamic=True)

    @staticmethod
    def _validate_sensor_configs(
        sensor_configs: list[dict[str, Any]],
    ) -> None:
        if len(sensor_configs) == 0:
            raise ValueError("sensor_configs must contain at least one sensor config")
        for idx, config in enumerate(sensor_configs):
            if not isinstance(config, dict):
                raise TypeError(
                    f"sensor_configs[{idx}] must be a dict, got {type(config).__name__}"
                )
            missing = [
                key
                for key in ("name", "nchannel", "nplatform")
                if key not in config
            ]
            if missing:
                raise ValueError(
                    f"sensor_configs[{idx}] is missing required key(s): {missing}"
                )


    def _reorder(self, x: torch.Tensor) -> torch.Tensor:
        r"""Reorder from NEST to HEALPIX_PAD_XY.

        Parameters
        ----------
        x : torch.Tensor
            Tensor with shape :math:`(..., N_{pix}, C)`.

        Returns
        -------
        torch.Tensor
            Tensor with shape :math:`(..., N_{pix}, C)`.
        """
        x = self.grid.reorder(
            HEALPIX_PAD_XY, x.transpose(-1, -2),
        ).transpose(-1, -2)
        return x

    def forward(
        self,
        obs: torch.Tensor,
        float_metadata: torch.Tensor,
        pix: torch.Tensor,
        local_channel: torch.Tensor,
        local_platform: torch.Tensor,
        obs_type: torch.Tensor,
        offsets: torch.Tensor,
        hpx_level: int,
    ) -> torch.Tensor:
        # Embed each sensor's observations separately
        obs_by_sensor = _split_by_sensor(
            obs=obs,
            float_metadata=float_metadata,
            pix=pix,
            local_channel=local_channel,
            local_platform=local_platform,
            obs_type=obs_type,
            offsets=offsets,
            expected_num_sensors=len(self.embedders),
        )
        sensor_embeddings = []

        for sensor_obs, embedder in zip(obs_by_sensor, self.embedders):
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
                hpx_level=hpx_level,
            )  # (b, t, x, c)
            sensor_embeddings.append(output)

        sensor_embeddings = torch.stack(
            sensor_embeddings, dim=0
        )

        # Fuse sensors
        num_sensors, b, t, x, c = sensor_embeddings.shape
        sensor_embeddings_flat = sensor_embeddings.view(num_sensors, b * t * x, c)
        fused_flat = self.sensor_fusion(sensor_embeddings_flat)  # (b*t*x, fusion_dim)

        out = fused_flat.view(b, t, x, self.fusion_dim)  # (b, t, x, fusion_dim)

        out = self._reorder(out)
        out = self.output_norm(out)
        out = out.permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        return out
