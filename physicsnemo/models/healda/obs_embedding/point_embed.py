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
import earth2grid
import math
import torch
import logging

import healda.profiling
from healda.datasets.da.types import (
    UnifiedObservation,
    split_by_sensor,
)
from healda.config.models import SensorEmbedderConfig, ModelSensorConfig
from healda.models.obs_embedding.scatter_infill_aggregators import ScatterAggregator


def _prod(shape):
    out = 1
    for s in shape:
        out *= s
    return out


GLOBAL_MAX_CHANNELS = 1024
GLOBAL_MAX_PLATFORM = 1024

logger = logging.getLogger(__name__)


class ObsTokenizer(torch.nn.Module):
    """Tokenizes individual observations using metadata + measurement + embedding tables into feature tokens.

    This creates intermediate token representations that will be aggregated and projected to final embeddings.

    Args:
        meta_dim: Dimension of static metadata features
        out_dim: Output token dimension
        platform_id_map: Tensor mapping global platform IDs to local indices
        n_embed: Size of observation type embedding table
        nchannel: Max number of channels (for channel embedding table)
        nplatform: Max number of platforms (for platform embedding table)
        embed_dim: Dimension of observation type embeddings
        use_channel_platform_embedding_table: Use channel and platform embedding tables
    """

    def __init__(
        self,
        meta_dim: int,
        out_dim: int,
        platform_id_map: torch.tensor,
        n_embed: int = 1024,
        nchannel: int = 1024,
        nplatform: int = 1024,
        embed_dim: int = 4,
        use_channel_platform_embedding_table: bool = True,
    ):
        super().__init__()

        if nchannel > GLOBAL_MAX_CHANNELS or nplatform > GLOBAL_MAX_PLATFORM:
            raise ValueError(
                f"nchannel {nchannel} or nplatform {nplatform} is greater than the global max {GLOBAL_MAX_CHANNELS} or {GLOBAL_MAX_PLATFORM}"
            )

        self.use_channel_platform_embedding_table = use_channel_platform_embedding_table
        if self.use_channel_platform_embedding_table:
            self.channel_embedding = torch.nn.Embedding(GLOBAL_MAX_CHANNELS, embed_dim)
            self.platform_embedding = torch.nn.Embedding(GLOBAL_MAX_PLATFORM, embed_dim)
        self.embed_table = torch.nn.Embedding(n_embed, embed_dim)

        self.register_buffer("platform_id_map", platform_id_map)

        mlp_in_dim = (
            1
            + meta_dim
            + embed_dim * (3 if self.use_channel_platform_embedding_table else 1)
        )
        mlp_out_dim = out_dim - 1
        hidden_dim = out_dim * 2 if out_dim <= 32 else out_dim

        self.meta_mlp = torch.nn.Sequential(
            torch.nn.Linear(mlp_in_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, mlp_out_dim),
        )

    def forward(self, obs: UnifiedObservation) -> torch.Tensor:
        """
        Tokenize observations into feature tokens.

        Args:
            obs: UnifiedObservation containing observations and metadata

        Returns:
            (nobs, out_dim)
        """
        # Extract columns from int_metadata (n_obs, 6)
        channel_id = obs.int_metadata[:, obs.bucket_index.local_channel]
        platform_id_global = obs.int_metadata[:, obs.bucket_index.platform]
        obs_type_id = obs.int_metadata[:, obs.bucket_index.obs_type]
        embed_vec = self.embed_table(obs_type_id)
        if self.use_channel_platform_embedding_table:
            channel_emb = self.channel_embedding(channel_id)
            local_platform_id = self.platform_id_map[platform_id_global]
            platform_emb = self.platform_embedding(local_platform_id)

        x_in = torch.cat(
            [
                obs.obs.unsqueeze(-1),
                obs.float_metadata,
                embed_vec,
                *(
                    [channel_emb, platform_emb]
                    if self.use_channel_platform_embedding_table
                    else []
                ),
            ],
            dim=-1,
        )
        mlp_out = self.meta_mlp(x_in)
        encoded = torch.cat([obs.obs.unsqueeze(-1), mlp_out], dim=-1)
        return encoded


# Sensor fusion module
class UniformFusion(torch.nn.Module):
    """
    Uniform weighting across all sensors with normalization for number of sensors.

    Simple averaging with 1/sqrt(N) scaling to maintain variance.
    """

    def __init__(self, fusion_dim: int = 256):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.norm = torch.nn.LayerNorm(self.fusion_dim)

    def forward(
        self, projected: torch.Tensor, sensor_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            projected: (num_sensors, ..., fusion_dim)
            sensor_ids: (num_sensors,) - not used, for API consistency
        Returns:
            (..., fusion_dim)
        """
        num_sensors = projected.shape[0]

        projected = self.norm(projected)
        return projected.sum(dim=0) / math.sqrt(num_sensors)


class SensorEmbedder(torch.nn.Module):
    """Unified sensor embedding for any observation source (satellite, conventional, etc.).

    Pipeline:
      1. Per-obs tokenization via MLP
      2. Scatter aggregation
      3. Final projection to output_dim

    Args:
        platform_ids: List of global platform IDs for this sensor
        sensor_embed_dim: Internal feature dimension
        output_dim: Final output dimension of a sensor embedding
        meta_dim: Dimension of static metadata features
        hpx_level: HEALPix grid level
        n_embed: Size of observation type embedding table
        embed_dim: Dimension of observation type embeddings
        nchannel: Max number of channels
        use_checkpoint: Apply gradient checkpointing
    """

    def __init__(
        self,
        platform_ids: list[int],
        sensor_embed_dim: int = 32,
        output_dim: int = 256,
        meta_dim: int = 32,
        hpx_level: int = 6,
        # Embedding table config
        n_embed: int = 1024,  # Large sparse table for observation types
        embed_dim: int = 4,
        nchannel: int = 1024,  # Max channels
        use_checkpoint: bool = False,
        use_channel_platform_embedding_table: bool = True,
    ):
        super().__init__()

        platform_id_map_size = GLOBAL_MAX_PLATFORM + 1
        # Map global platform IDs to local indices for embedding lookup
        if len(platform_ids) == 0:
            # Platform-agnostic sensor (e.g., conv): all platforms map to index 0
            # Use a map large enough to cover all possible platform IDs
            self.register_buffer(
                "platform_id_map",
                torch.zeros(
                    platform_id_map_size, dtype=torch.long
                ),  # All platforms → 0
            )
            nplatform = 1  # Single embedding for all platforms
        else:
            # Normal sensor: create lookup map for specific platforms
            self.register_buffer(
                "platform_id_map",
                torch.full((platform_id_map_size,), -1, dtype=torch.long),
            )
            for local_idx, global_id in enumerate(platform_ids):
                self.platform_id_map[global_id] = local_idx
            nplatform = len(platform_ids)

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
            platform_id_map=self.platform_id_map,
            n_embed=n_embed,
            nchannel=nchannel,
            nplatform=nplatform,
            embed_dim=embed_dim,
            use_channel_platform_embedding_table=use_channel_platform_embedding_table,
        )

        # Aggregation setup - outputs (nbatch, npix, output_dim)
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
        obs: UnifiedObservation,
        batch_idx: torch.Tensor,
        nbatch: int,
    ) -> torch.Tensor:
        """
        Aggregate observations to spatial grid and project to output dimension.

        Args:
            embedded_obs: (nobs, sensor_embed_dim) tokenized observations
            obs: UnifiedObservation
            batch_idx: (nobs,) batch index for each obs
            nbatch: product of batch dimensions

        Returns:
            (nbatch, npix, output_dim) aggregated spatial grid in HEALPIX_PAD_XY format
        """
        obs_pix = obs.int_metadata[:, obs.bucket_index.pix]
        channel = obs.int_metadata[:, obs.bucket_index.local_channel]
        platform_global = obs.int_metadata[:, obs.bucket_index.platform]
        # TODO remove....just let every sensor use platform_global
        # there are not many platforms. --would result in very large mlp in the aggregation channel/platform mixing layer
        platform = self.platform_id_map[platform_global]

        # Convert observation pixels to aggregator grid resolution
        pix = obs_pix // int(4.0 ** (obs.hpx_level - self.hpx_level))

        # Build combined bucket ID
        bucket_id = platform * self.nchannel + channel
        return self.scatter_infill_aggregator(
            obs_features=embedded_obs,
            batch_idx=batch_idx,
            pix=pix,
            bucket_id=bucket_id,
            nbatch=nbatch,
        )

    def _forward(self, obs: UnifiedObservation):
        batch_dims = obs.batch_dims  # () if offsets is None, (B, T) otherwise
        nbatch = _prod(batch_dims)  # 1 if batch_dims==(), B*T otherwise

        embedded_obs = self.obs_tokenizer(obs)

        batch_idx = obs.batch_idx

        # Aggregator handles empty batches internally to keep all parameters in the computation graph
        output = self.aggregate(
            embedded_obs, obs, batch_idx, nbatch
        )  # NEST (nbatch, npix, output_dim)

        if len(batch_dims) == 0:
            output = output.view(self.npix, self.output_dim)
        else:
            output = output.view(*batch_dims, self.npix, self.output_dim)

        return output

    @healda.profiling.nvtx(enabled=False)
    def forward(self, obs: UnifiedObservation) -> torch.Tensor:
        """
        Embed observations from a single sensor onto a spatial grid.

        Args:
            obs: UnifiedObservation for a single sensor. Observations are flattened
                 across batch/time dimensions; `obs.offsets` defines the structure.

        Returns:
            If offsets=None: (npix, output_dim) - single spatial grid in NEST order
            If offsets present: (*offsets.shape, npix, output_dim) - grid in NEST order
            e.g., (batch, time, npix, output_dim) if offsets is (batch, time)
        """
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, obs, use_reentrant=False
            )
        else:
            return self._forward(obs)


class MultiSensorObsEmbedding(torch.nn.Module):
    """Multi-sensor observation embedding.

    Args:
        sensor_embedder_config: Config with embedding hyperparameters
        sensors: Dict mapping sensor names to ModelSensorConfig
        hpx_level: HEALPix grid level for all sensors
        use_checkpoint: Apply gradient checkpointing
    """

    def __init__(
        self,
        sensor_embedder_config: SensorEmbedderConfig,
        sensors: dict[str, ModelSensorConfig],
        hpx_level: int,
        use_checkpoint: bool = False,
        compile: bool = True,
    ):
        super().__init__()

        # Store config values
        self.sensors = sensors
        self.sensor_names = list(self.sensors.keys())
        self.sensor_ids = [cfg.sensor_id for cfg in self.sensors.values()]
        self.fusion_dim = sensor_embedder_config.fusion_dim
        self.use_channel_platform_embedding_table = (
            sensor_embedder_config.use_channel_platform_embedding_table
        )
        self.hpx_level = hpx_level
        self.npix = 12 * 4**hpx_level

        # src grid of sensor embeddings
        self.grid = earth2grid.healpix.Grid(
            hpx_level, pixel_order=earth2grid.healpix.NEST
        )

        embed_cfg = sensor_embedder_config
        # Separate embedders for each sensor.
        self.embedder = torch.nn.ModuleDict(
            {
                str(sensor_cfg.sensor_id): SensorEmbedder(
                    sensor_embed_dim=embed_cfg.embed_dim,
                    meta_dim=embed_cfg.meta_dim,
                    output_dim=self.fusion_dim,
                    hpx_level=self.hpx_level,
                    nchannel=sensor_cfg.nchannel,
                    platform_ids=sensor_cfg.platform_ids,
                    use_checkpoint=use_checkpoint,
                    use_channel_platform_embedding_table=self.use_channel_platform_embedding_table,
                )
                for sensor_cfg in self.sensors.values()
            }
        )

        self.sensor_fusion = UniformFusion(fusion_dim=self.fusion_dim)

        self.output_norm = torch.nn.LayerNorm(self.fusion_dim)

        self.register_buffer(
            "sensor_ids_tensor", torch.tensor(self.sensor_ids, dtype=torch.int32)
        )
        if compile:
            self.forward = torch.compile(self.forward, dynamic=True)

    def _reorder(self, x: torch.Tensor) -> torch.Tensor:
        """Reorder from NEST to HEALPIX_PAD_XY. Input shape: (..., npix, c)"""
        x = self.grid.reorder(
            earth2grid.healpix.HEALPIX_PAD_XY, x.transpose(-1, -2)
        ).transpose(-1, -2)
        return x

    @healda.profiling.nvtx(enabled=False)
    def forward(self, obs: UnifiedObservation) -> torch.Tensor:
        """
        Args:
            obs: UnifiedObservation with flattened observations from all sensors

        Returns:
            (batch, fusion_dim, time, npix) in HEALPIX_PAD_XY format
        """
        if obs.batch_dims is None:
            raise ValueError(
                f"offset batch dimensions must be (batch, time) in MultiSensorObsEmbedding, got {obs.batch_dims}"
            )

        # Embed each sensor's obs separately
        obs_by_sensor = split_by_sensor(obs, self.sensor_ids)
        sensor_embeddings = []

        for sensor_id_str, embedder in self.embedder.items():
            sensor_id = int(sensor_id_str)
            sensor_obs: UnifiedObservation = obs_by_sensor[sensor_id]
            output = embedder(sensor_obs)  # (b, t, x, c)
            sensor_embeddings.append(output)

        sensor_embeddings = torch.stack(
            sensor_embeddings, dim=0
        )  # (num_sensors, b, t, x, c)

        # Fuse sensors
        num_sensors, b, t, x, c = sensor_embeddings.shape
        sensor_embeddings_flat = sensor_embeddings.view(num_sensors, b * t * x, c)
        fused_flat = self.sensor_fusion(
            sensor_embeddings_flat, self.sensor_ids_tensor
        )  # (b*t*x, fusion_dim)

        out = fused_flat.view(b, t, x, self.fusion_dim)  # (b, t, x, fusion_dim)

        out = self._reorder(out)
        out = self.output_norm(out)
        out = out.permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        return out
