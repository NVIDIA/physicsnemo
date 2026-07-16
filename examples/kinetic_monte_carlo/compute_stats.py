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

"""Compute z-score normalization stats for a preprocessed particles dataset."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import hydra
import torch
from dataset import ParticlesDataset
from dataset.dataset import _collate
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

# Floor added to delays before taking the log, for the log_delay companion stat.
_LOG_EPS: float = 1e-12


class Welford:
    """Numerically stable running mean / variance using Welford's algorithm.

    Stores ``(count, mean, M2)`` as Python floats so we can all-reduce them
    across ranks with Chan's parallel-combine formula.
    """

    def __init__(self) -> None:
        self.count: float = 0.0
        self.mean: float = 0.0
        self.M2: float = 0.0

    def update(self, batch: torch.Tensor) -> None:
        """Absorb a batch of scalar samples (any shape; flattened internally)."""
        x = batch.detach().reshape(-1)
        n_b = float(x.numel())
        if n_b == 0:
            return
        mean_b = float(x.mean().item())
        var_b = float(x.var(unbiased=False).item())
        m2_b = var_b * n_b

        if self.count == 0.0:
            self.count = n_b
            self.mean = mean_b
            self.M2 = m2_b
            return

        delta = mean_b - self.mean
        new_count = self.count + n_b
        self.mean = self.mean + delta * n_b / new_count
        self.M2 = self.M2 + m2_b + delta * delta * self.count * n_b / new_count
        self.count = new_count

    @property
    def std(self) -> float:
        """Population standard deviation derived from the running ``M2``."""
        if self.count < 2:
            return 0.0
        return math.sqrt(self.M2 / self.count)

    def all_reduce(self) -> None:
        """Combine statistics across all distributed ranks in place."""
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return
        world_size = torch.distributed.get_world_size()
        if world_size < 2:
            return

        # NCCL collectives require tensors on the distributed device (it rejects
        # CPU tensors), so build them on that device for multi-GPU runs. The
        # tolist() calls below move the gathered values back to the host.
        device = DistributedManager().device
        triples = torch.tensor(
            [self.count, self.mean, self.M2], dtype=torch.float64, device=device
        )
        gathered = [torch.zeros_like(triples) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, triples)

        total_count, total_mean, total_m2 = 0.0, 0.0, 0.0
        for t in gathered:
            c, m, m2 = t.tolist()
            if c == 0.0:
                continue
            if total_count == 0.0:
                total_count, total_mean, total_m2 = c, m, m2
                continue
            delta = m - total_mean
            new_count = total_count + c
            total_mean = total_mean + delta * c / new_count
            total_m2 = total_m2 + m2 + delta * delta * total_count * c / new_count
            total_count = new_count

        self.count, self.mean, self.M2 = total_count, total_mean, total_m2


@hydra.main(version_base="1.3", config_path="conf", config_name="config_compute_stats")
def main(cfg: DictConfig) -> None:
    """Compute dataset-wide z-score stats and dump them to JSON.

    Iterates the dataset once, accumulating a running mean/variance for the
    pooled spatial coordinates (``coords``, shared by every xyz column), each
    named particle feature, each named mesh field, the inter-event delay, and
    its log companion (``log_delay``). The resulting keys match the names the
    datapipe looks up at train / inference time.
    """
    DistributedManager.initialize()
    dist = DistributedManager()
    logger = PythonLogger("compute_stats")
    rank_zero = RankZeroLoggingWrapper(logger, dist)

    data_dir = to_absolute_path(str(cfg.dataset.data_dir))
    particle_feature_names = list(cfg.dataset.particle_feature_names)
    mesh_feature_names = list(cfg.dataset.mesh_feature_names)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silence "no stats_file" warning
        dataset = ParticlesDataset(
            data_dir,
            particle_feature_names=particle_feature_names,
            mesh_feature_names=mesh_feature_names,
            num_particles_max=cfg.dataset.num_particles_max,
            n_steps=1,
            stats_file=None,
        )
    rank_zero.info(f"Dataset size: {len(dataset)} samples")

    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=dist.world_size,
            rank=dist.rank,
            shuffle=False,
            drop_last=False,
        )
        if dist.world_size > 1
        else None
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.compute.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=cfg.compute.num_workers,
        collate_fn=_collate,
        pin_memory=False,
        drop_last=False,
    )

    coord_stats = Welford()
    delay_stats = Welford()
    log_delay_stats = Welford()
    particle_feature_stats = {name: Welford() for name in particle_feature_names}
    mesh_feature_stats = {name: Welford() for name in mesh_feature_names}

    for i, (batch, _meta) in enumerate(loader):
        mask = batch["particle_state"] > 0.5  # (B, S, K)

        # Spatial coords: pool mesh and real-particle coordinates (same unit).
        coord_stats.update(batch["mesh_coords"])  # (B, N, 3)
        if mask.any():
            coord_stats.update(batch["particle_coords"][mask])  # real particles only

        for name in mesh_feature_names:
            mesh_feature_stats[name].update(batch[name])  # (B, N)

        if mask.any():
            for name in particle_feature_names:
                particle_feature_stats[name].update(batch[name][mask])
            delays = batch["delay"][mask]
            delay_stats.update(delays)
            log_delay_stats.update(torch.log(delays.clamp(min=0.0) + _LOG_EPS))

        if i % cfg.io.logging_frequency == 0:
            rank_zero.info(
                f"Processed {i * cfg.compute.batch_size * max(1, dist.world_size)} / "
                f"{len(dataset)} samples"
            )

    coord_stats.all_reduce()
    delay_stats.all_reduce()
    log_delay_stats.all_reduce()
    for w in (*particle_feature_stats.values(), *mesh_feature_stats.values()):
        w.all_reduce()

    if dist.rank == 0:
        stats = {
            "coords": {"mean": float(coord_stats.mean), "std": float(coord_stats.std)},
            "delay": {"mean": float(delay_stats.mean), "std": float(delay_stats.std)},
            "log_delay": {
                "mean": float(log_delay_stats.mean),
                "std": float(log_delay_stats.std),
                "eps": _LOG_EPS,
            },
        }
        for name, w in {**particle_feature_stats, **mesh_feature_stats}.items():
            stats[name] = {"mean": float(w.mean), "std": float(w.std)}

        out_path = (
            Path(to_absolute_path(str(cfg.io.output_path)))
            if cfg.io.output_path is not None
            else Path(data_dir) / "stats.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)
        rank_zero.info(f"Wrote stats to {out_path}")
        rank_zero.info(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
