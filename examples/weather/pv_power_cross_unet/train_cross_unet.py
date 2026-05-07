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

"""Train the experimental :class:`CrossUnet` PV-power forecaster on synthetic data.

This example fabricates a deterministic multivariate time series that mimics
photovoltaic power forecasting (a diurnal irradiance signal plus a noisy
power channel and two lagged history channels). It demonstrates the
PhysicsNeMo training-loop primitives -- ``DistributedManager``,
``LaunchLogger``, ``save_checkpoint``/``load_checkpoint`` -- against the
multi-input ``CrossUnet`` forward signature without requiring any external
dataset download.
"""

from __future__ import annotations

import math

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from physicsnemo.distributed import DistributedManager
from physicsnemo.experimental.models.pv_power import CrossUnet
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import LaunchLogger, PythonLogger


def _build_long_series(
    *,
    n_steps: int,
    target_channels: int,
    weather_channels: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one long deterministic series the dataset slices into windows."""
    t = torch.arange(n_steps, dtype=torch.float32)

    # Diurnal irradiance (96 = 24 h at 15-min cadence).
    irradiance = torch.clamp(torch.sin(2 * math.pi * t / 96.0), min=0.0)
    # Slow modulation of conversion efficiency.
    efficiency = 0.7 + 0.3 * torch.sin(t / 97.0)
    # Noisy primary power signal in [0, 1].
    noise = 0.05 * torch.randn(n_steps, generator=generator)
    power = torch.clamp(irradiance * efficiency + noise, min=0.0, max=1.0)

    # Weather feature pool (truncated/padded to ``weather_channels``).
    temp_proxy = 0.5 + 0.5 * torch.cos(2 * math.pi * t / 96.0)
    humidity_proxy = 0.5 + 0.3 * torch.sin(2 * math.pi * t / 97.0)
    weather_pool = torch.stack([irradiance, temp_proxy, humidity_proxy], dim=-1)
    if weather_channels <= weather_pool.shape[-1]:
        weather = weather_pool[:, :weather_channels]
    else:
        pad = torch.zeros(n_steps, weather_channels - weather_pool.shape[-1])
        weather = torch.cat([weather_pool, pad], dim=-1)

    # Target features = (target_channels - 1) lagged copies of power + power itself.
    lagged = [
        torch.roll(power, shifts=lag).unsqueeze(-1) for lag in range(1, target_channels)
    ]
    targets = torch.cat([*lagged, power.unsqueeze(-1)], dim=-1)
    return (
        targets,
        weather,
    )  # shapes: (n_steps, target_channels), (n_steps, weather_channels)


class SyntheticPVDataset(Dataset):
    """Sliding-window dataset over a deterministic synthetic series."""

    def __init__(
        self,
        *,
        num_samples: int,
        seq_len: int,
        pred_len: int,
        target_channels: int,
        weather_channels: int,
        seed: int,
    ) -> None:
        # Generate enough data for ``num_samples`` non-overlapping windows.
        n_steps = num_samples * (seq_len + pred_len)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.targets, self.weather = _build_long_series(
            n_steps=n_steps,
            target_channels=target_channels,
            weather_channels=weather_channels,
            generator=generator,
        )
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_samples = num_samples
        self.target_channels = target_channels
        self.weather_channels = weather_channels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * (self.seq_len + self.pred_len)
        window_end = start + self.seq_len
        target_end = window_end + self.pred_len

        x_enc = self.targets[start:window_end]  # (L, target_channels)
        w_enc = self.weather[start:window_end]  # (L, weather_channels)
        # "Historical weather" reuses the encoder weather window for this
        # synthetic example; in real deployments this would be a separate
        # measurement stream.
        seq_w_nwp_hist = w_enc.clone()
        seq_x_hist = x_enc[:, :-1].clone()  # (L, target_channels - 1)
        target = self.targets[window_end:target_end]  # (H, target_channels)
        return {
            "x_enc": x_enc,
            "w_enc": w_enc,
            "seq_w_nwp_hist": seq_w_nwp_hist,
            "seq_x_hist": seq_x_hist,
            "target": target,
        }


def _move(batch: dict[str, torch.Tensor], device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()
    LaunchLogger.initialize()
    logger = PythonLogger("cross_unet")

    torch.manual_seed(cfg.seed)

    train_ds = SyntheticPVDataset(
        num_samples=cfg.num_train_samples,
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        target_channels=cfg.target_channels,
        weather_channels=cfg.weather_channels,
        seed=cfg.seed,
    )
    valid_ds = SyntheticPVDataset(
        num_samples=cfg.num_valid_samples,
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        target_channels=cfg.target_channels,
        weather_channels=cfg.weather_channels,
        seed=cfg.seed + 1,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size_valid, shuffle=False)

    model = CrossUnet(
        target_channels=cfg.target_channels,
        weather_channels=cfg.weather_channels,
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        seg_len=cfg.seg_len,
        e_layers=cfg.e_layers,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        attention_kind=cfg.attention_kind,
        merge_kind=cfg.merge_kind,
        use_bottleneck_in_decoder=cfg.use_bottleneck_in_decoder,
    ).to(dist.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.start_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=cfg.lr_scheduler_gamma
    )

    loaded_epoch = load_checkpoint(
        "./checkpoints",
        models=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=dist.device,
    )

    target_slice = slice(-cfg.target_channels, None)

    for epoch in range(max(1, loaded_epoch + 1), cfg.max_epochs + 1):
        model.train()
        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=len(train_loader),
            epoch_alert_freq=1,
        ) as log:
            for batch in train_loader:
                batch = _move(batch, dist.device)
                optimizer.zero_grad()
                pred = model(
                    batch["x_enc"],
                    batch["w_enc"],
                    batch["seq_w_nwp_hist"],
                    batch["seq_x_hist"],
                )
                loss = F.mse_loss(pred[..., target_slice], batch["target"])
                loss.backward()
                optimizer.step()
                log.log_minibatch({"loss": loss.detach()})
            scheduler.step()
            log.log_epoch({"learning_rate": optimizer.param_groups[0]["lr"]})

        # Validation
        model.eval()
        valid_losses = []
        with torch.no_grad():
            for batch in valid_loader:
                batch = _move(batch, dist.device)
                pred = model(
                    batch["x_enc"],
                    batch["w_enc"],
                    batch["seq_w_nwp_hist"],
                    batch["seq_x_hist"],
                )
                valid_losses.append(
                    F.mse_loss(pred[..., target_slice], batch["target"]).item()
                )
        with LaunchLogger("valid", epoch=epoch) as log:
            log.log_epoch({"loss": sum(valid_losses) / max(1, len(valid_losses))})

        if epoch % cfg.checkpoint_save_freq == 0:
            save_checkpoint(
                "./checkpoints",
                models=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
