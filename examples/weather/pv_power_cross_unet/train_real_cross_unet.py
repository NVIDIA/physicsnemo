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

"""Train or run inference with CrossUnet on the bundled PV-power CSV datasets."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from physicsnemo.experimental.models.pv_power import CrossUnet
from physicsnemo.utils.logging import PythonLogger

Split = Literal["train", "valid", "test"]
WeatherSource = Literal["nwp", "satellite"]


@dataclass
class RealPVDataConfig:
    """Configuration for the bundled real PV-power CSV datasets."""

    dataset_root: str | Path = "./dataset"
    station_name: str = "KDASC"
    seq_len: int = 96
    pred_len: int = 16
    weather_source: WeatherSource = "nwp"
    max_samples: int | None = None


@dataclass
class ArrayScaler:
    """Small serializable standard scaler for numpy arrays."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "ArrayScaler":
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std < 1.0e-6, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    @classmethod
    def from_state(cls, state: dict[str, list[float]]) -> "ArrayScaler":
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            std=np.asarray(state["std"], dtype=np.float32),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse_last_channel(self, values: np.ndarray) -> np.ndarray:
        return values * self.std[-1] + self.mean[-1]

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


@dataclass
class EarlyStopping:
    """Track validation loss and stop after stale epochs exceed patience."""

    patience: int = 5
    min_delta: float = 0.0
    best_loss: float = float("inf")
    stale_epochs: int = 0

    def __post_init__(self) -> None:
        if self.patience < 0:
            raise ValueError(f"patience must be non-negative, got {self.patience}.")

    def step(self, loss: float) -> tuple[bool, bool]:
        """Update state and return ``(improved, should_stop)``."""
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.stale_epochs = 0
            return True, False

        self.stale_epochs += 1
        return False, self.stale_epochs >= self.patience


def infer_station_columns(
    station_name: str,
    weather_source: WeatherSource,
) -> tuple[str, list[str], list[str]]:
    """Return target/history/weather columns following the upstream PV-power setup."""
    if station_name in {"KDASC", "yulara"}:
        return "Active_Power", ["Active_Power"], ["SWR"]

    target_col = "power"
    history_cols = [
        "lmd_totalirrad",
        "lmd_diffuseirrad",
        "lmd_temperature",
        "lmd_pressure",
        "lmd_winddirection",
        "lmd_windspeed",
        target_col,
    ]
    if weather_source == "satellite":
        weather_cols = ["SWR"]
    else:
        weather_cols = [
            "nwp_globalirrad",
            "nwp_directirrad",
            "nwp_temperature",
            "nwp_humidity",
            "nwp_windspeed",
            "nwp_winddirection",
        ]
    return target_col, history_cols, weather_cols


def _split_borders(n_rows: int, seq_len: int) -> dict[Split, tuple[int, int]]:
    num_train = int(n_rows * 0.8)
    num_test = int(n_rows * 0.1)
    num_valid = n_rows - num_train - num_test
    return {
        "train": (0, num_train),
        "valid": (max(0, num_train - seq_len), num_train + num_valid),
        "test": (max(0, n_rows - num_test - seq_len), n_rows),
    }


class RealPVDataset(Dataset):
    """Sliding-window dataset for bundled PV-power station CSV files."""

    def __init__(
        self,
        cfg: RealPVDataConfig,
        split: Split,
        target_scaler: ArrayScaler | None = None,
        weather_scaler: ArrayScaler | None = None,
    ) -> None:
        self.cfg = cfg
        self.split = split
        dataset_root = Path(cfg.dataset_root)
        station_path = dataset_root / "All_dataset" / f"{cfg.station_name}.csv"
        if not station_path.is_file():
            raise FileNotFoundError(
                f"Station file not found: {station_path}. "
                "Expected files under dataset/All_dataset."
            )

        target_col, history_cols, weather_cols = infer_station_columns(
            cfg.station_name, cfg.weather_source
        )
        self.target_col = target_col
        self.history_cols = history_cols
        self.weather_cols = weather_cols
        self.target_channels = len(history_cols)
        self.weather_channels = len(weather_cols)

        df = pd.read_csv(station_path)
        required_cols = ["Time", *history_cols, *weather_cols]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{station_path} is missing required columns: {missing}")

        df = df[required_cols].copy()
        df["Time"] = pd.to_datetime(df["Time"])
        df[history_cols + weather_cols] = df[history_cols + weather_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        df = df.dropna(subset=history_cols + weather_cols).reset_index(drop=True)
        if len(df) < cfg.seq_len * 2:
            raise ValueError(
                f"{station_path} has {len(df)} usable rows; at least "
                f"{cfg.seq_len * 2} are required."
            )

        borders = _split_borders(len(df), cfg.seq_len)
        train_start, train_end = borders["train"]
        target_values = df[history_cols].to_numpy(dtype=np.float32)
        weather_values = df[weather_cols].to_numpy(dtype=np.float32)
        self.target_scaler = target_scaler or ArrayScaler.fit(
            target_values[train_start:train_end]
        )
        self.weather_scaler = weather_scaler or ArrayScaler.fit(
            weather_values[train_start:train_end]
        )

        start, end = borders[split]
        self.times = df["Time"].iloc[start:end].to_numpy()
        self.target_data = self.target_scaler.transform(target_values[start:end])
        self.weather_data = self.weather_scaler.transform(weather_values[start:end])
        self.raw_target = target_values[start:end, -1:].astype(np.float32)
        future_len = max(cfg.seq_len, cfg.pred_len)
        usable = len(self.target_data) - cfg.seq_len - future_len + 1
        self._length = max(0, usable)
        if cfg.max_samples is not None:
            self._length = min(self._length, cfg.max_samples)
        if self._length <= 0:
            raise ValueError(
                f"Split {split!r} has no usable windows for seq_len={cfg.seq_len}."
            )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        seq_len = self.cfg.seq_len
        pred_len = self.cfg.pred_len
        s_begin = index
        s_end = s_begin + seq_len
        w_end = s_end + seq_len
        target_end = s_end + pred_len

        if target_end > len(self.target_data):
            raise IndexError(index)

        if index < seq_len:
            hist_begin, hist_end = s_begin, s_end
        else:
            hist_begin, hist_end = s_begin - seq_len, s_begin

        return {
            "x_enc": torch.from_numpy(self.target_data[s_begin:s_end]),
            "w_enc": torch.from_numpy(self.weather_data[s_end:w_end]),
            "seq_w_nwp_hist": torch.from_numpy(self.weather_data[s_begin:s_end]),
            "seq_x_hist": torch.from_numpy(self.target_data[hist_begin:hist_end]),
            "target": torch.from_numpy(self.target_data[s_end:target_end]),
        }

    def target_times(self, index: int) -> list[np.datetime64]:
        start = index + self.cfg.seq_len
        end = start + self.cfg.pred_len
        return list(self.times[start:end])

    def raw_primary_target(self, index: int) -> np.ndarray:
        start = index + self.cfg.seq_len
        end = start + self.cfg.pred_len
        return self.raw_target[start:end]


def _move(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _make_model(cfg: DictConfig, dataset: RealPVDataset) -> CrossUnet:
    return CrossUnet(
        target_channels=dataset.target_channels,
        weather_channels=dataset.weather_channels,
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        seg_len=cfg.seg_len,
        e_layers=cfg.e_layers,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        nonlinear_correlation_proj=cfg.nonlinear_correlation_proj,
        attention_kind=cfg.attention_kind,
        merge_kind=cfg.merge_kind,
        use_bottleneck_in_decoder=cfg.use_bottleneck_in_decoder,
    )


def _data_cfg_from_hydra(cfg: DictConfig, max_samples: int | None) -> RealPVDataConfig:
    return RealPVDataConfig(
        dataset_root=Path(to_absolute_path(str(cfg.dataset_root))),
        station_name=cfg.station_name,
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        weather_source=cfg.weather_source,
        max_samples=max_samples,
    )


def train(cfg: DictConfig) -> Path:
    """Train CrossUnet on one real station CSV and save a checkpoint."""
    logger = PythonLogger("real_cross_unet")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = RealPVDataset(
        _data_cfg_from_hydra(cfg, cfg.max_train_samples), split="train"
    )
    valid_ds = RealPVDataset(
        _data_cfg_from_hydra(cfg, cfg.max_valid_samples),
        split="valid",
        target_scaler=train_ds.target_scaler,
        weather_scaler=train_ds.weather_scaler,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size_valid, shuffle=False)

    model = _make_model(cfg, train_ds).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.start_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=cfg.lr_scheduler_gamma
    )

    early_stopping = EarlyStopping(patience=cfg.early_stop_patience)
    station_dir = Path(to_absolute_path(str(cfg.output_dir))) / cfg.station_name
    checkpoint_dir = station_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"real_cross_unet_{cfg.station_name}.pt"

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = _move(batch, device)
            optimizer.zero_grad()
            pred = model(
                batch["x_enc"],
                batch["w_enc"],
                batch["seq_w_nwp_hist"],
                batch["seq_x_hist"],
            )
            loss = F.mse_loss(pred[..., -1:], batch["target"][..., -1:])
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        valid_loss = evaluate(model, valid_loader, device)
        logger.info(
            f"epoch={epoch} train_loss={float(np.mean(train_losses)):.6f} "
            f"valid_loss={valid_loss:.6f}"
        )
        improved, should_stop = early_stopping.step(valid_loss)
        if improved:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "data_config": asdict(train_ds.cfg),
                    "model_config": {
                        key: OmegaConf.to_container(cfg)[key]
                        for key in [
                            "seg_len",
                            "e_layers",
                            "d_model",
                            "n_heads",
                            "d_ff",
                            "dropout",
                            "nonlinear_correlation_proj",
                            "attention_kind",
                            "merge_kind",
                            "use_bottleneck_in_decoder",
                        ]
                    },
                    "target_scaler": train_ds.target_scaler.state_dict(),
                    "weather_scaler": train_ds.weather_scaler.state_dict(),
                    "best_valid_loss": early_stopping.best_loss,
                    "epoch": epoch,
                    "early_stop_patience": cfg.early_stop_patience,
                },
                checkpoint_path,
            )
        if should_stop:
            logger.info(
                f"early stopping at epoch={epoch}; "
                f"best_valid_loss={early_stopping.best_loss:.6f}"
            )
            break
    logger.info(f"saved checkpoint to {checkpoint_path}")
    return checkpoint_path


@torch.no_grad()
def evaluate(model: CrossUnet, loader: DataLoader, device: torch.device) -> float:
    """Evaluate scaled primary-target MSE."""
    model.eval()
    losses = []
    for batch in loader:
        batch = _move(batch, device)
        pred = model(
            batch["x_enc"],
            batch["w_enc"],
            batch["seq_w_nwp_hist"],
            batch["seq_x_hist"],
        )
        losses.append(F.mse_loss(pred[..., -1:], batch["target"][..., -1:]).item())
    return float(np.mean(losses))


def save_prediction_csv(
    path: Path,
    station_name: str,
    target_times: list[list[np.datetime64]],
    predictions: np.ndarray,
    targets: np.ndarray,
) -> None:
    """Save primary-target predictions in a simple long CSV format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            ["station", "timestamp", "horizon_step", "prediction", "target"]
        )
        for sample_times, sample_pred, sample_target in zip(
            target_times, predictions, targets
        ):
            for step, (timestamp, pred, target) in enumerate(
                zip(sample_times, sample_pred[:, 0], sample_target[:, 0])
            ):
                writer.writerow(
                    [
                        station_name,
                        np.datetime_as_string(timestamp, unit="m"),
                        step,
                        float(pred),
                        float(target),
                    ]
                )


@torch.no_grad()
def predict(cfg: DictConfig) -> Path:
    """Load a checkpoint, run test-set inference, and save a prediction CSV."""
    logger = PythonLogger("real_cross_unet")
    if cfg.checkpoint_path is None:
        checkpoint_path = (
            Path(to_absolute_path(str(cfg.output_dir)))
            / cfg.station_name
            / "checkpoints"
            / f"real_cross_unet_{cfg.station_name}.pt"
        )
    else:
        checkpoint_path = Path(to_absolute_path(str(cfg.checkpoint_path)))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    saved_data_cfg = checkpoint["data_config"]
    data_cfg = RealPVDataConfig(
        dataset_root=Path(to_absolute_path(str(cfg.dataset_root))),
        station_name=saved_data_cfg["station_name"],
        seq_len=saved_data_cfg["seq_len"],
        pred_len=saved_data_cfg["pred_len"],
        weather_source=saved_data_cfg["weather_source"],
        max_samples=cfg.max_predict_samples,
    )
    target_scaler = ArrayScaler.from_state(checkpoint["target_scaler"])
    weather_scaler = ArrayScaler.from_state(checkpoint["weather_scaler"])
    test_ds = RealPVDataset(
        data_cfg,
        split="test",
        target_scaler=target_scaler,
        weather_scaler=weather_scaler,
    )

    merged_cfg = OmegaConf.create({**checkpoint["model_config"], **asdict(data_cfg)})
    model = _make_model(merged_cfg, test_ds)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    loader = DataLoader(test_ds, batch_size=cfg.batch_size_valid, shuffle=False)
    predictions = []
    targets = []
    target_times = []
    sample_offset = 0
    for batch in loader:
        batch_size = batch["x_enc"].shape[0]
        batch = _move(batch, device)
        pred = model(
            batch["x_enc"],
            batch["w_enc"],
            batch["seq_w_nwp_hist"],
            batch["seq_x_hist"],
        )
        pred_primary = pred[..., -1:].detach().cpu().numpy()
        target_primary = batch["target"][..., -1:].detach().cpu().numpy()
        predictions.append(target_scaler.inverse_last_channel(pred_primary))
        targets.append(target_scaler.inverse_last_channel(target_primary))
        target_times.extend(
            test_ds.target_times(idx)
            for idx in range(sample_offset, sample_offset + batch_size)
        )
        sample_offset += batch_size

    pred_array = np.concatenate(predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)
    if cfg.prediction_path is None:
        prediction_path = (
            Path(to_absolute_path(str(cfg.output_dir)))
            / data_cfg.station_name
            / "predictions"
            / f"{data_cfg.station_name}_predictions.csv"
        )
    else:
        prediction_path = Path(to_absolute_path(str(cfg.prediction_path)))
    save_prediction_csv(
        prediction_path,
        station_name=data_cfg.station_name,
        target_times=target_times,
        predictions=pred_array,
        targets=target_array,
    )
    mae = float(np.mean(np.abs(pred_array - target_array)))
    rmse = float(np.sqrt(np.mean((pred_array - target_array) ** 2)))
    logger.info(f"saved predictions to {prediction_path}")
    logger.info(f"test MAE={mae:.6f} RMSE={rmse:.6f}")
    return prediction_path


@hydra.main(version_base="1.2", config_path="conf", config_name="real_data")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint."""
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "predict":
        predict(cfg)
    else:
        raise ValueError(f"Unsupported mode={cfg.mode!r}; expected train or predict.")


if __name__ == "__main__":
    main()
