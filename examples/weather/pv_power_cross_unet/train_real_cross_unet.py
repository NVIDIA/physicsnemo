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

"""Train, predict, and evaluate CrossUnet on user-provided PV-power CSV data."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

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
Normalization = Literal["standard", "minmax"]

HORIZON_CONFIGS: dict[str, dict[str, int]] = {
    "4h": {"pred_len": 16, "seq_len": 96, "seg_len": 12},
    "1d": {"pred_len": 96, "seq_len": 96, "seg_len": 24},
    "7d": {"pred_len": 672, "seq_len": 672, "seg_len": 48},
}

METRIC_COLUMNS = [
    "data_file",
    "horizon_label",
    "seq_len",
    "pred_len",
    "weather_cols",
    "weather_channels",
    "mae",
    "mse",
    "rmse",
    "r2",
    "best_valid_loss",
    "epoch",
    "checkpoint_path",
    "prediction_path",
]


@dataclass
class RealPVDataConfig:
    """Configuration for one generic 15-minute PV-power CSV file."""

    data_file: str | Path
    time_col: str
    target_col: str
    weather_cols: list[str]
    freq_minutes: int = 15
    seq_len: int = 96
    pred_len: int = 16
    normalization: Normalization = "standard"
    max_samples: int | None = None


@dataclass
class ArrayScaler:
    """Small serializable scaler for numpy arrays."""

    mean: np.ndarray
    std: np.ndarray
    kind: Normalization = "standard"

    @classmethod
    def fit(cls, values: np.ndarray, kind: Normalization = "standard") -> "ArrayScaler":
        """Fit scaler statistics from a 2-D array of shape (N, C)."""
        if kind == "minmax":
            mean = values.min(axis=0)
            std = values.max(axis=0) - mean
        else:
            mean = values.mean(axis=0)
            std = values.std(axis=0)
        std = np.where(std < 1.0e-6, 1.0, std)
        return cls(
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            kind=kind,
        )

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "ArrayScaler":
        """Restore an ArrayScaler from a state dict produced by state_dict()."""
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            std=np.asarray(state["std"], dtype=np.float32),
            kind=state.get("kind", "standard"),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Normalize values using fitted mean and std."""
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Invert normalization for all channels."""
        return (values * self.std + self.mean).astype(np.float32)

    def inverse_last_channel(self, values: np.ndarray) -> np.ndarray:
        """Invert normalization for the last channel only (the power target)."""
        return values * self.std[-1] + self.mean[-1]

    def state_dict(self) -> dict[str, Any]:
        """Return serializable scaler state for checkpoint saving."""
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "kind": self.kind,
        }


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


def horizon_config(label: str) -> dict[str, int]:
    """Return the default CrossUnet window settings for a named horizon."""
    if label not in HORIZON_CONFIGS:
        raise ValueError(
            f"Unsupported horizon_label={label!r}; expected one of "
            f"{sorted(HORIZON_CONFIGS)}."
        )
    return dict(HORIZON_CONFIGS[label])


def _split_ranges(n_rows: int) -> dict[Split, tuple[int, int]]:
    num_train = int(n_rows * 0.8)
    num_test = int(n_rows * 0.1)
    num_valid = n_rows - num_train - num_test
    return {
        "train": (0, num_train),
        "valid": (num_train, num_train + num_valid),
        "test": (num_train + num_valid, num_train + num_valid + num_test),
    }


def _load_generic_csv(cfg: RealPVDataConfig) -> pd.DataFrame:
    data_file = Path(cfg.data_file)
    if not data_file.is_file():
        raise FileNotFoundError(f"data_file not found: {data_file}")
    if not cfg.weather_cols:
        raise ValueError("weather_cols must contain at least one column.")

    required_cols = [cfg.time_col, cfg.target_col, *cfg.weather_cols]
    frame = pd.read_csv(data_file)
    missing = [col for col in required_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"{data_file} is missing required columns: {missing}")

    frame = frame[required_cols].copy()
    frame[cfg.time_col] = pd.to_datetime(frame[cfg.time_col])
    numeric_cols = [cfg.target_col, *cfg.weather_cols]
    frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=[cfg.time_col, *numeric_cols]).sort_values(cfg.time_col)
    frame = frame.reset_index(drop=True)

    duplicated = frame[cfg.time_col].duplicated()
    if bool(duplicated.any()):
        raise ValueError(f"{data_file} has duplicate timestamps in {cfg.time_col!r}.")

    expected_delta = pd.Timedelta(minutes=cfg.freq_minutes)
    deltas = frame[cfg.time_col].diff().dropna()
    if not deltas.empty and bool((deltas != expected_delta).any()):
        raise ValueError(
            f"{data_file} expected {cfg.freq_minutes}-minute cadence in "
            f"{cfg.time_col!r}."
        )
    return frame


def compute_prediction_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """Compute flattened MAE, MSE, RMSE, and R2."""
    pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    error = pred - target
    mae = float(np.mean(np.abs(error)))
    mse = float(np.mean(error**2))
    target_mean = float(np.mean(target))
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((target - target_mean) ** 2))
    r2 = 0.0 if ss_tot <= 1.0e-12 else 1.0 - ss_res / ss_tot
    return {"mae": mae, "mse": mse, "rmse": float(np.sqrt(mse)), "r2": float(r2)}


class RealPVDataset(Dataset):
    """Sliding-window dataset for generic 15-minute PV-power CSV files."""

    def __init__(
        self,
        cfg: RealPVDataConfig,
        split: Split,
        target_scaler: ArrayScaler | None = None,
        weather_scaler: ArrayScaler | None = None,
    ) -> None:
        self.cfg = cfg
        self.split = split
        self.target_cols = [cfg.target_col]
        self.weather_cols = list(cfg.weather_cols)
        self.target_channels = 1
        self.weather_channels = len(cfg.weather_cols)

        frame = _load_generic_csv(cfg)
        if len(frame) < 2 * cfg.seq_len + max(cfg.seq_len, cfg.pred_len):
            raise ValueError(
                f"{cfg.data_file} has {len(frame)} usable rows; at least "
                f"{2 * cfg.seq_len + max(cfg.seq_len, cfg.pred_len)} are required."
            )

        ranges = _split_ranges(len(frame))
        train_start, train_end = ranges["train"]
        split_start, split_end = ranges[split]
        self.split_target_start_time = (
            frame[cfg.time_col].iloc[split_start].to_datetime64()
        )

        target_values = frame[[cfg.target_col]].to_numpy(dtype=np.float32)
        weather_values = frame[cfg.weather_cols].to_numpy(dtype=np.float32)
        self.target_scaler = target_scaler or ArrayScaler.fit(
            target_values[train_start:train_end], kind=cfg.normalization
        )
        self.weather_scaler = weather_scaler or ArrayScaler.fit(
            weather_values[train_start:train_end], kind=cfg.normalization
        )

        self.times = frame[cfg.time_col].to_numpy()
        self.target_data = self.target_scaler.transform(target_values)
        self.weather_data = self.weather_scaler.transform(weather_values)
        self.raw_target = target_values
        first_start = max(0, split_start - 2 * cfg.seq_len)
        last_by_split = split_end - cfg.pred_len - 2 * cfg.seq_len
        last_by_frame = len(frame) - max(cfg.seq_len, cfg.pred_len) - 2 * cfg.seq_len
        last_start = min(last_by_split, last_by_frame)
        self.window_starts = list(range(first_start, last_start + 1))
        if cfg.max_samples is not None:
            self.window_starts = self.window_starts[: cfg.max_samples]
        if not self.window_starts:
            raise ValueError(
                f"Split {split!r} has no usable windows for seq_len={cfg.seq_len}, "
                f"pred_len={cfg.pred_len}."
            )

    def __len__(self) -> int:
        return len(self.window_starts)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        start = self.window_starts[index]
        seq_len = self.cfg.seq_len
        pred_len = self.cfg.pred_len
        hist_begin = start
        enc_begin = start + seq_len
        target_begin = enc_begin + seq_len
        weather_end = target_begin + seq_len
        target_end = target_begin + pred_len

        return {
            "x_enc": torch.from_numpy(self.target_data[enc_begin:target_begin]),
            "w_enc": torch.from_numpy(self.weather_data[target_begin:weather_end]),
            "seq_w_nwp_hist": torch.from_numpy(
                self.weather_data[enc_begin:target_begin]
            ),
            "seq_x_hist": torch.from_numpy(self.target_data[hist_begin:enc_begin]),
            "target": torch.from_numpy(self.target_data[target_begin:target_end]),
        }

    def target_times(self, index: int) -> list[np.datetime64]:
        """Return the forecast timestamps for sample at index."""
        start = self.window_starts[index] + 2 * self.cfg.seq_len
        end = start + self.cfg.pred_len
        return list(self.times[start:end])

    def raw_primary_target(self, index: int) -> np.ndarray:
        """Return un-normalized power values for the forecast window at index."""
        start = self.window_starts[index] + 2 * self.cfg.seq_len
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
        data_file=Path(to_absolute_path(str(cfg.data_file))),
        time_col=cfg.time_col,
        target_col=cfg.target_col,
        weather_cols=list(cfg.weather_cols),
        freq_minutes=cfg.freq_minutes,
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        normalization=cfg.normalization,
        max_samples=max_samples,
    )


def _data_config_state(cfg: RealPVDataConfig) -> dict[str, Any]:
    state = asdict(cfg)
    state["data_file"] = str(state["data_file"])
    return state


def _checkpoint_name(data_file: str | Path, horizon_label: str | None) -> str:
    stem = Path(data_file).stem
    if horizon_label:
        return f"real_cross_unet_{stem}_{horizon_label}.pt"
    return f"real_cross_unet_{stem}.pt"


def _output_stem(data_file: str | Path, horizon_label: str | None) -> str:
    stem = Path(data_file).stem
    if horizon_label:
        return f"{stem}_{horizon_label}"
    return stem


def train(cfg: DictConfig) -> Path:
    """Train CrossUnet on one real CSV and save the best validation checkpoint."""
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
    output_dir = Path(to_absolute_path(str(cfg.output_dir)))
    checkpoint_dir = (
        output_dir / _output_stem(cfg.data_file, cfg.horizon_label) / "checkpoints"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / _checkpoint_name(
        cfg.data_file, cfg.horizon_label
    )

    last_epoch = 0
    for epoch in range(1, cfg.max_epochs + 1):
        last_epoch = epoch
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

        valid_loss = validation_loss(model, valid_loader, device)
        logger.info(
            f"epoch={epoch} train_loss={float(np.mean(train_losses)):.6f} "
            f"valid_loss={valid_loss:.6f}"
        )
        improved, should_stop = early_stopping.step(valid_loss)
        if improved:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "data_config": _data_config_state(train_ds.cfg),
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
                    "last_epoch": last_epoch,
                    "early_stop_patience": cfg.early_stop_patience,
                    "horizon_label": cfg.horizon_label,
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
def validation_loss(
    model: CrossUnet, loader: DataLoader, device: torch.device
) -> float:
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
    data_file: str,
    target_times: list[list[np.datetime64]],
    predictions: np.ndarray,
    targets: np.ndarray,
) -> None:
    """Save primary-target predictions in a simple long CSV format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            ["data_file", "timestamp", "horizon_step", "prediction", "target"]
        )
        for sample_times, sample_pred, sample_target in zip(
            target_times, predictions, targets
        ):
            for step, (timestamp, pred, target) in enumerate(
                zip(sample_times, sample_pred[:, 0], sample_target[:, 0])
            ):
                writer.writerow(
                    [
                        data_file,
                        np.datetime_as_string(timestamp, unit="m"),
                        step,
                        float(pred),
                        float(target),
                    ]
                )


def _checkpoint_path_from_cfg(cfg: DictConfig) -> Path:
    if cfg.checkpoint_path is not None:
        return Path(to_absolute_path(str(cfg.checkpoint_path)))
    return (
        Path(to_absolute_path(str(cfg.output_dir)))
        / _output_stem(cfg.data_file, cfg.horizon_label)
        / "checkpoints"
        / _checkpoint_name(cfg.data_file, cfg.horizon_label)
    )


def _prediction_path_from_cfg(cfg: DictConfig, data_file: str | Path) -> Path:
    if cfg.prediction_path is not None:
        return Path(to_absolute_path(str(cfg.prediction_path)))
    return (
        Path(to_absolute_path(str(cfg.output_dir)))
        / _output_stem(data_file, cfg.horizon_label)
        / "predictions"
        / f"{_output_stem(data_file, cfg.horizon_label)}_predictions.csv"
    )


@torch.no_grad()
def _run_prediction(cfg: DictConfig) -> tuple[Path, dict[str, float], dict[str, Any]]:
    checkpoint_path = _checkpoint_path_from_cfg(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    saved_data_cfg = checkpoint["data_config"]
    data_cfg = RealPVDataConfig(
        data_file=Path(to_absolute_path(str(saved_data_cfg["data_file"]))),
        time_col=saved_data_cfg["time_col"],
        target_col=saved_data_cfg["target_col"],
        weather_cols=list(saved_data_cfg["weather_cols"]),
        freq_minutes=saved_data_cfg.get("freq_minutes", 15),
        seq_len=saved_data_cfg["seq_len"],
        pred_len=saved_data_cfg["pred_len"],
        normalization=saved_data_cfg.get("normalization", "standard"),
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
    prediction_path = _prediction_path_from_cfg(cfg, data_cfg.data_file)
    save_prediction_csv(
        prediction_path,
        data_file=Path(data_cfg.data_file).name,
        target_times=target_times,
        predictions=pred_array,
        targets=target_array,
    )
    metrics = compute_prediction_metrics(pred_array, target_array)
    return prediction_path, metrics, checkpoint


def predict(cfg: DictConfig) -> Path:
    """Load a checkpoint, run test-set inference, and save a prediction CSV."""
    logger = PythonLogger("real_cross_unet")
    prediction_path, metrics, _ = _run_prediction(cfg)
    logger.info(f"saved predictions to {prediction_path}")
    logger.info(
        f"test MAE={metrics['mae']:.6f} MSE={metrics['mse']:.6f} "
        f"RMSE={metrics['rmse']:.6f} R2={metrics['r2']:.6f}"
    )
    return prediction_path


def evaluate(cfg: DictConfig) -> Path:
    """Run prediction and append one row to the metrics CSV."""
    logger = PythonLogger("real_cross_unet")
    prediction_path, metrics, checkpoint = _run_prediction(cfg)
    metrics_path = Path(to_absolute_path(str(cfg.metrics_csv_path)))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "data_file": Path(checkpoint["data_config"]["data_file"]).name,
        "horizon_label": checkpoint.get("horizon_label") or cfg.horizon_label or "",
        "seq_len": checkpoint["data_config"]["seq_len"],
        "pred_len": checkpoint["data_config"]["pred_len"],
        "weather_cols": "|".join(checkpoint["data_config"]["weather_cols"]),
        "weather_channels": len(checkpoint["data_config"]["weather_cols"]),
        "mae": metrics["mae"],
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "best_valid_loss": checkpoint.get("best_valid_loss", ""),
        "epoch": checkpoint.get("epoch", ""),
        "checkpoint_path": str(_checkpoint_path_from_cfg(cfg)),
        "prediction_path": str(prediction_path),
    }
    write_header = not metrics_path.exists()
    with metrics_path.open("a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=METRIC_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    logger.info(f"saved predictions to {prediction_path}")
    logger.info(f"appended metrics to {metrics_path}")
    return metrics_path


@hydra.main(version_base="1.2", config_path="conf", config_name="real_data")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint."""
    if cfg.horizon_label:
        for key, value in horizon_config(str(cfg.horizon_label)).items():
            cfg[key] = value
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "predict":
        predict(cfg)
    elif cfg.mode == "evaluate":
        evaluate(cfg)
    else:
        raise ValueError(
            f"Unsupported mode={cfg.mode!r}; expected train, predict, or evaluate."
        )


if __name__ == "__main__":
    main()
