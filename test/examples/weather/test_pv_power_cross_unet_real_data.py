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

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from examples.weather.pv_power_cross_unet.train_real_cross_unet import (
    ArrayScaler,
    EarlyStopping,
    RealPVDataConfig,
    RealPVDataset,
    compute_prediction_metrics,
    horizon_config,
    save_prediction_csv,
)

CONFIG_PATH = Path("examples/weather/pv_power_cross_unet/conf/real_data.yaml")


def _write_csv(
    path: Path,
    *,
    rows: int = 80,
    freq_minutes: int = 15,
    time_col: str = "timestamp",
    target_col: str = "power",
    weather_cols: tuple[str, ...] = ("irradiance", "temperature"),
) -> pd.DataFrame:
    times = pd.date_range("2020-01-01", periods=rows, freq=f"{freq_minutes}min")
    frame = pd.DataFrame({time_col: times, target_col: np.arange(rows, dtype=float)})
    for offset, col in enumerate(weather_cols, start=1):
        frame[col] = np.arange(rows, dtype=float) + offset * 100.0
    frame["ignored"] = "unused"
    frame.to_csv(path, index=False)
    return frame


def _dataset_cfg(
    path: Path,
    *,
    seq_len: int = 8,
    pred_len: int = 4,
    weather_cols: tuple[str, ...] = ("irradiance", "temperature"),
    max_samples: int | None = 3,
) -> RealPVDataConfig:
    return RealPVDataConfig(
        data_file=path,
        time_col="timestamp",
        target_col="power",
        weather_cols=list(weather_cols),
        freq_minutes=15,
        seq_len=seq_len,
        pred_len=pred_len,
        max_samples=max_samples,
    )


def test_real_pv_dataset_accepts_arbitrary_column_names_and_ignores_extras(tmp_path):
    data_file = tmp_path / "custom_station.csv"
    _write_csv(data_file, rows=80)

    dataset = RealPVDataset(_dataset_cfg(data_file), split="train")
    sample = dataset[0]

    assert dataset.target_channels == 1
    assert dataset.weather_channels == 2
    assert dataset.weather_cols == ["irradiance", "temperature"]
    assert sample["x_enc"].shape == (8, 1)
    assert sample["seq_x_hist"].shape == (8, 1)
    assert sample["w_enc"].shape == (8, 2)
    assert sample["seq_w_nwp_hist"].shape == (8, 2)
    assert sample["target"].shape == (4, 1)


def test_real_pv_dataset_rejects_missing_required_columns(tmp_path):
    data_file = tmp_path / "missing.csv"
    _write_csv(data_file, rows=80, weather_cols=("irradiance",))

    with pytest.raises(ValueError, match="missing required columns"):
        RealPVDataset(
            _dataset_cfg(
                data_file,
                weather_cols=("irradiance", "temperature"),
            ),
            split="train",
        )


def test_real_pv_dataset_rejects_duplicate_times(tmp_path):
    data_file = tmp_path / "duplicate.csv"
    frame = _write_csv(data_file, rows=80)
    frame.loc[10, "timestamp"] = frame.loc[9, "timestamp"]
    frame.to_csv(data_file, index=False)

    with pytest.raises(ValueError, match="duplicate timestamps"):
        RealPVDataset(_dataset_cfg(data_file), split="train")


def test_real_pv_dataset_rejects_non_15_minute_cadence(tmp_path):
    data_file = tmp_path / "hourly.csv"
    _write_csv(data_file, rows=80, freq_minutes=60)

    with pytest.raises(ValueError, match="expected 15-minute cadence"):
        RealPVDataset(_dataset_cfg(data_file), split="train")


def test_real_pv_dataset_uses_real_prior_history_without_target_leakage(tmp_path):
    data_file = tmp_path / "ordered.csv"
    _write_csv(data_file, rows=120)

    dataset = RealPVDataset(_dataset_cfg(data_file, max_samples=1), split="valid")
    sample = dataset[0]

    assert not np.array_equal(sample["seq_x_hist"].numpy(), sample["x_enc"].numpy())
    target_start = dataset.target_times(0)[0]
    assert target_start >= dataset.split_target_start_time


def test_compute_prediction_metrics_flattens_predictions_and_includes_rmse():
    pred = np.array([[[1.0], [3.0]], [[2.0], [4.0]]], dtype=np.float32)
    target = np.array([[[1.0], [1.0]], [[3.0], [5.0]]], dtype=np.float32)

    metrics = compute_prediction_metrics(pred, target)

    assert metrics["mae"] == 1.0
    assert metrics["mse"] == 1.5
    assert np.isclose(metrics["rmse"], np.sqrt(1.5))
    assert np.isclose(metrics["r2"], 0.4545454545454546)


def test_horizon_config_matches_requested_windows():
    assert horizon_config("4h") == {"pred_len": 16, "seq_len": 96, "seg_len": 12}
    assert horizon_config("1d") == {"pred_len": 96, "seq_len": 96, "seg_len": 24}
    assert horizon_config("7d") == {"pred_len": 672, "seq_len": 672, "seg_len": 48}


def test_array_scaler_supports_minmax_normalization():
    values = np.array([[2.0, 10.0], [4.0, 14.0], [6.0, 18.0]], dtype=np.float32)

    scaler = ArrayScaler.fit(values, kind="minmax")
    scaled = scaler.transform(values)
    restored = scaler.inverse_transform(scaled)

    assert scaler.kind == "minmax"
    assert np.allclose(scaled[0], [0.0, 0.0])
    assert np.allclose(scaled[-1], [1.0, 1.0])
    assert np.allclose(restored, values)


def test_early_stopping_respects_patience_and_resets_on_improvement():
    early_stop = EarlyStopping(patience=2)

    assert early_stop.step(1.0) == (True, False)
    assert early_stop.step(1.1) == (False, False)
    assert early_stop.step(0.9) == (True, False)
    assert early_stop.step(0.95) == (False, False)
    assert early_stop.step(0.96) == (False, True)


def test_real_data_config_uses_cross_unet_defaults_and_patience_five():
    cfg = OmegaConf.load(CONFIG_PATH)

    assert cfg.early_stop_patience == 5
    assert cfg.d_model == 256
    assert cfg.d_ff == 512
    assert cfg.n_heads == 4
    assert cfg.e_layers == 3
    assert cfg.start_lr == 1.0e-4
    assert cfg.max_epochs == 100
    assert cfg.seed == 2021
    assert cfg.nonlinear_correlation_proj is True
    assert cfg.use_bottleneck_in_decoder is True


def test_save_prediction_csv_writes_generic_columns(tmp_path):
    target_times = [
        [np.datetime64("2020-01-01T00:00"), np.datetime64("2020-01-01T00:15")]
    ]
    predictions = np.array([[[1.0], [2.0]]], dtype=np.float32)
    targets = np.array([[[1.5], [2.5]]], dtype=np.float32)
    out_path = tmp_path / "predictions.csv"

    save_prediction_csv(
        out_path,
        data_file="custom_station.csv",
        target_times=target_times,
        predictions=predictions,
        targets=targets,
    )

    text = out_path.read_text()
    assert "data_file,timestamp,horizon_step,prediction,target" in text
    assert "custom_station.csv,2020-01-01T00:00,0,1.0,1.5" in text
