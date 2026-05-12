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
from omegaconf import OmegaConf

from examples.weather.pv_power_cross_unet.train_real_cross_unet import (
    ArrayScaler,
    EarlyStopping,
    RealPVDataConfig,
    RealPVDataset,
    apply_paper_preset,
    compute_prediction_metrics,
    paper_seq_len_for_horizon,
    save_prediction_csv,
)

DATASET_ROOT = Path("examples/weather/pv_power_cross_unet/dataset")
CONFIG_PATH = Path("examples/weather/pv_power_cross_unet/conf/real_data.yaml")


def test_real_pv_dataset_kdasc_shapes():
    """KDASC real dataset should produce the CrossUnet five-tensor contract."""
    cfg = RealPVDataConfig(
        dataset_root=DATASET_ROOT,
        station_name="KDASC",
        seq_len=96,
        pred_len=16,
        max_samples=2,
    )
    train_ds = RealPVDataset(cfg, split="train")
    sample = train_ds[0]

    assert train_ds.target_channels == 1
    assert train_ds.weather_channels == 1
    assert sample["x_enc"].shape == (96, 1)
    assert sample["w_enc"].shape == (96, 1)
    assert sample["seq_w_nwp_hist"].shape == (96, 1)
    assert sample["seq_x_hist"].shape == (96, 1)
    assert sample["target"].shape == (16, 1)


def test_real_pv_dataset_nwp_s1_uses_seven_history_and_six_weather_channels():
    """Table 4 NWP stations should expose seven history and six forecast channels."""
    cfg = RealPVDataConfig(
        dataset_root=DATASET_ROOT,
        station_name="S-1",
        weather_source="nwp",
        seq_len=96,
        pred_len=16,
        max_samples=2,
    )

    train_ds = RealPVDataset(cfg, split="train")
    sample = train_ds[0]

    assert train_ds.target_channels == 7
    assert train_ds.weather_channels == 6
    assert train_ds.weather_cols == [
        "nwp_globalirrad",
        "nwp_directirrad",
        "nwp_temperature",
        "nwp_humidity",
        "nwp_windspeed",
        "nwp_winddirection",
    ]
    assert sample["x_enc"].shape == (96, 7)
    assert sample["w_enc"].shape == (96, 6)
    assert sample["seq_w_nwp_hist"].shape == (96, 6)


def test_real_pv_dataset_satellite_shapes_for_s1_and_kdasc():
    """Table 5 satellite rows use SWR with station-specific history channels."""
    s1_cfg = RealPVDataConfig(
        dataset_root=DATASET_ROOT,
        station_name="S-1",
        weather_source="satellite",
        seq_len=96,
        pred_len=16,
        max_samples=1,
    )
    kdasc_cfg = RealPVDataConfig(
        dataset_root=DATASET_ROOT,
        station_name="KDASC",
        weather_source="satellite",
        seq_len=96,
        pred_len=16,
        max_samples=1,
    )

    s1_ds = RealPVDataset(s1_cfg, split="train")
    kdasc_ds = RealPVDataset(kdasc_cfg, split="train")

    assert (s1_ds.target_channels, s1_ds.weather_channels) == (7, 1)
    assert (kdasc_ds.target_channels, kdasc_ds.weather_channels) == (1, 1)
    assert s1_ds.weather_cols == ["SWR"]
    assert kdasc_ds.weather_cols == ["SWR"]


def test_real_pv_dataset_ai_s1_maps_to_station_zero_and_uses_corrdiff():
    """Table 6 AI weather should use issued CorrDiff files without future leakage."""
    cfg = RealPVDataConfig(
        dataset_root=DATASET_ROOT,
        station_name="S-1",
        weather_source="ai",
        seq_len=96,
        pred_len=16,
        max_samples=2,
    )

    train_ds = RealPVDataset(cfg, split="train")
    sample = train_ds[0]

    assert train_ds.ai_station_id == "0"
    assert train_ds.target_channels == 1
    assert train_ds.weather_channels == 1
    assert train_ds.history_cols == ["power"]
    assert train_ds.weather_cols == ["ssrd_corrdiff"]
    assert sample["x_enc"].shape == (96, 1)
    assert sample["w_enc"].shape == (96, 1)
    assert sample["seq_w_nwp_hist"].shape == (96, 1)
    assert train_ds.issue_times[0] <= min(train_ds.target_times(0))


def test_real_pv_dataset_ai_supports_long_history_from_prior_issues():
    """Long-horizon AI rows should stitch historical weather from prior issues."""
    cfg = RealPVDataConfig(
        dataset_root=DATASET_ROOT,
        station_name="S-1",
        weather_source="ai",
        seq_len=384,
        pred_len=384,
        max_samples=1,
    )

    train_ds = RealPVDataset(cfg, split="train")
    sample = train_ds[0]

    assert sample["x_enc"].shape == (384, 1)
    assert sample["w_enc"].shape == (384, 1)
    assert sample["seq_w_nwp_hist"].shape == (384, 1)
    assert np.isfinite(sample["seq_w_nwp_hist"].numpy()).all()
    assert train_ds.issue_times[0] <= min(train_ds.target_times(0))


def test_real_pv_dataset_requires_real_prior_history_window():
    """P-corr samples should not reuse the current encoder window as prior history."""
    cfg = RealPVDataConfig(
        dataset_root=DATASET_ROOT,
        station_name="S-1",
        weather_source="nwp",
        seq_len=96,
        pred_len=16,
        max_samples=1,
    )
    train_ds = RealPVDataset(cfg, split="train")

    sample = train_ds[0]

    assert not np.array_equal(sample["seq_x_hist"].numpy(), sample["x_enc"].numpy())
    assert not np.array_equal(sample["seq_w_nwp_hist"].numpy(), sample["w_enc"].numpy())


def test_compute_prediction_metrics_flattens_predictions():
    """Paper comparison metrics should be MAE, MSE, and R2 over flattened arrays."""
    pred = np.array([[[1.0], [3.0]], [[2.0], [4.0]]], dtype=np.float32)
    target = np.array([[[1.0], [1.0]], [[3.0], [5.0]]], dtype=np.float32)

    metrics = compute_prediction_metrics(pred, target)

    assert metrics["mae"] == 1.0
    assert metrics["mse"] == 1.5
    assert np.isclose(metrics["r2"], 0.4545454545454546)


def test_array_scaler_supports_minmax_normalization():
    """Paper preset should be able to use training-range Min-Max scaling."""
    values = np.array([[2.0, 10.0], [4.0, 14.0], [6.0, 18.0]], dtype=np.float32)

    scaler = ArrayScaler.fit(values, kind="minmax")
    scaled = scaler.transform(values)
    restored = scaler.inverse_last_channel(scaled[:, -1:])

    assert scaler.kind == "minmax"
    assert np.allclose(scaled[0], [0.0, 0.0])
    assert np.allclose(scaled[-1], [1.0, 1.0])
    assert np.allclose(restored[:, 0], values[:, -1])


def test_paper_seq_len_for_horizon_matches_cross_unet_table_setup():
    """Paper reproduction uses 96 history steps unless the horizon is longer."""
    assert paper_seq_len_for_horizon(16) == 96
    assert paper_seq_len_for_horizon(48) == 96
    assert paper_seq_len_for_horizon(96) == 96
    assert paper_seq_len_for_horizon(384) == 384


def test_apply_paper_preset_uses_conservative_learning_rate():
    """Paper-size Cross-Unet should use the stable low-LR preset."""
    cfg = OmegaConf.create(
        {
            "paper_preset": True,
            "d_model": 64,
            "d_ff": 128,
            "n_heads": 4,
            "e_layers": 1,
            "seg_len": 6,
            "max_epochs": 5,
            "early_stop_patience": 5,
            "start_lr": 1.0e-3,
            "seed": 0,
            "nonlinear_correlation_proj": False,
        }
    )

    apply_paper_preset(cfg)

    assert cfg.d_model == 252
    assert cfg.d_ff == 492
    assert cfg.start_lr == 1.0e-4
    assert cfg.normalization == "minmax"
    assert cfg.nonlinear_correlation_proj is True


def test_early_stopping_respects_patience_and_resets_on_improvement():
    """Early stopping should trigger after patience consecutive stale epochs."""
    early_stop = EarlyStopping(patience=2)

    assert early_stop.step(1.0) == (True, False)
    assert early_stop.step(1.1) == (False, False)
    assert early_stop.step(0.9) == (True, False)
    assert early_stop.step(0.95) == (False, False)
    assert early_stop.step(0.96) == (False, True)


def test_real_data_config_exposes_default_early_stop_patience():
    """The user-facing real-data config should make patience tunable."""
    cfg = OmegaConf.load(CONFIG_PATH)
    assert cfg.early_stop_patience == 5


def test_save_prediction_csv_writes_expected_columns(tmp_path):
    """Prediction CSV should be directly inspectable by new users."""
    target_times = [
        [np.datetime64("2020-01-01T00:00"), np.datetime64("2020-01-01T00:15")]
    ]
    predictions = np.array([[[1.0], [2.0]]], dtype=np.float32)
    targets = np.array([[[1.5], [2.5]]], dtype=np.float32)
    out_path = tmp_path / "predictions.csv"

    save_prediction_csv(
        out_path,
        station_name="KDASC",
        target_times=target_times,
        predictions=predictions,
        targets=targets,
    )

    text = out_path.read_text()
    assert "station,timestamp,horizon_step,prediction,target" in text
    assert "KDASC,2020-01-01T00:00,0,1.0,1.5" in text
