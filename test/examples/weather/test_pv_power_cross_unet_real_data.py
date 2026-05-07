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
    EarlyStopping,
    RealPVDataConfig,
    RealPVDataset,
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
