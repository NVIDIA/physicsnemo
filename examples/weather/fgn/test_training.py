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
import torch
from hydra import compose, initialize
from inference import _allocate_members, run_inference
from train import run_training
from utils.trainer import find_latest_model_checkpoint


def _load_config(config_name: str):
    with initialize(version_base=None, config_path="config", job_name="test_fgn"):
        return compose(config_name=config_name)


def test_fgn_training_and_inference(tmp_path: Path):
    cfg = _load_config("test_fgn")
    cfg.training.outdir = str(tmp_path)
    cfg.training.experiment_name = "fgn-smoke"
    cfg.training.run_id = "0"
    cfg.training.rundir = str(tmp_path / "fgn-smoke" / "0")

    run_training(cfg)

    checkpoint = find_latest_model_checkpoint(
        Path(cfg.training.rundir) / cfg.training.checkpoint_dir
    )
    assert checkpoint.endswith(".mdlus")

    infer_cfg = _load_config("inference_fgn")
    infer_cfg.training.rundir = cfg.training.rundir
    infer_cfg.inference.checkpoint = "latest"
    infer_cfg.inference.output_path = str(tmp_path / "forecast.pt")

    result = run_inference(infer_cfg)
    assert Path(result["output_path"]).is_file()
    assert result["num_models"] == 1
    assert result["members_per_model"] == [int(infer_cfg.inference.num_trajectories)]

    payload = torch.load(result["output_path"], map_location="cpu")
    assert payload["trajectories"].ndim == 5
    assert payload["target"].ndim == 4


def test_fgn_deep_ensemble_inference(tmp_path: Path):
    """Two independently-trained seeds rolled out together (paper §2.2.1)."""
    # Train two seeds with distinct run_ids so checkpoints live in separate
    # directories we can point the ensemble inference path at.
    checkpoint_paths: list[str] = []
    for seed_idx, seed in enumerate([7, 13]):
        cfg = _load_config("test_fgn")
        cfg.training.outdir = str(tmp_path)
        cfg.training.experiment_name = "fgn-ensemble"
        cfg.training.run_id = f"seed{seed_idx}"
        cfg.training.rundir = str(tmp_path / "fgn-ensemble" / f"seed{seed_idx}")
        cfg.training.seed = seed
        run_training(cfg)
        checkpoint_paths.append(
            find_latest_model_checkpoint(
                Path(cfg.training.rundir) / cfg.training.checkpoint_dir
            )
        )
    assert len(checkpoint_paths) == 2

    infer_cfg = _load_config("inference_fgn")
    # rundir is unused when ``checkpoints`` is given; set something benign.
    infer_cfg.training.rundir = str(tmp_path / "fgn-ensemble" / "seed0")
    infer_cfg.inference.checkpoints = checkpoint_paths
    # 5 trajectories across 2 models -> [3, 2] (remainder on earlier model).
    infer_cfg.inference.num_trajectories = 5
    infer_cfg.inference.output_path = str(tmp_path / "ensemble_forecast.pt")

    result = run_inference(infer_cfg)
    assert result["num_models"] == 2
    assert result["members_per_model"] == [3, 2]

    payload = torch.load(result["output_path"], map_location="cpu")
    # Trajectories from both models concatenated on the leading axis.
    assert payload["trajectories"].shape[0] == 5
    assert payload["num_models"] == 2
    assert payload["checkpoint_paths"] == checkpoint_paths


def test_fgn_training_writes_validation_metrics(tmp_path: Path):
    """With ``training.validation_metrics: true`` the trainer should emit
    an .npz summary + PNG plots under ``rundir/validation/step=<step>/``.
    """
    cfg = _load_config("test_fgn")
    cfg.training.outdir = str(tmp_path)
    cfg.training.experiment_name = "fgn-diag"
    cfg.training.run_id = "0"
    cfg.training.rundir = str(tmp_path / "fgn-diag" / "0")
    cfg.training.validation_metrics = True
    cfg.training.validation_ensemble_size = 2

    run_training(cfg)

    val_root = Path(cfg.training.rundir) / "validation"
    assert val_root.is_dir()
    step_dirs = sorted(val_root.glob("step=*"))
    assert step_dirs, "expected at least one validation snapshot"
    npz_path = step_dirs[-1] / "metrics.npz"
    assert npz_path.is_file()
    data = np.load(npz_path, allow_pickle=True)
    # Paper Figure 2 core panels are all present.
    for key in (
        "crps_per_lead_per_channel",
        "rmse_per_lead_per_channel",
        "spread_skill_ratio",
        "rank_histograms",
        "power_spectrum_forecast",
        "power_spectrum_truth",
    ):
        assert key in data.files, key


def test_allocate_members_distribution():
    # Equal split when divisible.
    assert _allocate_members(16, 4) == [4, 4, 4, 4]
    # Remainder on the earlier models (paper-faithful default).
    assert _allocate_members(14, 4) == [4, 4, 3, 3]
    assert _allocate_members(1, 4) == [1, 0, 0, 0]
    assert _allocate_members(0, 4) == [0, 0, 0, 0]
    # Single-model degenerate case.
    assert _allocate_members(7, 1) == [7]
