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

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DistributedSampler
import xarray as xr
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

import train
from datasets.hrrr_era5 import HRRRERA5Dataset
from utils.nn import TimeScaledModel, build_condition, get_dit, sample_flow
from utils.trainer import Trainer

from physicsnemo.core import Module
from physicsnemo.diffusion.metrics.losses import MSEDSMLoss
from physicsnemo.diffusion.noise_schedulers import RectifiedFlowNoiseScheduler
from physicsnemo.distributed import DistributedManager

DistributedManager.initialize()

# Training and sampling reach t = 1, where the loss and the sampling flow stay finite
pytestmark = pytest.mark.filterwarnings(
    "ignore:RectifiedFlowNoiseScheduler was constructed with t_max=1.0"
)

CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints",
    "corrdiff_era5_hrrr.mdlus",
)


def _load_config(config_name: str) -> DictConfig:
    """Load a configuration through Hydra."""
    with initialize(version_base=None, config_path="config", job_name="test_training"):
        return compose(config_name=config_name)


@pytest.fixture
def cfg() -> DictConfig:
    """Return the mock-data test configuration."""
    return _load_config(config_name="test_diffusion")


def _setup_rundir(tmp_path: Path) -> str:
    """Return a temporary run directory path that all test ranks share."""
    rundir = str(tmp_path / "rundir")
    dist = DistributedManager()
    if dist.world_size > 1:
        output_list = [None]
        torch.distributed.scatter_object_list(
            output_list, [rundir] * dist.world_size, src=0
        )
        rundir = output_list[0]
    return rundir


@pytest.mark.parametrize("fp_optimizations", ["fp32", "amp-bf16"])
@pytest.mark.parametrize("lr_decay", ["cosine", "constant"])
def test_training(
    tmp_path: Path, cfg: DictConfig, fp_optimizations: str, lr_decay: str
):
    """Test that training runs and writes checkpoints and validation plots."""
    rundir = _setup_rundir(tmp_path)
    cfg.training.rundir = rundir
    cfg.training.batch_size = 2 * DistributedManager().world_size
    cfg.training.perf.fp_optimizations = fp_optimizations
    cfg.training.scheduler.lr_decay = lr_decay

    train.main(cfg)

    checkpoint_path = os.path.join(rundir, "checkpoints")
    assert os.path.isfile(os.path.join(checkpoint_path, "TimeScaledModel.0.10.mdlus"))
    assert os.path.isfile(os.path.join(checkpoint_path, "checkpoint.0.10.pt"))
    assert os.path.isfile(os.path.join(rundir, "images", "state_0_000000010.png"))


def test_checkpointing(tmp_path: Path, cfg: DictConfig):
    """Test that training resumes from the latest checkpoint."""
    rundir = _setup_rundir(tmp_path)
    cfg.training.rundir = rundir
    cfg.training.batch_size = 2 * DistributedManager().world_size

    cfg.training.checkpoints_to_keep = 2

    # Train for 10 steps, then continue the same run to 20 steps
    train.main(cfg)
    cfg.training.total_train_steps = 20
    trainer = Trainer(cfg)
    assert trainer.total_steps == 10
    assert trainer.scheduler.last_epoch == 10
    trainer.train()

    # Only the two most recent checkpoints remain
    checkpoint_path = os.path.join(rundir, "checkpoints")
    assert sorted(os.listdir(checkpoint_path)) == [
        "TimeScaledModel.0.15.mdlus",
        "TimeScaledModel.0.20.mdlus",
        "checkpoint.0.15.pt",
        "checkpoint.0.20.pt",
    ]


def test_resume_mid_epoch(tmp_path: Path, cfg: DictConfig):
    """Test that a run resumed in the middle of an epoch continues with the
    next batch of that epoch."""
    dist = DistributedManager()
    cfg.training.rundir = _setup_rundir(tmp_path)
    cfg.training.batch_size = 2 * dist.world_size
    cfg.training.total_train_steps = 3
    train.main(cfg)
    trainer = Trainer(cfg)

    # The order of the first epoch on this rank, and the batch that comes next
    order = list(
        DistributedSampler(
            trainer.dataset_train,
            num_replicas=dist.world_size,
            rank=dist.rank,
            shuffle=True,
            seed=cfg.training.seed,
            drop_last=True,
        )
    )
    batches_done = trainer.total_steps * trainer.num_accumulation_rounds
    assert 0 < batches_done < len(trainer.train_dataloader)
    batch_size = trainer.local_batch_size
    expected = order[batches_done * batch_size : (batches_done + 1) * batch_size]

    batch = next(trainer.dataset_iterator)
    expected_state = np.stack([trainer.dataset_train[i]["state"] for i in expected])
    assert torch.equal(batch["state"], torch.from_numpy(expected_state))


@pytest.mark.parametrize("t", [None, [0.0, 1.0]])
@pytest.mark.parametrize("amp", [False, True])
def test_loss_equals_x0_mse(
    tmp_path: Path, cfg: DictConfig, t: list[float] | None, amp: bool
):
    """Test that the training loss equals the x0-prediction loss of MSEDSMLoss.
    It covers sampled times and the endpoints t = 0 and 1, with and without
    bf16 autocast."""
    cfg.training.rundir = _setup_rundir(tmp_path)
    cfg.training.batch_size = 2 * DistributedManager().world_size
    trainer = Trainer(cfg)
    x0_mse = MSEDSMLoss(
        trainer.train_net,
        trainer.noise_scheduler,
        prediction_type="x0",
        reduction="none",
    )

    device = trainer.device
    (h, w) = cfg.dataset.image_size
    x0 = torch.randn(2, cfg.dataset.num_state_channels, h, w, device=device)
    background = torch.randn(
        2, cfg.dataset.num_background_channels, h, w, device=device
    )
    scalar_conditions = torch.randn(
        2, cfg.dataset.num_scalar_cond_channels, device=device
    )
    condition = build_condition(background, scalar_conditions)
    if t is not None:
        t = torch.tensor(t, device=device)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        torch.manual_seed(0)
        loss = trainer.loss_fn(x0, torch.ones_like(x0), t=t, condition=condition)
        torch.manual_seed(0)  # same times and noise as in the training loss
        expected = x0_mse(x0, t=t, condition=condition)
    assert torch.isfinite(loss).all()
    assert torch.equal(loss, expected)


def _write_hrrr_era5_data(path: Path) -> dict:
    """Write a small dataset in the layout that HRRRERA5Dataset expects. The ERA5
    fields are linear in latitude and longitude, so their bilinear
    interpolation to the HRRR grid is exact."""
    times = pd.date_range("2025-01-01", periods=6, freq="h")
    (y, x) = np.meshgrid(np.arange(12), np.arange(16), indexing="ij")
    lat = (35.0 + 0.5 * y + 0.1 * x).astype(np.float32)
    lon = (-105.0 + 0.6 * x + 0.05 * y).astype(np.float32)
    rng = np.random.default_rng(0)
    hrrr = rng.normal(size=(6, 2, 12, 16)).astype(np.float32)
    xr.Dataset(
        {
            "HRRR": (("time", "channel", "y", "x"), hrrr),
            "latitude": (("y", "x"), lat),
            "longitude": (("y", "x"), lon),
            "valid_data": ("time", np.array([True] * 5 + [False])),  # last hour invalid
        },
        coords={"time": times, "channel": np.array(["u10m", "t2m"])},
    ).to_zarr(path / "hrrr-conus" / "2025.zarr")

    # ERA5 fields c0 + c1 * lat + c2 * lon, without the hour at index 4
    (era5_lat, era5_lon) = (np.arange(50.0, 29.0, -1.0), np.arange(-110.0, -89.0))
    coefs = np.array([[1.0, 0.2, -0.3], [2.0, -0.1, 0.4], [3.0, 0.05, 0.1]])
    fields = (
        coefs[:, 0, None, None]
        + coefs[:, 1, None, None] * era5_lat[:, None]
        + coefs[:, 2, None, None] * era5_lon[None, :]
    )
    era5_times = times[[0, 1, 2, 3, 5]]
    xr.Dataset(
        {
            "data": (
                ("time", "channel", "latitude", "longitude"),
                np.repeat(fields[None], len(era5_times), axis=0).astype(np.float32),
            )
        },
        coords={
            "time": era5_times,
            "channel": np.array(["u10m", "tcwv", "sp"]),
            "latitude": era5_lat,
            "longitude": era5_lon,
        },
    ).to_zarr(path / "era5-conus" / "2025.zarr")

    invariant_names = ["elev_mean", "elev_std", "lsm_mean"]
    invariants = rng.normal(size=(3, 12, 16)).astype(np.float32)
    os.makedirs(path / "invariants")
    xr.Dataset(
        {v: (("y", "x"), invariants[k]) for (k, v) in enumerate(invariant_names)}
    ).to_netcdf(path / "invariants" / "invariants_HRRR.nc")

    stats = {
        "hrrr": {"u10m": {"mean": 0.5, "std": 2.0}, "t2m": {"mean": -1.0, "std": 4.0}},
        "era5": {v: {"mean": 1.0, "std": 3.0} for v in ["u10m", "tcwv", "sp"]},
        "invariant": {v: {"mean": 0.1, "std": 2.0} for v in invariant_names},
    }
    os.makedirs(path / "stats")
    with open(path / "stats" / "hrrr_era5_stats.json", "w") as f:
        json.dump(stats, f)
    return {
        "hrrr": hrrr,
        "lat": lat,
        "lon": lon,
        "coefs": coefs,
        "invariants": invariants,
        "times": times,
    }


def test_hrrr_era5_dataset(tmp_path: Path):
    """Test time filtering, cropping, interpolation, normalization, patches,
    and presence flags of HRRRERA5Dataset on a small synthetic dataset."""
    data = _write_hrrr_era5_data(tmp_path)
    params = dict(
        data_path=str(tmp_path),
        years=[2025],
        time_range_train=["2025-01-01T00:00:00", "2025-01-01T02:00:00"],
        time_range_valid=["2025-01-01T02:00:00", "2025-01-02T00:00:00"],
        crop_window=[[1, 11], [2, 14]],
        drop_variables=["tcwv", "sp"],
        drop_prob=1.0,
    )
    crop = (slice(1, 11), slice(2, 14))

    # Validation on the full domain: the dataset skips the invalid HRRR hour
    # and the hour missing in ERA5, and keeps every input
    valid = HRRRERA5Dataset(train=False, patch_shape=None, **params)
    assert list(valid.times) == list(data["times"][[2, 3]])
    assert valid.image_shape() == (10, 12)
    assert len(valid.background_channels()) == 3 + 1 + 3
    item = valid[0]
    np.testing.assert_allclose(
        item["state"],
        (data["hrrr"][2][:, crop[0], crop[1]] - np.array([0.5, -1.0])[:, None, None])
        / np.array([2.0, 4.0])[:, None, None],
        rtol=1e-6,
    )
    (lat, lon) = (data["lat"][crop], data["lon"][crop])
    coefs = data["coefs"]
    era5 = (
        coefs[:, 0, None, None]
        + coefs[:, 1, None, None] * lat
        + coefs[:, 2, None, None] * lon
    )
    np.testing.assert_allclose(item["background"][:3], (era5 - 1.0) / 3.0, atol=1e-4)
    np.testing.assert_allclose(
        item["background"][4:],
        (data["invariants"][:, crop[0], crop[1]] - 0.1) / 2.0,
        rtol=1e-6,
    )
    year_phase = 2 * np.pi * (2 / 24) / 365.25  # 02:00 on January 1
    np.testing.assert_allclose(
        item["scalar_conditions"],
        [np.sin(year_phase), np.cos(year_phase), 1, 1],
        rtol=1e-5,
    )

    # Training patches: with drop_prob = 1, the dataset replaces tcwv and sp by
    # their climatological mean (zero) and sets their presence flags to zero
    train_ds = HRRRERA5Dataset(train=True, patch_shape=[4, 4], **params)
    assert len(train_ds) == 2
    item = train_ds[1]
    assert item["state"].shape == (2, 4, 4)
    assert item["background"].shape == (7, 4, 4)
    assert np.all(item["background"][1:3] == 0)
    np.testing.assert_array_equal(item["scalar_conditions"][2:], [0, 0])


class GaussianX0Predictor(torch.nn.Module):
    """Exact x0-predictor E[x0 | x_t] for data x0 ~ N(mean, std^2) on the
    rectified-flow path x_t = (1 - t) x0 + t eps."""

    def __init__(self, mean: float, std: float):
        super().__init__()
        self.mean = mean
        self.var = std**2
        self.unused = torch.nn.Parameter(torch.zeros(()))  # sets the device

    def forward(self, x, t, condition=None):
        alpha = (1 - t).reshape(-1, 1, 1, 1)
        sigma = t.reshape(-1, 1, 1, 1)
        gain = alpha * self.var / (alpha**2 * self.var + sigma**2)
        return self.mean + gain * (x - alpha * self.mean)


@pytest.mark.parametrize(
    "solver, num_steps, time_shift",
    [("heun", 32, 1.0), ("heun", 64, 4.0), ("euler", 128, 1.0)],
)
def test_sampling_from_pure_noise(solver: str, num_steps: int, time_shift: float):
    """Test that sampling from pure noise (t = 1) with the exact x0-predictor
    of Gaussian data reproduces the data distribution."""
    (mean, std) = (2.0, 0.5)
    model = GaussianX0Predictor(mean, std).to(DistributedManager().device)
    torch.manual_seed(0)
    x = sample_flow(
        model,
        None,
        (4, 1, 64, 64),
        RectifiedFlowNoiseScheduler(t_max=1.0),
        num_steps=num_steps,
        solver=solver,
        time_shift=time_shift,
    )
    assert torch.isfinite(x).all()
    assert abs(x.mean().item() - mean) < 0.02
    assert abs(x.std().item() - std) < 0.02


@pytest.mark.skipif(
    not os.path.isfile(CHECKPOINT), reason="released checkpoint not available"
)
def test_released_checkpoint():
    """Test that the released weights match the model config and that
    sampling from pure noise works."""
    assert isinstance(Module.from_checkpoint(CHECKPOINT), TimeScaledModel)

    device = DistributedManager().device
    model_cfg = _load_config(config_name="diffusion").model
    net = get_dit(
        target_channels=99,
        conditional_channels=30,
        scalar_condition_channels=4,
        img_resolution=(256, 256),
        **OmegaConf.to_container(model_cfg.hyperparameters, resolve=True),
    )
    net.load(CHECKPOINT)  # strict: all weights must match the architecture
    net = net.to(device).eval()

    torch.manual_seed(0)
    condition = build_condition(
        torch.randn(1, 30, 256, 256, device=device),
        torch.tensor([[0.0, 1.0, 1.0, 1.0]], device=device),
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x = sample_flow(
            net,
            condition,
            (1, 99, 256, 256),
            RectifiedFlowNoiseScheduler(t_max=1.0),
            num_steps=4,
        )
    assert x.shape == (1, 99, 256, 256)
    assert torch.isfinite(x).all()
