# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import os

import hydra
from omegaconf import OmegaConf
import torch

from physicsnemo import Module
from physicsnemo.datapipes.climate.climate import ClimateDataSourceSpec
from physicsnemo.datapipes.climate.utils import invariant
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import LaunchLogger
from physicsnemo.launch.logging.mlflow import initialize_mlflow
from physicsnemo.models.afno import ModAFNO

from datapipe.climate_interp import InterpClimateDatapipe
from utils import distribute, loss
from utils.trainer import Trainer


def setup_datapipes(
    *,
    data_dir: str,
    dist_manager: DistributedManager,
    metadata_path: str,
    geopotential_filename: str | None = None,
    lsm_filename: str | None = None,
    use_latlon: bool = True,
    num_samples_per_year_train: int = 365 * 24 - 6,
    num_samples_per_year_valid: int = 4,
    batch_size_train: int = 4,
    batch_size_valid: int | None = None,
    num_workers: int = 4,
    valid_subdir: str = "test",
    valid_start_year: int = 2017,
    valid_shuffle: bool = False,
) -> tuple[InterpClimateDatapipe, InterpClimateDatapipe]:
    """Setup datapipes for training.

    The arguments passed to this function can be modified in the 'datapipe' section
    of the config.

    Args:
        data_dir: path to data directory
        dist_manager: an initialized DistributedManager instance
        geopotential_filename: path to NetCDF file with global geopotential on the 0.25 deg grid
        geopotential_filename: path to NetCDF file with global land-sea mask on the 0.25 deg grid
        use_latlon: if True, will return latitude and longitude from the datapipe
        num_samples_per_year_train: number of training samples per year
        num_samples_per_year_valid: number of validation samples per year
        batch_size_train: batch size per GPU for training
        batch_size_valid: batch size per GPU for validation, when None equal to batch_size_train
        num_workers: number of datapipe workers per training process
        valid_subdir: subdirectory in data_dir where validation data is found
        valid_shuffle: when True, shuffle order of validation set; recommend setting this to False for consistent validation results
    Returns:
        Tuple of training datapipe and validation datapipe.
    """
    if batch_size_valid is None:
        batch_size_valid = batch_size_train

    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, valid_subdir)
    mean_file = os.path.join(data_dir, "stats/global_means.npy")
    std_file = os.path.join(data_dir, "stats/global_stds.npy")

    spec_kwargs = dict(
        stats_files={"mean": mean_file, "std": std_file},
        use_cos_zenith=True,
        name="atmos",
        metadata_path=metadata_path,
        stride=6,
    )

    spec_train = ClimateDataSourceSpec(data_dir=train_dir, **spec_kwargs)

    spec_valid = ClimateDataSourceSpec(data_dir=valid_dir, **spec_kwargs)

    invariants = {}
    if use_latlon:
        invariants["latlon"] = invariant.LatLon()
    if geopotential_filename is not None:
        invariants["geopotential"] = invariant.FileInvariant(geopotential_filename, "Z")
    if lsm_filename is not None:
        invariants["land_sea_mask"] = invariant.FileInvariant(lsm_filename, "LSM")

    pipe_kwargs = dict(
        invariants=invariants,
        crop_window=((0, 720), (0, 1440)),
        num_workers=num_workers,
        device=dist_manager.device,
        dt=1.0,
    )

    pipe_train = InterpClimateDatapipe(
        [spec_train],
        batch_size=batch_size_train,
        num_samples_per_year=num_samples_per_year_train,
        process_rank=dist_manager.rank,
        world_size=dist_manager.world_size,
        **pipe_kwargs,
    )

    pipe_valid = InterpClimateDatapipe(
        [spec_valid],
        batch_size=batch_size_valid,
        num_samples_per_year=num_samples_per_year_valid,
        shuffle=valid_shuffle,
        start_year=valid_start_year,
        **pipe_kwargs,
    )

    return (pipe_train, pipe_valid)


# default parameters if not overridden by config
default_model_params = {
    "modafno": {
        "inp_shape": (720, 1440),
        "in_channels": 155,
        "out_channels": 73,
        "patch_size": (8, 8),
        "embed_dim": 768,
        "depth": 12,
        "num_blocks": 8,
    }
}


def setup_model(model_cfg: dict | None = None) -> Module:
    """Setup interpolation model.

    Args:
        model_cfg: model configuration dict
    Returns:
        Model object
    """
    if model_cfg is None:
        model_cfg = {}
    model_type = model_cfg.pop("model_type", "modafno")
    if model_type != "modafno":
        raise ValueError(
            "Model types other than 'modafno' are not currently supported."
        )
    model_name = model_cfg.pop("model_name")
    model_kwargs = default_model_params[model_type].copy()
    model_kwargs.update(model_cfg)
    if model_type == "modafno":
        model = ModAFNO(**model_kwargs)

    if model_name is not None:
        model.meta.name = model_name

    return model


@torch.no_grad()
def input_output_from_batch_data(
    batch: dict[str, torch.Tensor], time_scale: float = 6 * 3600.0
) -> tuple[[tuple[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """Function to convert the datapipe output dict to model input and output batches.

    Args:
        batch: The data dict returned by the datapipe.
        time_scale: Number of seconds between the interpolation endpoints (default 6 hours)
    Returns:
        Nested tuple in the form ((input, time), output)
    """
    batch = batch[0]
    # concatenate all input variables to a single tensor
    atmos_vars = batch["state_seq-atmos"]
    cos_zenith = batch["cos_zenith-atmos"].squeeze(dim=2)

    sincos_latlon = batch["latlon"]
    geop = batch["geopotential"]
    lsm = batch["land_sea_mask"]

    atmos_vars_in = torch.cat(
        [atmos_vars[:, 0], atmos_vars[:, 1], cos_zenith, sincos_latlon, geop, lsm],
        dim=1,
    )

    atmos_vars_out = atmos_vars[:, 2]

    time = batch["timestamps-atmos"]
    # normalize time coordinate
    time = (time[:, -1:] - time[:, :1]).to(dtype=torch.float32) / time_scale

    return ((atmos_vars_in, time), atmos_vars_out)


def setup_trainer(**cfg: dict) -> Trainer:
    """Setup training environment.

    Args:
        cfg: The configuration dict passed from hydra.
    Returns:
        The Trainer object for training the interpolation model.
    """

    # setup model
    model = setup_model(model_cfg=cfg["model"])
    (model, dist_manager) = distribute.distribute_model(model)

    # setup datapipes
    (train_datapipe, valid_datapipe) = setup_datapipes(
        **cfg["datapipe"],
        dist_manager=dist_manager,
    )

    mlflow_cfg = cfg.get("logging", {}).get("mlflow", {})
    if mlflow_cfg.pop("use_mlflow", False):
        initialize_mlflow(**mlflow_cfg)
        LaunchLogger.initialize(use_mlflow=True)

    # setup training loop
    loss_func = loss.GeometricL2Loss(num_lats_cropped=cfg["model"]["inp_shape"][0]).to(
        device=dist_manager.device
    )
    trainer = Trainer(
        model,
        dist_manager=dist_manager,
        loss=loss_func,
        train_datapipe=train_datapipe,
        valid_datapipe=valid_datapipe,
        input_output_from_batch_data=input_output_from_batch_data,
        **cfg["training"],
    )

    return trainer


@hydra.main(version_base=None, config_path="config")
def main(cfg):
    """Main function."""
    trainer = setup_trainer(**OmegaConf.to_container(cfg))
    trainer.fit()


if __name__ == "__main__":
    main()
