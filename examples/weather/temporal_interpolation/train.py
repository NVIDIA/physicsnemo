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

import os
import datetime
from typing import Any, Dict
import warnings

import hydra
from omegaconf import OmegaConf
import torch
import wandb

from physicsnemo import Module
from physicsnemo.datapipes.climate.climate import ClimateDataSourceSpec
from physicsnemo.datapipes.climate.utils import invariant
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import LaunchLogger
from physicsnemo.utils.logging.mlflow import initialize_mlflow
from physicsnemo.utils.logging.wandb import initialize_wandb
from physicsnemo.models.afno import ModAFNO
from physicsnemo.utils import load_checkpoint

from datapipe.climate_interp import InterpClimateDatapipe
from utils import distribute, loss
from utils.trainer import Trainer

try:
    from apex.optimizers import FusedAdam
except ImportError:
    warnings.warn("Apex is not installed, defaulting to PyTorch optimizers.")


def setup_datapipes(
    *,
    data_dir: str,
    dist_manager: DistributedManager,
    metadata_path: str,
    geopotential_filename: str | None = None,
    lsm_filename: str | None = None,
    use_latlon: bool = True,
    num_samples_per_year_train: int | None = None,
    num_samples_per_year_valid: int = 4,
    batch_size_train: int = 4,
    batch_size_valid: int | None = None,
    num_workers: int = 4,
    valid_subdir: str = "test",
    valid_start_year: int = 2017,
    valid_shuffle: bool = False,
) -> tuple[InterpClimateDatapipe, InterpClimateDatapipe, int]:
    """
    Setup datapipes for training.

    The arguments passed to this function can be modified in the 'datapipe' section
    of the config.

    Parameters
    ----------
    data_dir : str
        Path to data directory.
    dist_manager : DistributedManager
        An initialized DistributedManager instance.
    metadata_path : str
        Path to metadata file.
    geopotential_filename : str or None, optional
        Path to NetCDF file with global geopotential on the 0.25 deg grid.
    lsm_filename : str or None, optional
        Path to NetCDF file with global land-sea mask on the 0.25 deg grid.
    use_latlon : bool, optional
        If True, will return latitude and longitude from the datapipe.
    num_samples_per_year_train : int or None, optional
        Number of training samples per year, if None will use all available samples.
    num_samples_per_year_valid : int, optional
        Number of validation samples per year.
    batch_size_train : int, optional
        Batch size per GPU for training.
    batch_size_valid : int or None, optional
        Batch size per GPU for validation, when None equal to batch_size_train.
    num_workers : int, optional
        Number of datapipe workers per training process.
    valid_subdir : str, optional
        Subdirectory in data_dir where validation data is found.
    valid_start_year : int, optional
        Starting year for validation data.
    valid_shuffle : bool, optional
        When True, shuffle order of validation set; recommend setting to False
        for consistent validation results.

    Returns
    -------
    tuple of (InterpClimateDatapipe, InterpClimateDatapipe, int)
        Tuple of training datapipe and validation datapipe, and the number of auxiliary channels.
    """
    if batch_size_valid is None:
        batch_size_valid = batch_size_train

    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, valid_subdir)
    mean_file = os.path.join(data_dir, "stats/global_means.npy")
    std_file = os.path.join(data_dir, "stats/global_stds.npy")

    spec_kwargs: Dict[str, Any] = dict(
        stats_files={"mean": mean_file, "std": std_file},
        use_cos_zenith=True,
        name="atmos",
        metadata_path=metadata_path,
        stride=6,
    )

    spec_train = ClimateDataSourceSpec(data_dir=train_dir, **spec_kwargs)
    spec_valid = ClimateDataSourceSpec(data_dir=valid_dir, **spec_kwargs)

    invariants = {}
    num_aux_channels = 3  # 3 channels for cos_zenith
    if use_latlon:
        invariants["latlon"] = invariant.LatLon()
        num_aux_channels += 4
    if geopotential_filename is not None:
        invariants["geopotential"] = invariant.FileInvariant(geopotential_filename, "Z")
        num_aux_channels += 1
    if lsm_filename is not None:
        invariants["land_sea_mask"] = invariant.FileInvariant(lsm_filename, "LSM")
        num_aux_channels += 1

    pipe_kwargs = dict(
        invariants=invariants,
        crop_window=((0, 720), (0, 1440)),
        num_workers=num_workers,
        device=dist_manager.device,
        dt=1.0,
    )

    if num_samples_per_year_train is None:
        num_samples_per_year_train = 365 * 24 - 12  # -12 to prevent overflow

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

    return (pipe_train, pipe_valid, num_aux_channels)


# Default parameters if not overridden by config
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


def setup_model(
    num_variables: int, num_auxiliaries: int, model_cfg: dict | None = None
) -> Module:
    """
    Setup interpolation model.

    Parameters
    ----------
    num_variables : int
        Number of atmospheric variables in the model.
    num_auxiliaries : int
        Number of auxiliary input channels.
    model_cfg : dict or None, optional
        Model configuration dict.

    Returns
    -------
    Module
        Model object.
    """
    if model_cfg is None:
        model_cfg = {}
    model_type = model_cfg.pop("model_type", "modafno")
    if model_type != "modafno":
        raise ValueError(
            "Model types other than 'modafno' are not currently supported."
        )
    if model_cfg.get("in_channels") is None:
        model_cfg["in_channels"] = 2 * num_variables + num_auxiliaries
    if model_cfg.get("out_channels") is None:
        model_cfg["out_channels"] = num_variables
    model_name = model_cfg.pop("model_name", None)
    model_kwargs = default_model_params[model_type].copy()
    model_kwargs.update(model_cfg)
    if model_type == "modafno":
        model = ModAFNO(**model_kwargs)

    if model_name is not None:
        model.meta.name = model_name

    return model


def setup_optimizer(
    model: torch.nn.Module,
    max_epoch: int,
    opt_cls: type[torch.optim.Optimizer] | None = None,
    opt_params: dict | None = None,
    scheduler_cls: type[torch.optim.lr_scheduler.LRScheduler] | None = None,
    scheduler_params: dict[str, Any] | None = None,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Setup optimizer.

    Parameters
    ----------
    model : torch.nn.Module
        Model that optimizer is applied to.
    max_epoch : int
        Maximum number of training epochs (used for scheduler setup).
    opt_cls : type[torch.optim.Optimizer] or None, optional
        Optimizer class. When None, will setup apex.optimizers.FusedAdam
        if available, otherwise PyTorch Adam.
    opt_params : dict or None, optional
        Dict of parameters (e.g. learning rate) to pass to optimizer.
    scheduler_cls : type[torch.optim.lr_scheduler.LRScheduler] or None, optional
        Scheduler class. When None, will setup CosineAnnealingLR.
    scheduler_params : dict[str, Any] or None, optional
        Dict of parameters to pass to scheduler.

    Returns
    -------
    tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]
        The initialized optimizer and learning rate scheduler.
    """

    opt_kwargs = {"lr": 0.0005}
    if opt_params is not None:
        opt_kwargs.update(opt_params)
    if opt_cls is None:
        try:
            opt_cls = FusedAdam
        except NameError:  # in case we don't have apex
            opt_cls = torch.optim.Adam

    scheduler_kwargs = {}
    if scheduler_cls is None:
        scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingLR
        scheduler_kwargs["T_max"] = max_epoch
    if scheduler_params is not None:
        scheduler_kwargs.update(scheduler_params)

    optimizer = opt_cls(model.parameters(), **opt_kwargs)
    scheduler = scheduler_cls(optimizer, **scheduler_kwargs)
    return (optimizer, scheduler)


@torch.no_grad()
def input_output_from_batch_data(
    batch: list[dict[str, torch.Tensor]], time_scale: float = 6 * 3600.0
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Convert the datapipe output dict to model input and output batches.

    Parameters
    ----------
    batch : list[dict[str, torch.Tensor]]
        The list data dicts returned by the datapipe.
    time_scale : float, optional
        Number of seconds between the interpolation endpoints (default 6 hours).

    Returns
    -------
    tuple
        Nested tuple in the form ((input, time), output).
    """
    batch = batch[0]
    # Concatenate all input variables to a single tensor
    atmos_vars = batch["state_seq-atmos"]

    atmos_vars_in = [atmos_vars[:, 0], atmos_vars[:, 1]]
    if "cos_zenith-atmos" in batch:
        atmos_vars_in = atmos_vars_in + [batch["cos_zenith-atmos"].squeeze(dim=2)]
    if "latlon" in batch:
        atmos_vars_in = atmos_vars_in + [batch["latlon"]]
    if "geopotential" in batch:
        atmos_vars_in = atmos_vars_in + [batch["geopotential"]]
    if "land_sea_mask" in batch:
        atmos_vars_in = atmos_vars_in + [batch["land_sea_mask"]]
    atmos_vars_in = torch.cat(atmos_vars_in, dim=1)

    atmos_vars_out = atmos_vars[:, 2]

    time = batch["timestamps-atmos"]
    # Normalize time coordinate
    time = (time[:, -1:] - time[:, :1]).to(dtype=torch.float32) / time_scale

    return ((atmos_vars_in, time), atmos_vars_out)


def setup_trainer(**cfg: dict) -> Trainer:
    """
    Setup training environment.

    Parameters
    ----------
    **cfg : dict
        The configuration dict passed from hydra.

    Returns
    -------
    Trainer
        The Trainer object for training the interpolation model.
    """

    DistributedManager.initialize()

    # Setup datapipes
    (train_datapipe, valid_datapipe, num_aux_channels) = setup_datapipes(
        **cfg["datapipe"],
        dist_manager=DistributedManager(),
    )

    # Setup model
    model = setup_model(
        num_variables=len(train_datapipe.sources[0].variables),
        num_auxiliaries=num_aux_channels,
        model_cfg=cfg["model"],
    )
    (model, dist_manager) = distribute.distribute_model(model)

    # Setup optimizer and learning rate scheduler
    (optimizer, scheduler) = setup_optimizer(
        model,
        cfg["training"].get("max_epoch", 1),
        opt_params=cfg.get("optimizer_params", {}),
        scheduler_params=cfg.get("scheduler_params", {}),
    )

    # Initialize mlflow
    mlflow_cfg = cfg.get("logging", {}).get("mlflow", {})
    if mlflow_cfg.pop("use_mlflow", False):
        initialize_mlflow(**mlflow_cfg)
        LaunchLogger.initialize(use_mlflow=True)

    # Initialize wandb
    use_wandb = False
    wandb_cfg = cfg.get("logging", {}).get("wandb", {})
    if wandb_cfg.get("use_wandb", False):
        use_wandb = True
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # Get checkpoint directory
        checkpoint_dir = cfg.get("training", {}).get("checkpoint_dir")
        # Check if we need to resume from checkpoint
        wandb_id = None
        resume = None
        load_epoch = cfg.get("training", {}).get("load_epoch")
        if checkpoint_dir is not None and load_epoch is not None:
            metadata = {"wandb_id": None}
            load_checkpoint(checkpoint_dir, metadata_dict=metadata)
            wandb_id = metadata.get("wandb_id")
            if wandb_id is not None:
                resume = "must"

        initialize_wandb(
            project=wandb_cfg.get("project", "Temporal-Interpolation-Training"),
            entity=wandb_cfg.get("entity"),
            mode=wandb_cfg.get("mode", "offline"),
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
            results_dir=wandb_cfg.get("results_dir", "./wandb/"),
            wandb_id=wandb_id,
            resume=resume,
            save_code=True,
            name=f"train-{timestamp}",
            init_timeout=600,
        )

    # Setup training loop
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
        optimizer=optimizer,
        scheduler=scheduler,
        use_wandb=use_wandb,
        **cfg["training"],
    )

    return trainer


@hydra.main(version_base=None, config_path="config")
def main(cfg):
    """
    Main entry point for training the interpolation model.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """
    trainer = setup_trainer(**OmegaConf.to_container(cfg))
    trainer.fit()

    # Finish wandb logging if it was used
    use_wandb = cfg.get("logging", {}).get("wandb", {}).get("use_wandb", False)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
