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

from datetime import datetime, timedelta
from typing import Generator, Literal

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import xarray as xr

from train_interp import setup_trainer, Trainer


def setup_analysis(
    cfg: dict, checkpoint: str | None = None, shuffle: bool = False
) -> Trainer:
    """Setup trainer for validation analysis.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.
    checkpoint : str or None, optional
        Path to model checkpoint file.
    shuffle : bool, optional
        Whether to shuffle validation data.

    Returns
    -------
    Trainer
        Configured trainer instance.
    """
    cfg["datapipe"]["num_samples_per_year_valid"] = cfg["datapipe"][
        "num_samples_per_year_train"
    ]
    cfg["datapipe"]["batch_size_valid"] = 1
    cfg["datapipe"]["valid_shuffle"] = shuffle

    trainer = setup_trainer(**cfg)
    if checkpoint is not None:
        trainer.model.load(checkpoint)

    return trainer


@torch.no_grad()
def inference_model(
    trainer: Trainer,
    timesteps: int = 6,
    denorm: bool = True,
    method: Literal["fcinterp", "linear"] = "fcinterp",
) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    """Run inference on validation data.

    Parameters
    ----------
    trainer : Trainer
        Trainer instance containing model and datapipe.
    timesteps : int, optional
        Number of timesteps between interpolation endpoints.
    denorm : bool, optional
        Whether to denormalize outputs.
    method : {"fcinterp", "linear"}, optional
        Interpolation method to use.

    Yields
    ------
    tuple[torch.Tensor, torch.Tensor]
        True and predicted values for each batch.
    """
    for batch in trainer.valid_datapipe:
        y_true_step = []
        y_pred_step = []
        for step in range(timesteps + 1):
            (invar, outvar_true) = input_output_from_batch_data_analysis(batch, step)
            invar = tuple(v.detach() for v in invar)
            outvar_true = outvar_true.detach()
            y_true_step.append(outvar_true)
            if method == "fcinterp":
                y_pred_step.append(trainer.eval_step(invar))
            elif method == "linear":
                y_pred_step.append(linear_interp_batch_data(batch, step))

        y_true = torch.stack(y_true_step, dim=1)
        y_pred = torch.stack(y_pred_step, dim=1)
        if denorm:
            y_true = denormalize(trainer, y_true)
            y_pred = denormalize(trainer, y_pred)

        yield (y_true, y_pred)


@torch.no_grad()
def input_output_from_batch_data_analysis(
    batch: list[dict[str, torch.Tensor]], step: int, time_scale: float = 6 * 3600.0
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Convert batch data to model inputs and outputs for a specific timestep.

    Parameters
    ----------
    batch : list[dict[str, torch.Tensor]]
        Batch dictionary from datapipe.
    step : int
        Timestep index for output.
    time_scale : float, optional
        Length of the interpolation interval in seconds.

    Returns
    -------
    tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]
        Model inputs (atmospheric variables, time) and ground truth output.
    """
    batch = batch[0]

    # concatenate all input variables to a single tensor
    atmos_vars = batch["state_seq-atmos"]

    atmos_vars_in = [atmos_vars[:, 0], atmos_vars[:, -1]]
    if "cos_zenith-atmos" in batch:
        atmos_vars_in = atmos_vars_in + [batch["cos_zenith-atmos"].squeeze(dim=2)]
    if "latlon" in batch:
        atmos_vars_in = atmos_vars_in + [batch["latlon"]]
    if "geopotential" in batch:
        atmos_vars_in = atmos_vars_in + [batch["geopotential"]]
    if "land_sea_mask" in batch:
        atmos_vars_in = atmos_vars_in + [batch["land_sea_mask"]]
    atmos_vars_in = torch.cat(atmos_vars_in, dim=1)

    atmos_vars_out = atmos_vars[:, step]

    time = batch["timestamps-atmos"]
    # normalize time coordinate
    time = (time[:, step : step + 1] - time[:, :1]).to(dtype=torch.float32) / time_scale

    return ((atmos_vars_in, time), atmos_vars_out)


def linear_interp_batch_data(batch: dict[str, torch.Tensor], step: int) -> torch.Tensor:
    """Perform linear interpolation on batch data.

    Parameters
    ----------
    batch : dict[str, torch.Tensor]
        Batch dictionary from datapipe.
    step : int
        Timestep index for interpolation.

    Returns
    -------
    torch.Tensor
        Linearly interpolated atmospheric variables.
    """
    atmos_vars = batch[0]["state_seq-atmos"]
    x0 = atmos_vars[:, 0]
    x1 = atmos_vars[:, -1]
    alpha = step / (atmos_vars.shape[1] - 1)
    return (1 - alpha) * x0 + alpha * x1


def denormalize(trainer: Trainer, y: torch.Tensor) -> torch.Tensor:
    """Denormalize predictions using dataset statistics.

    Parameters
    ----------
    trainer : Trainer
        Trainer instance containing datapipe with statistics.
    y : torch.Tensor
        Normalized tensor to denormalize.

    Returns
    -------
    torch.Tensor
        Denormalized tensor.
    """
    mean = torch.Tensor(trainer.valid_datapipe.sources[0].mu).to(device=y.device)[
        :, None, ...
    ]
    std = torch.Tensor(trainer.valid_datapipe.sources[0].sd).to(device=y.device)[
        :, None, ...
    ]
    return y * std + mean


def error_by_time(
    cfg: dict,
    checkpoint: str | None = None,
    timesteps: int = 6,
    method: Literal["fcinterp", "linear"] = "fcinterp",
    max_error: float = 1.0,
    nbins: int = 10000,
    n_samples: int = 1000,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Compute error statistics for each interpolation step.

    Parameters
    ----------
    cfg : dict
        The configuration dict passed from hydra.
    checkpoint : str or None, optional
        Path to model checkpoint file.
    timesteps : int, optional
        Number of timesteps between interpolation endpoints.
    method : {"fcinterp", "linear"}, optional
        Interpolation method to use.
    max_error : float, optional
        Maximum error value for histogram bins.
    nbins : int, optional
        Number of histogram bins.
    n_samples : int, optional
        Number of samples to process.

    Returns
    -------
    tuple[list[torch.Tensor], torch.Tensor]
        Histogram counts for each timestep and bin edges.
    """
    trainer = setup_analysis(cfg=cfg, checkpoint=checkpoint, shuffle=True)

    lat = torch.linspace(90, -90, 721)[:-1].to(device=trainer.model.device)
    lat[0] = 0.5 * (lat[0] + lat[1])
    cos_lat = torch.cos(lat * (torch.pi / 180))[None, None, :, None]

    bins = torch.linspace(0, max_error, nbins + 1)

    def _hist(y_true, y_pred):
        err = (y_true - y_pred) ** 2
        weights = torch.ones_like(err) * cos_lat
        return torch.histogram(
            err.ravel().cpu(), bins=bins, weight=weights.ravel().cpu()
        )[0]

    hist_counts = [None] * (timesteps + 1)

    for i_sample, (y_true, y_pred) in enumerate(
        inference_model(trainer, timesteps=timesteps, denorm=False, method=method)
    ):
        if i_sample % 100 == 0:
            print(f"{i_sample}/{n_samples}")

        for step in range(timesteps + 1):
            hist_counts_step = _hist(y_true[:, step, ...], y_pred[:, step, ...])
            if hist_counts[step] is None:
                hist_counts[step] = hist_counts_step
            else:
                hist_counts[step] += hist_counts_step

        if i_sample >= n_samples:  # len(trainer.valid_datapipe):
            break

    return (hist_counts, bins)


def save_histogram(
    hist_counts: list[torch.Tensor], bins: torch.Tensor, output_path: str
) -> None:
    """Save histogram data to netCDF4 file.

    Parameters
    ----------
    hist_counts : list[torch.Tensor]
        List of histogram counts for each timestep.
    bins : torch.Tensor
        Bin edges for the histogram.
    output_path : str
        Path to output netCDF4 file.
    """
    # Convert torch tensors to numpy
    hist_counts_np = np.stack([h.cpu().numpy() for h in hist_counts], axis=0)
    bins_np = bins.cpu().numpy()

    # Compute bin centers from edges
    bin_centers = (bins_np[:-1] + bins_np[1:]) / 2

    # Create xarray Dataset
    ds = xr.Dataset(
        {
            "hist_counts": (["timestep", "bin"], hist_counts_np),
            "bin_edges": (["bin_edge"], bins_np),
        },
        coords={
            "timestep": np.arange(len(hist_counts)),
            "bin": bin_centers,
            "bin_edge": bins_np,
        },
        attrs={
            "description": "Histogram of squared errors for temporal interpolation",
            "created": datetime.now().isoformat(),
        },
    )

    # Save to netCDF4
    ds.to_netcdf(output_path, format="NETCDF4")
    print(f"Histogram saved to {output_path}")


@hydra.main(version_base=None, config_path="config")
def main(cfg: DictConfig):
    """Main entry point for validation and error analysis.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """
    cfg = OmegaConf.to_container(cfg)
    validation_cfg = cfg.pop("validation")
    output_path = validation_cfg.pop("output_path")
    (hist_counts, bins) = error_by_time(cfg, **validation_cfg)
    save_histogram(hist_counts, bins, output_path)


if __name__ == "__main__":
    main()
