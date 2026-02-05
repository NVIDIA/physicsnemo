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

from typing import Generator, Literal

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import xarray as xr

from train import input_output_from_batch_data, setup_trainer, Trainer


def setup_analysis(
    cfg: dict, checkpoint: str | None = None, shuffle: bool = False
) -> Trainer:
    """
    Setup trainer for validation analysis.

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
) -> Generator[tuple[torch.Tensor, torch.Tensor, int], None, None]:
    """
    Run inference on validation data.

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
    tuple[torch.Tensor, torch.Tensor, int]
        True values, predicted values, and timestep index for each batch.
    """
    for batch in trainer.valid_datapipe:
        y_true_step = []
        y_pred_step = []
        (invar, outvar_true) = input_output_from_batch_data(batch)
        invar = tuple(v.detach() for v in invar)
        outvar_true = outvar_true.detach()
        y_true_step.append(outvar_true)
        step = min(int(round(invar[1].item() * timesteps)), timesteps)
        if method == "fcinterp":
            y_pred_step.append(trainer.eval_step(invar))
        elif method == "linear":
            y_pred_step.append(linear_interp_batch_data(batch, step))

        y_true = torch.stack(y_true_step, dim=1)
        y_pred = torch.stack(y_pred_step, dim=1)
        if denorm:
            y_true = denormalize(trainer, y_true)
            y_pred = denormalize(trainer, y_pred)

        yield (y_true, y_pred, step)


def linear_interp_batch_data(
    batch: list[dict[str, torch.Tensor]], step: int
) -> torch.Tensor:
    """
    Perform linear interpolation on batch data.

    Parameters
    ----------
    batch : list[dict[str, torch.Tensor]]
        Batch data from datapipe (list containing a dictionary).
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
    """
    Denormalize predictions using dataset statistics.

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
    """
    Compute error statistics for each interpolation step. The error
    is computed as the squared difference of the prediction and truth
    and is area-weighted (i.e. multiplied by the cosine of the latitude).
    It is calculated on the values normalized to zero mean and unit variance,
    so that errors of all variables are comparable.

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
    trainer = setup_analysis(cfg=cfg, checkpoint=checkpoint)

    bins = torch.linspace(0, max_error, nbins + 1)

    def _hist(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        lat = torch.linspace(90, -90, y_true.shape[-2])[:-1].to(
            device=trainer.model.device
        )
        lat[0] = 0.5 * (lat[0] + lat[1])
        cos_lat = torch.cos(lat * (torch.pi / 180))[None, None, :, None]

        err = (y_true - y_pred) ** 2
        weights = torch.ones_like(err) * cos_lat
        return torch.histogram(
            err.ravel().cpu(), bins=bins, weight=weights.ravel().cpu()
        )[0]

    hist_counts = [
        torch.zeros(nbins, dtype=torch.float64) for _ in range(timesteps + 1)
    ]

    for i_sample, (y_true, y_pred, step) in enumerate(
        inference_model(trainer, timesteps=timesteps, denorm=False, method=method)
    ):
        if i_sample % 100 == 0:
            print(f"{i_sample}/{n_samples}")

        hist_counts_step = _hist(y_true[:, -1, ...], y_pred[:, -1, ...])
        hist_counts[step] += hist_counts_step

        if i_sample + 1 >= n_samples:
            break

    return (hist_counts, bins)


def save_histogram(
    hist_counts: list[torch.Tensor], bins: torch.Tensor, output_path: str
) -> None:
    """
    Save histogram data to netCDF4 file.

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
    """
    Run validation for interpolation error as a function of step.

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
