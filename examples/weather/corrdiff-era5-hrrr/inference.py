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

"""Downscale ERA5 to HRRR over the full domain: generate an ensemble of samples
for one time with a trained model, save them, and compare them with HRRR."""

import os
import warnings

import hydra
from matplotlib import pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import xarray as xr

from physicsnemo.diffusion.noise_schedulers import RectifiedFlowNoiseScheduler
from physicsnemo.utils.logging import PythonLogger

from datasets import dataset_classes
from utils.nn import build_condition, get_dit, sample_flow
from utils.plots import comparison_plot


@hydra.main(version_base=None, config_path="config", config_name="corrdiff_inference")
def main(cfg: DictConfig):
    """Generate an ensemble of samples for one time and compare them with HRRR."""

    # Sampling starts at t = 1, where the flow used by the sampler stays finite
    warnings.filterwarnings(
        "ignore", message="^RectifiedFlowNoiseScheduler was constructed with t_max=1.0"
    )
    logger = PythonLogger("inference")
    device = torch.device("cuda")
    rundir = cfg.inference.rundir
    os.makedirs(rundir, exist_ok=True)

    # Dataset prep
    params = OmegaConf.to_container(cfg.dataset, resolve=True)
    dataset = dataset_classes[params.pop("name")](train=False, **params)
    state_channels = dataset.state_channels()
    background_channels = dataset.background_channels()

    time = np.datetime64(cfg.inference.time)
    idx = np.flatnonzero(dataset.times == time)
    if len(idx) == 0:
        raise ValueError(
            f"No data found for {time}; inference.time must be a valid time within dataset.time_range_valid."
        )
    item = dataset[int(idx[0])]

    # Build the model for the inference domain and load the trained weights
    net = get_dit(
        target_channels=len(state_channels),
        conditional_channels=len(background_channels),
        scalar_condition_channels=len(dataset.scalar_condition_channels()),
        img_resolution=dataset.image_shape(),
        **OmegaConf.to_container(cfg.model.hyperparameters, resolve=True),
    )
    net.load(cfg.inference.checkpoint)
    net = net.to(device).eval().requires_grad_(False)

    # Generate the samples, starting from pure noise at t = 1
    noise_scheduler = RectifiedFlowNoiseScheduler(t_max=1.0)
    sampler_args = OmegaConf.to_container(cfg.sampler.args, resolve=True)
    scalar_conditions = item.get("scalar_conditions")
    if scalar_conditions is not None:
        scalar_conditions = torch.from_numpy(scalar_conditions)[None].to(device)
    condition = build_condition(
        torch.from_numpy(item["background"])[None].to(device), scalar_conditions
    )
    # Keep only the channels to save or plot, in physical units
    output_variables = list(cfg.inference.output_variables) or state_channels
    variables = list(
        dict.fromkeys(output_variables + list(cfg.inference.plot_variables))
    )
    kept_channels = [state_channels.index(v) for v in variables]

    shape = (1, len(state_channels)) + tuple(dataset.image_shape())
    samples = []
    for i in range(cfg.inference.num_ensembles):
        torch.manual_seed(cfg.inference.seed + i)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = sample_flow(net, condition, shape, noise_scheduler, **sampler_args)
        x = dataset.denormalize_state(x[0].float().cpu().numpy())
        samples.append(x[kept_channels])
        logger.info(f"Generated sample {i + 1}/{cfg.inference.num_ensembles}")

    samples = np.stack(samples)
    truth = dataset.denormalize_state(item["state"])[kept_channels]
    background = dataset.denormalize_background(item["background"])

    # Ensemble-mean error and ensemble spread compared with HRRR
    for variable in output_variables:
        c = variables.index(variable)
        rmse = np.sqrt(np.mean((samples[:, c].mean(axis=0) - truth[c]) ** 2))
        spread = np.sqrt(np.mean(samples[:, c].var(axis=0)))
        logger.info(f"{variable}: ensemble-mean RMSE {rmse:.4g}, spread {spread:.4g}")

    # Save the samples and the HRRR truth
    data_vars = {}
    for variable in output_variables:
        c = variables.index(variable)
        data_vars[variable] = (("ensemble", "y", "x"), samples[:, c])
        data_vars[f"{variable}_hrrr"] = (("y", "x"), truth[c])
    output = xr.Dataset(
        data_vars,
        coords={
            "time": time,
            "lat": (("y", "x"), dataset.latitude()),
            "lon": (("y", "x"), dataset.longitude()),
        },
    )
    output_file = os.path.join(rundir, "samples.nc")
    output.to_netcdf(output_file)
    logger.info(f"Saved samples to {output_file}")

    # Plot the ERA5 input (if available), up to two samples, and HRRR
    for variable in cfg.inference.plot_variables:
        c = variables.index(variable)
        panels = [
            (f"Sample {i + 1}", samples[i, c]) for i in range(min(len(samples), 2))
        ]
        panels.append(("HRRR", truth[c]))
        if variable in background_channels:
            panels.insert(0, ("ERA5", background[background_channels.index(variable)]))
        fig = comparison_plot(panels, variable, title=f"{variable}, {time}")
        fig.savefig(os.path.join(rundir, f"{variable}.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
