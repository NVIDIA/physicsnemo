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

"""Single-step diagnostic inference.

Evaluates a trained regression/diffusion pair against ``n_steps`` independent
samples from the dataset's own ``background`` (which, per the data-source
contract, already carries any past-state input the model needs -- see
``datasets/dataset.py``). This script does not perform autoregressive
rollout: each step is scored against its own dataset sample rather than fed
forward as input to the next step. For autoregressive rollout or more
elaborate inference workflows, bring your checkpoints to
[Earth2Studio](https://github.com/NVIDIA/earth2studio); see the README's
"Running inference" section.
"""

import matplotlib.pyplot as plt
import torch
from datetime import datetime
import pandas as pd
import hydra
from physicsnemo.distributed import DistributedManager
from omegaconf import DictConfig
from physicsnemo.core import Module

from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler

from datasets import dataset_classes
from utils.io import (
    init_inference_results_zarr,
    write_inference_results_zarr,
    save_inference_results_netcdf,
)
from utils.nn import build_network_condition_and_target, diffusion_model_forward
from utils.plots import inference_plot


@hydra.main(version_base=None, config_path="config", config_name="stormcast_inference")
def main(cfg: DictConfig):
    # Initialize
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    initial_time = datetime.fromisoformat(cfg.inference.initial_time)
    n_steps = cfg.inference.n_steps

    # Dataset prep
    dataset_cls = dataset_classes[cfg.dataset.name]
    dataset = dataset_cls(cfg.dataset, train=False)

    background_channels = dataset.background_channels()
    state_channels = dataset.state_channels()
    lead_time_steps = dataset.lead_time_steps

    invariant_array = dataset.get_invariants()
    invariant_tensor = (
        None
        if invariant_array is None
        else torch.from_numpy(invariant_array).to(device).repeat(1, 1, 1, 1)
    )

    if len(cfg.inference.output_state_channels) == 0:
        output_state_channels = state_channels.copy()
    else:
        output_state_channels = cfg.inference.output_state_channels

    vardict_state: dict[str, int] = {
        state_channel: i for i, state_channel in enumerate(state_channels)
    }

    vardict_background = {
        background_channel: i
        for i, background_channel in enumerate(background_channels)
    }

    hours_since_jan_01 = int(
        (initial_time - datetime(initial_time.year, 1, 1, 0, 0)).total_seconds() / 3600
    )

    # Load pretrained models
    if "regression" in cfg.model.diffusion_conditions:
        net = Module.from_checkpoint(cfg.inference.regression_checkpoint)
        regression_model = net.to(device)
    else:
        regression_model = None
    net = Module.from_checkpoint(cfg.inference.diffusion_checkpoint)
    diffusion_model = net.to(device)

    sa = dict(cfg.sampler.args)
    sampling_scheduler = EDMNoiseScheduler(
        sigma_min=sa.get("sigma_min", 0.002),
        sigma_max=sa.get("sigma_max", 80.0),
        rho=sa.get("rho", 7.0),
    )

    # initialize zarr
    (
        group,
        target_group,
        edm_prediction_group,
        noedm_prediction_group,
    ) = init_inference_results_zarr(
        dataset, cfg.inference.rundir, output_state_channels, n_steps
    )

    with torch.no_grad():
        for i in range(n_steps):
            data = dataset[i + hours_since_jan_01]

            background = data["background"].to(device=device, dtype=torch.float32)
            background = background.unsqueeze(0)
            target = data["state"].to(device=device, dtype=torch.float32)
            target = target.unsqueeze(0)

            lead_time_label = data.get("lead_time_label")
            if lead_time_label is not None:
                lead_time_label = lead_time_label.to(device=device, dtype=torch.int64)
                lead_time_label = lead_time_label.unsqueeze(0)

            # build diffusion condition and inference regression model
            (condition, _, reg_out) = build_network_condition_and_target(
                background,
                target,
                invariant_tensor,
                lead_time_label=lead_time_label,
                regression_net=regression_model,
                condition_list=cfg.model.diffusion_conditions,
                regression_condition_list=cfg.model.regression_conditions,
            )

            # regression-only estimate (the diffusion model's condition), or
            # zero if no regression model is used
            state_pred_noedm = (
                reg_out.clone() if reg_out is not None else torch.zeros_like(target)
            )

            # inference diffusion model: it predicts a residual around
            # `state_pred_noedm` (see build_network_condition_and_target)
            edm_corrected_outputs = diffusion_model_forward(
                diffusion_model,
                condition,
                target.shape,
                scheduler=sampling_scheduler,
                sampler_args=sa,
                lead_time_label=lead_time_label,
            )
            state_pred_edm = state_pred_noedm + edm_corrected_outputs.float()

            assert (
                state_pred_edm.shape == (1, len(state_channels)) + dataset.image_shape()
            )
            assert (
                state_pred_noedm.shape
                == (1, len(state_channels)) + dataset.image_shape()
            )
            # write zarr
            write_inference_results_zarr(
                dataset.denormalize_state(state_pred_edm.cpu().numpy())[0],
                dataset.denormalize_state(state_pred_noedm.cpu().numpy())[0],
                dataset.denormalize_state(target.cpu().numpy())[0],
                edm_prediction_group,
                noedm_prediction_group,
                target_group,
                output_state_channels,
                vardict_state,
                i,
            )

            varidx_state = vardict_state[cfg.inference.plot_var_state]
            varidx_background = vardict_background[cfg.inference.plot_var_background]

            background_arr = background.cpu().numpy()[0]
            state_true_arr = target.cpu().numpy()[0]
            state_pred_arr = state_pred_edm.cpu().numpy()[0]

            background_arr = dataset.denormalize_background(background_arr)
            state_true_arr = dataset.denormalize_state(state_true_arr)
            state_pred_arr = dataset.denormalize_state(state_pred_arr)

            fig = inference_plot(
                background_arr[varidx_background],
                state_pred_arr[varidx_state],
                state_true_arr[varidx_state],
                cfg.inference.plot_var_background,
                cfg.inference.plot_var_state,
                initial_time,
                i,
            )
            fig.savefig(f"{cfg.inference.rundir}/out_{i}.png")
            plt.close(fig)

    initial_time_pd = pd.to_datetime(initial_time)
    val_times = []
    for i in range(n_steps):
        val_times.append(initial_time_pd + pd.Timedelta(hours=i))

    save_inference_results_netcdf(
        ds_out_path=cfg.inference.rundir,
        zarr_group=group,
        vertical_vars=cfg.inference.save_vertical_vars,
        level_names=cfg.inference.save_vertical_levels,
        horizontal_vars=cfg.inference.save_horizontal_vars,
        val_times=val_times,
    )


if __name__ == "__main__":
    main()
