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

import matplotlib.pyplot as plt
import torch
from datetime import datetime
import pandas as pd
import hydra
from physicsnemo.distributed import DistributedManager
from omegaconf import DictConfig
from physicsnemo.models import Module

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

    invariant_array = dataset.get_invariants()
    invariant_tensor = torch.from_numpy(invariant_array).to(device).repeat(1, 1, 1, 1)

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

            if i == 0:
                state_pred = data["state"][0].to(device=device, dtype=torch.float32)
                state_pred = state_pred.unsqueeze(0)
                state_pred_edm = state_pred.clone()
                state_pred_noedm = state_pred.clone()

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
                dataset.denormalize_state(data["state"][0].cpu().numpy()),
                edm_prediction_group,
                noedm_prediction_group,
                target_group,
                output_state_channels,
                vardict_state,
                i,
            )

            # build diffusion condition and inference regression model, placing output into state_pred
            (condition, _, state_pred) = build_network_condition_and_target(
                background,
                [state_pred, state_pred],
                invariant_tensor,
                regression_net=regression_model,
                condition_list=cfg.model.diffusion_conditions,
                regression_condition_list=cfg.model.regression_conditions,
            )

            if state_pred is None:  # in case of no regression model
                state_pred = torch.zeros_like(state_pred_edm)

            state_pred_noedm = state_pred.clone()
            # inference diffusion model
            edm_corrected_outputs = diffusion_model_forward(
                diffusion_model,
                condition,
                state_pred.shape,
                sampler_args=dict(cfg.sampler.args),
            )


            state_pred[0, :] += edm_corrected_outputs[0].float()
            state_pred_edm = state_pred.clone()

            varidx_state = vardict_state[cfg.inference.plot_var_state]
            varidx_background = vardict_background[cfg.inference.plot_var_background]

            background_arr = background.cpu().numpy()[0]
            state_true_arr = data["state"][1].cpu().numpy()
            state_pred_arr = state_pred.cpu().numpy()[0]

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
        val_times.append(initial_time_pd + pd.Timedelta(seconds=i * hours_since_jan_01))

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
