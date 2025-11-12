# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

import datetime

from physicsnemo.utils.diffusion import convert_datetime_to_cftime

from datasets.dataset import init_dataset_from_config
from datasets.base import DownscalingDataset

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable
from tqdm.auto import tqdm
import nvtx
from physicsnemo.utils.corrdiff import regression_step
from physicsnemo.utils.generative import StackedRandomGenerator


def get_dataset_and_sampler(dataset_cfg, times, has_lead_time=False):
    """
    Get a dataset and sampler for generation.
    """
    (dataset, _) = init_dataset_from_config(dataset_cfg, batch_size=1)
    if has_lead_time:
        plot_times = times
    else:
        plot_times = [
            convert_datetime_to_cftime(
                datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")
            )
            for time in times
        ]
    all_times = dataset.time()
    time_indices = [all_times.index(t) for t in plot_times]
    sampler = time_indices

    return dataset, sampler


def save_images(
    writer,
    dataset: DownscalingDataset,
    times,
    image_out,
    image_tar,
    image_lr,
    time_index,
    dataset_index,
    solar
):
    """
    Saves inferencing result along with the baseline

    Parameters
    ----------

    writer (NetCDFWriter): Where the data is being written
    in_channels (List): List of the input channels being used
    input_channel_info (Dict): Description of the input channels
    out_channels (List): List of the output channels being used
    output_channel_info (Dict): Description of the output channels
    input_norm (Tuple): Normalization data for input
    target_norm (Tuple): Normalization data for the target
    image_out (torch.Tensor): Generated output data
    image_tar (torch.Tensor): Ground truth data
    image_lr (torch.Tensor): Low resolution input data
    time_index (int): Epoch number
    dataset_index (int): index where times are located
    """
    # weather sub-plot
    image_lr2 = image_lr[0].unsqueeze(0)
    image_lr2 = image_lr2.cpu().numpy()
    if solar: #In solar downscaling, we input the ear5 variables at two time (T, T+1)
        len_era5 = len(dataset.era5_input)
        era5_sample = image_lr2[:,0:2*len_era5,:,:]
    
        era5_sample_T = era5_sample[:,0::2, :, :]
        era5_sample_T_1 = era5_sample[:,1::2, :, :]

        image_lr2[:,0:len_era5,:,:] = dataset.denormalize_input(era5_sample_T)
        image_lr2[:,len_era5:2*len_era5,:,:] = dataset.denormalize_input(era5_sample_T_1)
    else:
        image_lr2 = dataset.denormalize_input(image_lr2)

    image_tar2 = image_tar[0].unsqueeze(0)
    image_tar2 = image_tar2.cpu().numpy()
    image_tar2 = dataset.denormalize_output(image_tar2)

    # some runtime assertions
    if image_tar2.ndim != 4:
        raise ValueError("image_tar2 must be 4-dimensional")

    for idx in range(image_out.shape[0]):
        image_out2 = image_out[idx].unsqueeze(0)
        if image_out2.ndim != 4:
            raise ValueError("image_out2 must be 4-dimensional")

        # Denormalize the input and outputs
        image_out2 = image_out2.cpu().numpy()
        image_out2 = dataset.denormalize_output(image_out2)

        time = times[dataset_index]
        writer.write_time(time_index, time)
        for channel_idx in range(image_out2.shape[1]):
            info = dataset.output_channels()[channel_idx]
            channel_name = info.name + info.level
            truth = image_tar2[0, channel_idx]

            writer.write_truth(channel_name, time_index, truth)
            writer.write_prediction(
                channel_name, time_index, idx, image_out2[0, channel_idx]
            )

    input_channel_info = dataset.input_channels()
    for channel_idx in range(len(input_channel_info)):
        info = input_channel_info[channel_idx]
        channel_name = info.name + info.level
        writer.write_input(channel_name, time_index, image_lr2[0, channel_idx])
        if channel_idx == image_lr2.shape[1] - 1:
            break



class MultiDiffusion(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    @torch.no_grad()
    def __call__(
        self,
        net: torch.nn.Module,
        img_lr: Tensor,
        regression_output: Tensor,
        class_labels: Optional[Tensor] = None,
        randn_like: Callable[[Tensor], Tensor] = torch.randn_like,
        windows:  Optional[Tensor] = None,
        lead_time_label: Optional[Tensor] = None,
        num_steps: int = 18,
        sigma_min: float = 0.002,
        sigma_max: float = 800,
        rho: float = 7,
        S_churn: float = 0,
        S_min: float = 0,
        S_max: float = float("inf"),
        S_noise: float = 1,
    ) -> Tensor:
        """
        
        Args:
            net (torch.nn.Module): the diffusion model
            regression_output (Tensor): output from regression model (B, C_cond, H, W)。
            randn_like (Callable): gaussian sampler
            windows : All windows
            stride (int): the stride between windows

        Returns:
            Tensor: (B, C_out, H, W)
        """
        
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)
        batch_size, _, height, width = regression_output.shape
        x_lr = torch.cat((regression_output,img_lr), dim=1)
        latents = randn_like(regression_output)
        
        step_indices = torch.arange(num_steps, device=self.device)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        views = windows
        
        value = torch.zeros_like(latents)
        count = torch.zeros_like(latents)

        optional_args = {}
        if lead_time_label is not None:
            optional_args["lead_time_label"] = lead_time_label

        x_next = latents * t_steps[0]
        
        for i, (t_cur, t_next) in enumerate(tqdm(zip(t_steps[:-1], t_steps[1:]), total=num_steps)):
            x_cur = x_next.clone()
            
            value.zero_()
            count.zero_()
            #print(f"x_cur:{x_cur.shape}")
            for view in views:
                h_start, h_end, w_start, w_end = int(view[0]),int(view[1]),int(view[2]),int(view[3])
                x_cur_view = x_cur[:, :, h_start:h_end, w_start:w_end]
                x_lr_view = x_lr[:, :, h_start:h_end, w_start:w_end]
                
                #(Churning)
                gamma = S_churn / num_steps if S_min <= t_cur <= S_max else 0
                t_hat = net.round_sigma(t_cur + gamma * t_cur)
                x_hat_view = x_cur_view + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur_view)
                #(Euler step part 1)
                denoised_view = net(
                    x_hat_view,
                    x_lr_view,
                    t_hat,
                    class_labels,
                    **optional_args,
                )
                
                d_cur_view = (x_hat_view - denoised_view) / t_hat
                
                x_next_view_first_order = x_hat_view + (t_next - t_hat) * d_cur_view

                if i < num_steps - 1:
                    denoised_prime_view = net(
                        x_next_view_first_order,
                        x_lr_view,
                        t_next,
                        class_labels,
                        **optional_args,
                    )
                    d_prime_view = (x_next_view_first_order - denoised_prime_view) / t_next
                    x_next_view = x_hat_view + (t_next - t_hat) * (0.5 * d_cur_view + 0.5 * d_prime_view)
                else:
                    x_next_view = x_next_view_first_order
                
                value[:, :, h_start:h_end, w_start:w_end] += x_next_view
                count[:, :, h_start:h_end, w_start:w_end] += 1
                #print(f"One window finished.")

            x_next = torch.where(count > 0, value / count, value)
        
        return x_next



def generate_solar(
    image_lr_full: torch.Tensor,
    image_tar_full: torch.Tensor, 
    windows: list,
    net_reg: torch.nn.Module,
    logger0, 
    img_out_channels: int = None,
    net_res: torch.nn.Module = None,
    seeds: list = None,
    lead_time_label: torch.Tensor = None,
):
    """
    A full generation function for solar downscaling
    Regression and selectabel Resiual in Multi-Diffusion way

    This function first run regression model on multi-windows to get the high-resolution output.
    if net_res is provided, Multi-Diffusion is performed for fine details
    Args:
        image_lr_full (torch.Tensor): Low-resolution input tensor
        image_tar_full (torch.Tensor): The target high-resolution tensor
        windows (list): The pre-defined windows
        net_reg (torch.nn.Module): Regression network
        logger0
        net_res (torch.nn.Module, optional): The resiual Diffusion network

    Returns:
        torch.Tensor or None: (B, C, H, W)。
    """
    with nvtx.annotate("generate_solar", color="blue"):
        device = image_lr_full.device
        
        image_reg_full = torch.zeros_like(image_tar_full).to(device=device).to(torch.float32)
        counts = torch.zeros_like(image_tar_full)
        logger0.info(f"Input LR shape: {image_lr_full.shape}")
        logger0.info(f"Target HR shape for stitching: {image_reg_full.shape}")

        with nvtx.annotate("solar_regression_stitching", color="green"):
            for window in windows:
                y_start, y_end, x_start, x_end = map(int, [window[0].item(), window[1].item(), window[2].item(), window[3].item()])
                
                image_lr_patch = image_lr_full[:, :, y_start:y_end, x_start:x_end]
                
                _, _, h, w = image_lr_patch.shape
                latents_shape = (image_lr_patch.shape[0], img_out_channels, h, w)

                with nvtx.annotate("regression_model_step", color="yellow"):
                    image_reg_patch = regression_step(
                        net=net_reg,
                        img_lr=image_lr_patch.to(memory_format=torch.channels_last),
                        latents_shape=latents_shape,
                        lead_time_label=lead_time_label,
                    )
                
                image_reg_full[:, :, y_start:y_end, x_start:x_end] += image_reg_patch
                counts[:, :, y_start:y_end, x_start:x_end] += 1
        
        counts = torch.where(counts == 0, torch.ones_like(counts), counts)
        image_reg_full = image_reg_full / counts
        logger0.info(f"Stitched regression image shape: {image_reg_full.shape}")
        
        final_output = image_reg_full

        if net_res:
            mdiff = MultiDiffusion(image_reg_full.device)
            with nvtx.annotate("solar_multidiffusion", color="purple"):
                logger0.info("Performing Multi-Diffusion step...")
                regression_output = image_reg_full
                ensemble_outputs = []
                
                for i in seeds:
                    rnd = StackedRandomGenerator(regression_output.device, [i])

                    image_res_out_full = mdiff(
                        net=net_res,
                        img_lr=image_lr_full,
                        regression_output=regression_output,
                        windows=windows,
                        randn_like=rnd.randn_like,
                    )
                    ensemble_outputs.append(regression_output + image_res_out_full)
                
                final_output = torch.cat(ensemble_outputs, dim=0)
                logger0.info(f"Final ensemble output shape: {final_output.shape}")
        else:
            logger0.info("Skipping diffusion step. Output is from the regression model.")

        
        return final_output  