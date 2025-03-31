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

import datetime

import cftime
import nvtx
import torch
import tqdm

from physicsnemo.utils.generative import StackedRandomGenerator, time_range

############################################################################
#                     CorrDiff Generation Utilities                        #
############################################################################


def regression_step(
    net: torch.nn.Module,
    img_lr: torch.Tensor,
    latents_shape: torch.Size,
    lead_time_label: torch.Tensor = None,
) -> torch.Tensor:
    """
    Given a low-res input, performs a regression step to produce ensemble mean.
    This function performs the regression on a single instance and then replicates
    the results across the batch dimension.

    Args:
        net (torch.nn.Module): U-Net model for regression.
        img_lr (torch.Tensor): Low-resolution input.
        latents_shape (torch.Size): Shape of the latent representation. Typically
        (batch_size, out_channels, image_shape_x, image_shape_y).


    Returns:
        torch.Tensor: Predicted output at the next time step.
    """
    # Create a tensor of zeros with the given shape and move it to the appropriate device
    x_hat = torch.zeros(latents_shape, dtype=torch.float64, device=net.device)
    t_hat = torch.tensor(1.0, dtype=torch.float64, device=net.device)

    # Perform regression on a single batch element
    with torch.inference_mode():
        if lead_time_label is not None:
            x = net(x_hat[0:1], img_lr, t_hat, lead_time_label=lead_time_label)
        else:
            x = net(x_hat[0:1], img_lr, t_hat)

    # If the batch size is greater than 1, repeat the prediction
    if x_hat.shape[0] > 1:
        x = x.repeat([d if i == 0 else 1 for i, d in enumerate(x_hat.shape)])

    return x


def diffusion_step(  # TODO generalize the module and add defaults
    net: torch.nn.Module,
    sampler_fn: callable,
    seed_batch_size: int,
    img_shape: tuple,
    img_out_channels: int,
    rank_batches: list,
    img_lr: torch.Tensor,
    rank: int,
    device: torch.device,
    hr_mean: torch.Tensor = None,
    lead_time_label: torch.Tensor = None,
) -> torch.Tensor:

    """
    Generate images using diffusion techniques as described in the relevant paper.

    Args:
        net (torch.nn.Module): The diffusion model network.
        sampler_fn (callable): Function used to sample images from the diffusion model.
        seed_batch_size (int): Number of seeds per batch.
        img_shape (tuple): Shape of the images, (height, width).
        img_out_channels (int): Number of output channels for the image.
        rank_batches (list): List of batches of seeds to process.
        img_lr (torch.Tensor): Low-resolution input image.
        rank (int): Rank of the current process for distributed processing.
        device (torch.device): Device to perform computations.
        mean_hr (torch.Tensor, optional): High-resolution mean tensor, to be used as an additional input. By default None.

    Returns:
        torch.Tensor: Generated images concatenated across batches.
    """

    img_lr = img_lr.to(memory_format=torch.channels_last)

    # Handling of the high-res mean
    additional_args = {}
    if hr_mean is not None:
        additional_args["mean_hr"] = hr_mean
    if lead_time_label is not None:
        additional_args["lead_time_label"] = lead_time_label
    additional_args["img_shape"] = img_shape

    # Loop over batches
    all_images = []
    for batch_seeds in tqdm.tqdm(rank_batches, unit="batch", disable=(rank != 0)):
        with nvtx.annotate(f"generate {len(all_images)}", color="rapids"):
            batch_size = len(batch_seeds)
            if batch_size == 0:
                continue

            # Initialize random generator, and generate latents
            rnd = StackedRandomGenerator(device, batch_seeds)
            latents = rnd.randn(
                [
                    seed_batch_size,
                    img_out_channels,
                    img_shape[0],
                    img_shape[1],
                ],
                device=device,
            ).to(memory_format=torch.channels_last)

            with torch.inference_mode():
                images = sampler_fn(
                    net, latents, img_lr, randn_like=rnd.randn_like, **additional_args
                )
            all_images.append(images)
    return torch.cat(all_images)


############################################################################
#                         CorrDiff writer utilities                        #
############################################################################


class NetCDFWriter:
    """NetCDF Writer"""

    def __init__(
        self, f, lat, lon, input_channels, output_channels, has_lead_time=False
    ):
        self._f = f
        self.has_lead_time = has_lead_time
        # create unlimited dimensions
        f.createDimension("time")
        f.createDimension("ensemble")

        if lat.shape != lon.shape:
            raise ValueError("lat and lon must have the same shape")
        ny, nx = lat.shape

        # create lat/lon grid
        f.createDimension("x", nx)
        f.createDimension("y", ny)

        v = f.createVariable("lat", "f", dimensions=("y", "x"))
        # NOTE rethink this for datasets whose samples don't have constant lat-lon.
        v[:] = lat
        v.standard_name = "latitude"
        v.units = "degrees_north"

        v = f.createVariable("lon", "f", dimensions=("y", "x"))
        v[:] = lon
        v.standard_name = "longitude"
        v.units = "degrees_east"

        # create time dimension
        if has_lead_time:
            v = f.createVariable("time", "str", ("time"))
        else:
            v = f.createVariable("time", "i8", ("time"))
            v.calendar = "standard"
            v.units = "hours since 1990-01-01 00:00:00"

        self.truth_group = f.createGroup("truth")
        self.prediction_group = f.createGroup("prediction")
        self.input_group = f.createGroup("input")

        for variable in output_channels:
            name = variable.name + variable.level
            self.truth_group.createVariable(name, "f", dimensions=("time", "y", "x"))
            self.prediction_group.createVariable(
                name, "f", dimensions=("ensemble", "time", "y", "x")
            )

        # setup input data in netCDF

        for variable in input_channels:
            name = variable.name + variable.level
            self.input_group.createVariable(name, "f", dimensions=("time", "y", "x"))

    def write_input(self, channel_name, time_index, val):
        """Write input data to NetCDF file."""
        self.input_group[channel_name][time_index] = val

    def write_truth(self, channel_name, time_index, val):
        """Write ground truth data to NetCDF file."""
        self.truth_group[channel_name][time_index] = val

    def write_prediction(self, channel_name, time_index, ensemble_index, val):
        """Write prediction data to NetCDF file."""
        self.prediction_group[channel_name][ensemble_index, time_index] = val

    def write_time(self, time_index, time):
        """Write time information to NetCDF file."""
        if self.has_lead_time:
            self._f["time"][time_index] = time
        else:
            time_v = self._f["time"]
            self._f["time"][time_index] = cftime.date2num(
                time, time_v.units, time_v.calendar
            )


############################################################################
#                          CorrDiff time utilities                         #
############################################################################


def get_time_from_range(times_range, time_format="%Y-%m-%dT%H:%M:%S"):
    """Generates a list of times within a given range.

    Args:
        times_range: A list containing start time, end time, and optional interval (hours).
        time_format: The format of the input times (default: "%Y-%m-%dT%H:%M:%S").

    Returns:
        A list of times within the specified range.
    """

    start_time = datetime.datetime.strptime(times_range[0], time_format)
    end_time = datetime.datetime.strptime(times_range[1], time_format)
    interval = (
        datetime.timedelta(hours=times_range[2])
        if len(times_range) > 2
        else datetime.timedelta(hours=1)
    )

    times = [
        t.strftime(time_format)
        for t in time_range(start_time, end_time, interval, inclusive=True)
    ]
    return times
