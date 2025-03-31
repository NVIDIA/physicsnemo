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


import sys

sys.path.append("./training")  # for loading pickled models which need training/utils
import os

import numpy as np
from concurrent.futures import ThreadPoolExecutor

import xarray as xr
import torch
import cftime
from netCDF4 import Dataset as NCDataset
from torch.distributed import gather
import argparse
import yaml
from configs.base import (
    val_station_path,
    isd_path,
    station_locations,
    path_to_hrrr,
    path_to_model_state,
)
from training.utils.diffusions.networks import get_preconditioned_architecture

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from sda.score import GaussianScore_from_denoiser, VPSDE_from_denoiser, VPSDE

from training.utils.diffusions.networks import get_preconditioned_architecture
from modulus.distributed import DistributedManager
import datetime
import netCDF4 as nc


# Initialize distributed manager
DistributedManager.initialize()
dist = DistributedManager()
device = dist.device


class NetCDFWriter:
    """NetCDF Writer"""

    def __init__(self, f, lat, lon, output_channels):
        self._f = f

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
        v[:] = lat
        v.standard_name = "latitude"
        v.units = "degrees_north"

        v = f.createVariable("lon", "f", dimensions=("y", "x"))
        v[:] = lon
        v.standard_name = "longitude"
        v.units = "degrees_east"

        # create time dimension
        v = f.createVariable("time", "i8", ("time"))
        v.calendar = "standard"
        v.units = "hours since 1990-01-01 00:00:00"

        self.truth_group = f.createGroup("truth")
        self.prediction_group = f.createGroup("prediction")
        self.validation_group = f.createGroup("validation")

        for name in output_channels:
            self.truth_group.createVariable(name, "f", dimensions=("time", "y", "x"))
            self.prediction_group.createVariable(
                name, "f", dimensions=("ensemble", "time", "y", "x")
            )
            self.validation_group.createVariable(
                name, "f", dimensions=("time", "y", "x")
            )

    def write_validation(self, channel_name, time_index, val):
        """Write input data to NetCDF file."""
        self.validation_group[channel_name][time_index] = val

    def write_truth(self, channel_name, time_index, val):
        """Write ground truth data to NetCDF file."""
        self.truth_group[channel_name][time_index] = val

    def write_prediction(self, channel_name, time_index, ensemble_index, val):
        """Write prediction data to NetCDF file."""
        self.prediction_group[channel_name][ensemble_index, time_index] = val

    def write_time(self, time_index, time):
        """Write time information to NetCDF file."""
        time_v = self._f["time"]
        self._f["time"][time_index] = cftime.date2num(
            time, time_v.units, time_v.calendar
        )


# Notice for now time are all 0 because of dataset
def main():
    print(dist.rank, dist.world_size)
    output_name = "full_inference_output.nc"
    n_steps = 64
    output_channel_map = ["10u", "10v", "tp"]
    num_ensembles = 16
    total_num_samples = 12 * 24 * 30
    actual_samples = list(np.arange(total_num_samples))
    num_samples = len(actual_samples)
    print(f"RUNNING SAMPLE {num_samples}")

    sample_indices = list(np.arange(num_samples))
    num_batches = ((len(sample_indices) - 1) // dist.world_size + 1) * dist.world_size
    all_batches = torch.as_tensor(sample_indices).tensor_split(num_batches)
    rank_batches = all_batches[dist.rank :: dist.world_size]

    hr = xr.open_zarr(path_to_hrrr, mask_and_scale=False)

    # load pretrained model
    model = get_preconditioned_architecture(
        name="ddpmpp-cwb-v0",
        resolution=128,
        target_channels=3,
        conditional_channels=0,
        label_dim=0,
        spatial_embedding=None,
    )

    modelpath = path_to_model_state
    load_state_dict = modelpath.endswith(".pth")

    state = torch.load(modelpath)
    if load_state_dict:
        model.load_state_dict(state, strict=True)
    else:
        model = state["net"]
    model.eval().to(device=device).to(memory_format=torch.channels_last)

    u10_mean, u10_std = -0.262, 2.372
    v10_mean, v10_std = 0.865, 4.115
    logtp_mean, logtp_std = -8.117, 2.489
    means, stds = np.array([u10_mean, v10_mean, logtp_mean]), np.array(
        [u10_std, v10_std, logtp_std]
    )

    def denorm(x):
        x *= stds[:, np.newaxis, np.newaxis]
        x += means[:, np.newaxis, np.newaxis]
        x[..., 2, :, :] = np.exp(x[..., 2, :, :] - 1e-4)
        return x

    def save_images(
        writer,
        channel_infos,
        image_out_rank,
        image_tar_rank,
        image_valid_rank,
        day_time,
    ):
        print(f"starting writing index: {day_time}")
        image_out_rank = denorm(image_out_rank.cpu())
        image_tar_rank = denorm(image_tar_rank.cpu())
        image_valid_rank = denorm(image_valid_rank.cpu())
        for rank in range(dist.world_size):
            time_index = day_time + rank
            image_out = image_out_rank[rank]
            image_tar = image_tar_rank[rank]
            image_valid = image_valid_rank[rank]

            for channel_idx in range(image_out.shape[1]):
                channel_name = channel_infos[channel_idx]
                truth = image_tar[0, channel_idx]
                writer.write_truth(channel_name, time_index, truth)
                writer.write_validation(
                    channel_name, time_index, image_valid[0, channel_idx]
                )
                for ens_idx in range(image_out.shape[0]):
                    writer.write_prediction(
                        channel_name,
                        time_index,
                        ens_idx,
                        image_out[ens_idx, channel_idx],
                    )

    ds_regrid = xr.open_dataset(isd_path)
    stat_loc = xr.open_dataarray(station_locations)

    valid = np.load(val_station_path)
    num_leave = 10
    valid = valid[:num_leave]

    bool_array = stat_loc.values.astype(bool)
    for indices in valid:
        bool_array[indices[0], indices[1]] = False
    tune = np.zeros_like(bool_array)

    for indices in valid:
        tune[indices[0], indices[1]] = True

    def process_day(daytime):
        daytime = int(daytime)
        target_time = slice(daytime, daytime + 1)
        u10 = ds_regrid.isel(DATE=target_time).u10.values
        v10 = ds_regrid.isel(DATE=target_time).v10.values
        tp = np.log(ds_regrid.isel(DATE=target_time).tp.values + 0.0001)

        obs = np.array([u10, v10, tp])
        obs -= means[:, np.newaxis, np.newaxis, np.newaxis]
        obs /= stds[:, np.newaxis, np.newaxis, np.newaxis]
        obs = obs.transpose(3, 0, 2, 1)
        obs = torch.tensor(obs)
        obs = obs.tile(num_ensembles, 1, 1, 1)

        u10 = hr.isel(time=target_time).sel(channel="10u").HRRR.values
        v10 = hr.isel(time=target_time).sel(channel="10v").HRRR.values
        tp = np.log(hr.isel(time=target_time).sel(channel="tp").HRRR.values + 0.0001)

        hrrr = np.array([u10, v10, tp])
        hrrr -= means[:, np.newaxis, np.newaxis, np.newaxis]
        hrrr /= stds[:, np.newaxis, np.newaxis, np.newaxis]
        hrrr = hrrr.transpose(1, 0, 2, 3)
        hrrr = torch.tensor(hrrr)

        inf_obs = obs.where(
            torch.tensor(np.tile(bool_array, (len(obs), 3, 1, 1))), np.nan
        )
        val_obs = obs.where(torch.tensor(np.tile(tune, (1, 3, 1, 1))), np.nan)

        mask = ~np.isnan(inf_obs).bool()

        def A(x):
            return x[mask]

        y_star = A(inf_obs).to(device=device)
        sde = VPSDE(  # from this we use sample()
            GaussianScore_from_denoiser(  # from this, sample() uses VPSDE.eps() which is GaussianScore_from_denoiser.forward()
                y_star,
                A=A,
                std=0.1,
                gamma=0.001,
                sde=VPSDE_from_denoiser(
                    model, shape=()
                ),  # which calls VPSDE_from_denoiser.eps_from_denoiser() which calls the network
            ).to(
                device=device
            ),
            shape=inf_obs.shape,
        ).to(device=device)

        x = sde.sample(steps=n_steps, corrections=2, tau=0.3, makefigs=False)
        val_obs = val_obs.to(device=device)
        hrrr = hrrr.to(device=device)

        if dist.world_size > 1:
            if dist.rank == 0:
                gathered_tensors = [
                    [
                        torch.zeros_like(x, dtype=x.dtype, device=x.device)
                        for _ in range(dist.world_size)
                    ],
                    [
                        torch.zeros_like(
                            val_obs, dtype=val_obs.dtype, device=val_obs.device
                        )
                        for _ in range(dist.world_size)
                    ],
                    [
                        torch.zeros_like(hrrr, dtype=hrrr.dtype, device=hrrr.device)
                        for _ in range(dist.world_size)
                    ],
                ]
            else:
                gathered_tensors = None

            torch.distributed.barrier()
            gather(
                x,
                gather_list=gathered_tensors[0] if dist.rank == 0 else None,
                dst=0,
            )
            gather(
                val_obs,
                gather_list=gathered_tensors[1] if dist.rank == 0 else None,
                dst=0,
            )
            gather(
                hrrr,
                gather_list=gathered_tensors[2] if dist.rank == 0 else None,
                dst=0,
            )

            if dist.rank == 0:
                x = torch.stack(gathered_tensors[0])
                val_obs = torch.stack(gathered_tensors[1])
                hrrr = torch.stack(gathered_tensors[2])
                return x, val_obs, hrrr
            else:
                return None, None, None

        else:
            return x.unsqueeze(0), val_obs.unsqueeze(0), hrrr.unsqueeze(0)

    # Synchronize
    if dist.world_size > 1:
        torch.distributed.barrier()

    if dist.rank == 0:
        f = NCDataset(output_name, "w")

        dummy_lon, dummy_lat = np.meshgrid(
            np.arange(bool_array.shape[1]), np.arange(bool_array.shape[0])
        )
        writer = NetCDFWriter(
            f,
            lat=dummy_lat.astype(np.float32),
            lon=dummy_lon.astype(np.float32),
            output_channels=output_channel_map,
        )

        writer_executor = ThreadPoolExecutor(max_workers=4)
        writer_threads = []

    for time_index, sample_index in enumerate(rank_batches):
        daytime = actual_samples[sample_index]
        x, val_obs, hrrr = process_day(daytime)

        if dist.rank == 0:
            print(f"Finishing running index: {time_index}")
            writer_threads.append(
                writer_executor.submit(
                    save_images,
                    writer,
                    output_channel_map,
                    x,
                    hrrr,
                    val_obs,
                    sample_index,
                )
            )

    if dist.rank == 0:
        # make sure all the workers are done writing
        for thread in list(writer_threads):
            thread.result()
            writer_threads.remove(thread)
        writer_executor.shutdown()
        f.close()
        print("rank 0 finishing")


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config-name", type=str)
    # args = parser.parse_args()

    main()
