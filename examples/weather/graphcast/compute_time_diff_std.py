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

import torch
import os
from physicsnemo.datapipes.climate import ERA5HDF5Datapipe
from physicsnemo.distributed import DistributedManager

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import numpy as np

from loss.utils import normalized_grid_cell_area


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # instantiate the training datapipe
    DataPipe = ERA5HDF5Datapipe  # [T,num_channel, 721, 1440], grid features
    datapipe = DataPipe(
        data_dir=to_absolute_path(os.path.join(cfg.dataset_path, "train")),
        stats_dir=to_absolute_path(os.path.join(cfg.dataset_path, "stats")),
        channels=[i for i in range(cfg.num_channels_climate)],
        latlon_resolution=cfg.latlon_res,
        num_samples_per_year=cfg.num_samples_per_year_train,
        num_steps=1,
        batch_size=1,
        num_workers=cfg.num_workers,
        device=dist.device,
        process_rank=dist.rank,
        world_size=dist.world_size,
        shuffle=False,
    )
    print(f"Loaded training datapipe of length {len(datapipe)}")

    area = (
        normalized_grid_cell_area(
            torch.linspace(-90, 90, steps=cfg.latlon_res[0]), unit="deg"
        )
        .unsqueeze(1)
        .to(dist.device)
    )

    mean, mean_sqr = 0, 0
    for i, data in enumerate(datapipe):
        invar = data[0]["invar"]
        outvar = data[0]["outvar"][0]
        diff = outvar - invar
        weighted_diff = area * diff
        weighted_diff_sqr = torch.square(weighted_diff)

        mean += torch.mean(weighted_diff, dim=(2, 3)) / len(datapipe)
        mean_sqr += torch.mean(weighted_diff_sqr, dim=(2, 3)) / len(datapipe)

        if i % 100 == 0 and i != 0 and dist.rank == 0:
            print("Number of iterations %d" % i)

    if dist.world_size > 1:
        torch.distributed.all_reduce(mean, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(mean_sqr, op=torch.distributed.ReduceOp.SUM)
        mean /= dist.world_size
        mean_sqr /= dist.world_size
    if dist.rank == 0:
        variance = mean_sqr - mean**2  # [1,num_channel, 1,1]
        std = torch.sqrt(variance)
        np.save("time_diff_std_new.npy", std.to(torch.device("cpu")).numpy())
        np.save("time_diff_mean_new.npy", mean.to(torch.device("cpu")).numpy())

    print("ended!")


if __name__ == "__main__":
    main()
