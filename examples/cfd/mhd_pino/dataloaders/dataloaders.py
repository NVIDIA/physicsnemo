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

import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader, Dataset
import os
import sys
import glob
import time
import h5py
from IPython.display import display

try:
    from .datasets import Dedalus2DDataset
except:
    from datasets import Dedalus2DDataset


class MHDDataloader(Dataset):
    "Dataloader for MHD Dataset with magnetic field"

    def __init__(
        self, dataset: Dedalus2DDataset, sub_x=1, sub_t=1, ind_x=None, ind_t=None
    ):
        self.dataset = dataset
        self.sub_x = sub_x
        self.sub_t = sub_t
        self.ind_x = ind_x
        self.ind_t = ind_t
        t, x, y = dataset.get_coords(0)
        self.x = x[:ind_x:sub_x]
        self.y = y[:ind_x:sub_x]
        self.t = t[:ind_t:sub_t]
        self.nx = len(self.x)
        self.ny = len(self.y)
        self.nt = len(self.t)
        self.num = num = len(self.dataset)
        self.x_slice = slice(0, self.ind_x, self.sub_x)
        self.t_slice = slice(0, self.ind_t, self.sub_t)

    def __len__(self):
        length = len(self.dataset)
        return length

    def __getitem__(self, index):
        "Gets input of dataloader, including data, t, x, and y"
        fields = self.dataset[index]

        # Data includes velocity and magnetic field
        velocity = fields["velocity"]
        magnetic_field = fields["magnetic field"]

        u = torch.from_numpy(
            velocity[
                : self.ind_t : self.sub_t,
                0,
                : self.ind_x : self.sub_x,
                : self.ind_x : self.sub_x,
            ]
        )
        v = torch.from_numpy(
            velocity[
                : self.ind_t : self.sub_t,
                1,
                : self.ind_x : self.sub_x,
                : self.ind_x : self.sub_x,
            ]
        )
        Bx = torch.from_numpy(
            magnetic_field[
                : self.ind_t : self.sub_t,
                0,
                : self.ind_x : self.sub_x,
                : self.ind_x : self.sub_x,
            ]
        )
        By = torch.from_numpy(
            magnetic_field[
                : self.ind_t : self.sub_t,
                1,
                : self.ind_x : self.sub_x,
                : self.ind_x : self.sub_x,
            ]
        )

        # shape is now (nt, nx, ny, nfields)
        data = torch.stack([u, v, Bx, By], dim=-1)
        data0 = data[0].reshape(1, self.nx, self.ny, -1).repeat(self.nt, 1, 1, 1)

        grid_t = (
            torch.from_numpy(self.t)
            .reshape(self.nt, 1, 1, 1)
            .repeat(1, self.nx, self.ny, 1)
        )
        grid_x = (
            torch.from_numpy(self.x)
            .reshape(1, self.nx, 1, 1)
            .repeat(self.nt, 1, self.ny, 1)
        )
        grid_y = (
            torch.from_numpy(self.y)
            .reshape(1, 1, self.ny, 1)
            .repeat(self.nt, self.nx, 1, 1)
        )

        inputs = torch.cat([grid_t, grid_x, grid_y, data0], dim=-1)
        outputs = data

        return inputs, outputs

    def create_dataloader(
        self,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        distributed=False,
    ):
        "Creates dataloader and sampler based on whether distributed training is on"
        if distributed:
            sampler = torch.utils.data.DistributedSampler(self)
            dataloader = DataLoader(
                self,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        else:
            sampler = None
            dataloader = DataLoader(
                self,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        return dataloader, sampler


class MHDDataloaderVecPot(MHDDataloader):
    "Dataloader for MHD Dataset with vector potential"

    def __init__(
        self, dataset: Dedalus2DDataset, sub_x=1, sub_t=1, ind_x=None, ind_t=None
    ):
        super().__init__(
            dataset=dataset, sub_x=sub_x, sub_t=sub_t, ind_x=ind_x, ind_t=ind_t
        )

    def __getitem__(self, index):
        "Gets input of dataloader, including data, t, x, and y"
        fields = self.dataset[index]

        # Data includes velocity and vector potential
        velocity = fields["velocity"]
        vector_potential = fields["vector potential"]

        u = torch.from_numpy(
            velocity[
                : self.ind_t : self.sub_t,
                0,
                : self.ind_x : self.sub_x,
                : self.ind_x : self.sub_x,
            ]
        )
        v = torch.from_numpy(
            velocity[
                : self.ind_t : self.sub_t,
                1,
                : self.ind_x : self.sub_x,
                : self.ind_x : self.sub_x,
            ]
        )
        A = torch.from_numpy(
            vector_potential[
                : self.ind_t : self.sub_t,
                : self.ind_x : self.sub_x,
                : self.ind_x : self.sub_x,
            ]
        )

        # shape is now (self.nt, self.nx, self.ny, nfields)
        data = torch.stack([u, v, A], dim=-1)
        data0 = data[0].reshape(1, self.nx, self.ny, -1).repeat(self.nt, 1, 1, 1)

        grid_t = (
            torch.from_numpy(self.t)
            .reshape(self.nt, 1, 1, 1)
            .repeat(1, self.nx, self.ny, 1)
        )
        grid_x = (
            torch.from_numpy(self.x)
            .reshape(1, self.nx, 1, 1)
            .repeat(self.nt, 1, self.ny, 1)
        )
        grid_y = (
            torch.from_numpy(self.y)
            .reshape(1, 1, self.ny, 1)
            .repeat(self.nt, self.nx, 1, 1)
        )

        inputs = torch.cat([grid_t, grid_x, grid_y, data0], dim=-1)
        outputs = data

        return inputs, outputs


if __name__ == "__main__":
    dataset = Dedalus2DDataset(
        data_path="../mhd_data/simulation_outputs_Re250",
        output_names="output-????",
        field_names=["magnetic field", "velocity", "vector potential"],
    )
    mhd_dataloader = MHDDataloader(dataset)
    mhd_vec_pot_dataloader = MHDDataloaderVecPot(dataset)

    data = mhd_dataloader[0]
    display(data)
