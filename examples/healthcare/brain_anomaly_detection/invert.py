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

import physicsnemo
from physicsnemo.sym.hydra import to_absolute_path
from physicsnemo.sym.distributed.manager import DistributedManager
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from typing import Union
import h5py
import hydra
from omegaconf import DictConfig
from physicsnemo.models.fno import FNO
from torch.utils.data import Dataset, DataLoader
from physicsnemo.launch.logging import PythonLogger, LaunchLogger
from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
import torch.nn.functional as F


class HDF5MapStyleDataset(Dataset):
    """Simple map-stype HDF5 dataset"""

    def __init__(
        self,
        file_path,
        device: Union[str, torch.device] = "cuda",
    ):
        self.file_path = file_path
        with h5py.File(file_path, "r") as f:
            self.keys = list(f.keys())

        # Set up device, needed for pipeline
        if isinstance(device, str):
            device = torch.device(device)
        # Need a index id if cuda
        if device.type == "cuda" and device.index == None:
            device = torch.device("cuda:0")
        self.device = device

    def __len__(self):
        with h5py.File(self.file_path, "r") as f:
            return len(f[self.keys[0]])

    def __getitem__(self, idx):
        data = {}
        with h5py.File(self.file_path, "r") as f:
            for key in self.keys:
                data[key] = np.array(f[key][idx])

        invar = torch.from_numpy(data["wavefield_in"])
        outvar = torch.from_numpy(data["wavefield_sol"])
        if self.device.type == "cuda":
            # Move tensors to GPU
            invar = invar.cuda()
            outvar = outvar.cuda()

        return invar, outvar


@hydra.main(version_base="1.3", config_path="./conf/", config_name="config_FNO_launch")
def main(cfg: DictConfig) -> None:
    """
    Invert the given wavefield to detect the brain anomaly, using trained FNO as
    surrogate forward operator to compute the gradient.
    """

    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    ### get the data
    train_path = to_absolute_path("./train_sets/data_scale_train.hdf5")
    train_dataset = HDF5MapStyleDataset(train_path, device=dist.device)
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True
    )

    test_path = to_absolute_path("./train_sets/data_scale_test.hdf5")
    # rewrite this into class
    test_dataset = HDF5MapStyleDataset(test_path, device=dist.device)
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.test.batch_size, shuffle=False
    )

    # define model, loss, optimiser, scheduler, data loader
    ## model
    model = FNO(
        in_channels=cfg.arch.fno.in_channels,
        out_channels=cfg.arch.decoder.out_features,
        decoder_layers=cfg.arch.decoder.layers,
        decoder_layer_size=cfg.arch.decoder.layer_size,
        dimension=cfg.arch.fno.dimension,
        latent_channels=cfg.arch.fno.latent_channels,
        num_fno_layers=cfg.arch.fno.fno_layers,
        num_fno_modes=cfg.arch.fno.fno_modes,
        padding=cfg.arch.fno.padding,
    ).to(dist.device)

    # load the trained model
    load_checkpoint("./checkpoints/", models=[model], device=dist.device)

    # test data path
    test_path = to_absolute_path("./train_sets/data_scale_test.hdf5")

    base_path = to_absolute_path("./train_sets/data_scale_base.hdf5")
    with h5py.File(test_path, "r") as f:
        custom_key = "wavefield_sol"
        extracted_output = f[custom_key][()]
        sol_numpy = extracted_output[1, :, :, :, :]

        custom_key = "wavefield_in"
        extracted_output = f[custom_key][()]
        input_test = extracted_output[0, :, :, :, :]
    with h5py.File(base_path, "r") as f:
        custom_key = "wavefield_in"
        extracted_output = f[custom_key][()]
        starting_input = extracted_output[0, :, :, :, :]

    test_wave_in = starting_input
    starting_dat = (
        torch.from_numpy(test_wave_in).unsqueeze(0).to(dist.device, torch.float32)
    )

    test_wave_in2 = input_test
    starting_dat2 = (
        torch.from_numpy(test_wave_in).unsqueeze(0).to(dist.device, torch.float32)
    )

    ### initial starting velocity
    velSlice = nn.Parameter(starting_dat[:, 20, :, :, :])
    test_data_in = {
        "wavefield_in": starting_dat,
    }
    # inject starting velocity to input tensor
    test_data_in["wavefield_in"][:, 20, :, :, :] = velSlice
    # put tensor rather than dict of tensor here
    prediction = model(starting_dat)
    prediction2 = model(starting_dat2)
    numpy_pred = prediction2.detach().cpu().numpy()
    np.save("./pred2.npy", numpy_pred)

    solTorch = torch.from_numpy(sol_numpy).to(dist.device, torch.float32)

    # define loss function
    loss_fn = nn.MSELoss()
    lr = 4000
    iterNum = 1200

    f = open(
        "loss_log_launch.txt", "w+"
    )  # located in working directory (e.g., ./outputs/<filename>/ )

    for i in range(iterNum):
        # inject updated elocity to input tensor
        test_data_in["wavefield_in"][:, 20, :, :, :] = velSlice
        # prediction =model(test_data_in)
        # inject input tensor
        prediction = model(test_data_in["wavefield_in"])
        outTorch = prediction
        loss = loss_fn(outTorch, solTorch.unsqueeze(1))

        if i % 10 == 0:
            print("iteration No. {}".format(i))
            print("loss: {}".format(loss))
        f.write("{}\n".format(loss))

        grad = torch.autograd.grad(loss, velSlice, torch.ones_like(loss))
        velSlice = velSlice - lr * grad[0][:, :, :, :]
        numpy_grad = grad[0].detach().cpu().numpy()

        # save gradient / updated velocity for every 100 step
        if i % 100 == 0:
            np.save("./grad_launch/numpy_grad_" + str(i).zfill(3), numpy_grad)
            np.save(
                "./grad_launch/vel_" + str(i).zfill(3), velSlice.detach().cpu().numpy()
            )


if __name__ == "__main__":
    main()
