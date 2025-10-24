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

import os
import tarfile
import urllib.request
import h5py
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from physicsnemo.models.rnn.rnn_one2many import One2ManyRNN
import torch.nn.functional as F
from typing import Union
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.launch.logging import PythonLogger, LaunchLogger
from hydra.utils import to_absolute_path
from pyevtk.hl import imageToVTK


def prepare_data(
    input_data_path,
    output_data_path,
    predict_nr_tsteps,
    start_timestep,
):
    """Data pre-processing"""
    if Path(output_data_path).is_file():
        pass
    else:
        data = h5py.File(input_data_path)
        list_data = []
        for i in range(len(list(data.keys()))):
            data_u = data[str(i)]["u"]
            data_v = data[str(i)]["v"]
            data_uv = np.stack([data_u, data_v], axis=0)
            data_uv = np.array(data_uv)
            list_data.append(data_uv)

        data.close()
        data_combined = np.stack(list_data, axis=0)

        h = h5py.File(output_data_path, "w")
        h.create_dataset(
            "invar",
            data=np.expand_dims(data_combined[:, :, start_timestep, ...], axis=2),
        )
        h.create_dataset(
            "outvar",
            data=data_combined[
                :, :, start_timestep + 1 : start_timestep + 1 + predict_nr_tsteps, ...
            ],
        )
        h.close()


def validation_step(model, dataloader, epoch):
    """Validation Step"""
    model.eval()

    for data in dataloader:
        invar, outvar = data
        predvar = model(invar)

    # convert data to numpy
    outvar = outvar.detach().cpu().numpy()
    predvar = predvar.detach().cpu().numpy()

    # plotting
    for t in range(outvar.shape[2]):
        cellData = {
            "outvar_chan0": outvar[0, 0, t, ...],
            "outvar_chan1": outvar[0, 1, t, ...],
            "predvar_chan0": predvar[0, 0, t, ...],
            "predvar_chan1": predvar[0, 1, t, ...],
        }
        imageToVTK(f"./test_{t}", cellData=cellData)


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

        invar = torch.from_numpy(data["invar"])
        outvar = torch.from_numpy(data["outvar"])
        if self.device.type == "cuda":
            # Move tensors to GPU
            invar = invar.cuda()
            outvar = outvar.cuda()

        return invar, outvar


@hydra.main(version_base="1.2", config_path="conf", config_name="config_3d")
def main(cfg: DictConfig) -> None:
    logger = PythonLogger("main")  # General python logger
    LaunchLogger.initialize()

    # Data download
    raw_train_data_path = to_absolute_path("./datasets/grayscott_training.hdf5")
    raw_test_data_path = to_absolute_path("./datasets/grayscott_test.hdf5")

    # Download data
    if Path(raw_train_data_path).is_file():
        pass
    else:
        logger.info("Data download starting...")
        url = "https://zenodo.org/record/5148524/files/grayscott_training.tar.gz"
        os.makedirs(to_absolute_path("./datasets/"), exist_ok=True)
        output_path = to_absolute_path("./datasets/grayscott_training.tar.gz")
        urllib.request.urlretrieve(url, output_path)
        logger.info("Data downloaded.")
        logger.info("Extracting data...")
        with tarfile.open(output_path, "r") as tar_ref:
            tar_ref.extractall(to_absolute_path("./datasets/"))
        logger.info("Data extracted")

    if Path(raw_test_data_path).is_file():
        pass
    else:
        logger.info("Data download starting...")
        url = "https://zenodo.org/record/5148524/files/grayscott_test.tar.gz"
        os.makedirs(to_absolute_path("./datasets/"), exist_ok=True)
        output_path = to_absolute_path("./datasets/grayscott_test.tar.gz")
        urllib.request.urlretrieve(url, output_path)
        logger.info("Data downloaded.")
        logger.info("Extracting data...")
        with tarfile.open(output_path, "r") as tar_ref:
            tar_ref.extractall(to_absolute_path("./datasets/"))
        logger.info("Data extracted")

    # Data pre-processing
    nr_tsteps_to_predict = 64
    nr_tsteps_to_test = 64
    start_timestep = 5

    train_save_path = "./train_data_gray_scott_one2many.hdf5"
    test_save_path = "./test_data_gray_scott_one2many.hdf5"

    # prepare data
    prepare_data(
        raw_train_data_path, train_save_path, nr_tsteps_to_predict, start_timestep
    )
    prepare_data(
        raw_test_data_path,
        test_save_path,
        nr_tsteps_to_test,
        start_timestep,
    )

    train_dataset = HDF5MapStyleDataset(train_save_path, device="cuda")
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True
    )
    test_dataset = HDF5MapStyleDataset(test_save_path, device="cuda")
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.batch_size_test, shuffle=False
    )

    # set device as GPU
    device = "cuda"

    # instantiate model
    arch = One2ManyRNN(
        input_channels=2,
        dimension=3,
        nr_tsteps=nr_tsteps_to_predict,
        nr_downsamples=2,
        nr_residual_blocks=2,
        nr_latent_channels=16,
    )

    if device == "cuda":
        arch.cuda()

    optimizer = torch.optim.Adam(
        arch.parameters(),
        betas=(0.9, 0.999),
        lr=cfg.start_lr,
        weight_decay=0.0,
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=cfg.lr_scheduler_gamma
    )

    loaded_epoch = load_checkpoint(
        "./checkpoints",
        models=arch,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cuda",
    )

    # Training loop
    for epoch in range(max(1, loaded_epoch + 1), cfg.max_epochs + 1):
        # wrap epoch in launch logger for console logs
        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=len(train_dataloader),
            epoch_alert_freq=1,
        ) as log:
            # go through the full dataset
            for i, data in enumerate(train_dataloader):
                invar, outvar = data
                optimizer.zero_grad()
                outpred = arch(invar)

                loss = F.mse_loss(outvar, outpred)
                loss.backward()
                optimizer.step()
                scheduler.step()
                log.log_minibatch({"loss": loss.detach()})

            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        with LaunchLogger("valid", epoch=epoch) as log:
            validation_step(arch, test_dataloader, epoch)

        if epoch % cfg.checkpoint_save_freq == 0:
            save_checkpoint(
                "./checkpoints",
                models=arch,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )

    logger.info("Finished Training")


if __name__ == "__main__":
    main()
