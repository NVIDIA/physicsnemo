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

import os
import sys
import zipfile
import h5py
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from physicsnemo.models.rnn.rnn_one2many import One2ManyRNN
from physicsnemo.models.rnn.rnn_seq2seq import Seq2SeqRNN
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Union
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, LaunchLogger
from hydra.utils import to_absolute_path


def prepare_data(
    input_data_path,
    output_data_path,
    input_nr_tsteps,
    predict_nr_tsteps,
    start_idx,
    num_samples,
):
    """Data pre-processing"""
    if Path(output_data_path).is_file():
        pass
    else:
        arrays = {}
        data = h5py.File(input_data_path)

        for k, v in data.items():
            arrays[k] = np.array(v)

        invar = arrays["u"][
            0:input_nr_tsteps,
            ...,
            start_idx : start_idx + num_samples,
        ]
        outvar = arrays["u"][
            input_nr_tsteps : input_nr_tsteps + predict_nr_tsteps,
            ...,
            start_idx : start_idx + num_samples,
        ]
        invar = np.moveaxis(invar, -1, 0)
        outvar = np.moveaxis(outvar, -1, 0)
        invar = np.expand_dims(invar, axis=1)
        outvar = np.expand_dims(outvar, axis=1)

        h = h5py.File(output_data_path, "w")
        h.create_dataset("invar", data=invar)
        h.create_dataset("outvar", data=outvar)
        h.close()


def validation_step(model, dataloader, epoch):
    """Validation step"""
    model.eval()

    loss_epoch = 0
    for data in dataloader:
        invar, outvar = data
        predvar = model(invar)
        loss_epoch += F.mse_loss(outvar, predvar)

    # convert data to numpy
    outvar = outvar.detach().cpu().numpy()
    predvar = predvar.detach().cpu().numpy()

    # plotting
    fig, ax = plt.subplots(2, outvar.shape[2], figsize=(5 * outvar.shape[2], 10))
    for t in range(outvar.shape[2]):
        ax[0, t].imshow(outvar[0, 0, t, ...])
        ax[1, t].imshow(predvar[0, 0, t, ...])
        ax[0, t].set_title(f"True: {t}")
        ax[1, t].set_title(f"Pred: {t}")

    fig.savefig(f"./test_{epoch}.png")
    plt.close()
    return loss_epoch / len(dataloader)


class HDF5MapStyleDataset(Dataset):
    """Simple map-style HDF5 dataset"""

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


@hydra.main(version_base="1.2", config_path="conf", config_name="config_2d")
def main(cfg: DictConfig) -> None:
    logger = PythonLogger("main")  # General python logger
    LaunchLogger.initialize()

    raw_data_path = to_absolute_path("./datasets/ns_V1e-3_N5000_T50.mat")
    # Download data
    if Path(raw_data_path).is_file():
        pass
    else:
        try:
            import gdown
        except:
            logger.error(
                "gdown package not found, install it using `pip install gdown`"
            )
            sys.exit()
        logger.info("Data download starting...")
        url = "https://drive.google.com/uc?id=1r3idxpsHa21ijhlu3QQ1hVuXcqnBTO7d"
        os.makedirs(to_absolute_path("./datasets/"), exist_ok=True)
        output_path = to_absolute_path("./datasets/navier_stokes.zip")
        gdown.download(url, output_path, quiet=False)
        logger.info("Data downloaded.")
        logger.info("Extracting data...")
        with zipfile.ZipFile(output_path, "r") as zip_ref:
            zip_ref.extractall(to_absolute_path("./datasets/"))
        logger.info("Data extracted")

    # Data pre-processing
    num_samples = 1000
    test_samples = 10
    nr_tsteps_to_predict = 16
    nr_tsteps_to_test = 16

    if cfg.model_type == "one2many":
        input_nr_tsteps = 1
    elif cfg.model_type == "seq2seq":
        input_nr_tsteps = nr_tsteps_to_predict
    else:
        logger.error("Invalid model type!")

    raw_data_path = to_absolute_path("./datasets/ns_V1e-3_N5000_T50.mat")
    train_save_path = "./train_data_" + str(cfg.model_type) + ".hdf5"
    test_save_path = "./test_data_" + str(cfg.model_type) + ".hdf5"

    # prepare data
    prepare_data(
        raw_data_path,
        train_save_path,
        input_nr_tsteps,
        nr_tsteps_to_predict,
        0,
        num_samples,
    )
    prepare_data(
        raw_data_path,
        test_save_path,
        input_nr_tsteps,
        nr_tsteps_to_test,
        num_samples,
        test_samples,
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
    if cfg.model_type == "one2many":
        arch = One2ManyRNN(
            input_channels=1,
            dimension=2,
            nr_tsteps=nr_tsteps_to_predict,
            nr_downsamples=3,
            nr_residual_blocks=2,
            nr_latent_channels=32,
        )
    elif cfg.model_type == "seq2seq":
        arch = Seq2SeqRNN(
            input_channels=1,
            dimension=2,
            nr_tsteps=nr_tsteps_to_predict,
            nr_downsamples=3,
            nr_residual_blocks=2,
            nr_latent_channels=32,
        )
    else:
        logger.error("Invalid model type!")

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
            epoch_alert_freq=10,
        ) as log:
            # go through the full dataset
            for data in train_dataloader:
                invar, outvar = data
                optimizer.zero_grad()
                outpred = arch(invar)

                loss = F.mse_loss(outvar, outpred)
                loss.backward()
                optimizer.step()
                scheduler.step()
                # log.log_minibatch({"loss": loss.detach()})

            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        with LaunchLogger("valid", epoch=epoch) as log:
            error = validation_step(arch, test_dataloader, epoch)
            log.log_epoch({"Validation error": error})

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
