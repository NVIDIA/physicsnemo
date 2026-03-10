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
from physicsnemo.models.dpot.dpot import DPOTNet
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
        try:
            data = h5py.File(input_data_path)
        except Exception as e:
            from scipy.io import loadmat

            data = loadmat(input_data_path)
        for k, v in data.items():
            arrays[k] = np.array(v)

        invar = arrays["u"][
            input_nr_tsteps : input_nr_tsteps + predict_nr_tsteps,
            ...,
            start_idx : start_idx + num_samples,
        ]
        outvar = arrays["u"][
            input_nr_tsteps + predict_nr_tsteps : input_nr_tsteps
            + 2 * predict_nr_tsteps,
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


def rel_l2_loss(pred, target):
    """Relative L2 loss"""
    return torch.sqrt(torch.sum((pred - target) ** 2)) / torch.sqrt(
        torch.sum(target**2)
    )


def validation_step(model, dataloader, epoch):
    """Validation step"""
    model.eval()

    loss_epoch = 0
    with torch.no_grad():
        for data in dataloader:
            invar, outvar = data
            pred = []
            for t in range(outvar.shape[-2]):
                predvar = model(invar)
                invar = torch.cat([invar[..., 1:, :], predvar], dim=-2)
                pred.append(predvar)
            predvar = torch.cat(pred, dim=-2)

            loss_epoch += rel_l2_loss(predvar, outvar).item()

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

        invar = torch.from_numpy(data["invar"]).permute(2, 3, 1, 0)
        outvar = torch.from_numpy(data["outvar"]).permute(2, 3, 1, 0)
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
    num_samples = 2000
    test_samples = 100
    nr_tsteps_to_predict = 10

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
        0,
        nr_tsteps_to_predict,
        0,
        num_samples,
    )
    prepare_data(
        raw_data_path,
        test_save_path,
        0,
        nr_tsteps_to_predict,
        0,
        test_samples,
    )

    train_dataset = HDF5MapStyleDataset(train_save_path, device=cfg.device)
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True
    )
    test_dataset = HDF5MapStyleDataset(test_save_path, device=cfg.device)
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.batch_size_test, shuffle=False
    )

    # set device as GPU
    device = cfg.device

    # instantiate model
    arch = DPOTNet(
        inp_shape=64,
        patch_size=8,
        in_channels=1,
        out_channels=1,
        in_timesteps=nr_tsteps_to_predict,
        out_timesteps=1,
        depth=4,
        embed_dim=128,
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
        device=cfg.device,
    )

    # Training loop
    for epoch in range(max(1, loaded_epoch + 1), cfg.max_epochs + 1):
        # wrap epoch in launch logger for console logs
        if cfg.train:
            arch.train()
            with LaunchLogger(
                "train",
                epoch=epoch,
                num_mini_batch=len(train_dataloader),
                epoch_alert_freq=10,
            ) as log:
                # go through the full dataset
                for i, seq in enumerate(train_dataloader):
                    # seq: (B,1,T,H,W)
                    x = torch.cat(seq, dim=-2)  # B, X, Y, T_total, C
                    T = x.shape[-2]

                    # --- Randomly sample a timestep  ---
                    t = torch.randint(
                        nr_tsteps_to_predict, T - 1, (1,), device=x.device
                    ).item()
                    x_t = x[..., t - nr_tsteps_to_predict : t, :]  # (B,1,H,W)
                    y_tp1 = x[..., t : t + 1, :]  # (B,1,H,W)

                    # noise scale
                    if cfg.noise_scale > 0:
                        norm_factor = torch.sqrt(
                            torch.sum(x_t**2, dim=(1, 2, 3), keepdim=True) + 1e-12
                        )
                        x_t = x_t + cfg.noise_scale * norm_factor * torch.randn_like(
                            x_t
                        )

                    pred = arch(x_t)
                    loss = rel_l2_loss(pred, y_tp1)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(arch.parameters(), cfg.grad_clip)
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
