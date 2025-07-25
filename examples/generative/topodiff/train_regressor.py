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
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm import trange
import numpy as np
import time, os


import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from physicsnemo.models.topodiff import Diffusion
from physicsnemo.models.topodiff import UNetEncoder
from physicsnemo.launch.logging import PythonLogger
from physicsnemo.launch.logging.wandb import initialize_wandb
from utils import load_data_topodiff, load_data_regressor


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    logger = PythonLogger("main")  # General Python Logger
    logger.log("Start running")

    topologies, load_imgs, vfs_stress_strain, labels = load_data_regressor(
        cfg.path_data_regressor_training
    )
    topologies = topologies * 2 - 1  # Normalize the range of image to be [-1, 1]
    """
    topologies = np.load(cfg.path_data + "Compliance/Training/topologies.npy").astype(np.float64)
    constraints = np.load(cfg.path_data + "Compliance/Training/constraints.npy", allow_pickle=True)
    stress = np.load(cfg.path_data+ "Compliance/Training/vonmises.npy", allow_pickle=True)
    strain = np.load(cfg.path_data + "Compliance/Training/strain_energy.npy", allow_pickle=True)
    load_imgs = np.load(cfg.path_data + "Compliance/Training/load_imgs.npy")
    bc_imgs = np.load(cfg.path_data + "Compliance/Training/bc_imgs.npy").astype(np.float64)
    Compliance = np.load(cfg.path_data + "Compliance/Training/compliance.npy").astype(np.float64)
    """

    device = torch.device("cuda:0")

    in_channels = 6
    regressor = UNetEncoder(in_channels=in_channels, out_channels=1).to(device)

    diffusion = Diffusion(n_steps=cfg.diffusion_steps, device=device)

    batch_size = cfg.batch_size
    """
    data = load_data_topodiff(
        topologies, vfs_stress_strain, load_imgs, batch_size= batch_size,deterministic=False
    )
    """
    lr = cfg.lr
    optimizer = AdamW(regressor.parameters(), lr=lr)
    scheduler = LinearLR(
        optimizer,
        start_factor=1,
        end_factor=0.001,
        total_iters=cfg.regressor_iterations,
    )

    loss_fn = nn.MSELoss()
    for i in range(cfg.regressor_iterations + 1):

        # get random batch from training data
        idx = np.random.choice(len(topologies), batch_size, replace=False)
        batch = (
            torch.tensor(topologies[idx]).float().unsqueeze(1).to(device) * 2 - 1
        )  # 4 x 1 x 64 x 64
        batch_pf = (
            torch.tensor(vfs_stress_strain[idx]).float().permute(0, 3, 1, 2).to(device)
        )
        batch_load = torch.tensor(load_imgs[idx]).float().permute(0, 3, 1, 2).to(device)

        batch_labels = torch.tensor(labels[idx]).float().to(device).unsqueeze(1)

        t = torch.randint(0, cfg.diffusion_steps, (batch.shape[0],)).to(device)
        batch = diffusion.q_sample(batch, t)

        batch = torch.cat((batch, batch_pf, batch_load), dim=1)

        logits = regressor(batch, time_steps=t)

        loss = loss_fn(logits, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 100 == 0:
            print("epoch: %d, loss: %.5f" % (i, loss.item()))

    torch.save(regressor.state_dict(), cfg.model_path + "regressor.pt")
    print("job done!")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
