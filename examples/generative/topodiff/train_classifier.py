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
import torch.nn.functional as F
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
from utils import load_data_topodiff, load_data_classifier


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    logger = PythonLogger("main")  # General Python Logger
    logger.log("Start running")

    train_img, train_labels = load_data_classifier(cfg.path_data_classifier_training)
    valid_img, valid_labels = load_data_classifier(cfg.path_data_classifier_validation)
    train_img = 2 * train_img - 1
    valid_img = 2 * valid_img - 1

    device = torch.device("cuda:1")

    classifier = UNetEncoder(in_channels=1, out_channels=2).to(device)

    diffusion = Diffusion(n_steps=cfg.diffusion_steps, device=device)

    batch_size = cfg.batch_size

    lr = cfg.lr
    optimizer = AdamW(classifier.parameters(), lr=lr)
    scheduler = LinearLR(
        optimizer,
        start_factor=1,
        end_factor=0.001,
        total_iters=cfg.classifier_iterations,
    )

    for i in range(cfg.classifier_iterations + 1):
        # get random batch from training data

        idx = np.random.choice(len(train_img), batch_size, replace=False)
        batch = torch.tensor(train_img[idx]).float().unsqueeze(1).to(device) * 2 - 1
        batch_labels = torch.tensor(train_labels[idx]).long().to(device)

        t = torch.randint(0, cfg.diffusion_steps, (batch.shape[0],)).to(device)
        batch = diffusion.q_sample(batch, t)
        logits = classifier(batch, time_steps=t)

        loss = F.cross_entropy(logits, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 100 == 0:
            with torch.no_grad():
                idx = np.random.choice(len(valid_img), batch_size, replace=False)
                batch = (
                    torch.tensor(valid_img[idx]).float().unsqueeze(1).to(device) * 2 - 1
                )
                batch_labels = torch.tensor(valid_labels[idx]).long().to(device)

                # Sample diffusion steps and get noised images
                t = torch.randint(0, cfg.diffusion_steps, (batch.shape[0],)).to(device)
                batch = diffusion.q_sample(batch, t)

                # Forward pass
                logits = classifier(batch, time_steps=t)

                # Compute accuracy
                predicted_labels = torch.argmax(logits, dim=1)
                correct_predictions = (predicted_labels == batch_labels).sum().item()
                accuracy = correct_predictions / batch_size

                print(
                    "epoch: %d, loss: %.5f, validation accuracy: %.3f"
                    % (i, loss.item(), accuracy)
                )
        # if i % 10000 == 0:
        #    torch.save(classifier.state_dict(), cfg.model_path + "classifier_" +str(i) + ".pt")
    torch.save(classifier.state_dict(), cfg.model_path + "classifier.pt")

    print("job done!")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
