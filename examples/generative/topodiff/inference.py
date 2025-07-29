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
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt


import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from physicsnemo.models.topodiff import TopoDiff, Diffusion
from physicsnemo.models.topodiff import UNetEncoder
from physicsnemo.launch.logging import PythonLogger
from physicsnemo.launch.logging.wandb import initialize_wandb
from utils import load_data_topodiff, load_data


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    logger = PythonLogger("main")  # General Python Logger
    logger.log("Job start!")

    topologies = np.random.randn(1800, 64, 64)
    vfs_stress_strain = load_data(
        cfg.path_test_data_diffusion,
        cfg.prefix_pf_file,
        ".npy",
        200,
        2000,
    )
    load_imgs = load_data(
        cfg.path_test_data_diffusion,
        cfg.prefix_load_file,
        ".npy",
        200,
        2000,
    )

    device = torch.device("cuda:1")
    model = TopoDiff(64, 6, 1, model_channels=128, attn_resolutions=[16, 8])
    model.load_state_dict(torch.load(cfg.model_path_diffusion))
    model.to(device)

    classifier = UNetEncoder(in_channels=1, out_channels=2)
    classifier.load_state_dict(torch.load(cfg.model_path_classifier))
    classifier.to(device)

    diffusion = Diffusion(n_steps=1000, device=device)
    batch_size = cfg.batch_size
    data = load_data_topodiff(
        topologies,
        vfs_stress_strain,
        load_imgs,
        batch_size=batch_size,
        deterministic=False,
    )

    _, cons = next(data)

    cons = cons.float().to(device)

    n_steps = 1000

    xt = torch.randn(batch_size, 1, 64, 64).to(device)
    floating_labels = torch.tensor([1] * batch_size).long().to(device)

    for i in reversed(trange(n_steps)):
        with torch.no_grad():
            t = torch.tensor([i] * batch_size, device=device)
            noisy = diffusion.p_sample(model, xt, t, cons)

        with torch.enable_grad():
            xt.requires_grad_(True)
            logits = classifier(xt, time_steps=t)
            loss = F.cross_entropy(logits, floating_labels)

            grad = torch.autograd.grad(loss, xt)[0]

        xt = (
            1
            / diffusion.alphas[i].sqrt()
            * (
                xt
                - noisy
                * (1 - diffusion.alphas[i])
                / (1 - diffusion.alpha_bars[i]).sqrt()
            )
        )

        if i > 0:
            z = torch.zeros_like(xt).to(device)
            xt = xt + diffusion.betas[i].sqrt() * (z * 0.8 + 0.2 * grad.float())

    result = (xt.cpu().detach().numpy() + 1) * 2

    np.save(cfg.generation_path + "results_topology.npy", result)

    # plot images for the generated samples
    fig, axes = plt.subplots(8, 8, figsize=(12, 6), dpi=300)

    for i in range(8):
        for j in range(8):
            img = result[i * 4 + j][0]
            axes[i, j].imshow(img, cmap="gray")
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    plt.gca().xaxis.set_visible(False)  # Optionally hide x-axis
    plt.gca().yaxis.set_visible(False)  # Optionally hide y-axis

    plt.savefig(
        cfg.generation_path + "grid_topology.png", bbox_inches="tight", pad_inches=0
    )


if __name__ == "__main__":
    main()
