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
# limitations under the License.cd ..

import hydra
import torch
from omegaconf import DictConfig
from torch.optim import AdamW
from tqdm import trange
from utils import DDPMLinearNoiseScheduler, load_data, load_data_topodiff

from physicsnemo.diffusion.metrics.losses import MSEDSMLoss
from physicsnemo.models.topodiff import TopoDiff
from physicsnemo.utils.logging import PythonLogger


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = PythonLogger("main")  # General Python Logger
    logger.log("Job start!")

    device = torch.device("cuda:0")
    model = TopoDiff(64, 6, 1, model_channels=128, attn_resolutions=[16, 8]).to(device)
    scheduler = DDPMLinearNoiseScheduler(n_steps=1000)

    # Adapt TopoDiff to DiffusionModel protocol (epsilon-predictor)
    def diffusion_model(x, t, condition=None, **kwargs):
        return model(x, condition, t.long())

    loss_fn = MSEDSMLoss(
        model=diffusion_model,
        noise_scheduler=scheduler,
        prediction_type="epsilon",
        epsilon_to_x0_fn=scheduler.epsilon_to_x0,
    )

    topologies = load_data(
        cfg.path_training_data_diffusion, cfg.prefix_topology_file, ".png", 0, 30000
    )
    vfs_stress_strain = load_data(
        cfg.path_training_data_diffusion, cfg.prefix_pf_file, ".npy", 0, 30000
    )
    load_imgs = load_data(
        cfg.path_training_data_diffusion, cfg.prefix_load_file, ".npy", 0, 30000
    )

    batch_size = cfg.batch_size
    data = load_data_topodiff(
        topologies,
        vfs_stress_strain,
        load_imgs,
        batch_size=batch_size,
        deterministic=False,
    )

    lr = cfg.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    logger.log("Start training!")

    prog = trange(cfg.epochs)

    for step in prog:
        tops, cons = next(data)

        tops = tops.float().to(device)
        cons = cons.float().to(device)

        losses = loss_fn(x0=tops, condition=cons)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if step % 100 == 0:
            logger.info("epoch: %d, loss: %.5f" % (step, losses.item()))

    torch.save(model.state_dict(), cfg.model_path + "topodiff_model.pt")
    logger.info("Training completed!")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
