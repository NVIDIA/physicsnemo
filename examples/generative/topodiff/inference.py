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

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from utils import (
    ClassifierGuidance,
    DDPMLinearNoiseScheduler,
    DDPMSolver,
    load_data,
    load_data_topodiff,
)

from physicsnemo.diffusion.guidance import DPSScorePredictor
from physicsnemo.diffusion.samplers import sample
from physicsnemo.models.topodiff import TopoDiff, UNetEncoder
from physicsnemo.utils.logging import PythonLogger


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

    device = torch.device("cuda:0")
    model = TopoDiff(64, 6, 1, model_channels=128, attn_resolutions=[16, 8])
    model.load_state_dict(torch.load(cfg.model_path_diffusion))
    model.to(device)

    classifier = UNetEncoder(in_channels=1, out_channels=2)
    classifier.load_state_dict(torch.load(cfg.model_path_classifier))
    classifier.to(device)

    n_steps = cfg.diffusion_steps
    scheduler = DDPMLinearNoiseScheduler(n_steps=n_steps)

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

    # Epsilon predictor (TopoDiff model with fixed conditions)
    def eps_predictor(x, t):
        with torch.no_grad():
            return model(x, cons, t.long())

    # X0 predictor (convert epsilon -> x0)
    def x0_predictor(x, t):
        eps = eps_predictor(x, t)
        return scheduler.epsilon_to_x0(eps, x, t)

    # Classifier guidance
    floating_labels = torch.tensor([1] * batch_size).long().to(device)
    guidance = ClassifierGuidance(classifier, floating_labels, scale=0.2)

    # DPS guided score predictor (framework component)
    dps_score = DPSScorePredictor(
        x0_predictor=x0_predictor,
        x0_to_score_fn=scheduler.x0_to_score,
        guidances=guidance,
    )

    # DDPM solver (no stochastic noise, matching original)
    solver = DDPMSolver(dps_score, scheduler, stochastic=False)

    # Generate samples
    xt = torch.randn(batch_size, 1, 64, 64).to(device)

    # Note: the denoiser arg is required by sample() but unused when a custom
    # Solver is provided — the DDPMSolver uses its own score_predictor internally.
    with torch.inference_mode(False):
        xt = sample(
            denoiser=scheduler.get_denoiser(score_predictor=dps_score),
            xN=xt,
            noise_scheduler=scheduler,
            num_steps=n_steps,
            solver=solver,
        )

    result = (xt.cpu().detach().numpy() + 1) * 2

    np.save(cfg.generation_path + "results_topology.npy", result)

    # plot images for the generated samples
    n_samples = result.shape[0]
    ncols = min(8, n_samples)
    nrows = min(8, (n_samples + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6), dpi=300, squeeze=False)

    for idx in range(min(nrows * ncols, n_samples)):
        r, c = divmod(idx, ncols)
        axes[r, c].imshow(result[idx][0], cmap="gray")
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])
    for idx in range(n_samples, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    plt.savefig(
        cfg.generation_path + "grid_topology.png", bbox_inches="tight", pad_inches=0
    )


if __name__ == "__main__":
    main()
