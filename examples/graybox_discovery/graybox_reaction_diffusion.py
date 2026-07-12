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
"""Gray-box discovery of a reaction/closure term in a PINN.

The governing equation is  u_t - D u_xx - R(u) = 0, where the structural terms
u_t and D u_xx are known but the reaction term R(u) is unknown. R is represented
by a second network that maps u -> R and is trained jointly with the field
network from point observations of u. A known-equilibria constraint (R(0)=R(1)=0)
is added as a physics prior; the observations do not cover u close to 1, so
without it R is under-determined there.

Run:
    python generate_data.py
    python graybox_reaction_diffusion.py
"""

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from sympy import Function, Symbol
from torch.optim import Adam, lr_scheduler

from physicsnemo.distributed import DistributedManager
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physicsnemo.utils.logging import PythonLogger


class GrayBoxReactionDiffusion(PDE):
    """u_t - D u_xx - R = 0 with R supplied by a second network.

    ``dim=1`` because only the ``x`` derivatives are auto-differentiated by
    :class:`PhysicsInformer`; the time derivative ``u__t`` is supplied
    directly (autodiff spatial gradients only cover x/y/z).
    """

    name = "GrayBoxReactionDiffusion"

    def __init__(self, D=0.02):
        self.dim = 1
        x, t = Symbol("x"), Symbol("t")
        u = Function("u")(x, t)
        r = Function("R")(x, t)  # value provided by the reaction network
        self.equations = {"reaction_diffusion": u.diff(t) - D * u.diff(x, 2) - r}


@hydra.main(version_base="1.3", config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """Train the field and reaction networks jointly and plot the recovered R(u)."""
    DistributedManager.initialize()
    dist = DistributedManager()

    log = PythonLogger(name="graybox_discovery")
    log.file_logging()

    data = np.load(cfg.data.data_file)
    D = float(data["D"])
    A_true = float(data["A"])
    u_obs = float(data["u_obs"])

    eq = GrayBoxReactionDiffusion(D=D)
    pi = PhysicsInformer(
        required_outputs=["reaction_diffusion"],
        equations=eq,
        grad_method="autodiff",
        device=dist.device,
    )

    # field network (x, t) -> u
    u_net = FullyConnected(
        in_features=2,
        out_features=1,
        layer_size=cfg.arch.layer_size,
        num_layers=cfg.arch.num_layers,
    ).to(dist.device)
    # reaction network u -> R -- the discoverable closure term
    r_net = FullyConnected(
        in_features=1,
        out_features=1,
        layer_size=cfg.arch.layer_size,
        num_layers=cfg.arch.num_layers,
    ).to(dist.device)

    def to_tensor(name):
        return torch.tensor(data[name], dtype=torch.float32, device=dist.device)

    x_data, t_data, u_data = (
        to_tensor("x_data"),
        to_tensor("t_data"),
        to_tensor("u_data"),
    )
    x_pde, t_pde = to_tensor("x_pde"), to_tensor("t_pde")
    u_eq = torch.tensor([[0.0], [1.0]], dtype=torch.float32, device=dist.device)

    n_data, n_pde = x_data.shape[0], x_pde.shape[0]
    batch_data = min(cfg.batch_size.data, n_data)
    batch_pde = min(cfg.batch_size.interior, n_pde)

    optimizer = Adam(
        list(u_net.parameters()) + list(r_net.parameters()),
        lr=cfg.scheduler.initial_lr,
    )
    per_step_gamma = cfg.scheduler.decay_rate ** (1.0 / cfg.scheduler.decay_steps)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=per_step_gamma)

    max_steps = cfg.training.max_steps
    log_freq = cfg.training.log_freq

    for step in range(max_steps):
        optimizer.zero_grad()

        # 1) data assimilation: fit the observed field
        idx = np.random.choice(n_data, size=batch_data, replace=False)
        u_pred = u_net(torch.cat([x_data[idx], t_data[idx]], dim=1))
        data_loss = torch.nn.functional.mse_loss(u_pred, u_data[idx])

        # 2) PDE residual on interior collocation points
        idx = np.random.choice(n_pde, size=batch_pde, replace=False)
        xb = x_pde[idx].clone().requires_grad_(True)
        tb = t_pde[idx].clone().requires_grad_(True)
        u_col = u_net(torch.cat([xb, tb], dim=1))
        u_t = torch.autograd.grad(
            u_col, tb, grad_outputs=torch.ones_like(u_col), create_graph=True
        )[0]
        r_col = r_net(u_col)
        residual = pi.forward({"u": u_col, "u__t": u_t, "R": r_col, "coordinates": xb})[
            "reaction_diffusion"
        ]
        pde_loss = (residual**2).mean()

        # 3) physics prior: u = 0 and u = 1 are known equilibria => R = 0 there.
        #    Evaluated directly on the reaction network (no field involved).
        r_eq = r_net(u_eq)
        equilibria_loss = (r_eq**2).mean()

        loss = (
            data_loss
            + cfg.loss_weights.pde * pde_loss
            + cfg.loss_weights.equilibria * equilibria_loss
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % log_freq == 0 or step == max_steps - 1:
            log.info(
                f"step {step:6d} | loss={loss.item():.6e} "
                f"data={data_loss.item():.6e} pde={pde_loss.item():.6e} "
                f"equilibria={equilibria_loss.item():.6e} "
                f"| lr={scheduler.get_last_lr()[0]:.6e}"
            )

    # Recover R(u) on a grid and compare against the true closure.
    u_grid = torch.linspace(0, 1, 200, device=dist.device).unsqueeze(1)
    with torch.no_grad():
        r_recovered = r_net(u_grid).cpu().numpy().ravel()
    u_np = u_grid.cpu().numpy().ravel()
    r_true = A_true * np.sin(np.pi * u_np)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(u_np, r_true, "k--", label="true R(u)")
    ax.plot(u_np, r_recovered, "C0", label="recovered R(u)")
    ax.axvspan(u_obs, 1.0, color="0.9", label="unobserved (u > u_obs)")
    ax.set_xlabel("u")
    ax.set_ylabel("R(u)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("recovered_R.png", dpi=150)
    log.info("saved recovered_R.png")


if __name__ == "__main__":
    main()
