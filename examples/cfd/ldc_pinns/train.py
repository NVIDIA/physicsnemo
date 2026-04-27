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
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import PythonLogger
from physicsnemo.models.fno import FNO
from physicsnemo.models.mlp.fully_connected import FullyConnected
from sympy import Function, Number, Symbol

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.planar.structured_grid import (
    load as load_structured_grid,
)
from physicsnemo.mesh.sampling import sample_random_points_on_cells
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physicsnemo.utils import StaticCaptureEvaluateNoGrad, StaticCaptureTraining
from omegaconf import DictConfig
from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler


class NavierStokes(PDE):
    """Incompressible Navier-Stokes equations (steady, 2D).

    Simplified from the compressible form in physicsnemo-sym for the case
    where ``rho`` is constant and ``time=False``.

    Reference: https://turbmodels.larc.nasa.gov/implementrans.html
    """

    def __init__(self, nu=0.01, rho=1.0, dim=2, time=False):
        self.dim = dim
        x, y = Symbol("x"), Symbol("y")
        iv = {"x": x, "y": y}
        u = Function("u")(*iv.values())
        v = Function("v")(*iv.values())
        p = Function("p")(*iv.values())
        nu, rho = Number(nu), Number(rho)
        self.equations = {
            "continuity": u.diff(x) + v.diff(y),
            "momentum_x": (
                u * u.diff(x)
                + v * u.diff(y)
                + (1 / rho) * p.diff(x)
                - nu * u.diff(x, 2)
                - nu * u.diff(y, 2)
            ),
            "momentum_y": (
                u * v.diff(x)
                + v * v.diff(y)
                + (1 / rho) * p.diff(y)
                - nu * v.diff(x, 2)
                - nu * v.diff(y, 2)
            ),
        }


@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def ldc_trainer(cfg: DictConfig) -> None:
    """Main function for the LDC PINNs."""
    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    # initialize monitoring
    log = PythonLogger(name="ldc")
    log.file_logging()

    # domain geometry using physicsnemo.mesh
    height = 0.1
    width = 0.1
    x_min, x_max = -width / 2, width / 2
    y_min, y_max = -height / 2, height / 2

    interior_mesh = load_structured_grid(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        n_x=50,
        n_y=50,
        device=dist.device,
    )
    boundary_mesh = interior_mesh.get_boundary_mesh()

    def sample_boundary(n_points, device):
        """Sample on the rectangle boundary using physicsnemo.mesh."""
        cell_indices = torch.randint(
            0, boundary_mesh.n_cells, (n_points,), device=device
        )
        pts = sample_random_points_on_cells(boundary_mesh, cell_indices)
        return {"x": pts[:, 0], "y": pts[:, 1]}

    def sample_interior(n_points, device):
        """Sample inside the rectangle using physicsnemo.mesh, with analytical SDF."""
        cell_indices = torch.randint(
            0, interior_mesh.n_cells, (n_points,), device=device
        )
        pts = sample_random_points_on_cells(interior_mesh, cell_indices)
        x, y = pts[:, 0], pts[:, 1]
        sdf = torch.min(
            torch.stack([x - x_min, x_max - x, y - y_min, y_max - y], dim=-1),
            dim=-1,
        ).values
        return {"x": x, "y": y, "sdf": sdf}

    model = FullyConnected(
        in_features=2, out_features=3, num_layers=6, layer_size=512
    ).to(dist.device)

    ns = NavierStokes(nu=0.01, rho=1.0, dim=2, time=False)
    phy_inf = PhysicsInformer(
        required_outputs=["continuity", "momentum_x", "momentum_y"],
        equations=ns,
        grad_method="autodiff",
        device=dist.device,
    )

    optimizer = Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 0.9999871767586216**step
    )

    # inference geometry
    x = np.linspace(-0.05, 0.05, 512)
    y = np.linspace(-0.05, 0.05, 512)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    xx, yy = (
        torch.from_numpy(xx).to(torch.float).to(dist.device),
        torch.from_numpy(yy).to(torch.float).to(dist.device),
    )

    for i in range(10000):
        optimizer.zero_grad()

        bc_data = sample_boundary(2000, dist.device)
        int_data = sample_interior(4000, dist.device)

        y_vals = bc_data["y"]
        mask_top_wall = y_vals >= height / 2 - 1e-7
        mask_no_slip = ~mask_top_wall

        no_slip_xy = torch.stack(
            [bc_data["x"][mask_no_slip], bc_data["y"][mask_no_slip]], dim=-1
        )
        top_wall_x = bc_data["x"][mask_top_wall].unsqueeze(-1)
        top_wall_xy = torch.stack(
            [bc_data["x"][mask_top_wall], bc_data["y"][mask_top_wall]], dim=-1
        )

        int_x = int_data["x"].unsqueeze(-1).requires_grad_(True)
        int_y = int_data["y"].unsqueeze(-1).requires_grad_(True)
        int_sdf = int_data["sdf"].unsqueeze(-1)
        coords = torch.cat([int_x, int_y], dim=1)

        no_slip_out = model(no_slip_xy)
        top_wall_out = model(top_wall_xy)
        interior_out = model(coords)

        u_no_slip = torch.mean(no_slip_out[:, 0:1] ** 2)
        v_no_slip = torch.mean(no_slip_out[:, 1:2] ** 2)
        u_slip = torch.mean(
            ((top_wall_out[:, 0:1] - 1.0) ** 2) * (1 - 20 * torch.abs(top_wall_x))
        )
        v_slip = torch.mean(top_wall_out[:, 1:2] ** 2)

        phy_loss_dict = phy_inf.forward(
            {
                "coordinates": coords,
                "u": interior_out[:, 0:1],
                "v": interior_out[:, 1:2],
                "p": interior_out[:, 2:3],
            }
        )

        cont = phy_loss_dict["continuity"] * int_sdf
        mom_x = phy_loss_dict["momentum_x"] * int_sdf
        mom_y = phy_loss_dict["momentum_y"] * int_sdf

        phy_loss = (
            torch.mean(cont**2)
            + torch.mean(mom_x**2)
            + torch.mean(mom_y**2)
            + u_no_slip
            + v_no_slip
            + u_slip
            + v_slip
        )
        phy_loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 1000 == 0:
            with torch.no_grad():
                inf_out = model(
                    torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1)
                )
                print(
                    f"Loss: {phy_loss.detach()}, LR: {optimizer.param_groups[0]['lr']}"
                )
                fig, axes = plt.subplots(1, 4, figsize=(12, 4))

                out_np = inf_out.detach().cpu().numpy()
                im = axes[0].imshow(out_np[:, 0].reshape(512, 512), origin="lower")
                fig.colorbar(im, ax=axes[0])
                axes[0].set_title("u")

                im = axes[1].imshow(out_np[:, 1].reshape(512, 512), origin="lower")
                fig.colorbar(im, ax=axes[1])
                axes[1].set_title("v")

                im = axes[2].imshow(out_np[:, 2].reshape(512, 512), origin="lower")
                fig.colorbar(im, ax=axes[2])
                axes[2].set_title("p")

                im = axes[3].imshow(
                    ((out_np[:, 0] ** 2 + out_np[:, 1] ** 2).reshape(512, 512)) ** 0.5,
                    origin="lower",
                )
                fig.colorbar(im, ax=axes[3])
                axes[3].set_title("u_mag")

                plt.savefig(f"./outputs/outputs_pc_{i}.png")
                plt.close()


if __name__ == "__main__":
    ldc_trainer()
