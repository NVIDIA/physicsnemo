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

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger
from physicsnemo.models.fno import FNO
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physicsnemo.sym.geometry.geometry_dataloader import GeometryDatapipe
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.utils import StaticCaptureEvaluateNoGrad, StaticCaptureTraining
from omegaconf import DictConfig
from sympy import Abs, Eq, Symbol
from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler


@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def ldc_trainer(cfg: DictConfig) -> None:
    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    # initialize monitoring
    log = PythonLogger(name="ldc")
    log.file_logging()

    # make geometry
    height = 0.1
    width = 0.1
    rec = Rectangle((-width / 2, -height / 2), (width / 2, height / 2))

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
    xx, yy = torch.from_numpy(xx).to(torch.float).to(dist.device), torch.from_numpy(
        yy
    ).to(torch.float).to(dist.device)

    # bc dataloader
    bc_dataloader = GeometryDatapipe(
        geom_objects=[rec],
        batch_size=1,
        num_points=2000,
        sample_type="surface",
        device=dist.device,
        num_workers=1,
        requested_vars=["x", "y"],
    )

    # interior dataloader
    interior_dataloader = GeometryDatapipe(
        geom_objects=[rec],
        batch_size=1,
        num_points=4000,
        sample_type="volume",
        device=dist.device,
        num_workers=1,
        requested_vars=["x", "y", "sdf"],
    )

    for i in range(10000):
        for bc_data, int_data in zip(bc_dataloader, interior_dataloader):

            optimizer.zero_grad()

            # subsample points:
            no_slip = {}
            top_wall = {}
            y_vals = bc_data[0]["y"]
            mask_no_slip = y_vals < height / 2
            mask_top_wall = y_vals == height / 2

            for k in bc_data[0].keys():
                no_slip[k] = (bc_data[0][k][mask_no_slip]).reshape(-1, 1)
                top_wall[k] = (bc_data[0][k][mask_top_wall]).reshape(-1, 1)

            interior = {}
            for k, v in int_data[0].items():
                # set requires_grad to true to enable gradient computation using autodiff
                if k in ["x", "y"]:
                    requires_grad = True
                else:
                    requires_grad = False
                interior[k] = v.reshape(-1, 1).requires_grad_(requires_grad)

            # apply BC constraints
            coords = torch.cat([interior["x"], interior["y"]], dim=1)
            no_slip_out = model(torch.cat([no_slip["x"], no_slip["y"]], dim=1))
            top_wall_out = model(torch.cat([top_wall["x"], top_wall["y"]], dim=1))
            interior_out = model(coords)

            v_no_slip = torch.mean(no_slip_out[:, 1:2] ** 2)
            u_no_slip = torch.mean(no_slip_out[:, 0:1] ** 2)
            u_slip = torch.mean(
                ((top_wall_out[:, 0:1] - 1.0) ** 2)
                * (1 - 20 * torch.abs(top_wall["x"]))
            )  # weight the edges zero.
            v_slip = torch.mean(top_wall_out[:, 1:2] ** 2)

            # apply interior constraints
            phy_loss_dict = phy_inf.forward(
                {
                    "coordinates": coords,
                    "u": interior_out[:, 0:1],
                    "v": interior_out[:, 1:2],
                    "p": interior_out[:, 2:3],
                }
            )

            cont = phy_loss_dict["continuity"] * interior["sdf"]
            mom_x = phy_loss_dict["momentum_x"] * interior["sdf"]
            mom_y = phy_loss_dict["momentum_y"] * interior["sdf"]

            phy_loss = (
                1 * torch.mean(cont**2)
                + 1 * torch.mean(mom_x**2)
                + 1 * torch.mean(mom_y**2)
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
