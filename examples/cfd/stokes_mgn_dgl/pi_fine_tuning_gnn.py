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

import os
import time

import hydra
import numpy as np
import torch
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

try:
    import apex
except:
    pass

try:
    import pyvista as pv
except:
    raise ImportError(
        "Stokes Dataset requires the pyvista library. Install with "
        + "pip install pyvista"
    )

from collections import OrderedDict
from typing import Dict, Optional

from physicsnemo.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
)
from physicsnemo.launch.logging.wandb import initialize_wandb
from physicsnemo.models.meshgraphnet import MeshGraphNet
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physicsnemo.sym.eq.spatial_grads.spatial_grads import compute_connectivity_tensor
from sympy import Function, Number, Symbol

from utils import get_dataset, relative_lp_error


class Stokes(PDE):
    """Incompressible Stokes flow"""

    def __init__(self, nu, dim=3):
        # set params
        self.dim = dim

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z}
        if self.dim == 2:
            input_variables.pop("z")

        # velocity componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        if self.dim == 3:
            w = Function("w")(*input_variables)
        else:
            w = Number(0)

        # pressure
        p = Function("p")(*input_variables)

        # kinematic viscosity
        if isinstance(nu, str):
            nu = Function(nu)(*input_variables)
        elif isinstance(nu, (float, int)):
            nu = Number(nu)

        # set equations
        self.equations = {}
        self.equations["continuity"] = u.diff(x) + v.diff(y) + w.diff(z)
        self.equations["momentum_x"] = +p.diff(x) - nu * (
            u.diff(x).diff(x) + u.diff(y).diff(y) + u.diff(z).diff(z)
        )
        self.equations["momentum_y"] = +p.diff(y) - nu * (
            v.diff(x).diff(x) + v.diff(y).diff(y) + v.diff(z).diff(z)
        )
        self.equations["momentum_z"] = +p.diff(z) - nu * (
            w.diff(x).diff(x) + w.diff(y).diff(y) + w.diff(z).diff(z)
        )

        if self.dim == 2:
            self.equations.pop("momentum_z")


class PhysicsInformedFineTuner:
    """
    Class to define all the physics informed utils and inference.
    """

    def __init__(
        self,
        cfg,
        device,
        gnn_u,
        gnn_v,
        gnn_p,
        coords,
        coords_inflow,
        coords_noslip,
        nu,
        ref_u,
        ref_v,
        ref_p,
        dgl_graph,
    ):
        super().__init__()

        self.device = device
        self.nu = nu
        self.dgl_graph = dgl_graph.to(self.device)
        edge_tensor = torch.stack(
            [dgl_graph.edges()[0], dgl_graph.edges()[1]], dim=1
        ).to(self.device)
        self.connectivity_tensor = compute_connectivity_tensor(
            dgl_graph.nodes(), edge_tensor
        )
        self.connectivity_tensor = tuple(
            t.to(self.device) for t in self.connectivity_tensor
        )

        self.ref_u = torch.tensor(ref_u).float().to(self.device)
        self.ref_v = torch.tensor(ref_v).float().to(self.device)
        self.ref_p = torch.tensor(ref_p).float().to(self.device)

        self.gnn_u = torch.tensor(gnn_u).float().to(self.device)
        self.gnn_v = torch.tensor(gnn_v).float().to(self.device)
        self.gnn_p = torch.tensor(gnn_p).float().to(self.device)

        self.coords = torch.tensor(coords, requires_grad=True).float().to(self.device)
        self.coords_inflow = (
            torch.tensor(coords_inflow, requires_grad=True).float().to(self.device)
        )
        self.coords_noslip = (
            torch.tensor(coords_noslip, requires_grad=True).float().to(self.device)
        )

        self.model = MeshGraphNet(
            cfg.input_dim_nodes
            + 128,  # additional 128 node features from fourier features
            cfg.input_dim_edges,
            cfg.output_dim,
            aggregation=cfg.aggregation,
            hidden_dim_node_encoder=cfg.hidden_dim_node_encoder,
            hidden_dim_edge_encoder=cfg.hidden_dim_edge_encoder,
            hidden_dim_node_decoder=cfg.hidden_dim_node_decoder,
        ).to(self.device)

        self.node_pde = Stokes(nu=self.nu, dim=2)

        # note: this example uses the PhysicsInformer class from PhysicsNeMo Sym to
        # construct the computational graph. This allows you to leverage PhysicsNeMo Sym's
        # optimized derivative backend to compute the derivatives, along with other
        # benefits like symbolic definition of PDEs and leveraging the PDEs from PhysicsNeMo
        # Sym's PDE module.

        self.phy_informer = PhysicsInformer(
            required_outputs=["continuity", "momentum_x", "momentum_y"],
            equations=self.node_pde,
            grad_method="least_squares",
            device=self.device,
            compute_connectivity=False,
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.pi_lr,
            fused=True if torch.cuda.is_available() else False,
        )

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.99935
        )

    def parabolic_inflow(self, y, U_max=0.3):
        u = 4 * U_max * y * (0.4 - y) / (0.4**2)
        v = torch.zeros_like(y)
        return u, v

    def loss(self):
        out = self.model(
            self.dgl_graph.ndata["x"], self.dgl_graph.edata["x"], self.dgl_graph
        )

        # inflow points
        mask_inflow = (
            self.dgl_graph.ndata["marker"]
            == torch.tensor([0, 1, 0, 0, 0]).to(self.device)
        ).all(dim=1)
        results_inflow = {
            k: out[:, i : i + 1][mask_inflow] for i, k in enumerate(["u", "v", "p"])
        }
        pred_u_in, pred_v_in = results_inflow["u"], results_inflow["v"]

        # no-slip points
        mask_1 = (
            self.dgl_graph.ndata["marker"]
            == torch.tensor([0, 0, 0, 1, 0]).to(self.device)
        ).all(dim=1)
        mask_2 = (
            self.dgl_graph.ndata["marker"]
            == torch.tensor([0, 0, 0, 0, 1]).to(self.device)
        ).all(dim=1)
        mask_noslip = torch.logical_or(mask_1, mask_2)
        results_noslip = {
            k: out[:, i : i + 1][mask_noslip] for i, k in enumerate(["u", "v", "p"])
        }
        pred_u_noslip, pred_v_noslip = results_noslip["u"], results_noslip["v"]

        # interior points
        mask_int = (
            self.dgl_graph.ndata["marker"]
            == torch.tensor([1, 0, 0, 0, 0]).to(self.device)
        ).all(dim=1)
        model_out = {
            k: out[:, i : i + 1][mask_int] for i, k in enumerate(["u", "v", "p"])
        }
        results_int = self.phy_informer.forward(
            {
                "coordinates": self.dgl_graph.ndata["pos"][:, 0:2],
                "u": out[:, 0:1],
                "v": out[:, 1:2],
                "p": out[:, 2:3],
                "connectivity_tensor": self.connectivity_tensor,
            }
        )
        pred_mom_u, pred_mom_v, pred_cont = (
            results_int["momentum_x"][mask_int],
            results_int["momentum_y"][mask_int],
            results_int["continuity"][mask_int],
        )
        pred_u, pred_v, pred_p = model_out["u"], model_out["v"], model_out["p"]

        u_in, v_in = self.parabolic_inflow(self.coords_inflow[:, 1:2])

        # Compute losses
        # data loss
        loss_u = torch.mean((self.gnn_u[mask_int] - pred_u) ** 2)
        loss_v = torch.mean((self.gnn_v[mask_int] - pred_v) ** 2)
        loss_p = torch.mean((self.gnn_p[mask_int] - pred_p) ** 2)

        # inflow boundary condition loss
        loss_u_in = torch.mean((u_in - pred_u_in) ** 2)
        loss_v_in = torch.mean((v_in - pred_v_in) ** 2)

        # noslip boundary condition loss
        loss_u_noslip = torch.mean(pred_u_noslip**2)
        loss_v_noslip = torch.mean(pred_v_noslip**2)

        # pde loss
        loss_mom_u = torch.mean(pred_mom_u**2)
        loss_mom_v = torch.mean(pred_mom_v**2)
        loss_cont = torch.mean(pred_cont**2)

        return (
            loss_u,
            loss_v,
            loss_p,
            loss_u_in,
            loss_v_in,
            loss_u_noslip,
            loss_v_noslip,
            loss_mom_u,
            loss_mom_v,
            loss_cont,
        )

    def train(self):
        """PINN based fine-tuning"""
        (
            loss_u,
            loss_v,
            loss_p,
            loss_u_in,
            loss_v_in,
            loss_u_noslip,
            loss_v_noslip,
            loss_mom_u,
            loss_mom_v,
            loss_cont,
        ) = self.loss()

        # Add custom weights to the different losses. The weights are chosen after
        # investigating the relative magnitudes of individual losses and their
        # convergence behavior.
        loss = (
            1 * loss_u
            + 1 * loss_v
            + 1 * loss_p
            + 10 * loss_u_in
            + 10 * loss_v_in
            + 10 * loss_u_noslip
            + 10 * loss_v_noslip
            + 1 * loss_mom_u
            + 1 * loss_mom_v
            + 10 * loss_cont
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return (
            loss_u,
            loss_v,
            loss_p,
            loss_u_in,
            loss_v_in,
            loss_u_noslip,
            loss_v_noslip,
            loss_mom_u,
            loss_mom_v,
            loss_cont,
            self.optimizer.param_groups[0]["lr"],
        )

    def validation(self):
        """Validation during the PINN fine-tuning step"""
        self.model.eval()
        with torch.no_grad():
            out = self.model(
                self.dgl_graph.ndata["x"], self.dgl_graph.edata["x"], self.dgl_graph
            )
            model_out = {k: out[:, i : i + 1] for i, k in enumerate(["u", "v", "p"])}
            pred_u, pred_v, pred_p = (
                model_out["u"],
                model_out["v"],
                model_out["p"],
            )
            error_u = torch.linalg.norm(self.ref_u - pred_u) / torch.linalg.norm(
                self.ref_u
            )
            error_v = torch.linalg.norm(self.ref_v - pred_v) / torch.linalg.norm(
                self.ref_v
            )
            error_p = torch.linalg.norm(self.ref_p - pred_p) / torch.linalg.norm(
                self.ref_p
            )
            wandb.log(
                {
                    "test_u_error (%)": error_u.detach().cpu().numpy(),
                    "test_v_error (%)": error_v.detach().cpu().numpy(),
                    "test_p_error (%)": error_p.detach().cpu().numpy(),
                }
            )
            return error_u, error_v, error_p


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # CUDA support
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # initialize loggers
    initialize_wandb(
        project="PhysicsNeMo-Launch",
        entity="PhysicsNeMo",
        name="Stokes-Physics-Informed-Fine-Tuning",
        group="Stokes-DDP-Group",
        mode=cfg.wandb_mode,
    )

    logger = PythonLogger("main")  # General python logger
    logger.file_logging()

    # Get dataset
    path = os.path.join(to_absolute_path(cfg.results_dir), cfg.graph_path)

    # get_dataset() function here provides the true values (ref_*) and the gnn
    # predictions (gnn_*) along with other data required for the PINN training.
    (
        ref_u,
        ref_v,
        ref_p,
        gnn_u,
        gnn_v,
        gnn_p,
        coords,
        coords_inflow,
        coords_outflow,
        coords_wall,
        coords_polygon,
        nu,
        dgl_graph,
    ) = get_dataset(path, return_graph=True)
    coords_noslip = np.concatenate([coords_wall, coords_polygon], axis=0)

    dgl_graph = dgl_graph.to(device)

    # Initialize model
    pi_fine_tuner = PhysicsInformedFineTuner(
        cfg,
        device,
        gnn_u,
        gnn_v,
        gnn_p,
        coords,
        coords_inflow,
        coords_noslip,
        nu,
        ref_u,
        ref_v,
        ref_p,
        dgl_graph,
    )

    logger.info("Inference (with physics-informed training for fine-tuning) started...")
    for iters in range(cfg.pi_iters):
        # Start timing the iteration
        start_iter_time = time.time()

        (
            loss_u,
            loss_v,
            loss_p,
            loss_u_in,
            loss_v_in,
            loss_u_noslip,
            loss_v_noslip,
            loss_mom_u,
            loss_mom_v,
            loss_cont,
            current_lr,
        ) = pi_fine_tuner.train()

        if iters % 100 == 0:
            error_u, error_v, error_p = pi_fine_tuner.validation()

            # Print losses
            logger.info(f"Iteration: {iters}")
            logger.info(f"Loss u: {loss_u.detach().cpu().numpy():.3e}")
            logger.info(f"Loss v: {loss_v.detach().cpu().numpy():.3e}")
            logger.info(f"Loss p: {loss_p.detach().cpu().numpy():.3e}")
            logger.info(f"Loss u_in: {loss_u_in.detach().cpu().numpy():.3e}")
            logger.info(f"Loss v_in: {loss_v_in.detach().cpu().numpy():.3e}")
            logger.info(f"Loss u noslip: {loss_u_noslip.detach().cpu().numpy():.3e}")
            logger.info(f"Loss v noslip: {loss_v_noslip.detach().cpu().numpy():.3e}")
            logger.info(f"Loss momentum u: {loss_mom_u.detach().cpu().numpy():.3e}")
            logger.info(f"Loss momentum v: {loss_mom_v.detach().cpu().numpy():.3e}")
            logger.info(f"Loss continuity: {loss_cont.detach().cpu().numpy():.3e}")
            logger.info(f"Learning Rate: {current_lr}")

            # Print errors
            logger.info(f"Error u: {error_u:.3e}")
            logger.info(f"Error v: {error_v:.3e}")
            logger.info(f"Error p: {error_p:.3e}")

            # Print iteration time
            end_iter_time = time.time()
            logger.info(
                f"This iteration took {end_iter_time - start_iter_time:.2f} seconds"
            )
            logger.info("-" * 50)  # Add a separator for clarity

    logger.info("Physics-informed fine-tuning training completed!")

    # Save results
    # Final inference call after fine-tuning predictions using the PINN model
    with torch.no_grad():
        out = pi_fine_tuner.model(dgl_graph.ndata["x"], dgl_graph.edata["x"], dgl_graph)
        results_int_inf = {k: out[:, i : i + 1] for i, k in enumerate(["u", "v", "p"])}
        pred_u_inf, pred_v_inf, pred_p_inf = (
            results_int_inf["u"],
            results_int_inf["v"],
            results_int_inf["p"],
        )

        pred_u_inf = pred_u_inf.detach().cpu().numpy()
        pred_v_inf = pred_v_inf.detach().cpu().numpy()
        pred_p_inf = pred_p_inf.detach().cpu().numpy()

        polydata = pv.read(path)
        polydata["filtered_u"] = pred_u_inf
        polydata["filtered_v"] = pred_v_inf
        polydata["filtered_p"] = pred_p_inf
        print(path)
        polydata.save(path)

    logger.info("Inference completed!")


if __name__ == "__main__":
    main()
