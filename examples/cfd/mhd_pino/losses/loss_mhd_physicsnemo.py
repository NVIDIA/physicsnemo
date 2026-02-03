# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

import numpy as np
import torch
import torch.nn.functional as F
import math
from .losses import LpLoss, fourier_derivatives_lap, fourier_derivatives_ptot
from .mhd_pde import MHD_PDE
from physicsnemo.nn.module.spectral_layers import fourier_derivatives


class LossMHD_PhysicsNeMo(object):
    "Calculate loss for MHD equations with magnetic field, using physicsnemo derivatives"

    def __init__(
        self,
        nu=1e-4,
        eta=1e-4,
        rho0=1.0,
        data_weight=1.0,
        ic_weight=1.0,
        pde_weight=1.0,
        constraint_weight=1.0,
        use_data_loss=True,
        use_ic_loss=True,
        use_pde_loss=True,
        use_constraint_loss=True,
        u_weight=1.0,
        v_weight=1.0,
        Bx_weight=1.0,
        By_weight=1.0,
        Du_weight=1.0,
        Dv_weight=1.0,
        DBx_weight=1.0,
        DBy_weight=1.0,
        div_B_weight=1.0,
        div_vel_weight=1.0,
        Lx=1.0,
        Ly=1.0,
        tend=1.0,
        use_weighted_mean=False,
        **kwargs,
    ):  # add **kwards so that we ignore unexpected kwargs when passing a config dict
        self.nu = nu
        self.eta = eta
        self.rho0 = rho0
        self.data_weight = data_weight
        self.ic_weight = ic_weight
        self.pde_weight = pde_weight
        self.constraint_weight = constraint_weight
        self.use_data_loss = use_data_loss
        self.use_ic_loss = use_ic_loss
        self.use_pde_loss = use_pde_loss
        self.use_constraint_loss = use_constraint_loss
        self.u_weight = u_weight
        self.v_weight = v_weight
        self.Bx_weight = Bx_weight
        self.By_weight = By_weight
        self.Du_weight = Du_weight
        self.Dv_weight = Dv_weight
        self.DBx_weight = DBx_weight
        self.DBy_weight = DBy_weight
        self.div_B_weight = div_B_weight
        self.div_vel_weight = div_vel_weight
        self.Lx = Lx
        self.Ly = Ly
        self.tend = tend
        self.use_weighted_mean = use_weighted_mean
        # Define 2D MHD PDEs
        self.mhd_pde_eq = MHD_PDE(self.nu, self.eta, self.rho0)
        self.mhd_pde_node = self.mhd_pde_eq.make_nodes()

        if not self.use_data_loss:
            self.data_weight = 0
        if not self.use_ic_loss:
            self.ic_weight = 0
        if not self.use_pde_loss:
            self.pde_weight = 0
        if not self.use_constraint_loss:
            self.constraint_weight = 0

    def __call__(self, pred, true, inputs, return_loss_dict=False):
        if not return_loss_dict:
            loss = self.compute_loss(pred, true, inputs)
            return loss
        else:
            loss, loss_dict = self.compute_losses(pred, true, inputs)
            return loss, loss_dict

    def compute_loss(self, pred, true, inputs):
        "Compute weighted loss"
        pred = pred.reshape(true.shape)
        u = pred[..., 0]
        v = pred[..., 1]
        Bx = pred[..., 2]
        By = pred[..., 3]

        # Data
        if self.use_data_loss:
            loss_data = self.data_loss(pred, true)
        else:
            loss_data = 0
        # IC
        if self.use_ic_loss:
            loss_ic = self.ic_loss(pred, inputs)
        else:
            loss_ic = 0

        # PDE
        if self.use_pde_loss:
            Du, Dv, DBx, DBy = self.mhd_pde(u, v, Bx, By)
            loss_pde = self.mhd_pde_loss(Du, Dv, DBx, DBy)
        else:
            loss_pde = 0

        # Constraints
        if self.use_constraint_loss:
            div_vel, div_B = self.mhd_constraint(u, v, Bx, By)
            loss_constraint = self.mhd_constraint_loss(div_vel, div_B)
        else:
            loss_constraint = 0

        if self.use_weighted_mean:
            weight_sum = (
                self.data_weight
                + self.ic_weight
                + self.pde_weight
                + self.constraint_weight
            )
        else:
            weight_sum = 1.0

        loss = (
            self.data_weight * loss_data
            + self.ic_weight * loss_ic
            + self.pde_weight * loss_pde
            + self.constraint_weight * loss_constraint
        ) / weight_sum
        return loss

    def compute_losses(self, pred, true, inputs):
        "Compute weighted loss and dictionary"
        pred = pred.reshape(true.shape)
        u = pred[..., 0]
        v = pred[..., 1]
        Bx = pred[..., 2]
        By = pred[..., 3]

        loss_dict = {}

        # Data
        if self.use_data_loss:
            loss_data, loss_u, loss_v, loss_Bx, loss_By = self.data_loss(
                pred, true, return_all_losses=True
            )
            loss_dict["loss_data"] = loss_data
            loss_dict["loss_u"] = loss_u
            loss_dict["loss_v"] = loss_v
            loss_dict["loss_Bx"] = loss_Bx
            loss_dict["loss_By"] = loss_By
        else:
            loss_data = 0
        # IC
        if self.use_ic_loss:
            loss_ic, loss_u_ic, loss_v_ic, loss_Bx_ic, loss_By_ic = self.ic_loss(
                pred, inputs, return_all_losses=True
            )
            loss_dict["loss_ic"] = loss_ic
            loss_dict["loss_u_ic"] = loss_u_ic
            loss_dict["loss_v_ic"] = loss_v_ic
            loss_dict["loss_Bx_ic"] = loss_Bx_ic
            loss_dict["loss_By_ic"] = loss_By_ic
        else:
            loss_ic = 0

        # PDE
        if self.use_pde_loss:
            Du, Dv, DBx, DBy = self.mhd_pde(u, v, Bx, By)
            loss_pde, loss_Du, loss_Dv, loss_DBx, loss_DBy = self.mhd_pde_loss(
                Du, Dv, DBx, DBy, return_all_losses=True
            )
            loss_dict["loss_pde"] = loss_pde
            loss_dict["loss_Du"] = loss_Du
            loss_dict["loss_Dv"] = loss_Dv
            loss_dict["loss_DBx"] = loss_DBx
            loss_dict["loss_DBy"] = loss_DBy
        else:
            loss_pde = 0

        # Constraints
        if self.use_constraint_loss:
            div_vel, div_B = self.mhd_constraint(u, v, Bx, By)
            loss_constraint, loss_div_vel, loss_div_B = self.mhd_constraint_loss(
                div_vel, div_B, return_all_losses=True
            )
            loss_dict["loss_constraint"] = loss_constraint
            loss_dict["loss_div_vel"] = loss_div_vel
            loss_dict["loss_div_B"] = loss_div_B
        else:
            loss_constraint = 0

        if self.use_weighted_mean:
            weight_sum = (
                self.data_weight
                + self.ic_weight
                + self.pde_weight
                + self.constraint_weight
            )
        else:
            weight_sum = 1.0

        loss = (
            self.data_weight * loss_data
            + self.ic_weight * loss_ic
            + self.pde_weight * loss_pde
            + self.constraint_weight * loss_constraint
        ) / weight_sum
        loss_dict["loss"] = loss
        return loss, loss_dict

    def data_loss(self, pred, true, return_all_losses=False):
        "Compute data loss"
        lploss = LpLoss(size_average=True)
        u_pred = pred[..., 0]
        v_pred = pred[..., 1]
        Bx_pred = pred[..., 2]
        By_pred = pred[..., 3]

        u_true = true[..., 0]
        v_true = true[..., 1]
        Bx_true = true[..., 2]
        By_true = true[..., 3]

        loss_u = lploss(u_pred, u_true)
        loss_v = lploss(v_pred, v_true)
        loss_Bx = lploss(Bx_pred, Bx_true)
        loss_By = lploss(By_pred, By_true)

        if self.use_weighted_mean:
            weight_sum = self.u_weight + self.v_weight + self.Bx_weight + self.By_weight
        else:
            weight_sum = 1.0

        loss_data = (
            self.u_weight * loss_u
            + self.v_weight * loss_v
            + self.Bx_weight * loss_Bx
            + self.By_weight * loss_By
        ) / weight_sum

        if return_all_losses:
            return loss_data, loss_u, loss_v, loss_Bx, loss_By
        else:
            return loss_data

    def ic_loss(self, pred, inputs, return_all_losses=False):
        "Compute initial condition loss"
        lploss = LpLoss(size_average=True)
        ic_pred = pred[:, 0]
        ic_true = inputs[:, 0, ..., 3:]
        u_ic_pred = ic_pred[..., 0]
        v_ic_pred = ic_pred[..., 1]
        Bx_ic_pred = ic_pred[..., 2]
        By_ic_pred = ic_pred[..., 3]

        u_ic_true = ic_true[..., 0]
        v_ic_true = ic_true[..., 1]
        Bx_ic_true = ic_true[..., 2]
        By_ic_true = ic_true[..., 3]

        loss_u_ic = lploss(u_ic_pred, u_ic_true)
        loss_v_ic = lploss(v_ic_pred, v_ic_true)
        loss_Bx_ic = lploss(Bx_ic_pred, Bx_ic_true)
        loss_By_ic = lploss(By_ic_pred, By_ic_true)

        if self.use_weighted_mean:
            weight_sum = weight_sum = (
                self.u_weight + self.v_weight + self.Bx_weight + self.By_weight
            )
        else:
            weight_sum = 1.0

        loss_ic = (
            self.u_weight * loss_u_ic
            + self.v_weight * loss_v_ic
            + self.Bx_weight * loss_Bx_ic
            + self.By_weight * loss_By_ic
        ) / weight_sum

        if return_all_losses:
            return loss_ic, loss_u_ic, loss_v_ic, loss_Bx_ic, loss_By_ic
        else:
            return loss_ic

    def mhd_pde_loss(self, Du, Dv, DBx, DBy, return_all_losses=None):
        "Compute PDE loss"
        Du_val = torch.zeros_like(Du)
        Dv_val = torch.zeros_like(Dv)
        DBx_val = torch.zeros_like(DBx)
        DBy_val = torch.zeros_like(DBy)

        loss_Du = F.mse_loss(Du, Du_val)
        loss_Dv = F.mse_loss(Dv, Dv_val)
        loss_DBx = F.mse_loss(DBx, DBx_val)
        loss_DBy = F.mse_loss(DBy, DBy_val)

        if self.use_weighted_mean:
            weight_sum = (
                self.Du_weight + self.Dv_weight + self.DBx_weight + self.DBy_weight
            )
        else:
            weight_sum = 1.0

        loss_pde = (
            self.Du_weight * loss_Du
            + self.Dv_weight * loss_Dv
            + self.DBx_weight * loss_DBx
            + self.DBy_weight * loss_DBy
        ) / weight_sum

        if return_all_losses:
            return loss_pde, loss_Du, loss_Dv, loss_DBx, loss_DBy
        else:
            return loss_pde

    def mhd_constraint_loss(self, div_vel, div_B, return_all_losses=False):
        "Compute constraint loss"
        div_vel_val = torch.zeros_like(div_vel)
        div_B_val = torch.zeros_like(div_B)

        loss_div_vel = F.mse_loss(div_vel, div_vel_val)
        loss_div_B = F.mse_loss(div_B, div_B_val)

        if self.use_weighted_mean:
            weight_sum = self.div_vel_weight + self.div_B_weight
        else:
            weight_sum = 1.0

        loss_constraint = (
            self.div_vel_weight * loss_div_vel + self.div_B_weight * loss_div_B
        ) / weight_sum

        if return_all_losses:
            return loss_constraint, loss_div_vel, loss_div_B
        else:
            return loss_constraint

    def mhd_constraint(self, u, v, Bx, By):
        "Compute constraints"
        batchsize = u.size(0)
        nt = u.size(1)
        nx = u.size(2)
        ny = u.size(3)
        device = u.device
        dt = self.tend / (nt - 1)
        dx = self.Lx / nx
        dy = self.Ly / ny

        f_du, _ = fourier_derivatives(u, [self.Lx, self.Ly])
        f_dv, _ = fourier_derivatives(v, [self.Lx, self.Ly])
        f_dBx, _ = fourier_derivatives(Bx, [self.Lx, self.Ly])
        f_dBy, _ = fourier_derivatives(By, [self.Lx, self.Ly])

        u_x = f_du[:, 0:nt, :nx, :ny]
        v_y = f_dv[:, nt : 2 * nt, :nx, :ny]
        Bx_x = f_dBx[:, 0:nt, :nx, :ny]
        By_y = f_dBy[:, nt : 2 * nt, :nx, :ny]

        div_B = self.mhd_pde_node[12].evaluate({"Bx__x": Bx_x, "By__y": By_y})["div_B"]
        div_vel = self.mhd_pde_node[13].evaluate({"u__x": u_x, "v__y": v_y})["div_vel"]

        return div_vel, div_B

    def mhd_pde(self, u, v, Bx, By, p=None):
        "Compute PDEs for MHD using magnetic field"
        batchsize = u.size(0)
        nt = u.size(1)
        nx = u.size(2)
        ny = u.size(3)
        device = u.device
        dt = self.tend / (nt - 1)
        dx = self.Lx / nx
        dy = self.Ly / ny

        B2 = Bx**2 + By**2
        B2_h = torch.fft.fftn(B2, dim=[2, 3])

        # compute fourier derivatives
        f_du, _ = fourier_derivatives(u, [self.Lx, self.Ly])
        f_dv, _ = fourier_derivatives(v, [self.Lx, self.Ly])
        f_dBx, _ = fourier_derivatives(Bx, [self.Lx, self.Ly])
        f_dBy, _ = fourier_derivatives(By, [self.Lx, self.Ly])

        u_x = f_du[:, 0:nt, :nx, :ny]
        u_y = f_du[:, nt : 2 * nt, :nx, :ny]
        v_x = f_dv[:, 0:nt, :nx, :ny]
        v_y = f_dv[:, nt : 2 * nt, :nx, :ny]
        Bx_x = f_dBx[:, 0:nt, :nx, :ny]
        Bx_y = f_dBx[:, nt : 2 * nt, :nx, :ny]
        By_x = f_dBy[:, 0:nt, :nx, :ny]
        By_y = f_dBy[:, nt : 2 * nt, :nx, :ny]

        u_lap = fourier_derivatives_lap(u, [self.Lx, self.Ly])
        v_lap = fourier_derivatives_lap(v, [self.Lx, self.Ly])
        Bx_lap = fourier_derivatives_lap(Bx, [self.Lx, self.Ly])
        By_lap = fourier_derivatives_lap(By, [self.Lx, self.Ly])

        # note that for pressure, the zero mode (the mean) cannot be zero for invertability so it is set to 1
        div_vel_grad_vel = u_x**2 + 2 * u_y * v_x + v_y**2
        div_B_grad_B = Bx_x**2 + 2 * Bx_y * By_x + By_y**2
        f_dptot = fourier_derivatives_ptot(
            p, div_vel_grad_vel, div_B_grad_B, B2_h, self.rho0, [self.Lx, self.Ly]
        )
        ptot_x = f_dptot[:, 0:nt, :nx, :ny]
        ptot_y = f_dptot[:, nt : 2 * nt, :nx, :ny]

        # Plug inputs into dictionary
        all_inputs = {
            "u": u,
            "u__x": u_x,
            "u__y": u_y,
            "v": v,
            "v__x": v_x,
            "v__y": v_y,
            "Bx": Bx,
            "Bx__x": Bx_x,
            "Bx__y": Bx_y,
            "By": By,
            "By__x": By_x,
            "By__y": By_y,
            "ptot__x": ptot_x,
            "ptot__y": ptot_y,
            "u__lap": u_lap,
            "v__lap": v_lap,
            "Bx__lap": Bx_lap,
            "By__lap": By_lap,
        }

        # Substitute values into PDE equations
        u_rhs = self.mhd_pde_node[14].evaluate(all_inputs)["u_rhs"]
        v_rhs = self.mhd_pde_node[15].evaluate(all_inputs)["v_rhs"]
        Bx_rhs = self.mhd_pde_node[16].evaluate(all_inputs)["Bx_rhs"]
        By_rhs = self.mhd_pde_node[17].evaluate(all_inputs)["By_rhs"]

        u_t = self.Du_t(u, dt)
        v_t = self.Du_t(v, dt)
        Bx_t = self.Du_t(Bx, dt)
        By_t = self.Du_t(By, dt)

        # Find difference
        Du = self.mhd_pde_node[18].evaluate({"u__t": u_t, "u_rhs": u_rhs[:, 1:-1]})[
            "Du"
        ]
        Dv = self.mhd_pde_node[19].evaluate({"v__t": v_t, "v_rhs": v_rhs[:, 1:-1]})[
            "Dv"
        ]
        DBx = self.mhd_pde_node[20].evaluate(
            {"Bx__t": Bx_t, "Bx_rhs": Bx_rhs[:, 1:-1]}
        )["DBx"]
        DBy = self.mhd_pde_node[21].evaluate(
            {"By__t": By_t, "By_rhs": By_rhs[:, 1:-1]}
        )["DBy"]

        return Du, Dv, DBx, DBy

    def Du_t(self, u, dt):
        "Compute time derivative"
        u_t = (u[:, 2:] - u[:, :-2]) / (2 * dt)
        return u_t
