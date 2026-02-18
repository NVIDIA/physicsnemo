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

import numpy as np
import torch
import torch.nn.functional as F
import math
from .losses import LpLoss


class LossMHD(object):
    "Calculate loss for MHD equations with magnetic field, using original derivatives in paper"

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
    ):  # add **kwargs so that we ignore unexpected kwargs when passing a config dict
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
        k_max = nx // 2
        k_x = (
            2
            * np.pi
            / self.Lx
            * torch.cat(
                [
                    torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device),
                ],
                0,
            )
            .reshape(nx, 1)
            .repeat(1, ny)
            .reshape(1, 1, nx, ny)
        )
        k_y = (
            2
            * np.pi
            / self.Ly
            * torch.cat(
                [
                    torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device),
                ],
                0,
            )
            .reshape(1, ny)
            .repeat(nx, 1)
            .reshape(1, 1, nx, ny)
        )

        u_h = torch.fft.fftn(u, dim=[2, 3])
        v_h = torch.fft.fftn(v, dim=[2, 3])
        Bx_h = torch.fft.fftn(Bx, dim=[2, 3])
        By_h = torch.fft.fftn(By, dim=[2, 3])

        ux_h = self.Du_i(u_h, k_x)
        vy_h = self.Du_i(v_h, k_y)
        Bx_x_h = self.Du_i(Bx_h, k_x)
        By_y_h = self.Du_i(By_h, k_y)

        ux = torch.fft.irfftn(ux_h[..., : k_max + 1], dim=[2, 3])
        vy = torch.fft.irfftn(vy_h[..., : k_max + 1], dim=[2, 3])
        Bx_x = torch.fft.irfftn(Bx_x_h[..., : k_max + 1], dim=[2, 3])
        By_y = torch.fft.irfftn(By_y_h[..., : k_max + 1], dim=[2, 3])

        div_vel = ux + vy
        div_B = Bx_x + By_y
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
        k_max = nx // 2
        # make wavenumbers
        k_x = (
            2
            * np.pi
            / self.Lx
            * torch.cat(
                [
                    torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device),
                ],
                0,
            )
            .reshape(nx, 1)
            .repeat(1, ny)
            .reshape(1, 1, nx, ny)
        )
        k_y = (
            2
            * np.pi
            / self.Ly
            * torch.cat(
                [
                    torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device),
                ],
                0,
            )
            .reshape(1, ny)
            .repeat(nx, 1)
            .reshape(1, 1, nx, ny)
        )

        B2 = Bx**2 + By**2

        # compute fourier transform
        u_h = torch.fft.fftn(u, dim=[2, 3])
        v_h = torch.fft.fftn(v, dim=[2, 3])
        Bx_h = torch.fft.fftn(Bx, dim=[2, 3])
        By_h = torch.fft.fftn(By, dim=[2, 3])
        B2_h = torch.fft.fftn(B2, dim=[2, 3])

        # compute laplacian in fourier space
        ux_h = self.Du_i(u_h, k_x)
        uy_h = self.Du_i(u_h, k_y)
        vx_h = self.Du_i(v_h, k_x)
        vy_h = self.Du_i(v_h, k_y)

        Bx_x_h = self.Du_i(Bx_h, k_x)
        Bx_y_h = self.Du_i(Bx_h, k_y)
        By_x_h = self.Du_i(By_h, k_x)
        By_y_h = self.Du_i(By_h, k_y)

        u_lap_h = self.Lap(u_h, k_x, k_y)
        v_lap_h = self.Lap(v_h, k_x, k_y)
        Bx_lap_h = self.Lap(Bx_h, k_x, k_y)
        By_lap_h = self.Lap(By_h, k_x, k_y)

        u_lap_x_h = self.Du_i(u_lap_h, k_x)
        v_lap_y_h = self.Du_i(v_lap_h, k_y)
        div_vel_lap_h = u_lap_x_h + v_lap_y_h
        vel_grad_u_h = u_h * ux_h + v_h * uy_h
        vel_grad_v_h = u_h * vx_h + v_h * vy_h

        vel_grad_u_x_h = self.Du_i(vel_grad_u_h, k_x)
        vel_grad_v_y_h = self.Du_i(vel_grad_v_h, k_y)

        div_vel_h = ux_h + vy_h

        # note that for pressure, the zero mode (the mean) cannot be zero for invertability so it is set to 1
        lap = -(k_x**2 + k_y**2)
        lap[..., 0, 0] = -1.0

        # inverse fourier transform out
        ux = torch.fft.irfftn(ux_h[..., : k_max + 1], dim=[2, 3])
        uy = torch.fft.irfftn(uy_h[..., : k_max + 1], dim=[2, 3])
        vx = torch.fft.irfftn(vx_h[..., : k_max + 1], dim=[2, 3])
        vy = torch.fft.irfftn(vy_h[..., : k_max + 1], dim=[2, 3])
        Bx_x = torch.fft.irfftn(Bx_x_h[..., : k_max + 1], dim=[2, 3])
        Bx_y = torch.fft.irfftn(Bx_y_h[..., : k_max + 1], dim=[2, 3])
        By_x = torch.fft.irfftn(By_x_h[..., : k_max + 1], dim=[2, 3])
        By_y = torch.fft.irfftn(By_y_h[..., : k_max + 1], dim=[2, 3])
        u_lap = torch.fft.irfftn(u_lap_h[..., : k_max + 1], dim=[2, 3])
        v_lap = torch.fft.irfftn(v_lap_h[..., : k_max + 1], dim=[2, 3])
        Bx_lap = torch.fft.irfftn(Bx_lap_h[..., : k_max + 1], dim=[2, 3])
        By_lap = torch.fft.irfftn(By_lap_h[..., : k_max + 1], dim=[2, 3])

        # calculate derivatives for ptot
        if p is None:
            div_vel_grad_vel = ux**2 + 2 * uy * vx + vy**2
            div_B_grad_B = Bx_x**2 + 2 * Bx_y * By_x + By_y**2
            div_vel_grad_vel_h = torch.fft.fftn(div_vel_grad_vel, dim=[2, 3])
            div_B_grad_B_h = torch.fft.fftn(div_B_grad_B, dim=[2, 3])
            ptot_h = (div_B_grad_B_h - self.rho0 * div_vel_grad_vel_h) / lap
            ptot_h[..., 0, 0] = B2_h[..., 0, 0] / 2.0
            p_h = ptot_h - B2_h / 2.0
        else:
            p_h = torch.fft.fftn(p, dim=[2, 3])
            ptot_h = p_h + B2_h / 2.0
        ptot_x_h = self.Du_i(ptot_h, k_x)
        ptot_y_h = self.Du_i(ptot_h, k_y)

        p = torch.fft.irfftn(p_h[..., : k_max + 1], dim=[2, 3])
        ptot = torch.fft.irfftn(ptot_h[..., : k_max + 1], dim=[2, 3])
        ptot_x = torch.fft.irfftn(ptot_x_h[..., : k_max + 1], dim=[2, 3])
        ptot_y = torch.fft.irfftn(ptot_y_h[..., : k_max + 1], dim=[2, 3])

        # Substitute values into PDE equations
        vel_grad_u = u * ux + v * uy
        vel_grad_v = u * vx + v * vy

        B_grad_u = Bx * ux + By * uy
        B_grad_v = Bx * vx + By * vy

        vel_grad_Bx = u * Bx_x + v * Bx_y
        vel_grad_By = u * By_x + v * By_y

        B_grad_Bx = Bx * Bx_x + By * Bx_y
        B_grad_By = Bx * By_x + By * By_y

        uBy_x = u * By_x + By * ux
        vBx_x = v * Bx_x + Bx * vx
        uBy_y = u * By_y + By * uy
        vBx_y = v * Bx_y + Bx * vy

        div_B = Bx_x + By_y
        div_vel = ux + vy

        curl_vel_cross_B_x = uBy_y - vBx_y
        curl_vel_cross_B_y = -(uBy_x - vBx_x)

        u_rhs = (
            -vel_grad_u - ptot_x / self.rho0 + B_grad_Bx / self.rho0 + self.nu * u_lap
        )
        v_rhs = (
            -vel_grad_v - ptot_y / self.rho0 + B_grad_By / self.rho0 + self.nu * v_lap
        )

        Bx_rhs = curl_vel_cross_B_x + self.eta * Bx_lap
        By_rhs = curl_vel_cross_B_y + self.eta * By_lap

        u_t = self.Du_t(u, dt)
        v_t = self.Du_t(v, dt)
        Bx_t = self.Du_t(Bx, dt)
        By_t = self.Du_t(By, dt)

        # Find difference
        Du = u_t - u_rhs[:, 1:-1]
        Dv = v_t - v_rhs[:, 1:-1]
        DBx = Bx_t - Bx_rhs[:, 1:-1]
        DBy = By_t - By_rhs[:, 1:-1]

        return Du, Dv, DBx, DBy

    def Du_t(self, u, dt):
        "Compute time derivative"
        u_t = (u[:, 2:] - u[:, :-2]) / (2 * dt)
        return u_t

    def Lap(self, u_h, k_i, k_j):
        "Helper method to calculate laplacian of variable"
        lap = -(k_i**2 + k_j**2)
        u_lap_h = lap * u_h
        return u_lap_h

    def Du_i(self, u_h, k_i):
        u_i_h = (1j * k_i) * u_h
        return u_i_h

    def Du_ij(self, u_h, k_i, k_j):
        u_ij_h = (1j * k_i) * (1j * k_j) * u_h
        return u_ij_h

    def Du_ii(self, u_h, k_i):
        u_ii_h = self.Du_ij(u_h, k_i, k_i)
        return u_ii_h
