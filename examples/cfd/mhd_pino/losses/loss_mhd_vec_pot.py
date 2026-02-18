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
from .loss_mhd import LossMHD


class LossMHDVecPot(LossMHD):
    "Calculate loss for MHD equations with vector potential, using original derivatives in paper"

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
        A_weight=1.0,
        Du_weight=1.0,
        Dv_weight=1.0,
        DA_weight=1.0,
        div_B_weight=1.0,
        div_vel_weight=1.0,
        Lx=1.0,
        Ly=1.0,
        tend=1.0,
        use_weighted_mean=False,
        **kwargs,
    ):  # add **kwargs so that we ignore unexpected kwargs when passing a config dict):
        super().__init__(
            nu=nu,
            eta=eta,
            rho0=rho0,
            data_weight=data_weight,
            ic_weight=ic_weight,
            pde_weight=pde_weight,
            constraint_weight=constraint_weight,
            use_data_loss=use_data_loss,
            use_ic_loss=use_ic_loss,
            use_pde_loss=use_pde_loss,
            use_constraint_loss=use_constraint_loss,
            u_weight=u_weight,
            v_weight=v_weight,
            Du_weight=Du_weight,
            Dv_weight=Dv_weight,
            div_B_weight=div_B_weight,
            div_vel_weight=div_vel_weight,
            Lx=Lx,
            Ly=Ly,
            tend=tend,
            use_weighted_mean=use_weighted_mean,
        )

        self.A_weight = A_weight
        self.DA_weight = DA_weight

    def compute_loss(self, pred, true, inputs):
        "Compute weighted loss"
        pred = pred.reshape(true.shape)
        u = pred[..., 0]
        v = pred[..., 1]
        A = pred[..., 2]

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
            Du, Dv, DA = self.mhd_pde(u, v, A)
            loss_pde = self.mhd_pde_loss(Du, Dv, DA)
        else:
            loss_pde = 0

        # Constraints
        if self.use_constraint_loss:
            div_vel, div_B = self.mhd_constraint(u, v, A)
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
        A = pred[..., 2]

        loss_dict = {}

        # Data
        if self.use_data_loss:
            loss_data, loss_u, loss_v, loss_A = self.data_loss(
                pred, true, return_all_losses=True
            )
            loss_dict["loss_data"] = loss_data
            loss_dict["loss_u"] = loss_u
            loss_dict["loss_v"] = loss_v
            loss_dict["loss_A"] = loss_A
        else:
            loss_data = 0
        # IC
        if self.use_ic_loss:
            loss_ic, loss_u_ic, loss_v_ic, loss_A_ic = self.ic_loss(
                pred, inputs, return_all_losses=True
            )
            loss_dict["loss_ic"] = loss_ic
            loss_dict["loss_u_ic"] = loss_u_ic
            loss_dict["loss_v_ic"] = loss_v_ic
            loss_dict["loss_A_ic"] = loss_A_ic
        else:
            loss_ic = 0

        # PDE
        if self.use_pde_loss:
            Du, Dv, DA = self.mhd_pde(u, v, A)
            loss_pde, loss_Du, loss_Dv, loss_DA = self.mhd_pde_loss(
                Du, Dv, DA, return_all_losses=True
            )
            loss_dict["loss_pde"] = loss_pde
            loss_dict["loss_Du"] = loss_Du
            loss_dict["loss_Dv"] = loss_Dv
            loss_dict["loss_DA"] = loss_DA
        else:
            loss_pde = 0

        # Constraints
        if self.use_constraint_loss:
            div_vel, div_B = self.mhd_constraint(u, v, A)
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
        A_pred = pred[..., 2]

        u_true = true[..., 0]
        v_true = true[..., 1]
        A_true = true[..., 2]

        loss_u = lploss(u_pred, u_true)
        loss_v = lploss(v_pred, v_true)
        loss_A = lploss(A_pred, A_true)

        if self.use_weighted_mean:
            weight_sum = self.u_weight + self.v_weight + self.A_weight
        else:
            weight_sum = 1.0

        loss_data = (
            self.u_weight * loss_u + self.v_weight * loss_v + self.A_weight * loss_A
        ) / weight_sum

        if return_all_losses:
            return loss_data, loss_u, loss_v, loss_A
        else:
            return loss_data

    def ic_loss(self, pred, input, return_all_losses=False):
        "Compute initial condition loss"
        lploss = LpLoss(size_average=True)
        ic_pred = pred[:, 0]
        ic_true = input[:, 0, ..., 3:]
        u_ic_pred = ic_pred[..., 0]
        v_ic_pred = ic_pred[..., 1]
        A_ic_pred = ic_pred[..., 2]

        u_ic_true = ic_true[..., 0]
        v_ic_true = ic_true[..., 1]
        A_ic_true = ic_true[..., 2]

        loss_u_ic = lploss(u_ic_pred, u_ic_true)
        loss_v_ic = lploss(v_ic_pred, v_ic_true)
        loss_A_ic = lploss(A_ic_pred, A_ic_true)

        if self.use_weighted_mean:
            weight_sum = self.u_weight + self.v_weight + self.A_weight
        else:
            weight_sum = 1.0

        loss_ic = (
            self.u_weight * loss_u_ic
            + self.v_weight * loss_v_ic
            + self.A_weight * loss_A_ic
        ) / weight_sum

        if return_all_losses:
            return loss_ic, loss_u_ic, loss_v_ic, loss_A_ic
        else:
            return loss_ic

    def mhd_pde_loss(self, Du, Dv, DA, return_all_losses=None):
        "Compute PDE loss"
        Du_val = torch.zeros_like(Du)
        Dv_val = torch.zeros_like(Dv)
        DA_val = torch.zeros_like(DA)

        loss_Du = F.mse_loss(Du, Du_val)
        loss_Dv = F.mse_loss(Dv, Dv_val)
        loss_DA = F.mse_loss(DA, DA_val)

        if self.use_weighted_mean:
            weight_sum = self.Du_weight + self.Dv_weight + self.DA_weight
        else:
            weight_sum = 1.0

        loss_pde = (
            self.Du_weight * loss_Du
            + self.Dv_weight * loss_Dv
            + self.DA_weight * loss_DA
        ) / weight_sum

        if return_all_losses:
            return loss_pde, loss_Du, loss_Dv, loss_DA
        else:
            return loss_pde

    def mhd_constraint(self, u, v, A):
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
        A_h = torch.fft.fftn(A, dim=[2, 3])

        ux_h = self.Du_i(u_h, k_x)
        vy_h = self.Du_i(v_h, k_y)

        Ax_h = self.Du_i(A_h, k_x)
        Ay_h = self.Du_i(A_h, k_y)

        Bx_h = Ay_h
        By_h = -Ax_h

        Bx_x_h = self.Du_i(Bx_h, k_x)
        By_y_h = self.Du_i(By_h, k_y)

        ux = torch.fft.irfftn(ux_h[..., : k_max + 1], dim=[2, 3])
        vy = torch.fft.irfftn(vy_h[..., : k_max + 1], dim=[2, 3])
        Bx_x = torch.fft.irfftn(Bx_x_h[..., : k_max + 1], dim=[2, 3])
        By_y = torch.fft.irfftn(By_y_h[..., : k_max + 1], dim=[2, 3])

        div_vel = ux + vy
        div_B = Bx_x + By_y

        return div_vel, div_B

    def mhd_pde(self, u, v, A, p=None):
        "Compute PDEs for MHD using vector potential"
        batchsize = u.size(0)
        nt = u.size(1)
        nx = u.size(2)
        ny = u.size(3)
        device = u.device
        dt = self.tend / (nt - 1)
        dx = self.Lx / nx
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
        lap = -(k_x**2 + k_y**2)
        lap[..., 0, 0] = -1.0

        # compute fourier transform
        u_h = torch.fft.fftn(u, dim=[2, 3])
        v_h = torch.fft.fftn(v, dim=[2, 3])
        A_h = torch.fft.fftn(A, dim=[2, 3])

        # compute laplacian in fourier space
        ux_h = self.Du_i(u_h, k_x)
        uy_h = self.Du_i(u_h, k_y)
        vx_h = self.Du_i(v_h, k_x)
        vy_h = self.Du_i(v_h, k_y)

        Ax_h = self.Du_i(A_h, k_x)
        Ay_h = self.Du_i(A_h, k_y)

        Bx_h = Ay_h
        By_h = -Ax_h

        B2_h = Bx_h**2 + By_h**2

        Bx_x_h = self.Du_i(Bx_h, k_x)
        Bx_y_h = self.Du_i(Bx_h, k_y)
        By_x_h = self.Du_i(By_h, k_x)
        By_y_h = self.Du_i(By_h, k_y)

        u_lap_h = self.Lap(u_h, k_x, k_y)
        v_lap_h = self.Lap(v_h, k_x, k_y)
        A_lap_h = self.Lap(A_h, k_x, k_y)

        # inverse fourier transform out
        ux = torch.fft.irfftn(ux_h[..., : k_max + 1], dim=[2, 3])
        uy = torch.fft.irfftn(uy_h[..., : k_max + 1], dim=[2, 3])
        vx = torch.fft.irfftn(vx_h[..., : k_max + 1], dim=[2, 3])
        vy = torch.fft.irfftn(vy_h[..., : k_max + 1], dim=[2, 3])
        Ax = torch.fft.irfftn(Ax_h[..., : k_max + 1], dim=[2, 3])
        Ay = torch.fft.irfftn(Ay_h[..., : k_max + 1], dim=[2, 3])
        Bx = torch.fft.irfftn(Bx_h[..., : k_max + 1], dim=[2, 3])
        By = torch.fft.irfftn(By_h[..., : k_max + 1], dim=[2, 3])
        B2 = torch.fft.irfftn(B2_h[..., : k_max + 1], dim=[2, 3])
        Bx_x = torch.fft.irfftn(Bx_x_h[..., : k_max + 1], dim=[2, 3])
        Bx_y = torch.fft.irfftn(Bx_y_h[..., : k_max + 1], dim=[2, 3])
        By_x = torch.fft.irfftn(By_x_h[..., : k_max + 1], dim=[2, 3])
        By_y = torch.fft.irfftn(By_y_h[..., : k_max + 1], dim=[2, 3])
        u_lap = torch.fft.irfftn(u_lap_h[..., : k_max + 1], dim=[2, 3])
        v_lap = torch.fft.irfftn(v_lap_h[..., : k_max + 1], dim=[2, 3])
        A_lap = torch.fft.irfftn(A_lap_h[..., : k_max + 1], dim=[2, 3])

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

        B_grad_Bx = Bx * Bx_x + By * Bx_y
        B_grad_By = Bx * By_x + By * By_y

        vel_grad_A = u * Ax + v * Ay

        u_rhs = (
            -vel_grad_u - ptot_x / self.rho0 + B_grad_Bx / self.rho0 + self.nu * u_lap
        )
        v_rhs = (
            -vel_grad_v - ptot_y / self.rho0 + B_grad_By / self.rho0 + self.nu * v_lap
        )
        A_rhs = -vel_grad_A + self.eta * A_lap

        u_t = self.Du_t(u, dt)
        v_t = self.Du_t(v, dt)
        A_t = self.Du_t(A, dt)

        # Find difference
        Du = u_t - u_rhs[:, 1:-1]
        Dv = v_t - v_rhs[:, 1:-1]
        DA = A_t - A_rhs[:, 1:-1]

        return Du, Dv, DA
