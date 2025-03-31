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
import torch.nn.functional as F
import numpy as np
from physicsnemo.models.layers.spectral_layers import fourier_derivatives


class LpLoss(object):
    "Relative MSE loss used for data loss with multiple equations"

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def swe_loss(s_pred, s_true, H=1.0, use_sum=True):
    "Data Loss"
    h_pred = s_pred[..., 0] - H
    u_pred = s_pred[..., 1]
    v_pred = s_pred[..., 2]
    h_true = s_true[..., 0] - H
    u_true = s_true[..., 1]
    v_true = s_true[..., 2]
    lploss = LpLoss(size_average=True)
    loss_h = lploss(h_pred, h_true)
    loss_u = lploss(u_pred, u_true)
    loss_v = lploss(v_pred, v_true)
    loss_s = torch.stack([loss_h, loss_u, loss_v], dim=-1)
    if use_sum:
        data_loss = torch.sum(loss_s)
    else:
        data_loss = torch.mean(loss_s)
    return data_loss


def ic_loss(out, s0):
    "Initial Condition Loss"
    batchsize = out.size(0)
    nx = out.size(1)
    ny = out.size(2)
    nt = out.size(3)
    s = out.reshape(batchsize, nx, ny, nt, 3)

    lploss = LpLoss(size_average=True)
    s_ic = s[..., 0, 0].reshape(s0.shape)
    loss_ic = lploss(s_ic, s0)
    return loss_ic


def fdm_swe_nonlin(h, u, v, D=1, g=1.0, nu=1.0e-3, device=0):
    "Original method in paper to calculate fourier derivatives"
    batchsize = u.size(0)
    nx = u.size(1)
    ny = u.size(2)
    nt = u.size(3)
    h = h.reshape(batchsize, nx, ny, nt)
    u = u.reshape(batchsize, nx, ny, nt)
    v = v.reshape(batchsize, nx, ny, nt)
    dt = D / (nt - 1)
    dx = D / (nx)

    # Variables to differentiate
    hu = h * u
    hv = h * v
    huu = h * u**2
    huv = h * u * v
    hvv = h * v**2
    hh = h**2

    # Compute fourier transform
    hu_h = torch.fft.fftn(hu, dim=[1, 2])
    hv_h = torch.fft.fftn(hv, dim=[1, 2])
    huu_h = torch.fft.fftn(huu, dim=[1, 2])
    huv_h = torch.fft.fftn(huv, dim=[1, 2])
    hvv_h = torch.fft.fftn(hvv, dim=[1, 2])
    hh_h = torch.fft.fftn(hh, dim=[1, 2])
    u_h = torch.fft.fftn(u, dim=[1, 2])
    v_h = torch.fft.fftn(v, dim=[1, 2])

    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = torch.cat(
        (
            torch.arange(start=0, end=k_max, step=1, device=device),
            torch.arange(start=-k_max, end=0, step=1, device=device),
        ),
        0,
    )
    k_x = k_x[None, :, None, None]
    k_x = k_x.expand(1, N, N, 1).clone()

    k_y = torch.cat(
        (
            torch.arange(start=0, end=k_max, step=1, device=device),
            torch.arange(start=-k_max, end=0, step=1, device=device),
        ),
        0,
    )
    k_y = k_y[None, None, :, None]
    k_y = k_y.expand(1, N, N, 1).clone()

    # Compute laplacian in fourier space
    hux_h = 2j * np.pi * k_x * hu_h
    hvy_h = 2j * np.pi * k_y * hv_h

    huux_h = 2j * np.pi * k_x * huu_h
    hvvy_h = 2j * np.pi * k_y * hvv_h

    huvx_h = 2j * np.pi * k_x * huv_h
    huvy_h = 2j * np.pi * k_y * huv_h

    hhx_h = 2j * np.pi * k_x * hh_h
    hhy_h = 2j * np.pi * k_y * hh_h

    ux_h = 2j * np.pi * k_x * u_h
    uxx_h = 2j * np.pi * k_x * ux_h
    uy_h = 2j * np.pi * k_y * u_h
    uyy_h = 2j * np.pi * k_y * uy_h

    vx_h = 2j * np.pi * k_x * v_h
    vxx_h = 2j * np.pi * k_x * vx_h
    vy_h = 2j * np.pi * k_y * v_h
    vyy_h = 2j * np.pi * k_y * vy_h

    # Inverse fourier transform out and compute time derivatives
    hux = torch.fft.irfftn(hux_h[:, :, : k_max + 1], dim=[1, 2])
    hvy = torch.fft.irfftn(hvy_h[:, :, : k_max + 1], dim=[1, 2])
    huux = torch.fft.irfftn(huux_h[:, :, : k_max + 1], dim=[1, 2])
    hvvy = torch.fft.irfftn(hvvy_h[:, :, : k_max + 1], dim=[1, 2])
    huvx = torch.fft.irfftn(huvx_h[:, :, : k_max + 1], dim=[1, 2])
    huvy = torch.fft.irfftn(huvy_h[:, :, : k_max + 1], dim=[1, 2])
    hhx = torch.fft.irfftn(hhx_h[:, :, : k_max + 1], dim=[1, 2])
    hhy = torch.fft.irfftn(hhy_h[:, :, : k_max + 1], dim=[1, 2])
    ht = (h[..., 2:] - h[..., :-2]) / (2 * dt)

    ux = torch.fft.irfftn(ux_h[:, :, : k_max + 1], dim=[1, 2])
    uy = torch.fft.irfftn(uy_h[:, :, : k_max + 1], dim=[1, 2])
    uxx = torch.fft.irfftn(uxx_h[:, :, : k_max + 1], dim=[1, 2])
    uyy = torch.fft.irfftn(uyy_h[:, :, : k_max + 1], dim=[1, 2])
    ut = (u[..., 2:] - u[..., :-2]) / (2 * dt)

    vx = torch.fft.irfftn(vx_h[:, :, : k_max + 1], dim=[1, 2])
    vy = torch.fft.irfftn(vy_h[:, :, : k_max + 1], dim=[1, 2])
    vxx = torch.fft.irfftn(vxx_h[:, :, : k_max + 1], dim=[1, 2])
    vyy = torch.fft.irfftn(vyy_h[:, :, : k_max + 1], dim=[1, 2])
    vt = (v[..., 2:] - v[..., :-2]) / (2 * dt)

    # Plug into PDEs
    Dh = ht + (hux + hvy)[..., 1:-1]
    Du = ut + ((huux + 0.5 * g * hhx) + huvy - nu * (uxx + uyy))[..., 1:-1]
    Dv = vt + (huvx + (hvvy + 0.5 * g * hhy) - nu * (vxx + vyy))[..., 1:-1]

    return Dh, Du, Dv


def pino_loss_swe_nonlin(
    s, g=1.0, nu=0.001, h_weight=1.0, u_weight=1.0, v_weight=1.0, device=0
):
    "Calculate PDE Loss using fdm_swe_nonlin like original paper"
    batchsize = s.size(0)
    nx = s.size(1)
    ny = s.size(2)
    nt = s.size(3)
    s = s.reshape(batchsize, nx, ny, nt, 3)

    h = s[..., 0]
    u = s[..., 1]
    v = s[..., 2]
    Dh, Du, Dv = fdm_swe_nonlin(h, u, v, g=g, nu=nu, device=device)
    Dh *= h_weight
    Du *= u_weight
    Dv *= v_weight
    Ds = torch.stack([Dh, Du, Dv], dim=-1)
    f_ = torch.zeros(
        Ds.shape, device=s.device
    )  # use f_ to distinguish from corriolous const f
    loss_f = F.mse_loss(Ds, f_)

    return loss_f


def physicsnemo_fdm_swe_nonlin(h, u, v, pde_node, D=1, device=0):
    "Calculate fourier derivatives using physicsnemo"
    batchsize = u.size(0)
    nx = u.size(1)
    ny = u.size(2)
    nt = u.size(3)
    h = h.reshape(batchsize, nx, ny, nt).permute(0, 3, 1, 2)
    u = u.reshape(batchsize, nx, ny, nt).permute(0, 3, 1, 2)
    v = v.reshape(batchsize, nx, ny, nt).permute(0, 3, 1, 2)

    # Compute variables to differentiate
    hu = h * u
    hv = h * v
    hh = h * h
    huu = h * u * u
    huv = h * u * v
    hvv = h * v * v

    # Compute time derivatives
    dt = 1.0 / (nt - 1)
    h_t = (h[:, 2:, ...] - h[:, :-2, ...]) / (2 * dt)
    u_t = (u[:, 2:, ...] - u[:, :-2, ...]) / (2 * dt)
    v_t = (v[:, 2:, ...] - v[:, :-2, ...]) / (2 * dt)

    # Compute fourier derivatives using physicsnemo
    _, f_ddu = fourier_derivatives(u, [1.0, 1.0])
    _, f_ddv = fourier_derivatives(v, [1.0, 1.0])
    f_dhu, _ = fourier_derivatives(hu, [1.0, 1.0])
    f_dhv, _ = fourier_derivatives(hv, [1.0, 1.0])
    f_dhh, _ = fourier_derivatives(hh, [1.0, 1.0])
    f_dhuu, _ = fourier_derivatives(huu, [1.0, 1.0])
    f_dhuv, _ = fourier_derivatives(huv, [1.0, 1.0])
    f_dhvv, _ = fourier_derivatives(hvv, [1.0, 1.0])

    u_x_x = f_ddu[:, 1 : nt - 1, :nx, :ny]
    u_y_y = f_ddu[:, nt + 1 : 2 * nt - 1, :nx, :ny]
    v_x_x = f_ddv[:, 1 : nt - 1, :nx, :ny]
    v_y_y = f_ddv[:, nt + 1 : 2 * nt - 1, :nx, :ny]

    hu_x = f_dhu[:, 1 : nt - 1, :nx, :ny]
    hv_y = f_dhv[:, nt + 1 : 2 * nt - 1, :nx, :ny]
    hh_x = f_dhh[:, 1 : nt - 1, :nx, :ny]
    hh_y = f_dhh[:, nt + 1 : 2 * nt - 1, :nx, :ny]

    huu_x = f_dhuu[:, 1 : nt - 1, :nx, :ny]
    hvv_y = f_dhvv[:, nt + 1 : 2 * nt - 1, :nx, :ny]
    huv_x = f_dhuv[:, 1 : nt - 1, :nx, :ny]
    huv_y = f_dhuv[:, nt + 1 : 2 * nt - 1, :nx, :ny]

    # Compute PDEs using PhysicsNeMo-Sym
    pde_Dh = pde_node[0].evaluate({"h__t": h_t, "hu__x": hu_x, "hv__y": hv_y})
    pde_Du = pde_node[1].evaluate(
        {
            "u__t": u_t,
            "huu__x": huu_x,
            "hh__x": hh_x,
            "huv__y": huv_y,
            "u__x__x": u_x_x,
            "u__y__y": u_y_y,
        }
    )
    pde_Dv = pde_node[2].evaluate(
        {
            "v__t": v_t,
            "hvv__y": hvv_y,
            "hh__y": hh_y,
            "huv__x": huv_x,
            "v__x__x": v_x_x,
            "v__y__y": v_y_y,
        }
    )

    Dh = pde_Dh["Dh"][1:-1]
    Du = pde_Du["Du"][1:-1]
    Dv = pde_Dv["Dv"][1:-1]

    return Dh, Du, Dv


def physicsnemo_fourier(
    s, pde_node, h_weight=1.0, u_weight=1.0, v_weight=1.0, device=0
):
    "Calculate PDE Loss using physicsnemo_fdm_swe_nonlin with PhysicsNeMo functions"
    batchsize = s.size(0)
    nx = s.size(1)
    ny = s.size(2)
    nt = s.size(3)
    s = s.reshape(batchsize, nx, ny, nt, 3)

    h = s[..., 0]
    u = s[..., 1]
    v = s[..., 2]

    Dh, Du, Dv = physicsnemo_fdm_swe_nonlin(h, u, v, pde_node, device=device)
    Dh *= h_weight
    Du *= u_weight
    Dv *= v_weight
    Ds = torch.stack([Dh, Du, Dv], dim=-1)
    f_ = torch.zeros(
        Ds.shape, device=s.device
    )  # use f_ to distinguish from corriolous const f
    loss_f = F.mse_loss(Ds, f_)

    return loss_f
