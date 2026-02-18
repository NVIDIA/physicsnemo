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

import torch
import torch.nn.functional as F
import math
from math import pi, gamma, sqrt
import numpy as np

torch.manual_seed(0)


class GRF_Mattern(object):
    """Generate Random Fields"""

    def __init__(
        self,
        dim,
        size,
        length=1.0,
        nu=None,
        l=0.1,
        sigma=1.0,
        boundary="periodic",
        constant_eig=None,
        device=None,
    ):
        self.dim = dim
        self.device = device
        self.bc = boundary

        a = sqrt(2 / length)
        if self.bc == "dirichlet":
            constant_eig = None

        if nu is not None:
            kappa = sqrt(2 * nu) / l
            alpha = nu + 0.5 * dim
            self.eta2 = (
                size**dim
                * sigma
                * (4.0 * pi) ** (0.5 * dim)
                * gamma(alpha)
                / (kappa**dim * gamma(nu))
            )
        else:
            self.eta2 = size**dim * sigma * (sqrt(2.0 * pi) * l) ** dim

        k_max = size // 2
        if self.bc == "periodic":
            const = (4.0 * (pi**2)) / (length**2)
        else:
            const = (pi**2) / (length**2)

        if dim == 1:
            k = torch.cat(
                (
                    torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device),
                ),
                0,
            )

            k2 = k**2
            if nu is not None:
                eigs = 1.0 + (const / (kappa * length) ** 2 * k2)
                self.sqrt_eig = self.eta2 / (length**dim) * eigs ** (-alpha / 2.0)
            else:
                self.sqrt_eig = (
                    self.eta2
                    / (length**dim)
                    * torch.exp(-((l) ** 2) * const * k2 / 4.0)
                )

            if constant_eig is not None:
                self.sqrt_eig[0] = constant_eig  # (size**dim)*sigma*(tau**(-alpha))
            else:
                self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat(
                (
                    torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device),
                ),
                0,
            ).repeat(size, 1)

            k_x = wavenumers.transpose(0, 1)
            k_y = wavenumers

            k2 = k_x**2 + k_y**2
            if nu is not None:
                eigs = 1.0 + (const / (kappa * length) ** 2 * k2)
                self.sqrt_eig = self.eta2 / (length**dim) * eigs ** (-alpha / 2.0)
            else:
                self.sqrt_eig = (
                    self.eta2
                    / (length**dim)
                    * torch.exp(-((l) ** 2) * const * k2 / 4.0)
                )

            if constant_eig is not None:
                self.sqrt_eig[0, 0] = constant_eig  # (size**dim)*sigma*(tau**(-alpha))
            else:
                self.sqrt_eig[0, 0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat(
                (
                    torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device),
                ),
                0,
            ).repeat(size, size, 1)

            k_x = wavenumers.transpose(1, 2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0, 2)

            k2 = k_x**2 + k_y**2 + k_z**2
            if nu is not None:
                eigs = 1.0 + (const / (kappa * length) ** 2 * k2)
                self.sqrt_eig = self.eta2 / (length**dim) * eigs ** (-alpha / 2.0)
            else:
                self.sqrt_eig = (
                    self.eta2
                    / (length**dim)
                    * torch.exp(-((l) ** 2) * const * k2 / 4.0)
                )

            if constant_eig is not None:
                self.sqrt_eig[0, 0, 0] = (
                    constant_eig  # (size**dim)*sigma*(tau**(-alpha))
                )
            else:
                self.sqrt_eig[0, 0, 0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):
        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        if self.bc == "dirichlet":
            coeff.real[:] = 0
        if self.bc == "neumann":
            coeff.imag[:] = 0
        coeff = self.sqrt_eig * coeff

        u = torch.fft.irfftn(coeff, self.size, norm="backward")
        return u


if __name__ == "__main__":
    from hydra import compose, initialize
    import h5py
    import os
    import matplotlib.pyplot as plt

    initialize(version_base=None, config_path=".", job_name="generate_field")
    cfg = compose(config_name="example_field")

    N = cfg.num_samples
    n = cfg.num_points
    dim = cfg.dim
    L = cfg.length
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    grf = GRF_Mattern(
        dim=cfg.dim,
        size=cfg.num_points,
        length=cfg.length,
        nu=cfg.nu,
        l=cfg.length_scale,
        sigma=cfg.sigma,
        boundary=cfg.boundary_condition,
        constant_eig=cfg.mean,
        device=device,
    )
    U = grf.sample(N)
    # convert to pad periodically
    pad_width = [(0, 0)] + [(0, 1) for _ in range(dim)]

    u = np.pad(U.cpu().numpy(), pad_width, mode="wrap")
    x = np.linspace(0, L, n + 1)
    digits = int(math.log10(N)) + 1
    basefile = cfg.file
    if basefile:
        filedir, file = os.path.split(basefile)
        if filedir:
            os.makedirs(filedir, exist_ok=True)

        for i, u0 in enumerate(u):
            filename = f"{basefile}-{i:0{digits}d}.h5"
            with h5py.File(filename, "w") as hf:
                hf.create_dataset("u", data=u0)
                for j in range(dim):
                    coord_name = f"x{j + 1}"
                    hf.create_dataset(coord_name, data=x)

    if cfg.plot:
        # coords = [x for _ in dim]
        # X = np.meshgrid(*coords, indexing='ij')
        if dim == 2:
            X, Y = np.meshgrid(x, x, indexing="ij")
            plt.close("all")
            fig = plt.figure()
            pmesh = plt.pcolormesh(X, Y, u[0], cmap="jet", shading="gouraud")
            plt.colorbar(pmesh)
            plt.axis("square")
            plt.title("Random Initial Data")
            plt.show()
