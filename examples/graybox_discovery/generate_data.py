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
"""Generate synthetic data for the gray-box discovery example.

Solves a 1D reaction-diffusion problem

    u_t = D u_xx + R(u),   R(u) = A sin(pi u)

with an explicit finite-difference scheme, then writes noisy point observations
and a set of interior collocation points to graybox_data.npz. The reaction term
R is non-polynomial on purpose (so a fixed polynomial library cannot represent
it), and observations are restricted to u <= U_OBS so the closure is
under-determined for larger u.
"""

import os

import numpy as np

# Anchor the output next to this script so the data lands in the example
# directory regardless of the caller's working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))

D, A, L = 0.02, 3.0, 1.0
NX, DT, T = 161, 5e-4, 0.9
U_OBS = 0.75
NOISE = 0.01


def solve():
    """Integrate the reaction-diffusion PDE with explicit finite differences.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(x, t, u)`` -- the spatial grid, the time grid, and the solution
        ``u`` with shape ``(len(t), len(x))``.
    """
    dx = L / (NX - 1)
    nt = int(T / DT) + 1
    x = np.linspace(0, L, NX)
    u = np.zeros((nt, NX))
    u[0] = 0.15 * np.exp(-(((x - 0.5) / 0.1) ** 2))
    for n in range(nt - 1):
        lap = np.zeros(NX)
        lap[1:-1] = (u[n, 2:] - 2 * u[n, 1:-1] + u[n, :-2]) / dx**2
        u[n + 1, 1:-1] = u[n, 1:-1] + DT * (
            D * lap[1:-1] + A * np.sin(np.pi * np.clip(u[n, 1:-1], 0, 1))
        )
        u[n + 1, 0] = u[n + 1, 1]
        u[n + 1, -1] = u[n + 1, -2]
    return x, np.linspace(0, T, nt), u


def main():
    """Simulate the PDE, sample noisy observations, and write ``graybox_data.npz``."""
    rng = np.random.default_rng(0)
    x, t, u = solve()
    X, Tm = np.meshgrid(x, t)
    obs = u <= U_OBS

    x_data = X[obs][:, None]
    t_data = Tm[obs][:, None]
    u_data = (u[obs] + NOISE * u.std() * rng.standard_normal(obs.sum()))[:, None]

    # thin the observations and sample interior collocation points
    keep = rng.choice(len(x_data), size=min(4000, len(x_data)), replace=False)
    x_data, t_data, u_data = x_data[keep], t_data[keep], u_data[keep]

    ci = rng.choice(X.size, size=4000, replace=False)
    x_pde, t_pde = X.ravel()[ci][:, None], Tm.ravel()[ci][:, None]

    out_path = os.path.join(_HERE, "graybox_data.npz")
    np.savez(
        out_path,
        x_data=x_data,
        t_data=t_data,
        u_data=u_data,
        x_pde=x_pde,
        t_pde=t_pde,
        u_obs=np.array(U_OBS),
        D=np.array(D),
        A=np.array(A),
    )
    print(f"wrote {out_path}: {len(x_data)} obs (u<={U_OBS}), {len(x_pde)} collocation")


if __name__ == "__main__":
    main()
