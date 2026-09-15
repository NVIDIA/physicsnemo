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
"""
Standalone, dependency-light reference implementation (numpy + sympy only) for
gray-box term discovery with physics-native verifiers.

This file DOES NOT require PhysicsNeMo. It validates the mechanism that the
PhysicsNeMo example (`examples/graybox_discovery/`) implements against the real
`physicsnemo.sym` API, so reviewers and CI can reproduce the core result without
a GPU/torch environment.

Result reproduced (see methods note / issue):
  * true closure R_true(u) = A*sin(pi*u) is OUTSIDE any polynomial library
  * observations cover only u <= U_OBS  ->  the closure is under-determined for u > U_OBS
  * an ensemble of unconstrained modules DISAGREES in the unobserved region
    (empirical non-identifiability), while physics verifiers collapse that spread
    and recover the unseen closure.
"""

from __future__ import annotations

import numpy as np
import sympy as sp


def _simulate(nu=0.02, A=3.0, L=1.0, nx=161, dt=5e-4, T=0.9, seed=0):
    dx = L / (nx - 1)
    nt = int(T / dt) + 1
    x = np.linspace(0, L, nx)
    U = np.zeros((nt, nx))
    U[0] = 0.15 * np.exp(-(((x - 0.5) / 0.10) ** 2))
    for n in range(nt - 1):
        u = U[n]
        uxx = np.zeros_like(u)
        uxx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
        U[n + 1, 1:-1] = u[1:-1] + dt * (
            nu * uxx[1:-1] + A * np.sin(np.pi * np.clip(u[1:-1], 0, 1))
        )
        U[n + 1, 0] = U[n + 1, 1]
        U[n + 1, -1] = U[n + 1, -2]
    return x, np.linspace(0, T, nt), U, dx, dt


class _MLP:
    """Tiny MLP with hand-written autodiff (stands in for the PhysicsNeMo f-network)."""

    def __init__(self, H=24, seed=0):
        r = np.random.default_rng(seed)

        def s(a, b):
            return r.standard_normal((a, b)) * np.sqrt(2 / (a + b))

        self.P = [s(1, H), np.zeros(H), s(H, H), np.zeros(H), s(H, 1), np.zeros(1)]

    def fwd(self, X, cache=False):
        W1, b1, W2, b2, W3, b3 = self.P
        z1 = X @ W1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2
        a2 = np.tanh(z2)
        y = a2 @ W3 + b3
        if cache:
            self.c = (X, a1, a2)
        return y

    def bwd(self, dY):
        X, a1, a2 = self.c
        W1, b1, W2, b2, W3, b3 = self.P
        gW3 = a2.T @ dY
        gb3 = dY.sum(0)
        da2 = (dY @ W3.T) * (1 - a2**2)
        gW2 = a1.T @ da2
        gb2 = da2.sum(0)
        da1 = (da2 @ W2.T) * (1 - a1**2)
        gW1 = X.T @ da1
        gb1 = da1.sum(0)
        return [gW1, gb1, gW2, gb2, gW3, gb3]


def run_wedge(seed_count=4, u_obs=0.75, iters=3500, verbose=False):
    """Run the wedge experiment; return a metrics dict."""
    nu, A = 0.02, 3.0
    x, t, U, dx, dt = _simulate(nu=nu, A=A)

    rng = np.random.default_rng(0)
    Un = U + 0.01 * np.std(U) * rng.standard_normal(U.shape)
    B = Un.copy()
    for _ in range(5):
        B[:, 1:-1] = 0.5 * B[:, 1:-1] + 0.25 * (B[:, 2:] + B[:, :-2])
        B[1:-1, :] = 0.5 * B[1:-1, :] + 0.25 * (B[2:, :] + B[:-2, :])
    u_t = (B[2:, :] - B[:-2, :]) / (2 * dt)
    u_xx = (B[:, 2:] - 2 * B[:, 1:-1] + B[:, :-2]) / dx**2
    u_i = B[1:-1, 1:-1]
    g = (u_t[:, 1:-1] - nu * u_xx[1:-1, :]).reshape(-1)
    uu = u_i.reshape(-1)

    obs = uu <= u_obs
    uo, go = uu[obs], g[obs]
    k = rng.choice(uo.size, size=min(1800, uo.size), replace=False)
    uo, go = uo[k][:, None], go[k][:, None]

    # symbolic seam (mirrors physicsnemo SympyToTorch Add.make_args)
    u_s, ut_s, uxx_s = sp.symbols("u u_t u_xx")
    R = sp.Function("R")(u_s)
    residual = ut_s - nu * uxx_s - R
    discover_idx = [i for i, tm in enumerate(sp.Add.make_args(residual)) if tm.has(R)]

    ug = np.linspace(0, 1, 200)[:, None]
    R_true = A * np.sin(np.pi * ug)
    anchors = np.array([[0.0], [1.0]])
    ugp = np.linspace(0, 1, 50)[:, None]

    def train(seed, verifier, lr=3e-3, lam=8.0, lpos=4.0):
        net = _MLP(seed=seed)
        m = [np.zeros_like(p) for p in net.P]
        v = [np.zeros_like(p) for p in net.P]
        for it in range(iters):
            r = net.fwd(uo, cache=True) - go
            gr = net.bwd((2 / len(go)) * r)
            if verifier:
                pv = net.fwd(anchors, cache=True)
                gr = [a + b for a, b in zip(gr, net.bwd((2 * lam / 2) * pv))]
                viol = np.minimum(net.fwd(ugp, cache=True), 0.0)
                gr = [a + b for a, b in zip(gr, net.bwd((2 * lpos / len(ugp)) * viol))]
            for i, gg in enumerate(gr):
                m[i] = 0.9 * m[i] + 0.1 * gg
                v[i] = 0.999 * v[i] + 0.001 * gg * gg
                net.P[i] -= (
                    lr
                    * (m[i] / (1 - 0.9 ** (it + 1)))
                    / (np.sqrt(v[i] / (1 - 0.999 ** (it + 1))) + 1e-8)
                )
        return net.fwd(ug).ravel()

    ens_no = np.stack([train(s, False) for s in range(seed_count)])
    ens_ver = np.stack([train(s, True) for s in range(seed_count)])
    held = ug.ravel() > u_obs
    out = dict(
        discover_idx=discover_idx,
        gaming_no=float(ens_no.std(0)[held].mean()),
        gaming_ver=float(ens_ver.std(0)[held].mean()),
        err_no=float(
            np.linalg.norm(ens_no.mean(0)[:, None] - R_true) / np.linalg.norm(R_true)
        ),
        err_ver=float(
            np.linalg.norm(ens_ver.mean(0)[:, None] - R_true) / np.linalg.norm(R_true)
        ),
    )
    out["gaming_reduction"] = out["gaming_no"] / max(out["gaming_ver"], 1e-9)
    if verbose:
        for k_, v_ in out.items():
            print(f"{k_:16s}: {v_}")
    return out


if __name__ == "__main__":
    run_wedge(verbose=True)
