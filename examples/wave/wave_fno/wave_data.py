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

"""On-the-fly 2D wave equation data generator using leapfrog finite differences.

Generates random initial wavefields from a superposition of Fourier modes and
evolves them forward in time using the standard second-order leapfrog scheme
with periodic boundary conditions.
"""

import numpy as np
import torch


def generate_wave_batch(
    batch_size: int,
    resolution: int,
    wave_speed: float = 1.0,
    target_time: float = 0.5,
    nr_modes: int = 5,
    cfl: float = 0.25,
    device: str = "cuda",
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of 2D wave equation initial conditions and solutions.

    Parameters
    ----------
    batch_size : int
        Number of samples to generate
    resolution : int
        Spatial resolution (NxN grid)
    wave_speed : float
        Wave propagation speed c
    target_time : float
        Time at which to evaluate the solution
    nr_modes : int
        Number of Fourier modes per axis for random initial conditions
    cfl : float
        CFL number (dt = cfl * dx / c)
    device : str
        Device to return tensors on
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (initial_condition, target_solution) each of shape (batch, 1, N, N)
    """
    rng = np.random.default_rng(seed)
    dx = 1.0 / resolution
    dt = cfl * dx / wave_speed
    n_steps = int(np.ceil(target_time / dt))
    dt = target_time / n_steps  # adjust for exact target time

    # Coordinate grids
    x = np.linspace(0, 1, resolution, endpoint=False)
    y = np.linspace(0, 1, resolution, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    c2_ratio = (wave_speed * dt / dx) ** 2

    u0_all = np.zeros((batch_size, resolution, resolution), dtype=np.float32)
    uT_all = np.zeros((batch_size, resolution, resolution), dtype=np.float32)

    for b in range(batch_size):
        # Random superposition of Fourier modes
        u = np.zeros((resolution, resolution), dtype=np.float64)
        for _ in range(nr_modes):
            kx = rng.integers(-nr_modes, nr_modes + 1)
            ky = rng.integers(-nr_modes, nr_modes + 1)
            amp = rng.standard_normal()
            phase = rng.uniform(0, 2 * np.pi)
            u += amp * np.sin(
                2 * np.pi * (kx * xx + ky * yy) + phase
            )

        # Normalize to unit variance
        std = np.std(u)
        if std > 1e-10:
            u /= std

        u0_all[b] = u.astype(np.float32)

        # Leapfrog time integration with zero initial velocity
        u_prev = u.copy()
        # Taylor expansion for first step: u(dt) = u(0) + 0.5*dt^2*c^2*laplacian(u)
        lap = (
            np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0)
            + np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1)
            - 4.0 * u
        ) / dx**2
        u_curr = u + 0.5 * dt**2 * wave_speed**2 * lap

        for _ in range(n_steps - 1):
            lap = (
                np.roll(u_curr, 1, axis=0) + np.roll(u_curr, -1, axis=0)
                + np.roll(u_curr, 1, axis=1) + np.roll(u_curr, -1, axis=1)
                - 4.0 * u_curr
            ) / dx**2
            u_next = 2.0 * u_curr - u_prev + dt**2 * wave_speed**2 * lap
            u_prev = u_curr
            u_curr = u_next

        uT_all[b] = u_curr.astype(np.float32)

    # Convert to tensors: (batch, 1, N, N)
    initial = torch.from_numpy(u0_all).unsqueeze(1).to(device)
    target = torch.from_numpy(uT_all).unsqueeze(1).to(device)
    return initial, target


class WaveDataLoader:
    """Iterable data loader that generates wave equation samples on the fly.

    Parameters
    ----------
    resolution : int
        Spatial resolution
    batch_size : int
        Batch size
    wave_speed : float
        Wave speed c
    target_time : float
        Target evolution time T
    nr_modes : int
        Number of Fourier modes for initial conditions
    cfl : float
        CFL number for time stepping
    normaliser : dict or None
        Normalisation parameters {"input": (mean, std), "output": (mean, std)}
    device : str
        Device for output tensors
    """

    def __init__(
        self,
        resolution: int = 128,
        batch_size: int = 32,
        wave_speed: float = 1.0,
        target_time: float = 0.5,
        nr_modes: int = 5,
        cfl: float = 0.25,
        normaliser: dict | None = None,
        device: str = "cuda",
    ):
        self.resolution = resolution
        self.batch_size = batch_size
        self.wave_speed = wave_speed
        self.target_time = target_time
        self.nr_modes = nr_modes
        self.cfl = cfl
        self.normaliser = normaliser
        self.device = device

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        initial, target = generate_wave_batch(
            batch_size=self.batch_size,
            resolution=self.resolution,
            wave_speed=self.wave_speed,
            target_time=self.target_time,
            nr_modes=self.nr_modes,
            cfl=self.cfl,
            device=self.device,
        )
        if self.normaliser is not None:
            im, isd = self.normaliser.get("input", (0.0, 1.0))
            om, osd = self.normaliser.get("output", (0.0, 1.0))
            initial = (initial - im) / isd
            target = (target - om) / osd
        return {"initial": initial, "target": target}
