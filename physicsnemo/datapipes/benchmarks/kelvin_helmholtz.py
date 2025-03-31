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

import sys
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import numpy as np
import torch
import warp as wp

from ..datapipe import Datapipe
from ..meta import DatapipeMetaData
from .kernels.finite_volume import (
    euler_apply_flux_batched_2d,
    euler_conserved_to_primitive_batched_2d,
    euler_extrapolation_batched_2d,
    euler_get_flux_batched_2d,
    euler_primitive_to_conserved_batched_2d,
    initialize_kelvin_helmoltz_batched_2d,
)
from .kernels.initialization import init_uniform_random_2d

Tensor = torch.Tensor
# TODO unsure if better to remove this
wp.init()


@dataclass
class MetaData(DatapipeMetaData):
    name: str = "KelvinHelmholtz2D"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = True
    # Parallel
    ddp_sharding: bool = False


class KelvinHelmholtz2D(Datapipe):
    """Kelvin-Helmholtz instability benchmark problem datapipe.

    This datapipe continuously generates samples with random initial conditions. All samples
    are generated on the fly and is meant to be a benchmark problem for testing data driven
    models. Initial conditions are given in the form of small perturbations. The solution
    is obtained using a GPU enabled Finite Volume Method.

    Parameters
    ----------
    resolution : int, optional
        Resolution to run simulation at, by default 512
    batch_size : int, optional
        Batch size of simulations, by default 16
    seq_length : int, optional
        Sequence length of output samples, by default 8
    nr_perturbation_freq : int, optional
        Number of frequencies to use for generating random initial perturbations, by default 5
    perturbation_range : float, optional
        Range to use for random perturbations. This value will be the max amplitude of the
        initial perturbation, by default 0.1
    nr_snapshots : int, optional
        Number of snapshots of simulation to generate for data generation. This will
        control how long the simulation is run for, by default 256
    iteration_per_snapshot : int, optional
         Number of finite volume steps to take between each snapshot. Each step size is
         fixed as the smallest possible value that satisfies the Courant-Friedrichs-Lewy
         condition, by default 32
    gamma : float, optional
        Heat capacity ratio, by default 5.0/3.0
    normaliser : Union[Dict[str, Tuple[float, float]], None], optional
        Dictionary with keys `density`, `velocity`, and `pressure`. The values for these keys are two floats corresponding to mean and std `(mean, std)`.
    device : Union[str, torch.device], optional
        Device for datapipe to run place data on, by default "cuda"
    """

    def __init__(
        self,
        resolution: int = 512,
        batch_size: int = 16,
        seq_length: int = 8,
        nr_perturbation_freq: int = 5,
        perturbation_range: float = 0.1,
        nr_snapshots: int = 256,
        iteration_per_snapshot: int = 32,
        gamma: float = 5.0 / 3.0,
        normaliser: Union[Dict[str, Tuple[float, float]], None] = None,
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__(meta=MetaData())

        # simulation params
        self.resolution = resolution
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.nr_perturbation_freq = nr_perturbation_freq
        self.perturbation_range = perturbation_range
        self.nr_snapshots = nr_snapshots
        self.iteration_per_snapshot = iteration_per_snapshot
        self.gamma = gamma
        self.courant_fac = 0.4  # hard set
        self.normaliser = normaliser

        # check normaliser keys
        if self.normaliser is not None:
            if not {"density", "velocity", "pressure"}.issubset(
                set(self.normaliser.keys())
            ):
                raise ValueError(
                    "normaliser need to have keys `density`, `velocity` and `pressure` with mean and std"
                )

        # Set up device for warp, warp has same naming convention as torch.
        if isinstance(device, torch.device):
            device = str(device)
        self.device = device

        # spatial dims
        self.dx = 1.0 / resolution
        self.dt = (
            self.courant_fac * self.dx / (np.sqrt(self.gamma * 5.0) + 2.0)
        )  # hard set to smallest possible step needed
        self.vol = self.dx**2
        self.dim = (self.batch_size, self.resolution, self.resolution)

        # allocate array for initial freq perturbation
        self.w = wp.zeros(
            (self.batch_size, self.nr_perturbation_freq),
            dtype=float,
            device=self.device,
        )

        # allocate conservation quantities
        self.mass = wp.zeros(self.dim, dtype=float, device=self.device)
        self.mom = wp.zeros(self.dim, dtype=wp.vec2, device=self.device)
        self.e = wp.zeros(self.dim, dtype=float, device=self.device)

        # allocate primitive quantities
        self.rho = wp.zeros(self.dim, dtype=float, device=self.device)
        self.vel = wp.zeros(self.dim, dtype=wp.vec2, device=self.device)
        self.p = wp.zeros(self.dim, dtype=float, device=self.device)

        # allocate flux values for computation
        self.mass_flux_x = wp.zeros(self.dim, dtype=float, device=self.device)
        self.mass_flux_y = wp.zeros(self.dim, dtype=float, device=self.device)
        self.mom_flux_x = wp.zeros(self.dim, dtype=wp.vec2, device=self.device)
        self.mom_flux_y = wp.zeros(self.dim, dtype=wp.vec2, device=self.device)
        self.e_flux_x = wp.zeros(self.dim, dtype=float, device=self.device)
        self.e_flux_y = wp.zeros(self.dim, dtype=float, device=self.device)

        # allocate extrapolation values for computation
        self.rho_xl = wp.zeros(self.dim, dtype=float, device=self.device)
        self.rho_xr = wp.zeros(self.dim, dtype=float, device=self.device)
        self.rho_yl = wp.zeros(self.dim, dtype=float, device=self.device)
        self.rho_yr = wp.zeros(self.dim, dtype=float, device=self.device)
        self.vel_xl = wp.zeros(self.dim, dtype=wp.vec2, device=self.device)
        self.vel_xr = wp.zeros(self.dim, dtype=wp.vec2, device=self.device)
        self.vel_yl = wp.zeros(self.dim, dtype=wp.vec2, device=self.device)
        self.vel_yr = wp.zeros(self.dim, dtype=wp.vec2, device=self.device)
        self.p_xl = wp.zeros(self.dim, dtype=float, device=self.device)
        self.p_xr = wp.zeros(self.dim, dtype=float, device=self.device)
        self.p_yl = wp.zeros(self.dim, dtype=float, device=self.device)
        self.p_yr = wp.zeros(self.dim, dtype=float, device=self.device)

        # allocate arrays for storing results
        self.seq_rho = [
            wp.zeros(self.dim, dtype=float, device=self.device)
            for _ in range(self.nr_snapshots)
        ]
        self.seq_vel = [
            wp.zeros(self.dim, dtype=wp.vec2, device=self.device)
            for _ in range(self.nr_snapshots)
        ]
        self.seq_p = [
            wp.zeros(self.dim, dtype=float, device=self.device)
            for _ in range(self.nr_snapshots)
        ]

        self.output_rho = None
        self.output_vel = None
        self.output_p = None

    def initialize_batch(self) -> None:
        """Initializes arrays for new batch of simulations"""

        # initialize random Fourier freq
        seed = np.random.randint(np.iinfo(np.uint64).max, dtype=np.uint64)
        wp.launch(
            init_uniform_random_2d,
            dim=[self.batch_size, self.nr_perturbation_freq],
            inputs=[self.w, -self.perturbation_range, self.perturbation_range, seed],
            device=self.device,
        )

        # initialize fields
        wp.launch(
            initialize_kelvin_helmoltz_batched_2d,
            dim=self.dim,
            inputs=[
                self.rho,
                self.vel,
                self.p,
                self.w,
                0.05 / np.sqrt(2.0),
                self.dim[1],
                self.dim[2],
                self.nr_perturbation_freq,
            ],
            device=self.device,
        )
        wp.launch(
            euler_primitive_to_conserved_batched_2d,
            dim=self.dim,
            inputs=[
                self.rho,
                self.vel,
                self.p,
                self.mass,
                self.mom,
                self.e,
                self.gamma,
                self.vol,
                self.dim[1],
                self.dim[2],
            ],
            device=self.device,
        )

    def generate_batch(self) -> None:
        """Solve for new batch of simulations"""

        # initialize tensors with random coef
        self.initialize_batch()

        # run solver
        for s in range(self.nr_snapshots):
            # save arrays for
            wp.copy(self.seq_rho[s], self.rho)
            wp.copy(self.seq_vel[s], self.vel)
            wp.copy(self.seq_p[s], self.p)

            # iterations
            for i in range(self.iteration_per_snapshot):
                # compute primitives
                wp.launch(
                    euler_conserved_to_primitive_batched_2d,
                    dim=self.dim,
                    inputs=[
                        self.mass,
                        self.mom,
                        self.e,
                        self.rho,
                        self.vel,
                        self.p,
                        self.gamma,
                        self.vol,
                        self.dim[1],
                        self.dim[2],
                    ],
                    device=self.device,
                )

                # compute extrapolations to faces
                wp.launch(
                    euler_extrapolation_batched_2d,
                    dim=self.dim,
                    inputs=[
                        self.rho,
                        self.vel,
                        self.p,
                        self.rho_xl,
                        self.rho_xr,
                        self.rho_yl,
                        self.rho_yr,
                        self.vel_xl,
                        self.vel_xr,
                        self.vel_yl,
                        self.vel_yr,
                        self.p_xl,
                        self.p_xr,
                        self.p_yl,
                        self.p_yr,
                        self.gamma,
                        self.dx,
                        self.dt,
                        self.dim[1],
                        self.dim[2],
                    ],
                    device=self.device,
                )

                # compute fluxes
                wp.launch(
                    euler_get_flux_batched_2d,
                    dim=self.dim,
                    inputs=[
                        self.rho_xl,
                        self.rho_xr,
                        self.rho_yl,
                        self.rho_yr,
                        self.vel_xl,
                        self.vel_xr,
                        self.vel_yl,
                        self.vel_yr,
                        self.p_xl,
                        self.p_xr,
                        self.p_yl,
                        self.p_yr,
                        self.mass_flux_x,
                        self.mass_flux_y,
                        self.mom_flux_x,
                        self.mom_flux_y,
                        self.e_flux_x,
                        self.e_flux_y,
                        self.gamma,
                        self.dim[1],
                        self.dim[2],
                    ],
                    device=self.device,
                )

                # apply fluxes
                wp.launch(
                    euler_apply_flux_batched_2d,
                    dim=self.dim,
                    inputs=[
                        self.mass_flux_x,
                        self.mass_flux_y,
                        self.mom_flux_x,
                        self.mom_flux_y,
                        self.e_flux_x,
                        self.e_flux_y,
                        self.mass,
                        self.mom,
                        self.e,
                        self.dx,
                        self.dt,
                        self.dim[1],
                        self.dim[2],
                    ],
                    device=self.device,
                )

    def __iter__(self) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Yields
        ------
        Iterator[Tuple[Tensor, Tensor]]
            Infinite iterator that returns a batch of timeseries with (density, velocity, pressure)
            fields of size [batch, seq_length, dim, resolution, resolution]
        """
        # infinite generator
        while True:
            # run simulation
            self.generate_batch()

            # return all samples generated before rerunning simulation
            batch_ind = [
                np.arange(self.nr_snapshots - self.seq_length)
                for _ in range(self.batch_size)
            ]
            for b_ind in batch_ind:
                np.random.shuffle(b_ind)
            for bb in range(self.nr_snapshots - self.seq_length):
                # run over batch to gather samples
                batched_seq_rho = []
                batched_seq_vel = []
                batched_seq_p = []
                for b in range(self.batch_size):
                    # gather seq from each batch
                    seq_rho = []
                    seq_vel = []
                    seq_p = []
                    for s in range(self.seq_length):
                        # get variables
                        rho = wp.to_torch(self.seq_rho[batch_ind[b][bb] + s])[b]
                        vel = wp.to_torch(self.seq_vel[batch_ind[b][bb] + s])[b]
                        p = wp.to_torch(self.seq_p[batch_ind[b][bb] + s])[b]

                        # add channels
                        rho = torch.unsqueeze(rho, 0)
                        vel = torch.permute(vel, (2, 0, 1))
                        p = torch.unsqueeze(p, 0)

                        # normalize values
                        if self.normaliser is not None:
                            rho = (
                                rho - self.normaliser["density"][0]
                            ) / self.normaliser["density"][1]
                            vel = (
                                vel - self.normaliser["velocity"][0]
                            ) / self.normaliser["velocity"][1]
                            p = (p - self.normaliser["pressure"][0]) / self.normaliser[
                                "pressure"
                            ][1]

                        # store for producing seq
                        seq_rho.append(rho)
                        seq_vel.append(vel)
                        seq_p.append(p)

                    # concat seq
                    batched_seq_rho.append(torch.stack(seq_rho, axis=0))
                    batched_seq_vel.append(torch.stack(seq_vel, axis=0))
                    batched_seq_p.append(torch.stack(seq_p, axis=0))

                # CUDA graphs static copies
                if self.output_rho is None:
                    # concat batches
                    self.output_rho = torch.stack(batched_seq_rho, axis=0)
                    self.output_vel = torch.stack(batched_seq_vel, axis=0)
                    self.output_p = torch.stack(batched_seq_p, axis=0)
                else:
                    self.output_rho.data.copy_(torch.stack(batched_seq_rho, axis=0))
                    self.output_vel.data.copy_(torch.stack(batched_seq_vel, axis=0))
                    self.output_p.data.copy_(torch.stack(batched_seq_p, axis=0))

                yield {
                    "density": self.output_rho,
                    "velocity": self.output_vel,
                    "pressure": self.output_p,
                }

    def __len__(self):
        return sys.maxsize
