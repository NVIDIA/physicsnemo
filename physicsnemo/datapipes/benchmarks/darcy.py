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
from .kernels.finite_difference import (
    darcy_mgrid_jacobi_iterative_batched_2d,
    mgrid_inf_residual_batched_2d,
)
from .kernels.initialization import init_uniform_random_4d
from .kernels.utils import (
    bilinear_upsample_batched_2d,
    fourier_to_array_batched_2d,
    threshold_3d,
)

Tensor = torch.Tensor
# TODO unsure if better to remove this. Keeping this in for now
wp.init()


@dataclass
class MetaData(DatapipeMetaData):
    name: str = "Darcy2D"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = True
    # Parallel
    ddp_sharding: bool = False


class Darcy2D(Datapipe):
    """2D Darcy flow benchmark problem datapipe.

    This datapipe continuously generates solutions to the 2D Darcy equation with variable
    permeability. All samples are generated on the fly and is meant to be a benchmark
    problem for testing data driven models. Permeability is drawn from a random Fourier
    series and threshold it to give a piecewise constant function. The solution is obtained
    using a GPU enabled multi-grid Jacobi iterative method.

    Parameters
    ----------
    resolution : int, optional
        Resolution to run simulation at, by default 256
    batch_size : int, optional
        Batch size of simulations, by default 64
    nr_permeability_freq : int, optional
        Number of frequencies to use for generating random permeability. Higher values
        will give higher freq permeability fields., by default 5
    max_permeability : float, optional
        Max permeability, by default 2.0
    min_permeability : float, optional
        Min permeability, by default 0.5
    max_iterations : int, optional
        Maximum iterations to use for each multi-grid, by default 30000
    convergence_threshold : float, optional
        Solver L-Infinity convergence threshold, by default 1e-6
    iterations_per_convergence_check : int, optional
        Number of Jacobi iterations to run before checking convergence, by default 1000
    nr_multigrids : int, optional
        Number of multi-grid levels, by default 4
    normaliser : Union[Dict[str, Tuple[float, float]], None], optional
        Dictionary with keys `permeability` and `darcy`. The values for these keys are two floats corresponding to mean and std `(mean, std)`.
    device : Union[str, torch.device], optional
        Device for datapipe to run place data on, by default "cuda"

    Raises
    ------
    ValueError
        Incompatable multi-grid and resolution settings
    """

    def __init__(
        self,
        resolution: int = 256,
        batch_size: int = 64,
        nr_permeability_freq: int = 5,
        max_permeability: float = 2.0,
        min_permeability: float = 0.5,
        max_iterations: int = 30000,
        convergence_threshold: float = 1e-6,
        iterations_per_convergence_check: int = 1000,
        nr_multigrids: int = 4,
        normaliser: Union[Dict[str, Tuple[float, float]], None] = None,
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__(meta=MetaData())

        # simulation params
        self.resolution = resolution
        self.batch_size = batch_size
        self.nr_permeability_freq = nr_permeability_freq
        self.max_permeability = max_permeability
        self.min_permeability = min_permeability
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.iterations_per_convergence_check = iterations_per_convergence_check
        self.nr_multigrids = nr_multigrids
        self.normaliser = normaliser

        # check normaliser keys
        if self.normaliser is not None:
            if not {"permeability", "darcy"}.issubset(set(self.normaliser.keys())):
                raise ValueError(
                    "normaliser need to have keys permeability and darcy with mean and std"
                )

        # Set up device for warp, warp has same naming convention as torch.
        if isinstance(device, torch.device):
            device = str(device)
        self.device = device

        # spatial dims
        self.dx = 1.0 / (self.resolution + 1)  # pad edges by 1 for multi-grid
        self.dim = (self.batch_size, self.resolution + 1, self.resolution + 1)
        self.fourier_dim = (
            4,
            self.batch_size,
            self.nr_permeability_freq,
            self.nr_permeability_freq,
        )

        # assert resolution is compatible with multi-grid method
        if (resolution % 2 ** (nr_multigrids - 1)) != 0:
            raise ValueError("Resolution is incompatible with number of sub grids.")

        # allocate arrays for constructing dataset
        self.darcy0 = wp.zeros(self.dim, dtype=float, device=self.device)
        self.darcy1 = wp.zeros(self.dim, dtype=float, device=self.device)
        self.permeability = wp.zeros(self.dim, dtype=float, device=self.device)
        self.rand_fourier = wp.zeros(self.fourier_dim, dtype=float, device=self.device)
        self.inf_residual = wp.zeros([1], dtype=float, device=self.device)

        # Output tenors
        self.output_k = None
        self.output_p = None

    def initialize_batch(self) -> None:
        """Initializes arrays for new batch of simulations"""

        # initialize permeability
        self.permeability.zero_()
        seed = np.random.randint(np.iinfo(np.uint64).max, dtype=np.uint64)
        wp.launch(
            kernel=init_uniform_random_4d,
            dim=self.fourier_dim,
            inputs=[self.rand_fourier, -1.0, 1.0, seed],
            device=self.device,
        )
        wp.launch(
            kernel=fourier_to_array_batched_2d,
            dim=self.dim,
            inputs=[
                self.permeability,
                self.rand_fourier,
                self.nr_permeability_freq,
                self.resolution,
                self.resolution,
            ],
            device=self.device,
        )
        wp.launch(
            kernel=threshold_3d,
            dim=self.dim,
            inputs=[
                self.permeability,
                0.0,
                self.min_permeability,
                self.max_permeability,
            ],
            device=self.device,
        )

        # zero darcy arrays
        self.darcy0.zero_()
        self.darcy1.zero_()

    def generate_batch(self) -> None:
        """Solve for new batch of simulations"""

        # initialize tensors with random permeability
        self.initialize_batch()

        # run solver
        for res in range(self.nr_multigrids):
            # calculate grid reduction factor and reduced dim
            grid_reduction_factor = 2 ** (self.nr_multigrids - res - 1)
            if grid_reduction_factor > 1:
                multigrid_dim = tuple(
                    [self.batch_size] + 2 * [(self.resolution) // grid_reduction_factor]
                )
            else:
                multigrid_dim = self.dim

            # run till max steps is reached
            for k in range(
                self.max_iterations // self.iterations_per_convergence_check
            ):
                # run jacobi iterations
                for s in range(self.iterations_per_convergence_check):
                    # iterate solver
                    wp.launch(
                        kernel=darcy_mgrid_jacobi_iterative_batched_2d,
                        dim=multigrid_dim,
                        inputs=[
                            self.darcy0,
                            self.darcy1,
                            self.permeability,
                            1.0,
                            self.dim[1],
                            self.dim[2],
                            self.dx,
                            grid_reduction_factor,
                        ],
                        device=self.device,
                    )

                    # swap buffers
                    (self.darcy0, self.darcy1) = (self.darcy1, self.darcy0)

                # compute residual
                self.inf_residual.zero_()
                wp.launch(
                    kernel=mgrid_inf_residual_batched_2d,
                    dim=multigrid_dim,
                    inputs=[
                        self.darcy0,
                        self.darcy1,
                        self.inf_residual,
                        grid_reduction_factor,
                    ],
                    device=self.device,
                )
                normalized_inf_residual = self.inf_residual.numpy()[0]

                # check if converged
                if normalized_inf_residual < (
                    self.convergence_threshold * grid_reduction_factor
                ):
                    break

            # upsample to higher resolution
            if grid_reduction_factor > 1:
                wp.launch(
                    kernel=bilinear_upsample_batched_2d,
                    dim=self.dim,
                    inputs=[
                        self.darcy0,
                        self.dim[1],
                        self.dim[2],
                        grid_reduction_factor,
                    ],
                    device=self.device,
                )

    def __iter__(self) -> Tuple[Tensor, Tensor]:
        """
        Yields
        ------
        Iterator[Tuple[Tensor, Tensor]]
            Infinite iterator that returns a batch of (permeability, darcy pressure)
            fields of size [batch, resolution, resolution]
        """
        # infinite generator
        while True:
            # run simulation
            self.generate_batch()

            # convert warp arrays to pytorch
            permeability = wp.to_torch(self.permeability)
            darcy = wp.to_torch(self.darcy0)

            # add channel dims
            permeability = torch.unsqueeze(permeability, axis=1)
            darcy = torch.unsqueeze(darcy, axis=1)

            # crop edges by 1 from multi-grid TODO messy
            permeability = permeability[:, :, : self.resolution, : self.resolution]
            darcy = darcy[:, :, : self.resolution, : self.resolution]

            # normalize values
            if self.normaliser is not None:
                permeability = (
                    permeability - self.normaliser["permeability"][0]
                ) / self.normaliser["permeability"][1]
                darcy = (darcy - self.normaliser["darcy"][0]) / self.normaliser[
                    "darcy"
                ][1]

            # CUDA graphs static copies
            if self.output_k is None:
                self.output_k = permeability
                self.output_p = darcy
            else:
                self.output_k.data.copy_(permeability)
                self.output_p.data.copy_(darcy)

            yield {"permeability": self.output_k, "darcy": self.output_p}

    def __len__(self):
        return sys.maxsize
