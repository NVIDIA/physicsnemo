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
import scipy.io as scio

from physicsnemo.datapipes.datapipe import Datapipe
from physicsnemo.datapipes.meta import DatapipeMetaData
from physicsnemo.datapipes.benchmarks.kernels.finite_difference import (
    darcy_mgrid_jacobi_iterative_batched_2d,
    mgrid_inf_residual_batched_2d,
)
from physicsnemo.datapipes.benchmarks.kernels.initialization import (
    init_uniform_random_4d,
)
from physicsnemo.datapipes.benchmarks.kernels.utils import (
    bilinear_upsample_batched_2d,
    fourier_to_array_batched_2d,
    threshold_3d,
)

Tensor = torch.Tensor
# TODO unsure if better to remove this. Keeping this in for now
wp.init()


class UnitTransformer:
    """Unit transformer class for normalizing and denormalizing data."""

    def __init__(self, X):
        self.mean = X.mean(dim=(0, 1), keepdim=True)
        self.std = X.std(dim=(0, 1), keepdim=True) + 1e-8

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

    def encode(self, x):
        x = (x - self.mean) / (self.std)
        return x

    def decode(self, x):
        return x * self.std + self.mean

    def transform(self, X, inverse=True, component="all"):
        if component == "all" or "all-reduce":
            if inverse:
                orig_shape = X.shape
                return (X * (self.std - 1e-8) + self.mean).view(orig_shape)
            else:
                return (X - self.mean) / self.std
        else:
            if inverse:
                orig_shape = X.shape
                return (
                    X * (self.std[:, component] - 1e-8) + self.mean[:, component]
                ).view(orig_shape)
            else:
                return (X - self.mean[:, component]) / self.std[:, component]


@dataclass
class MetaData(DatapipeMetaData):
    name: str = "Darcy2D"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = True
    # Parallel
    ddp_sharding: bool = False


class Darcy2D_fix(Datapipe):
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
        train_path: str = None,
        is_test: bool = False,
        x_normalizer: UnitTransformer = None,
        y_normalizer: UnitTransformer = None,
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
        # if (resolution % 2 ** (nr_multigrids - 1)) != 0:
        #     raise ValueError("Resolution is incompatible with number of sub grids.")

        # allocate arrays for constructing dataset
        self.darcy0 = wp.zeros(self.dim, dtype=float, device=self.device)
        self.darcy1 = wp.zeros(self.dim, dtype=float, device=self.device)
        self.permeability = wp.zeros(self.dim, dtype=float, device=self.device)
        self.rand_fourier = wp.zeros(self.fourier_dim, dtype=float, device=self.device)
        self.inf_residual = wp.zeros([1], dtype=float, device=self.device)
        self.train_path = train_path
        self.downsample = 5
        self.r = self.downsample
        self.h = int(((421 - 1) / self.r) + 1)
        self.s = self.h
        # print(f"=============={self.s}===============")
        self.dx = 1.0 / self.s

        # Output tenors
        self.output_k = None
        self.output_p = None

        self.is_test = is_test

        if not self.is_test:
            n_train = 1000
        else:
            n_train = 200
        self.n_train = n_train

        if self.train_path is not None:
            self.__get_data__()

        if not self.is_test:
            self.x_normalizer = UnitTransformer(self.x_train)
            self.y_normalizer = UnitTransformer(self.y_train)

            self.x_train = self.x_normalizer.encode(self.x_train)
            self.y_train = self.y_normalizer.encode(self.y_train)
        else:
            self.x_train = x_normalizer.encode(self.x_train)

    def __get_normalizer__(self):
        return self.x_normalizer, self.y_normalizer

    def __get_data__(self):
        x = np.linspace(0, 1, self.s)
        y = np.linspace(0, 1, self.s)
        x, y = np.meshgrid(x, y)
        pos = np.c_[x.ravel(), y.ravel()]
        pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0).cuda()
        self.x_train = scio.loadmat(self.train_path)["coeff"][
            : self.n_train, :: self.r, :: self.r
        ][:, : self.s, : self.s]
        self.x_train = self.x_train.reshape(self.n_train, -1)
        self.x_train = torch.from_numpy(self.x_train).float().cuda()
        self.y_train = scio.loadmat(self.train_path)["sol"][
            : self.n_train, :: self.r, :: self.r
        ][:, : self.s, : self.s]
        self.y_train = self.y_train.reshape(self.n_train, -1)
        self.y_train = torch.from_numpy(self.y_train).float().cuda()
        self.pos_train = pos.repeat(self.n_train, 1, 1)

    def __iter__(self):
        """
        Yields
        ------
        Iterator[Tuple[Tensor, Tensor]]
            Infinite iterator that returns a batch of (permeability, darcy pressure)
            fields of size [batch, resolution, resolution]
        """
        # infinite generator
        while True:
            idx = np.random.choice(200, self.batch_size)
            x = self.x_train[idx]
            y = self.y_train[idx]
            pos = self.pos_train[idx]
            yield pos, x, y

    def __len__(self):
        return self.n_train // self.batch_size
