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

import os
import torch
from torch import vmap

from solver.swe import SWE_Nonlinear
from solver.my_random_fields import GRF_Mattern

from train_utils.utils import load_config, download_SWE_NL_dataset

config_file = "config_pino.yaml"
config = load_config(config_file)
# Load config values
Nsamples = config["data"]["total_num"]
N = config["data"]["nx"]
Nt = config["data"]["nt"]
g = config["data"]["g"]
H = config["data"]["H"]
nu = config["data"]["nu"]
Nx = N
Ny = N
dim = 2
l = 0.1
L = 1.0
sigma = 0.2  # 2.0
Nu = None  # 2.0
dt = 1.0e-3
tend = 1.0
save_int = int(tend / dt / Nt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Creating data...")

# Generate random fields
grf = GRF_Mattern(
    dim, N, length=L, nu=Nu, l=l, sigma=sigma, boundary="periodic", device=device
)
H0 = H + grf.sample(Nsamples)  # add 1 for mean surface pressure

# Evolve the SWE
swe_eq = SWE_Nonlinear(Nx=Nx, Ny=Ny, g=g, nu=nu, dt=dt, tend=tend, device=device)
H, U, V = vmap(swe_eq.driver, in_dims=(0, None))(H0, save_int)


h = H.cpu().float()
u = U.cpu().float()
v = V.cpu().float()

torch.cuda.empty_cache()

# Option to use Gaussian test
use_gaussian_test = False
data = torch.stack([h, u, v], dim=-1)
if use_gaussian_test:
    swe_test = SWE_Nonlinear(Nx=Nx, Ny=Ny, g=g, nu=nu, dt=dt, tend=tend, device=device)
    h0_test = swe_test.initialize_gaussian(amp=0.1)
    htest, utest, vtest = swe_test.driver(h0_test, save_interval=save_int)
    htest = htest.cpu().float()
    utest = utest.cpu().float()
    vtest = vtest.cpu().float()

    data_test = torch.stack([htest, utest, vtest], dim=-1)[None]  # custom gaussian
    data = torch.cat([data, data_test], dim=0)

# Create dataset directory
directory = "datasets/"
if not os.path.exists(directory):
    os.makedirs(directory)

print("Downloading data")
# Downloading dataset into directory
download_SWE_NL_dataset(data, outdir="datasets/")
print("Done!")
