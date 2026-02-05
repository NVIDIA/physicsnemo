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
Dedalus script simulating a 2D periodic incompressible MHD flow with a passive
tracer field for visualization. This script demonstrates solving a 2D periodic
initial value problem. This script is meant to be ran in parallel, and uses the
built-in analysis framework to save data snapshots to HDF5 files.
The simulation should take at least 100 gpu-minutes to run.

The initial flow is in the x-direction and depends only on z. The problem is
non-dimensionalized usign the shear-layer spacing and velocity jump, so the
resulting viscosity and tracer diffusivity are related to the Reynolds and
Schmidt numbers as:

    nu = 1 / Re
    eta = 1 / ReM
    D = nu / Schmidt

To run this script:
    $ python dedalus_mhd_parallel.py
"""

import os
import glob
import h5py
import numpy as np
import functools
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
import argparse
import multiprocessing as mp
import dedalus
import dedalus.public as d3
from dedalus.extras import plot_tools
import pathlib
from docopt import docopt
from dedalus.tools import logging
from dedalus.tools import post
from dedalus.tools.parallel import Sync
import logging
import math
from IPython.display import display
import imageio
from importlib import reload
from my_random_fields import GRF_Mattern
import torch
from functorch import vmap
from hydra import compose, initialize
from hydra.utils import get_class

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# display(device)


def check_if_complete(sim_outputs, Nt=101):
    try:
        files = sorted(glob.glob(sim_outputs))
        file = files[0]
        with h5py.File(file, mode="r") as h5file:
            data_file = h5file["tasks"]
            keys = list(data_file.keys())
            dims = data_file[keys[0]].dims
            t = dims[0]["sim_time"][:]
        if len(t) == Nt:
            return True
        else:
            return False
    except Exception:
        return False


if __name__ == "__main__":
    initialize(version_base=None, config_path=".", job_name="generate_mhd_field")
    cfg = compose(config_name="mhd_field")

    # Parameters
    Lx, Ly = cfg.Lx, cfg.Ly
    Nx, Ny = cfg.Nx, cfg.Ny
    Re = cfg.Re  # 1e4
    ReM = cfg.ReM  # 1e4
    Schmidt = cfg.Schmidt  # 1
    rho0 = cfg.rho0  # 1.0
    dealias = cfg.dealias  # 3/2
    stop_sim_time = cfg.tend
    timestepper = get_class(cfg.timestepper)  # d3.RK443 #d3.RK222
    Dt = cfg.Dt  #  1e-3
    max_timestep = cfg.max_timestep  #  1e-2
    output_dt = cfg.output_dt  #  1e-2 # 1e-1
    log_iter = cfg.log_iter  # 10
    dtype = get_class(cfg.dtype)  #  np.float64
    max_writes = cfg.max_writes  #  None
    logger = logging.getLogger(__name__)
    output_dir = cfg.output_dir  # 'outputs_random'
    movie_dir = cfg.movie_dir  # 'MHD_test_random/movie/'
    use_cfl = cfg.use_cfl  # False
    skip_exists = cfg.skip_exists  # False

    ## ID Parameters
    L = cfg.L  # 1
    dim = 2
    Nsamples = cfg.N  # 1
    l_u = cfg.l_u  # 0.1
    l_A = cfg.l_A  # 0.1
    Nu = cfg.Nu  # None
    sigma_u = cfg.sigma_u  # 0.1
    sigma_A = cfg.sigma_A  # 5e-3

    # Generate Random Initial Data
    grf_u = GRF_Mattern(
        dim,
        Nx,
        length=Lx,
        nu=Nu,
        l=l_u,
        sigma=sigma_u,
        boundary="periodic",
        device=device,
    )
    grf_A = GRF_Mattern(
        dim,
        Nx,
        length=Lx,
        nu=Nu,
        l=l_A,
        sigma=sigma_A,
        boundary="periodic",
        device=device,
    )

    u0_pot = grf_u.sample(Nsamples).cpu().numpy().reshape(Nsamples, Nx, Ny)
    A0 = grf_A.sample(Nsamples).cpu().numpy().reshape(Nsamples, Nx, Ny)
    digits = int(math.log10(Nsamples)) + 1

    # expected number of time steps
    Nt = len(np.arange(0, stop_sim_time + Dt, output_dt))
    indices = list(range(Nsamples))

    if skip_exists:
        completed_list = []
        for j in range(Nsamples):
            # print('hi')
            sim_output_dir = os.path.join(output_dir, f"output-{j:0{digits}}")
            sim_outputs = os.path.join(sim_output_dir, "*.h5")
            # skip if the next output directory exists and if the output is complete
            if os.path.exists(sim_output_dir):
                completed = check_if_complete(sim_outputs, Nt=Nt)
            else:
                completed = False
            completed_list.append(completed)
        indices = [j for j, completed in enumerate(completed_list) if not completed]
    print(indices)

    def run_simulation(
        i,
        Lx=Lx,
        Ly=Ly,
        Nx=Nx,
        Ny=Ny,
        Re=Re,
        ReM=ReM,
        Schmidt=Schmidt,
        rho0=rho0,
        dealias=dealias,
        stop_sim_time=stop_sim_time,
        timestepper=timestepper,
        Dt=Dt,
        max_timestep=max_timestep,
        output_dt=output_dt,
        log_iter=log_iter,
        dtype=dtype,
        max_writes=max_writes,
        logger=logger,
        output_dir=output_dir,
        use_cfl=use_cfl,
        L=L,
        dim=dim,
        Nsamples=Nsamples,
        l_u=l_u,
        l_A=l_A,
        Nu=Nu,
        sigma_u=sigma_u,
        sigma_A=sigma_A,
        grf_u=grf_u,
        grf_A=grf_A,
        u0_pot=u0_pot,
        A0=A0,
        digits=digits,
        Nt=Nt,
    ):
        sim_output_dir = os.path.join(output_dir, f"output-{i:0{digits}}")
        sim_outputs = os.path.join(sim_output_dir, "*.h5")
        print(
            f"Running simulation {i:0{digits}} with outputs in {sim_output_dir}",
            flush=True,
        )
        # Bases
        coords = d3.CartesianCoordinates("x", "y")
        dist = d3.Distributor(coords, dtype=dtype)
        xbasis = d3.RealFourier(coords["x"], size=Nx, bounds=(0, Lx), dealias=dealias)
        ybasis = d3.RealFourier(coords["y"], size=Ny, bounds=(0, Ly), dealias=dealias)

        # Fields
        p = dist.Field(name="p", bases=(xbasis, ybasis))
        s = dist.Field(name="s", bases=(xbasis, ybasis))
        u = dist.VectorField(coords, name="u", bases=(xbasis, ybasis))
        B = dist.VectorField(coords, name="B", bases=(xbasis, ybasis))
        A = dist.Field(name="A", bases=(xbasis, ybasis))
        B2 = dist.Field(name="B2", bases=(xbasis, ybasis))
        u_pot = dist.Field(name="u_pot", bases=(xbasis, ybasis))
        Ax = dist.Field(name="Ax", bases=(xbasis, ybasis))
        Ay = dist.Field(name="Ay", bases=(xbasis, ybasis))
        Bx = dist.Field(name="Bx", bases=(xbasis, ybasis))
        By = dist.Field(name="By", bases=(xbasis, ybasis))
        u0 = dist.VectorField(coords, name="u0", bases=(xbasis, ybasis))
        ux = dist.Field(name="ux", bases=(xbasis, ybasis))
        uy = dist.Field(name="uy", bases=(xbasis, ybasis))
        tau_p = dist.Field(name="tau_p")

        # Substitutions
        nu = 1 / Re
        D = nu / Schmidt
        eta = 1 / ReM
        x, y = dist.local_grids(xbasis, ybasis)
        X, Y = np.meshgrid(x, y, indexing="ij")
        ex, ey = coords.unit_vector_fields(dist)
        # ez = d3.CrossProduct(ex, ey)
        curl2d_scalar = lambda x: -d3.skew(d3.grad(x))
        curl2d_vector = lambda x: -d3.div(d3.skew(x))
        B = curl2d_scalar(A)
        B2 = d3.dot(B, B)
        Bx = B @ ex
        By = B @ ey
        ux = u @ ex
        uy = u @ ey

        # Problem
        problem = d3.IVP([u, p, A, tau_p, s], namespace=locals())
        problem.add_equation(
            "dt(u) + grad(p)/rho0 - nu*lap(u) = - 0.5*grad(B2)/rho0 - u@grad(u) + B@grad(B)/rho0"
        )
        problem.add_equation("dt(s) - D*lap(s) = - u@grad(s)")
        problem.add_equation("dt(A) - eta*lap(A) = - u@grad(A)")
        problem.add_equation("div(u) + tau_p = 0")
        problem.add_equation("integ(p) = 0")  # Pressure gauge

        # Solver
        solver = problem.build_solver(timestepper)
        # solver.stop_sim_time = stop_sim_time
        solver.stop_sim_time = (
            stop_sim_time + Dt
        )  # Make sure we record the last timestep

        # Initial conditions
        u_pot["g"] = u0_pot[i]
        u0 = curl2d_scalar(u_pot).evaluate()
        u0.change_scales(1)
        u["g"] = u0["g"]
        ux = u @ ex
        uy = u @ ey
        B2 = d3.dot(B, B)
        # s.set_global_data(u0_pot[i])
        s["g"] = u0_pot[i]
        # A.set_global_data(A0[i])
        A["g"] = A0[i]

        # Analysis (This overwrites existing files)
        os.makedirs(sim_output_dir, exist_ok=True)
        snapshots = solver.evaluator.add_file_handler(
            sim_output_dir, sim_dt=output_dt, max_writes=max_writes
        )

        snapshots.add_task(s, name="tracer")
        snapshots.add_task(A, name="vector potential")
        snapshots.add_task(B, name="magnetic field")

        snapshots.add_task(u, name="velocity")
        snapshots.add_task(p, name="pressure")

        # CFL (Don't actually use this.  Use constant timestep instead)
        CFL = d3.CFL(
            solver,
            initial_dt=max_timestep,
            cadence=10,
            safety=0.2,
            threshold=0.1,
            max_change=1.5,
            min_change=0.5,
            max_dt=max_timestep,
        )
        CFL.add_velocity(u)

        # Flow properties
        flow = d3.GlobalFlowProperty(solver, cadence=10)
        flow.add_property(d3.dot(u, u), name="w2")
        flow.add_property(d3.dot(B, B), name="B2")
        flow.add_property(d3.div(B), name="divB")

        # Main loop
        try:
            logger.info("Starting main loop")
            while solver.proceed:
                if use_cfl:
                    timestep = CFL.compute_timestep()
                else:
                    timestep = Dt
                solver.step(timestep)
                if (solver.iteration) % 10 == 0:
                    max_w = np.sqrt(flow.max("w2"))
                    max_B = np.sqrt(flow.max("B2"))
                    max_divB = flow.max("divB")
                    logger.info(
                        f"Iteration={solver.iteration}, Time={solver.sim_time:#.3g}, dt={timestep:#.3g}, max(w)={max_w:#.3g}, max(B)={max_B:#.3g}, max(div_B)={max_divB:#.3g}"
                    )
            print(
                f"Finished simulation {i:0{digits}} with outputs in {sim_output_dir}",
                flush=True,
            )
        except:
            logger.error("Exception raised, triggering end of main loop.")
            raise
        solver.log_stats()

    # Run in parallel
    with mp.Pool(mp.cpu_count() - 1) as pool:
        pool.map(run_simulation, indices, chunksize=10)
