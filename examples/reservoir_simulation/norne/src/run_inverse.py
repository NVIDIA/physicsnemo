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

# ruff: noqa: RUF001, RUF002
# Greek letters appear in docstrings/log strings to match conventional
# algorithm notation (alpha-REKI); ruff's confusable-character rule is
# suppressed for the file.
"""
SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

=====================================================================
          NVIDIA PHYSICSNEMO RESERVOIR SIMULATION INVERSE UQ MODELLING
=====================================================================

This module implements Bayesian inverse uncertainty quantification (UQ) for reservoir
simulation using NVIDIA PhysicsNeMo. It provides a comprehensive framework for solving
inverse problems in reservoir engineering with advanced data assimilation methods.

Key Features:
- Physics-informed neural operators for black oil reservoir simulation
- Bayesian inverse problem workflow with ensemble Kalman methods
- Weighted Adaptive REKI (α-REKI) with covariance localization
- Support for multiple measurement types (WOPR, WWPR, WGPR)
- Integration with PhysicsNeMo's neural operator surrogates
- Comprehensive uncertainty quantification and analysis

Data Assimilation Methods:
- Weighted Adaptive REKI - Adaptive Regularised Ensemble Kalman (α-REKI)
- Inversion with covariance localization
- 66 Measurements to be matched: 22 WOPR, 22 WWPR, 22 WGPR
- Field configuration: 22 producers, 9 water injectors, 4 gas injectors

Usage:
    python run_inverse.py --config-path=conf --config-name=INVERSE_CONFIG

Inputs:
    - Configuration file with inverse problem parameters
    - Observation data (production rates, pressure measurements)
    - Prior ensemble of reservoir models
    - Neural operator surrogate models

Outputs:
    - Posterior ensemble of reservoir models
    - Uncertainty quantification results
    - Model parameter estimates with confidence intervals
    - Visualization plots for analysis

@Author : Clement Etienam
"""

import multiprocessing
import random
import os
import sys
import time
import pickle
import logging
import gzip
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import yaml

import torch
import hydra
from hydra.utils import to_absolute_path
from physicsnemo.distributed import DistributedManager
from omegaconf import DictConfig

from data_extract.opm_extract_rates import (
    read_compdats2,
    read_compdats,
    extract_qs,
    get_dyna2,
)

from inverse.utils.percentile_ensemble import plot_percentile_models
from inverse.utils.ensemble_generation import (
    setup_models_and_data,
    generate_ensemble,
)
from inverse.utils.ensemble_results import process_final_results
from inverse.utils.inverse_config import (
    EnsembleRuntime,
    EnsembleSetup,
    FlowArrays,
    GridConfig,
    InversionParams,
    NormBounds,
    PermBounds,
    PriorEnsembles,
    SurrogateConfig,
    TimeArrays,
    WellConfig,
)

from inverse.history_matching import run_history_matching_loop
from utils.io_utils import is_available, initialize_environment


def setup_logging() -> logging.Logger:
    """Configure and return the main logger with green INFO console output."""
    logger = logging.getLogger("inverse problem")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers on re-entry

    formatter = logging.Formatter(" %(asctime)s - %(levelname)s - %(message)s")

    # File handler — plain, no colors (keeps log file clean)
    f_handler = logging.FileHandler(filename="read_vectors.log", mode="w")
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)

    # Console handler — colored
    class _ColorFormatter(logging.Formatter):
        COLORS = {
            logging.DEBUG: "\033[0;36m",  # cyan
            logging.INFO: "\033[0;32m",  # green
            logging.WARNING: "\033[1;33m",  # yellow
            logging.ERROR: "\033[0;31m",  # red
            logging.CRITICAL: "\033[1;31m",  # bold red
        }
        RESET = "\033[0m"

        def format(self, record):
            msg = super().format(record)
            if sys.stdout.isatty():
                color = self.COLORS.get(record.levelno, "")
                return f"{color}{msg}{self.RESET}"
            return msg

    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(
        _ColorFormatter(" %(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(c_handler)

    return logger


def load_configuration_data(cfg: DictConfig, logger: logging.Logger) -> dict[str, Any]:
    """Load configuration and experimental data."""
    exper = sio.loadmat(to_absolute_path("../data/exper.mat"))
    experts = exper["expert"]

    # Load conversion data
    mat = sio.loadmat(to_absolute_path("../data/conversions.mat"))
    minK = mat["minK"]
    maxK = mat["maxK"]
    minT = mat["minT"]
    maxT = mat["maxT"]
    minP = mat["minP"]
    maxP = mat["maxP"]
    minQw = mat["minQW"]
    maxQw = mat["maxQw"]
    minQg = mat["minQg"]
    maxQg = mat["maxQg"]
    minQ = mat["minQ"]
    maxQ = mat["maxQ"]

    return {
        "experts": experts,
        "minK": minK,
        "maxK": maxK,
        "minT": minT,
        "maxT": maxT,
        "minP": minP,
        "maxP": maxP,
        "minQw": minQw,
        "maxQw": maxQw,
        "minQg": minQg,
        "maxQg": maxQg,
        "minQ": minQ,
        "maxQ": maxQ,
    }


def setup_ensemble_data(cfg: DictConfig, logger: logging.Logger) -> dict[str, Any]:
    """Setup ensemble data and parameters."""
    # Load ensemble data
    try:
        with gzip.open(
            to_absolute_path("../data/data_train_peaceman.pkl.gz"), "rb"
        ) as f:
            X_data1 = pickle.load(f)
    except (pickle.PickleError, EOFError, FileNotFoundError) as e:
        logger.error(f"Error loading pickle file: {e}")
        raise

    # Extract ensemble data
    perm_ensemble = X_data1["ensemble"]
    poro_ensemble = X_data1["ensemblep"]
    fault_ensemble = X_data1["ensemblefault"]

    # Load effective data
    source_dir = cfg.custom.file_location
    active_mask_3d = np.genfromtxt(Path(source_dir) / "actnum.out", dtype="float")
    effec = active_mask_3d.reshape(-1, 1)

    return {
        "perm_ensemble": perm_ensemble,
        "poro_ensemble": poro_ensemble,
        "fault_ensemble": fault_ensemble,
        "active_mask_3d": active_mask_3d,
        "effec": effec,
    }


@hydra.main(version_base="1.2", config_path="conf", config_name="DECK_CONFIG")
def main(cfg: DictConfig) -> None:
    """Main function for inverse problem solving."""
    # Initialize environment and logging
    gpu_available, logger = initialize_environment()
    DistributedManager.initialize()
    dist = DistributedManager()

    if torch.cuda.is_available():
        torch.cuda.set_device(dist.rank % torch.cuda.device_count())
    device = dist.device
    oldfolder = os.getcwd()

    if dist.rank == 0:
        logger.info(f"World size: {dist.world_size}, Rank: {dist.rank}")
    logger.info(
        f"Rank {dist.rank}: device = {device}, "
        f"cuda current = {torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'}"
    )

    if dist.rank == 0:
        navail = multiprocessing.cpu_count()
        logger.info(f"Available CPU cores: {navail}")
    njobs = max(1, multiprocessing.cpu_count() // 5)
    num_cores = njobs

    DEFAULT = cfg.custom.INVERSE_PROBLEM.DEFAULT
    well_measurements = cfg.custom.well_measurements
    if DEFAULT == "Yes" and dist.rank == 0:
        logger.info(
            "Default configuration selected for inverse modelling, sit back and relax....."
        )
    TEMPLATEFILE = {}
    TEMPLATEFILE["Surrogate model"] = "PINO" if cfg.custom.fno_type == "PINO" else "FNO"

    # ── Load conversion data ─────────────────────────────────────────────────
    exper = sio.loadmat(to_absolute_path("../data/exper.mat"))
    experts = exper["expert"]
    mat = sio.loadmat(to_absolute_path("../data/conversions.mat"))
    minK = mat["minK"]
    maxK = mat["maxK"]
    minT = mat["minT"]
    maxT = mat["maxT"]
    minP = mat["minP"]
    maxP = mat["maxP"]
    minQw = mat["minQW"]
    maxQw = mat["maxQW"]
    minQg = mat["minQg"]
    maxQg = mat["maxQg"]
    minQ = mat["minQ"]
    maxQ = mat["maxQ"]
    min_inn_fcn = mat["min_inn_fcn"]
    max_inn_fcn = mat["max_inn_fcn"]
    min_out_fcn = mat["min_out_fcn"]
    max_out_fcn = mat["max_out_fcn"]
    min_inn_fcn2 = mat["min_inn_fcn2"]
    max_inn_fcn2 = mat["max_inn_fcn2"]
    min_out_fcn2 = mat["min_out_fcn2"]
    max_out_fcn2 = mat["max_out_fcn2"]
    steppi = int(mat["steppi"])
    steppi_indices = mat["steppi_indices"].flatten()
    effective = mat["effective"]
    target_min = 0.01
    target_max = 1
    nx = cfg.custom.PROPS.nx
    ny = cfg.custom.PROPS.ny
    nz = cfg.custom.PROPS.nz
    effective = np.reshape(effective, (nx * ny * nz, -1), "F")
    effec = np.reshape(effective[:, 0], (-1, 1), "F")

    # ── Load static ensemble data ────────────────────────────────────────────
    try:
        with gzip.open(to_absolute_path("../data/static.pkl.gz"), "rb") as f2:
            mat = pickle.load(f2)
    except (pickle.PickleError, EOFError, FileNotFoundError) as e:
        logger.error(f"Error loading static pickle file: {e}")
        raise
    X_data1 = mat
    if dist.rank == 0:
        for key, value in X_data1.items():
            logger.info(
                "****************************************************************"
            )
            logger.info(f"For key '{key}':")
            logger.info(f"\tContains inf: {np.isinf(value).any()}")
            logger.info(f"\tContains -inf: {np.isinf(-value).any()}")
            logger.info(f"\tContains NaN: {np.isnan(value).any()}")
            logger.info(
                "****************************************************************"
            )

    # Seed RNGs identically across ranks
    seed = int(cfg.custom.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    perm_ensembley = X_data1["ensemble"]
    poro_ensembley = X_data1["ensemblep"]
    fault_ensemblepy = X_data1["ensemblefault"]
    source_dir = cfg.custom.file_location
    active_mask_3d = np.genfromtxt(Path(source_dir) / "actnum.out", dtype="float")
    effec = active_mask_3d.reshape(-1, 1)
    active_mask_3d = np.reshape(active_mask_3d, (nx, ny, nz), "F")
    active_cells_ensemble = active_mask_3d
    experts = int(cfg.custom.Type_of_experts)
    lenwels = len(well_measurements)
    if dist.rank == 0:
        logger.info(f"lenwels = {lenwels}")

    filename = cfg.custom.COMPLETIONS_DATA
    gas_injectors, producers, injectors = read_compdats2(
        cfg.custom.COMPLETIONS_DATA, cfg.custom.SUMMARY_DATA
    )
    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|                         PRINT WELLS                           : |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info("gas injectors wells")
        logger.info(str(gas_injectors))
        logger.info("producer well")
        logger.info(str(producers))
        logger.info("water injector well")
        logger.info(str(injectors))
        logger.info("****************************************************************")

    N_injw = len(injectors)
    N_pr = len(producers)
    N_injg = len(gas_injectors)
    if dist.rank == 0:
        logger.info(f"N_pr = {N_pr}")
    well_names = [entry[-1] for entry in producers]
    well_namesg = [entry[-1] for entry in gas_injectors]
    well_namesw = [entry[-1] for entry in injectors]

    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|                         PRINT WELL NAMES                      : |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info("producer well names")
        logger.info(str(well_names))
        logger.info("gas injectors well names")
        logger.info(str(well_namesg))
        logger.info("water injector well names")
        logger.info(str(well_namesw))
        logger.info("****************************************************************")

    compdat_data = read_compdats(filename, well_names)
    compdat_datag = read_compdats(filename, well_namesg)
    compdat_dataw = read_compdats(filename, well_namesw)
    filenamea = os.path.basename(cfg.custom.DECK)
    filenameui = os.path.splitext(filenamea)[0]

    Qg = np.zeros((steppi, nx, ny, nz))
    Qw = np.zeros((steppi, nx, ny, nz))
    Qo = np.zeros((steppi, nx, ny, nz))

    os.chdir(to_absolute_path("../RESULTS/FORWARD_RESULTS/RESULTS/True_Flow"))
    seeg, seew = extract_qs(
        steppi, steppi_indices, filenameui, injectors, gas_injectors, filename
    )
    os.chdir(oldfolder)
    awater, agas, aoil = get_dyna2(
        steppi, compdat_dataw, compdat_datag, compdat_data, Qw, Qg, Qo, seew, seeg
    )
    aqq = awater + agas + aoil

    TEMPLATEFILE["Kalman update"] = "Exotic"
    TEMPLATEFILE["weighting"] = "Non Weighted innovation"
    excel = 2
    use_pretrained = (
        "Yes" if DEFAULT == "Yes" else cfg.custom.INVERSE_PROBLEM["Pretrained Model"]
    )
    TEMPLATEFILE["Use pretrained model"] = use_pretrained
    input_variables = cfg.custom.input_properties
    output_variables = cfg.custom.output_properties

    Ne = 500 if DEFAULT == "Yes" else int(cfg.custom.INVERSE_PROBLEM.Ensemble_size)
    N_ens = Ne

    (
        models,
        TEMPLATEFILE,
        True_data,
        True_mat,
        _True_dataTI,
        rows_to_remove,
        Time_unie1,
        timestep,
        indii,
        Low_K1,
        High_K1,
        Low_K,
        High_K,
        Low_P,
        High_P,
        pred_type,
        degg,
        _rho,
        Trainmoe,
        _BASSE,
        Time,
        True_K,
    ) = setup_models_and_data(
        input_variables=input_variables,
        output_variables=output_variables,
        runtime=EnsembleRuntime(
            cfg=cfg,
            dist=dist,
            device=device,
            oldfolder=oldfolder,
            DEFAULT=DEFAULT,
            excel=excel,
            TEMPLATEFILE=TEMPLATEFILE,
        ),
        grid=GridConfig(
            nx=nx,
            ny=ny,
            nz=nz,
            steppi=steppi,
            steppi_indices=steppi_indices,
        ),
        well=WellConfig(
            N_pr=N_pr,
            lenwels=lenwels,
            well_names=well_names,
        ),
        priors=PriorEnsembles(
            perm=perm_ensembley,
            poro=poro_ensembley,
            fault=fault_ensemblepy,
        ),
        Ne=Ne,
        minK=minK,
        maxK=maxK,
    )

    # path = os.getcwd()
    os.chdir(oldfolder)
    noise_level = 25 if DEFAULT == "Yes" else cfg.custom.INVERSE_PROBLEM["Noise_level"]
    noise_level = noise_level / 100
    if DEFAULT == "Yes":
        Deccor = "No"
        if dist.rank == 0:
            logger.info("No initial ensemble decorrrlation")
    else:
        Deccor = cfg.custom.INVERSE_PROBLEM.Decorrelationn
    if Deccor == "Yes":
        TEMPLATEFILE["Ensemble decorrelation"] = "ensemble decorrelation = Yes"
    else:
        TEMPLATEFILE["Ensemble decorrelation"] = "ensemble decorrelation = No"
    if DEFAULT == "Yes" and dist.rank == 0:
        logger.info("Using reccomended alpha value")

    if DEFAULT == "Yes" and dist.rank == 0:
        logger.info("Random generated ensemble")

    TEMPLATEFILE["Data assimilation method"] = (
        "ADAPT_REKI (Vanilla Adaptive Ensemble Kalman Inversion)\n"
    )
    Termm = 20 if DEFAULT == "Yes" else cfg.custom.INVERSE_PROBLEM.iteration_count
    TEMPLATEFILE["Iterations"] = Termm

    if DEFAULT == "Yes":
        Do_parametrisation = "No"
        Do_param_method = cfg.custom.INVERSE_PROBLEM.Do_param_method
    else:
        Do_parametrisation = cfg.custom.INVERSE_PROBLEM.parametrization_options
        Do_param_method = cfg.custom.INVERSE_PROBLEM.Do_param_method
    if Do_parametrisation == "No":
        TEMPLATEFILE["Domain parametrisation"] = (
            "domain parametrisation during inverse problem = No"
        )
        if DEFAULT == "Yes":
            do_localisation = "Yes"
            if dist.rank == 0:
                logger.info("Doing covariance localisation")
        else:
            do_localisation = cfg.custom.INVERSE_PROBLEM.Covariance_localisation
        if do_localisation == "Yes":
            TEMPLATEFILE["Covariance localisation"] = "Covariance localisaion = Yes"
        else:
            TEMPLATEFILE["Covariance localisation"] = "Covariance localisaion = No"
        sizedct = cfg.custom.INVERSE_PROBLEM.DCT
        sizedct = sizedct / 100
        size1, size2 = int(np.ceil(int(sizedct * nx))), int(np.ceil(int(sizedct * ny)))
    else:
        do_localisation = "No"
        TEMPLATEFILE["Do_parametrisation"] = (
            "domain parametrisation during inverse problem = yes"
        )
        if Do_param_method == "DCT":
            TEMPLATEFILE["parametrisation method"] = "Discrete cosine transform\n"
        else:
            TEMPLATEFILE["parametrisation method"] = "Variational Conv. Autoencoder\n"
        sizedct = cfg.custom.INVERSE_PROBLEM.DCT
        sizedct = sizedct / 100
        size1, size2 = int(np.ceil(int(sizedct * nx))), int(np.ceil(int(sizedct * ny)))
    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|                 SOLVE INVERSE PROBLEM WITH WEIGHTED α-REKI:     |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )
    if do_localisation == "No":
        logger.info(
            "History Matching using the Adaptive Regularised Ensemble Kalman Inversion (α-REKI)"
        )
        logger.info("Novel Implementation by Clement Etienam, DevTech Energy - NVIDIA")
    else:
        logger.info(
            "History Matching using the Adaptive Regularised Ensemble Kalman Inversion (α-REKI) with covariance localisation"
        )
        logger.info("Novel Implementation by Clement Etienam, DevTech Energy - NVIDIA")

    # ── build grouped config objects used by all four pipeline functions ──────
    grid = GridConfig(
        nx=nx,
        ny=ny,
        nz=nz,
        steppi=steppi,
        steppi_indices=steppi_indices,
    )
    norm = NormBounds(
        minK=minK,
        maxK=maxK,
        minT=minT,
        maxT=maxT,
        minP=minP,
        maxP=maxP,
        minQ=minQ,
        maxQ=maxQ,
        minQw=minQw,
        maxQw=maxQw,
        minQg=minQg,
        maxQg=maxQg,
        min_inn_fcn=min_inn_fcn,
        max_inn_fcn=max_inn_fcn,
        min_out_fcn=min_out_fcn,
        max_out_fcn=max_out_fcn,
        min_inn_fcn2=min_inn_fcn2,
        max_inn_fcn2=max_inn_fcn2,
        min_out_fcn2=min_out_fcn2,
        max_out_fcn2=max_out_fcn2,
        target_min=target_min,
        target_max=target_max,
    )
    perm = PermBounds(
        High_K=High_K,
        Low_K=Low_K,
        High_K1=High_K1,
        Low_K1=Low_K1,
        High_P=High_P,
        Low_P=Low_P,
    )
    ens_setup = EnsembleSetup(
        Ne=Ne,
        N_ens=N_ens,
        effective=effective,
        effec=effec,
        active_cells_ensemble=active_cells_ensemble,
        active_mask_3d=active_mask_3d,
        rows_to_remove=rows_to_remove,
        indii=indii,
    )
    well = WellConfig(
        producers=producers,
        injectors=injectors,
        gas_injectors=gas_injectors,
        well_names=well_names,
        N_pr=N_pr,
        N_injw=N_injw,
        N_injg=N_injg,
        lenwels=lenwels,
        compdat_data=compdat_data,
    )
    surrogate = SurrogateConfig(
        models=models,
        Trainmoe=Trainmoe,
        pred_type=pred_type,
        degg=degg,
        experts=experts,
        num_cores=num_cores,
    )
    inversion = InversionParams(
        Termm=Termm,
        Do_parametrisation=Do_parametrisation,
        Do_param_method=Do_param_method,
        do_localisation=do_localisation,
        size1=size1,
        size2=size2,
        noise_level=noise_level,
        Deccor=Deccor,
    )
    time_arr = TimeArrays(Time=Time, Time_unie1=Time_unie1, timestep=timestep)
    flow = FlowArrays(awater=awater, agas=agas, aoil=aoil, aqq=aqq)

    (
        ensemble,
        ensemblep,
        ensemblef,
        ini_ensemble,
        _ini_ensemblep,
        _ini_ensemblefault,
        True_data,
        True_mat,
        dt,
        _Nop,
        CDd,
        perturbations,
        start_time,
    ) = generate_ensemble(
        cfg=cfg,
        dist=dist,
        device=device,
        oldfolder=oldfolder,
        TEMPLATEFILE=TEMPLATEFILE,
        gpu_available=gpu_available,
        grid=grid,
        perm=perm,
        ens=ens_setup,
        well=well,
        time_arr=time_arr,
        inversion=inversion,
    )

    iteration_converged = 0
    iteration_count = 0

    (
        use_k,
        use_p,
        use_f,
        mean_cost,
        best_cost,
        ensemble_bestK,
        ensemble_meanK,
        ensemble_bestP,
        ensemble_meanP,
        ensemble_bestf,
        ensemble_meanf,
        iteration_count,
        iteration_converged,
        _alpha_big,
        ensemble,
        ensemblep,
        ensemblef,
        chm,
        cc_ini,
        ensemble_dict,
        base_k,
        base_p,
        base_f,
    ) = run_history_matching_loop(
        dist=dist,
        logger=logger,
        cfg=cfg,
        device=device,
        oldfolder=oldfolder,
        gpu_available=gpu_available,
        iteration_converged=iteration_converged,
        iteration_count=iteration_count,
        input_variables=input_variables,
        output_variables=output_variables,
        ensemble=ensemble,
        ensemblep=ensemblep,
        ensemblef=ensemblef,
        CDd=CDd,
        True_mat=True_mat,
        perturbations=perturbations,
        grid=grid,
        norm=norm,
        perm=perm,
        ens=ens_setup,
        well=well,
        surrogate=surrogate,
        inversion=inversion,
        time_arr=time_arr,
        flow=flow,
    )

    yes_best = {}
    ensemble_best = {}
    yes_mean = {}
    ensemble_mean = {}
    all_ensemble = {}

    X_data1 = process_final_results(
        cfg=cfg,
        dist=dist,
        device=device,
        oldfolder=oldfolder,
        input_variables=input_variables,
        output_variables=output_variables,
        ensemble_bestK=ensemble_bestK,
        ensemble_meanK=ensemble_meanK,
        ensemble_bestP=ensemble_bestP,
        ensemble_meanP=ensemble_meanP,
        ensemble_bestf=ensemble_bestf,
        ensemble_meanf=ensemble_meanf,
        ensemble=ensemble,
        ensemblep=ensemblep,
        ensemblef=ensemblef,
        use_k=use_k,
        use_p=use_p,
        use_f=use_f,
        chm=chm,
        dt=dt,
        cc_ini=cc_ini,
        mean_cost=mean_cost,
        best_cost=best_cost,
        ini_ensemble=ini_ensemble,
        ensemble_dict=ensemble_dict,
        yes_best=yes_best,
        ensemble_best=ensemble_best,
        yes_mean=yes_mean,
        ensemble_mean=ensemble_mean,
        all_ensemble=all_ensemble,
        True_mat=True_mat,
        True_data=True_data,
        True_K=True_K,
        grid=grid,
        norm=norm,
        perm=perm,
        ens=ens_setup,
        well=well,
        surrogate=surrogate,
        time_arr=time_arr,
        flow=flow,
    )
    os.chdir(oldfolder)
    ensembleout = {}
    ensembleoutf1 = None
    if "PERM" in input_variables:
        ensembleout1 = np.hstack(
            [
                X_data1["P10_Perm"],
                X_data1["P50_Perm"],
                X_data1["P90_Perm"],
                X_data1["yes_best"]["PERM"],
                X_data1["yes_mean"]["PERM"],
                base_k,
            ]
        )
        ensembleout["PERM"] = ensembleout1
    if "PORO" in input_variables:
        ensembleoutp1 = np.hstack(
            [
                X_data1["P10_Poro"],
                X_data1["P50_Poro"],
                X_data1["P90_Poro"],
                X_data1["yes_best"]["PORO"],
                X_data1["yes_mean"]["PORO"],
                base_p,
            ]
        )
        ensembleout["PORO"] = ensembleoutp1
    if "FAULT" in input_variables:
        ensembleoutf1 = np.hstack(
            [
                X_data1["P10_Fault"],
                X_data1["P50_Fault"],
                X_data1["P90_Fault"],
                X_data1["yes_best"]["FAULT"],
                X_data1["yes_mean"]["FAULT"],
                base_f,
            ]
        )
        ensembleout["FAULT"] = ensembleoutf1

    plot_percentile_models(
        cfg=cfg,
        dist=dist,
        device=device,
        oldfolder=oldfolder,
        input_variables=input_variables,
        output_variables=output_variables,
        ensembleout=ensembleout,
        ensembleoutf1=ensembleoutf1,
        base_k=base_k,
        X_data1=X_data1,
        True_mat=True_mat,
        True_K=True_K,
        grid=grid,
        norm=norm,
        ens=ens_setup,
        well=well,
        surrogate=surrogate,
        time_arr=time_arr,
        flow=flow,
    )
    os.chdir(oldfolder)
    if dist.rank == 0:
        logger.info("****************************************************************")
        logger.info("              SECTION ADAPTIVE REKI (α-REKI) ENDED              ")
        logger.info("****************************************************************")
    elapsed_time_secs = time.time() - start_time
    comment = "Adaptive Regularised Ensemble Kalman Inversion"
    comment2 = "PINO-CCR" if Trainmoe == "MoE" else "PINO-FNO"
    if dist.rank == 0:
        logger.info("Inverse problem solution used =: " + comment)
        logger.info("Forward model surrogate =: " + comment2)
        logger.info("Ensemble size = " + str(Ne))
    msg = "Execution took: {} secs (Wall clock time)".format(
        timedelta(seconds=round(elapsed_time_secs))
    )
    if dist.rank == 0:
        logger.info(msg)
    TEMPLATEFILE["Inverse problem solution used =: "] = comment
    TEMPLATEFILE["Forward model surrogate =: "] = comment2
    TEMPLATEFILE["Ensemble size = "] = Ne
    TEMPLATEFILE["Execution in secs = "] = timedelta(seconds=round(elapsed_time_secs))
    if dist.rank == 0:
        logger.info("****************************************************************")
        logger.info("        HISTORY MATCHING OPERATIONAL CONDITIONS                 ")
        logger.info("****************************************************************")
        for key, value in TEMPLATEFILE.items():
            logger.info(f"{key}: {value}")
    yaml_filename = to_absolute_path(
        "../RESULTS/HM_RESULTS/History_Matching_Template_file.yaml"
    )
    if dist.rank == 0:
        with open(yaml_filename, "w") as yaml_file:
            yaml.dump(TEMPLATEFILE, yaml_file)
    if dist.rank == 0:
        logger.info(
            "-------------------PROGRAM EXECUTED-------------------------------------"
        )


if __name__ == "__main__":
    try:
        avail_code = is_available()
        setup_logging().info(
            "GPU Available with CUDA" if avail_code == 0 else "No GPU Available"
        )
    except Exception:
        setup_logging().info("No GPU Available")
    main()
