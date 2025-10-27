"""
SPDX-FileCopyrightText: Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
simulation using NVIDIA PhyNeMo. It provides a comprehensive framework for solving
inverse problems in reservoir engineering with advanced data assimilation methods.

Key Features:
- Physics-informed neural operators for black oil reservoir simulation
- Bayesian inverse problem workflow with ensemble Kalman methods
- Weighted Adaptive REKI (α-REKI) with covariance localization
- Support for multiple measurement types (WOPR, WWPR, WGPR)
- Integration with PhyNeMo's neural operator surrogates
- Comprehensive uncertainty quantification and analysis

Data Assimilation Methods:
- Weighted Adaptive REKI - Adaptive Regularised Ensemble Kalman (α-REKI)
- Inversion with covariance localization
- 66 Measurements to be matched: 22 WOPR, 22 WWPR, 22 WGPR
- Field configuration: 22 producers, 9 water injectors, 4 gas injectors

Usage:
    python Inverse_problem.py --config-path=conf --config-name=INVERSE_CONFIG

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

# 🛠 Standard Library
import multiprocessing
import os
import time
import pickle
import logging
import gzip
import random as ra
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Dict, Any

# 🔧 Third-party Libraries
import numpy as np
import numpy.matlib
import scipy.io as sio
import yaml

# 🔥 PhyNeMo & ML Libraries
import torch
import hydra
from hydra.utils import to_absolute_path
from physicsnemo.distributed import DistributedManager
from omegaconf import DictConfig

# 📦 Local Modules

from data_extract.opm_extract_rates import read_compdats2, read_compdats

from inverse.utils.percentile_ensemble import plot_percentile_models
from inverse.utils.ensemble_generation import (
    setup_models_and_data,
    generate_ensemble,
)
from inverse.utils.ensemble_results import process_final_results

from inverse.history_matching import run_history_matching_loop


def setup_logging() -> logging.Logger:
    """Configure and return the main logger."""
    logger = logging.getLogger("Inverse problem")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    warnings.filterwarnings("ignore")
    return logger


def initialize_environment() -> logging.Logger:
    """Initialize the environment and return logger."""
    logger = setup_logging()
    logger.info(__doc__)
    return logger


def load_configuration_data(cfg: DictConfig, logger: logging.Logger) -> Dict[str, Any]:
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


def setup_ensemble_data(cfg: DictConfig, logger: logging.Logger) -> Dict[str, Any]:
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
    effective_abi = np.genfromtxt(Path(source_dir) / "actnum.out", dtype="float")
    effec = effective_abi.reshape(-1, 1)

    return {
        "perm_ensemble": perm_ensemble,
        "poro_ensemble": poro_ensemble,
        "fault_ensemble": fault_ensemble,
        "effective_abi": effective_abi,
        "effec": effec,
    }


def is_available() -> int:
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        code = result.returncode
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        code = 1
    return code


@hydra.main(version_base="1.2", config_path="conf", config_name="DECK_CONFIG")
def main(cfg: DictConfig) -> None:
    """Main function for inverse problem solving."""
    # Initialize environment and logging
    logger = initialize_environment()
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    oldfolder = os.getcwd()
    os.chdir(oldfolder)
    gpu_available = is_available()
    # cur_dir unused
    if dist.rank == 0:
        navail = multiprocessing.cpu_count()
        logger.info(f"Available CPU cores: {navail}")
    njobs = max(1, multiprocessing.cpu_count() // 5)  # Ensure at least 1 core is used
    num_cores = njobs
    DEFAULT = cfg.custom.INVERSE_PROBLEM.DEFAULT
    well_measurements = cfg.custom.well_measurements
    if DEFAULT == "Yes":
        if dist.rank == 0:
            logger.info(
                "Default configuration selected for inverse modelling, sit back and relax....."
            )
    else:
        pass
    TEMPLATEFILE = {}
    # surrogate unused
    if cfg.custom.fno_type == "PINO":
        TEMPLATEFILE["Surrogate model"] = "PINO"
    else:
        TEMPLATEFILE["Surrogate model"] = "FNO"
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
    # train_interest unused
    target_min = 0.01
    target_max = 1
    nx = cfg.custom.PROPS.nx
    ny = cfg.custom.PROPS.ny
    nz = cfg.custom.PROPS.nz
    effective = np.reshape(effective, (nx * ny * nz, -1), "F")
    effec = np.reshape(effective[:, 0], (-1, 1), "F")
    try:
        with gzip.open(to_absolute_path("../data/static.pkl.gz"), "rb") as f2:
            mat = pickle.load(f2)
    except (pickle.PickleError, EOFError, FileNotFoundError) as e:
        logger.error(f"Error loading static pickle file: {e}")
        raise
    X_data1 = mat
    for key, value in X_data1.items():
        if dist.rank == 0:
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
    perm_ensembley = X_data1["ensemble"]
    poro_ensembley = X_data1["ensemblep"]
    fault_ensemblepy = X_data1["ensemblefault"]
    source_dir = cfg.custom.file_location
    # filenamea = os.path.basename(cfg.custom.DECK)
    # filenameui = os.path.splitext(filenamea)[0]
    effective_abi = np.genfromtxt(Path(source_dir) / "actnum.out", dtype="float")
    effec = effective_abi.reshape(-1, 1)
    effective_abi = np.reshape(effective_abi, (nx, ny, nz), "F")
    effectiveuse = effective_abi
    # remove unused min/max vars
    experts = int(cfg.custom.Type_of_experts)
    # len_scale = cfg.custom.geostats.len_scale
    well_measurements = cfg.custom.well_measurements
    lenwels = len(well_measurements)
    logger.info(str(lenwels))
    filename = cfg.custom.COMPLETIONS_DATA
    gass, producers, injectors = read_compdats2(
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
        logger.info(str(gass))
        logger.info("producer well")
        logger.info(str(producers))
        logger.info("water injector well")
        logger.info(str(injectors))
        logger.info("****************************************************************")
    N_injw = len(injectors)
    N_pr = len(producers)  # Number of producers
    logger.info(str(N_pr))
    N_injg = len(gass)
    well_names = [entry[-1] for entry in producers]
    well_namesg = [entry[-1] for entry in gass]  # Adjust index as needed
    well_namesw = [entry[-1] for entry in injectors]  # Adjust index as needed
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
    seed = 1  # 1 is the best
    ra.seed(seed)
    torch.manual_seed(seed)
    # input_channel = 5  # [K,phi,fault,Pini,Sini]
    # output_channel = 4  # [ SWAT,SOIL,SGAS,PRESURE]
    TEMPLATEFILE["Kalman update"] = "Exotic"
    TEMPLATEFILE["weighting"] = "Non Weighted innovation"
    excel = 2
    if DEFAULT == "Yes":
        use_pretrained = "Yes"
    else:
        use_pretrained = cfg.custom.INVERSE_PROBLEM["Pretrained Model"]
    TEMPLATEFILE["Use pretrained model"] = use_pretrained
    input_variables = cfg.custom.input_properties
    output_variables = cfg.custom.output_properties

    if DEFAULT == "Yes":
        Ne = 500  # int(cfg.custom["ntrain"])
    else:
        Ne = int(cfg.custom.INVERSE_PROBLEM.Ensemble_size)
    N_ens = Ne

    (
        models,
        TEMPLATEFILE,
        quant_big,
        True_data,
        True_mat,
        True_dataTI,
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
        rho,
        Trainmoe,
        BASSE,
        Time,
        True_K,
    ) = setup_models_and_data(
        input_variables=input_variables,
        output_variables=output_variables,
        TEMPLATEFILE=TEMPLATEFILE,
        cfg=cfg,
        dist=dist,
        Ne=Ne,
        steppi=steppi,
        N_pr=N_pr,
        lenwels=lenwels,
        device=device,
        excel=excel,
        oldfolder=oldfolder,
        DEFAULT=DEFAULT,
        perm_ensembley=perm_ensembley,
        poro_ensembley=poro_ensembley,
        fault_ensemblepy=fault_ensemblepy,
        nx=nx,
        ny=ny,
        nz=nz,
        steppi_indices=steppi_indices,
        well_names=well_names,
        minK=minK,
        maxK=maxK,
    )

    # path = os.getcwd()
    os.chdir(oldfolder)
    if DEFAULT == "Yes":
        noise_level = 25
    else:
        noise_level = cfg.custom.INVERSE_PROBLEM["Noise_level"]
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
    if DEFAULT == "Yes":
        # De_alpha = "Yes"
        if dist.rank == 0:
            logger.info("Using reccomended alpha value")

    if DEFAULT == "Yes":
        # afresh = "pretrained"
        if dist.rank == 0:
            logger.info("Random generated ensemble")

    TEMPLATEFILE["Data assimilation method"] = (
        "ADAPT_REKI (Vanilla Adaptive Ensemble Kalman Inversion)\n"
    )
    if DEFAULT == "Yes":
        Termm = 20
    else:
        Termm = cfg.custom.INVERSE_PROBLEM.iteration_count
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

    (
        ensemble,
        ensemblep,
        ensemblef,
        ini_ensemble,
        ini_ensemblep,
        ini_ensemblefault,
        True_data,
        True_mat,
        dt,
        Nop,
        CDd,
        perturbations,
        start_time,
    ) = generate_ensemble(
        Ne,
        cfg,
        dist,
        nx,
        ny,
        nz,
        steppi,
        steppi_indices,
        N_pr,
        indii,
        lenwels,
        quant_big,
        rows_to_remove,
        High_K,
        Low_K,
        High_P,
        Low_P,
        effec,
        N_ens,
        High_K1,
        Low_K1,
        timestep,
        well_names,
        Time_unie1,
        oldfolder,
        TEMPLATEFILE,
        gpu_available,
        device,
        noise_level,
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
        alpha_big,
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
        dist,
        logger,
        cfg,
        iteration_converged,
        iteration_count,
        Termm,
        input_variables,
        ensemble,
        ensemblep,
        ensemblef,
        nx,
        ny,
        nz,
        Ne,
        effective,
        oldfolder,
        target_min,
        target_max,
        minK,
        maxK,
        minT,
        maxT,
        minP,
        maxP,
        minQ,
        maxQ,
        minQw,
        maxQw,
        minQg,
        maxQg,
        steppi,
        device,
        steppi_indices,
        min_inn_fcn,
        max_inn_fcn,
        models,
        min_out_fcn,
        max_out_fcn,
        Time,
        effectiveuse,
        Trainmoe,
        num_cores,
        pred_type,
        degg,
        experts,
        min_out_fcn2,
        max_out_fcn2,
        min_inn_fcn2,
        max_inn_fcn2,
        producers,
        compdat_data,
        output_variables,
        quant_big,
        N_pr,
        lenwels,
        effective_abi,
        rows_to_remove,
        timestep,
        Time_unie1,
        well_names,
        CDd,
        gpu_available,
        Do_parametrisation,
        Do_param_method,
        size1,
        size2,
        Low_K1,
        High_K1,
        High_P,
        Low_P,
        N_ens,
        do_localisation,
        gass,
        injectors,
        effec,
        True_mat,
        perturbations,
    )

    # best_cost1 = np.vstack(best_cost)
    # chb = np.argmin(best_cost1)
    yes_best = {}
    ensemble_best = {}
    yes_mean = {}
    ensemble_mean = {}
    all_ensemble = {}

    X_data1 = process_final_results(
        input_variables,
        output_variables,
        ensemble_bestK,
        ensemble_meanK,
        ensemble_bestP,
        ensemble_meanP,
        ensemble_bestf,
        ensemble_meanf,
        ensemble,
        ensemblep,
        ensemblef,
        use_k,
        use_p,
        use_f,
        chm,
        Ne,
        nx,
        ny,
        nz,
        N_ens,
        High_K1,
        Low_K1,
        High_P,
        Low_P,
        effec,
        effective,
        oldfolder,
        target_min,
        target_max,
        minK,
        maxK,
        minT,
        maxT,
        minP,
        maxP,
        minQ,
        maxQ,
        minQw,
        maxQw,
        minQg,
        maxQg,
        steppi,
        device,
        steppi_indices,
        cfg,
        models,
        min_inn_fcn,
        max_inn_fcn,
        min_out_fcn,
        max_out_fcn,
        Time,
        effectiveuse,
        Trainmoe,
        num_cores,
        pred_type,
        degg,
        experts,
        min_out_fcn2,
        max_out_fcn2,
        min_inn_fcn2,
        max_inn_fcn2,
        producers,
        compdat_data,
        quant_big,
        N_pr,
        lenwels,
        effective_abi,
        rows_to_remove,
        True_mat,
        True_data,
        Time_unie1,
        well_names,
        dt,
        dist,
        cc_ini,
        mean_cost,
        best_cost,
        ini_ensemble,
        N_injw,
        N_injg,
        injectors,
        gass,
        True_K,
        yes_best,
        ensemble_best,
        yes_mean,
        ensemble_mean,
        all_ensemble,
        ensemble_dict,
    )
    os.chdir(oldfolder)
    ensembleout = {}
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
        ensembleout,
        ensembleoutf1,
        nx,
        ny,
        nz,
        effective,
        oldfolder,
        target_min,
        target_max,
        minK,
        maxK,
        minT,
        maxT,
        minP,
        maxP,
        minQ,
        maxQ,
        minQw,
        maxQw,
        minQg,
        maxQg,
        steppi,
        device,
        steppi_indices,
        input_variables,
        cfg,
        models,
        min_inn_fcn,
        max_inn_fcn,
        min_out_fcn,
        max_out_fcn,
        Time,
        effectiveuse,
        Trainmoe,
        num_cores,
        pred_type,
        degg,
        experts,
        min_out_fcn2,
        max_out_fcn2,
        min_inn_fcn2,
        max_inn_fcn2,
        producers,
        compdat_data,
        output_variables,
        quant_big,
        N_pr,
        lenwels,
        effective_abi,
        rows_to_remove,
        True_mat,
        Time_unie1,
        well_names,
        True_K,
        base_k,
        X_data1,
        dist,
        N_injw,
        N_injg,
        injectors,
        gass,
    )
    os.chdir(oldfolder)
    if dist.rank == 0:
        logger.info("****************************************************************")
        logger.info("              SECTION ADAPTIVE REKI (α-REKI) ENDED              ")
        logger.info("****************************************************************")
    elapsed_time_secs = time.time() - start_time
    comment = "Adaptive Regularised Ensemble Kalman Inversion"
    if Trainmoe == "MoE":
        comment2 = "PINO-CCR"
    else:
        comment2 = "PINO-FNO"
    if dist.rank == 0:
        logger.info("Inverse problem solution used =: " + comment)
        logger.info("Forward model surrogate =: " + comment2)
        logger.info("Ensemble size = " + str(Ne))
    msg = "Execution took: %s secs (Wall clock time)" % timedelta(
        seconds=round(elapsed_time_secs)
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
