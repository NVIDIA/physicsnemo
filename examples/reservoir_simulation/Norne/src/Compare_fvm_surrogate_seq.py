"""
SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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
                    FVM SURROGATE COMPARISON - SEQUENTIAL PROCESSING
=====================================================================

This module provides sequential processing capabilities for comparing FVM (Finite Volume Method)
surrogate models with actual simulation results. It includes functions for ensemble
processing, result comparison, and performance analysis.

Key Features:
- Sequential processing of ensemble simulations
- FVM surrogate model comparison
- Performance metrics calculation
- Result visualization and analysis
- Memory optimization for large datasets

Usage:
    from Compare_fvm_surrogate_seq import (
        run_sequential_comparison,
        process_ensemble_results,
        analyze_performance_metrics
    )

Inputs:
    - Configuration parameters
    - Ensemble data arrays
    - Surrogate model specifications
    - Comparison settings

Outputs:
    - Comparison results
    - Performance metrics
    - Visualization outputs
    - Logged analysis results

@Author : Clement Etienam
"""

# 🛠 Standard Library
import os
import time
import gzip
import shutil
import logging
import pickle
import warnings
import multiprocessing
import subprocess
from datetime import timedelta

# 🔧 Third-party Libraries
import numpy as np
import numpy.matlib
import scipy.io as sio
import torch
import hydra
from omegaconf import DictConfig

# 📦 Local Modules
from hydra.utils import to_absolute_path
from physicsnemo.distributed import DistributedManager

from compare.sequential.misc_plotting import simulation_data_types
from compare.sequential.misc_plotting_utils import read_compdats2
from compare.sequential.misc_model import load_modell, create_fno_model
from compare.sequential.misc_gather import read_compdats, extract_qs, get_dyna2
from compare.sequential.misc_forward_utils import Get_data_FFNN1
from compare.sequential.misc_gather_utils import (
    Geta_all,
    ensemble_pytorch,
    copy_files,
    save_files,
    Run_simulator,
)
from compare.sequential.utils.misc_utils import compare_and_analyze_results
from compare.sequential.misc_forward_enact import Forward_model_ensemble

warnings.filterwarnings("ignore")


def setup_logging() -> logging.Logger:
    """Configure and return the main logger."""
    logger = logging.getLogger("Forward problem")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def is_gpu_available() -> bool:
    """Check if NVIDIA GPU is available using `nvidia-smi`."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


@hydra.main(version_base="1.2", config_path="conf", config_name="DECK_CONFIG")
def main(cfg: DictConfig) -> None:
    """Main function for FVM surrogate comparison sequential processing."""
    # Initialize logging
    logger = setup_logging()

    # Initialize simulation data types
    (
        type_dict,
        ecl_extensions,
        dynamic_props,
        ecl_vectors,
        static_props,
        SUPPORTED_DATA_TYPES,
    ) = simulation_data_types()
    DistributedManager.initialize()
    dist = DistributedManager()
    if "RANK" not in os.environ:
        os.environ["RANK"] = str(dist.rank)
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(dist.rank % torch.cuda.device_count())
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        device_id = dist.rank % gpu_count  # Map rank to available GPUs
        torch.cuda.set_device(device_id)
        logger.info(
            f"Process {dist.rank} is using GPU {device_id}: {torch.cuda.get_device_name(device_id)}"
        )
    else:
        logger.info(f"Process {dist.rank} is using CPU")
    device = dist.device
    if not os.path.exists(to_absolute_path("../RESULTS/FORWARD_RESULTS_SEQUENTIAL")):
        os.makedirs(
            to_absolute_path("../RESULTS/FORWARD_RESULTS_SEQUENTIAL"), exist_ok=True
        )
    else:
        shutil.rmtree(to_absolute_path("../RESULTS/FORWARD_RESULTS_SEQUENTIAL"))
        os.makedirs(
            to_absolute_path("../RESULTS/FORWARD_RESULTS_SEQUENTIAL"), exist_ok=True
        )
    oldfolder = os.getcwd()
    os.chdir(oldfolder)

    Trainmoe = "MoE"
    logger.info("-----------------------------------------------------------------")
    logger.info(
        "Using Cluster Classify Regress (CCR) for peacemann model -Hard Prediction          "
    )

    logger.info("References for CCR include: ")
    logger.info(
        " (1): David E. Bernholdt, Mark R. Cianciosa, David L. Green, Jin M. Park,\n\
Kody J. H. Law, and Clement Etienam. Cluster, classify, regress:A general\n\
method for learning discontinuous functions.Foundations of Data Science,\n\
1(2639-8001-2019-4-491):491, 2019.\n"
    )

    logger.info(
        "(2): Clement Etienam, Kody Law, Sara Wade. Ultra-fast Deep Mixtures of\n\
Gaussian Process Experts. arXiv preprint arXiv:2006.13309, 2020.\n"
    )
    logger.info(
        "-----------------------------------------------------------------------"
    )
    pred_type = 1
    if cfg.custom.fno_type == "PINO":
        folderr = os.path.join(
            oldfolder,
            "..",
            "RESULTS",
            "FORWARD_RESULTS_SEQUENTIAL",
            "RESULTS",
            "COMPARE_RESULTS",
            "PINO",
            "PEACEMANN_CCR",
        )
    else:
        folderr = os.path.join(
            oldfolder,
            "..",
            "RESULTS",
            "FORWARD_RESULTS_SEQUENTIAL",
            "RESULTS",
            "COMPARE_RESULTS",
            "FNO",
            "PEACEMANN_CCR",
        )
    os.makedirs(to_absolute_path(folderr), exist_ok=True)
    absolute_path = os.path.abspath(folderr)
    logger.info(f"Resolved path: {absolute_path}")
    os.makedirs(folderr, exist_ok=True)
    logger.info(f"Directory created: {folderr}")
    degg = 3
    num_cores = multiprocessing.cpu_count()
    njobs = (num_cores // 4) - 1
    num_cores = njobs

    exper = sio.loadmat(to_absolute_path("../data/exper.mat"))
    experts = exper["expert"]
    mat = sio.loadmat(to_absolute_path("../data/conversions.mat"))
    for key, value in mat.items():
        logger.info(f"For key '{key}':")
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
    N_ens = int(mat["N_ens"])
    steppi_indices = mat["steppi_indices"].flatten()
    effective = mat["effective"]
    steppi = int(mat["steppi"])
    steppi_indices = mat["steppi_indices"].flatten()
    target_min = 0.01
    target_max = 1
    input_variables = cfg.custom.input_properties
    output_variables = cfg.custom.output_properties
    nx = cfg.custom.PROPS.nx
    ny = cfg.custom.PROPS.ny
    nz = cfg.custom.PROPS.nz
    sourc_dir = cfg.custom.file_location
    source_dir = to_absolute_path(sourc_dir)
    effectiveuse = np.reshape(effective, (nx * ny * nz, N_ens), "F")
    effective = effectiveuse[:, 0].reshape(-1, 1)
    effective_abi = np.genfromtxt(os.path.join(source_dir, "actnum.out"), dtype="float")
    effective_abi = np.reshape(effective_abi, (nx, ny, nz), "F")
    Ne = 1
    with gzip.open(to_absolute_path("../data/static.pkl.gz"), "rb") as f2:
        X_data1 = pickle.load(f2)
    for key, value in X_data1.items():
        logger.info(f"For key '{key}':")
        logger.info("\tContains inf: %s", np.isinf(value).any())
        logger.info("\tContains -inf: %s", np.isinf(-value).any())
        logger.info("\tContains NaN: %s", np.isnan(value).any())
    perm_ensemble = X_data1["ensemble"]
    logger.info("Permeability ensemble shape: %s", perm_ensemble.shape)
    poro_ensemble = X_data1["ensemblep"]
    fault_ensemble = X_data1["ensemblefault"]
    index = 20
    perm_use = perm_ensemble[:, index].reshape(-1, 1)
    poro_use = poro_ensemble[:, index].reshape(-1, 1)
    fault_use = fault_ensemble[:, index].reshape(-1, 1)
    effectiveuse = np.reshape(effective, (nx, ny, nz), "F")
    filename = cfg.custom.COMPLETIONS_DATA
    filename_fault = to_absolute_path(cfg.custom.FAULT_DATA)
    FAULT_INCLUDE = cfg.custom.FAULT_INCLUDE
    logger.info("Fault include: %s", FAULT_INCLUDE)
    PERMX_INCLUDE = cfg.custom.PERMX_INCLUDE
    PORO_INCLUDE = cfg.custom.PORO_INCLUDE
    gass, producers, injectors = read_compdats2(
        cfg.custom.COMPLETIONS_DATA, cfg.custom.SUMMARY_DATA
    )  # filename
    logger.info("|-----------------------------------------------------------------|")
    logger.info("|                         PRINT WELLS                           : |")
    logger.info("|-----------------------------------------------------------------|")
    logger.info("gas injectors wells: %s", gass)
    logger.info("producer well: %s", producers)
    logger.info("water injector well: %s", injectors)
    N_injw = len(injectors)
    N_pr = len(producers)  # Number of producers
    logger.info("Number of producers: %s", N_pr)
    N_injg = len(gass)
    well_names = [entry[-1] for entry in producers]
    well_namesg = [entry[-1] for entry in gass]  # Adjust index as needed
    well_namesw = [entry[-1] for entry in injectors]  # Adjust index as needed
    logger.info("|-----------------------------------------------------------------|")
    logger.info("|                         PRINT WELL NAMES                      : |")
    logger.info("|-----------------------------------------------------------------|")
    logger.info("producer well names: %s", well_names)
    logger.info("gas injectors well names: %s", well_namesg)
    logger.info("water injector well names: %s", well_namesw)
    columns = well_names
    compdat_data = read_compdats(filename, well_names)
    compdat_datag = read_compdats(filename, well_namesg)
    compdat_dataw = read_compdats(filename, well_namesw)
    filenamea = os.path.basename(cfg.custom.DECK)
    filenameui = os.path.splitext(filenamea)[0]
    if cfg.custom["numerical_solver"] == "flow":
        string_simulation_command = (
            f"mpirun --oversubscribe --allow-run-as-root -np 32 "
            f"flow "
            f"{filenamea} "
            f"--parsing-strictness=low "
            f"--enable-ecl-output=true "
        )
    else:
        string_simulation_command = f"ecl100 {filenamea} "
    oldfolder2 = os.getcwd()
    path_out = "../RESULTS/FORWARD_RESULTS_SEQUENTIAL/RESULTS/True_Flow"
    os.makedirs(to_absolute_path(path_out), exist_ok=True)
    copy_files(cfg.custom.file_location, path_out)
    save_files(
        perm_use,
        poro_use,
        fault_use,
        path_out,
        oldfolder2,
        FAULT_INCLUDE,
        PERMX_INCLUDE,
        PORO_INCLUDE,
    )
    excel = 1
    logger.info("---------------------------------------------------------------------")
    logger.info("")
    logger.info("|-----------------------------------------------------------------|")
    logger.info("|                 RUN FLOW SIMULATOR                              |")
    logger.info("|-----------------------------------------------------------------|")
    start_time_plots1 = time.time()
    Run_simulator(path_out, oldfolder2, string_simulation_command)
    elapsed_time_secs = (time.time() - start_time_plots1) / 2
    msg = "Reservoir simulation with FLOW  took: %s secs (Wall clock time)" % timedelta(
        seconds=round(elapsed_time_secs)
    )
    logger.info(msg)

    logger.info("Finished FLOW NUMERICAL simulations")
    input_variables = cfg.custom.input_properties
    output_variables = cfg.custom.output_properties
    input_variables2 = cfg.custom.input_properties2
    # output_variables2 = cfg.custom.output_properties2
    logger.info("|-----------------------------------------------------------------|")
    logger.info("|                 DATA CURRATION IN PROCESS                       |")
    logger.info("|-----------------------------------------------------------------|")
    N = Ne
    pressure = []
    Sgas = []
    Swater = []
    Soil = []
    Time = []
    permeability = np.zeros((N, 1, nx, ny, nz))
    porosity = np.zeros((N, 1, nx, ny, nz))
    actnumm = np.zeros((N, 1, nx, ny, nz))
    folder = path_out
    return_values = Geta_all(
        folder,
        nx,
        ny,
        nz,
        oldfolder,
        steppi,
        steppi_indices,
        filenameui,
        injectors,
        gass,
        filename,
        compdat_datag,
        compdat_dataw,
        compdat_data,
        input_variables,
        output_variables,
        filename_fault,
        FAULT_INCLUDE,
    )
    pressure_true = (
        return_values.get("PRESSURE", None) if "PRESSURE" in output_variables else None
    )
    Swater_true = (
        return_values.get("SWAT", None) if "SWAT" in output_variables else None
    )
    Sgas_true = return_values.get("SGAS", None) if "SGAS" in output_variables else None
    Soil_true = return_values.get("SOIL", None) if "SOIL" in output_variables else None
    if pressure_true is not None:
        pressure_true = pressure_true[None, :, :, :, :]
    if Swater_true is not None:
        Swater_true = Swater_true[None, :, :, :, :]
    if Sgas_true is not None:
        Sgas_true = Sgas_true[None, :, :, :, :]
    if Soil_true is not None:
        Soil_true = Soil_true[None, :, :, :, :]
    Time = return_values.get("Time")  # Mandatory value
    # Extract rates for potential future use
    return_values.get("QG")  # Extract gas rates
    return_values.get("QW")  # Extract water rates
    return_values.get("QO")  # Extract oil rates
    return_values.get("FAULT", None) if "FAULT" in input_variables else None
    Time = np.stack(Time, axis=0)
    Time = Time[None, :, :, :, :]
    effec_abbi = np.zeros((1, 1, nx, ny, nz), dtype=np.float32)
    effec_abbi[0, 0, :, :, :] = effective_abi
    Soil_true = (np.ones_like(pressure_true) * effec_abbi) - (
        (Sgas_true * effec_abbi) + (Swater_true * effec_abbi)
    )
    Soil_true = np.clip(Soil_true, 0, 1)
    permeability[0, 0, :, :, :] = np.reshape(perm_ensemble[:, index], (nx, ny, nz), "F")
    porosity[0, 0, :, :, :] = np.reshape(poro_ensemble[:, index], (nx, ny, nz), "F")
    actnumm[0, 0, :, :, :] = np.reshape(effective, (nx, ny, nz), "F")
    well_measurements = cfg.custom.well_measurements
    lenwels = len(well_measurements)
    _, out_fcn_true = Get_data_FFNN1(
        folder,
        oldfolder2,
        N,
        pressure_true,
        Sgas_true,
        Swater_true,
        Soil_true,
        permeability,
        Time,
        steppi,
        steppi_indices,
        N_pr,
        producers,
        compdat_data,
        filenameui,
        well_measurements,
        lenwels,
    )

    logger.info("|-----------------------------------------------------------------|")
    logger.info("|                 DATA CURRATION FINISHED                         |")
    logger.info("|-----------------------------------------------------------------|")
    logger.info("---------------------------------------------------------------------")
    logger.info("")
    logger.info("|-----------------------------------------------------------------|")
    logger.info(
        "|          RUN  NVIDIA PHYSICSNEMO RESERVOIR SIMULATION SURROGATE     |"
    )
    logger.info("|-----------------------------------------------------------------|")
    os.chdir(folder)
    Qg = np.zeros((steppi, nx, ny, nz))
    Qw = np.zeros((steppi, nx, ny, nz))
    Qo = np.zeros((steppi, nx, ny, nz))
    seeg, seew = extract_qs(
        steppi, steppi_indices, filenameui, injectors, gass, filename
    )
    os.chdir(oldfolder2)
    awater, agas, aoil = get_dyna2(
        steppi, compdat_dataw, compdat_datag, compdat_data, Qw, Qg, Qo, seew, seeg
    )
    aqq = awater + agas + aoil
    param_temp = {}
    if "FAULT" in input_variables:
        param_temp["FAULT"] = fault_use
    if "PERM" in input_variables:
        param_temp["PERM"] = perm_use
    if "PORO" in input_variables:
        param_temp["PORO"] = poro_use
    inn = ensemble_pytorch(
        param_temp,
        nx,
        ny,
        nz,
        Ne,
        effectiveuse,
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
    )
    for key, value in inn.items():
        logger.info(
            "---------------------------------------------------------------------"
        )
        logger.info(f"For key '{key}':")

    logger.info("Finished constructing Pytorch inputs")
    logger.info("*******************Load the trained Forward models*******************")
    input_variables2 = cfg.custom.input_properties2
    # output_variables2 = cfg.custom.output_properties2
    input_keys = []
    if "PERM" in input_variables:
        input_keys.append("perm")
    if "PORO" in input_variables:
        input_keys.append("poro")
    if "PINI" in input_variables:
        input_keys.append("pini")
    if "SINI" in input_variables:
        input_keys.append("sini")
    if "SGINI" in input_variables:
        input_keys.append("sgini")
    if "SOINI" in input_variables:
        input_keys.append("soini")
    if "FAULT" in input_variables:
        input_keys.append("fault")
    if "WTIR" in input_variables2:
        input_keys.append("Q")
    if "WGIR" in input_variables2:
        input_keys.append("Qg")
    if "WWIR" in input_variables2:
        input_keys.append("Qw")
    if "DELTA_TIME" in input_variables2:
        input_keys.append("dt")
    # input_keys_peacemann = ["X"]
    output_keys_peacemann = ["Y"]
    output_keys_pressure = []
    if "PRESSURE" in output_variables:
        output_keys_pressure.append("pressure")
    output_keys_gas = []
    if "SGAS" in output_variables:
        output_keys_gas.append("gas_sat")
    output_keys_saturation = []
    if "SWAT" in output_variables:
        output_keys_saturation.append("water_sat")
    output_keys_oil = []
    if "SOIL" in output_variables:
        output_keys_oil.append("oil_sat")
    if "PRESSURE" in output_variables:
        fno_supervised_pressure = create_fno_model(
            len(input_keys),
            1,
            len(output_keys_pressure),
            device,
            num_fno_modes=16,
            latent_channels=32,
            decoder_layer_size=32,
            padding=22,
            decoder_layers=4,
            dimension=3,
        )
    if "SGAS" in output_variables:
        fno_supervised_gas = create_fno_model(
            len(input_keys),
            1,
            len(output_keys_gas),
            device,
            num_fno_modes=16,
            latent_channels=32,
            decoder_layer_size=32,
            padding=22,
            decoder_layers=4,
            dimension=3,
        )
    fno_supervised_peacemann = create_fno_model(
        2 + (4 * N_pr),
        (lenwels * N_pr),
        len(output_keys_peacemann),
        device,
        num_fno_modes=13,
        latent_channels=64,
        decoder_layer_size=32,
        padding=20,
        decoder_layers=4,
        num_fno_layers=5,
        dimension=1,
    )
    if "SWAT" in output_variables:
        fno_supervised_saturation = create_fno_model(
            len(input_keys),
            1,
            len(output_keys_saturation),
            device,
            num_fno_modes=16,
            latent_channels=32,
            decoder_layer_size=32,
            padding=22,
            decoder_layers=4,
            dimension=3,
        )
    if "SOIL" in output_variables:
        fno_supervised_oil = create_fno_model(
            len(input_keys),
            1,
            len(output_keys_oil),
            device,
            num_fno_modes=16,
            latent_channels=32,
            decoder_layer_size=32,
            padding=22,
            decoder_layers=4,
            dimension=3,
        )
    if cfg.custom.fno_type == "FNO":
        os.chdir("../MODELS/FNO")
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|                     FNO MODEL LEARNING    :                     |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )

        if cfg.custom.model_saturation == "FNO":
            logger.info(
                "|-------------------------------------------------------------------------|"
            )
            logger.info(
                "|   PRESSURE MODEL = FNO;   SATUARATION MODEL = FNO; PEACEMAN MODEL = FNO |"
            )
            logger.info(
                "|-------------------------------------------------------------------------|"
            )
        models = {}
        base_paths = {
            "pressure": "./checkpoints_pressure_seq",
            "gas": "./checkpoints_gas_seq",
            "peacemann": "./checkpoints_peacemann_seq",
            "saturation": "./checkpoints_saturation_seq",
            "oil": "./checkpoints_oil_seq",
        }
        if "PRESSURE" in output_variables:
            logger.info("🟢 Loading Surrogate Model for Pressure")
            if excel == 1:
                model_path = os.path.join(
                    base_paths["pressure"], "fno_pressure_forward_model.pth"
                )
            else:
                model_path = os.path.join(base_paths["pressure"], "checkpoint.pth")
            fno_supervised_pressure = load_modell(
                fno_supervised_pressure,
                model_path,
                cfg.custom.model_Distributed,
                device,
                excel,
                "PRESSURE",
            )
            models["pressure"] = fno_supervised_pressure
        if "SGAS" in output_variables:
            logger.info("🟠 Loading Surrogate Model for Gas")
            if excel == 1:
                model_path = os.path.join(
                    base_paths["gas"], "fno_gas_forward_model.pth"
                )
            else:
                model_path = os.path.join(base_paths["gas"], "checkpoint.pth")

            fno_supervised_gas = load_modell(
                fno_supervised_gas,
                model_path,
                cfg.custom.model_Distributed,
                device,
                excel,
                "SGAS",
            )
            models["gas"] = fno_supervised_gas
        logger.info("🔵 Loading Surrogate Model for Peacemann")
        if excel == 1:
            model_path = os.path.join(
                base_paths["peacemann"], "fno_peacemann_forward_model.pth"
            )
        else:
            model_path = os.path.join(base_paths["peacemann"], "checkpoint.pth")
        fno_supervised_peacemann = load_modell(
            fno_supervised_peacemann,
            model_path,
            cfg.custom.model_Distributed,
            device,
            excel,
            "PEACEMANN",
        )
        models["peacemann"] = fno_supervised_peacemann
        if "SWAT" in output_variables:
            logger.info("🟣 Loading Surrogate Model for Saturation")
            if cfg.custom.model_saturation == "FNO":
                if excel == 1:
                    model_path = os.path.join(
                        base_paths["saturation"], "fno_saturation_forward_model.pth"
                    )
                else:
                    model_path = os.path.join(
                        base_paths["saturation"], "checkpoint.pth"
                    )
            else:
                if excel == 1:
                    model_path = os.path.join(
                        base_paths["saturation"], "GNN_saturation_forward_model.pth"
                    )
                else:
                    model_path = os.path.join(
                        base_paths["saturation"], "checkpoint.pth"
                    )
            fno_supervised_saturation = load_modell(
                fno_supervised_saturation,
                model_path,
                cfg.custom.model_Distributed,
                device,
                excel,
                "SWAT",
            )
            models["saturation"] = fno_supervised_saturation
        if "SOIL" in output_variables:
            logger.info("🟣 Loading Surrogate Model for oil")
            if excel == 1:
                model_path = os.path.join(
                    base_paths["oil"], "fno_oil_forward_model.pth"
                )
            else:
                model_path = os.path.join(base_paths["oil"], "checkpoint.pth")
            fno_supervised_oil = load_modell(
                fno_supervised_oil,
                model_path,
                cfg.custom.model_Distributed,
                device,
                excel,
                "SWAT",
            )
            models["oil"] = fno_supervised_oil
    else:
        os.chdir("../MODELS/PINO")
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|                     PINO MODEL LEARNING    :                     |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )

        if cfg.custom.model_saturation == "FNO":
            logger.info(
                "|-------------------------------------------------------------------------|"
            )
            logger.info(
                "|   PRESSURE MODEL = FNO;   SATUARATION MODEL = FNO; PEACEMAN MODEL = FNO |"
            )
            logger.info(
                "|-------------------------------------------------------------------------|"
            )

        models = {}
        base_paths = {
            "pressure": "./checkpoints_pressure_seq",
            "gas": "./checkpoints_gas_seq",
            "peacemann": "./checkpoints_peacemann_seq",
            "saturation": "./checkpoints_saturation_seq",
            "oil": "./checkpoints_oil_seq",
        }
        if "PRESSURE" in output_variables:
            logger.info("🟢 Loading Surrogate Model for Pressure")
            if excel == 1:
                model_path = os.path.join(
                    base_paths["pressure"], "pino_pressure_forward_model.pth"
                )
            else:
                model_path = os.path.join(base_paths["pressure"], "checkpoint.pth")

            fno_supervised_pressure = load_modell(
                fno_supervised_pressure,
                model_path,
                cfg.custom.model_Distributed,
                device,
                excel,
                "PRESSURE",
            )
            models["pressure"] = fno_supervised_pressure
        if "SGAS" in output_variables:
            logger.info("🟠 Loading Surrogate Model for Gas")
            if excel == 1:
                model_path = os.path.join(
                    base_paths["gas"], "pino_gas_forward_model.pth"
                )
            else:
                model_path = os.path.join(base_paths["gas"], "checkpoint.pth")
            fno_supervised_gas = load_modell(
                fno_supervised_gas,
                model_path,
                cfg.custom.model_Distributed,
                device,
                excel,
                "SGAS",
            )
            models["gas"] = fno_supervised_gas
        logger.info("🔵 Loading Surrogate Model for Peacemann")
        if excel == 1:
            model_path = os.path.join(
                base_paths["peacemann"], "pino_peacemann_forward_model.pth"
            )
        else:
            model_path = os.path.join(base_paths["peacemann"], "checkpoint.pth")
        fno_supervised_peacemann = load_modell(
            fno_supervised_peacemann,
            model_path,
            cfg.custom.model_Distributed,
            device,
            excel,
            "PEACEMANN",
        )
        models["peacemann"] = fno_supervised_peacemann
        if "SWAT" in output_variables:
            logger.info("🟣 Loading Surrogate Model for Saturation")
            if excel == 1:
                model_path = os.path.join(
                    base_paths["saturation"], "pino_saturation_forward_model.pth"
                )
            else:
                model_path = os.path.join(base_paths["saturation"], "checkpoint.pth")
            fno_supervised_saturation = load_modell(
                fno_supervised_saturation,
                model_path,
                cfg.custom.model_Distributed,
                device,
                excel,
                "SWAT",
            )
            models["saturation"] = fno_supervised_saturation
        if "SOIL" in output_variables:
            logger.info("🟣 Loading Surrogate Model for oil")
            if excel == 1:
                model_path = os.path.join(
                    base_paths["oil"], "pino_oil_forward_model.pth"
                )
            else:
                model_path = os.path.join(base_paths["oil"], "checkpoint.pth")
            fno_supervised_oil = load_modell(
                fno_supervised_oil,
                model_path,
                cfg.custom.model_Distributed,
                device,
                excel,
                "SWAT",
            )
            models["oil"] = fno_supervised_oil
    os.chdir(oldfolder)
    logger.info("********************Model Loaded*************************************")
    start_time_plots2 = time.time()
    simout = Forward_model_ensemble(
        Ne,
        inn,
        steppi,
        min_inn_fcn,
        max_inn_fcn,
        target_min,
        target_max,
        minK,
        maxK,
        minT,
        maxT,
        minP,
        maxP,
        models,
        device,
        min_out_fcn,
        max_out_fcn,
        Time,
        effective_abi,
        Trainmoe,
        num_cores,
        pred_type,
        oldfolder,
        degg,
        experts,
        min_out_fcn2,
        max_out_fcn2,
        min_inn_fcn2,
        max_inn_fcn2,
        producers,
        compdat_data,
        output_variables,
        well_measurements,
        cfg,
        N_pr,
        lenwels,
        effective_abi,
        awater,
        agas,
        aoil,
        aqq,
        nx,
        ny,
        nz,
        minQ,
        maxQ,
        minQw,
        maxQw,
        minQg,
        maxQg,
    )
    elapsed_time_secs2 = time.time() - start_time_plots2
    msg = (
        "Reservoir simulation with NVidia PhyNeMo (CCR - Hard prediction)  took: %s secs (Wall clock time)"
        % timedelta(seconds=round(elapsed_time_secs2))
    )
    logger.info(msg)

    if "PRESSURE" in output_variables:
        pressure = simout["PRESSURE"]
    if "SWAT" in output_variables:
        Swater = simout["SWAT"]
    if "SOIL" in output_variables:
        Soil = simout["SOIL"]
    if "SGAS" in output_variables:
        Sgas = simout["SGAS"]
    ouut_peacemann = simout["ouut_p"]
    physicsnemo_time = elapsed_time_secs2
    flow_time = elapsed_time_secs

    compare_and_analyze_results(
        physicsnemo_time=physicsnemo_time,
        flow_time=flow_time,
        nx=nx,
        ny=ny,
        nz=nz,
        steppi=steppi,
        steppi_indices=steppi_indices,
        Ne=Ne,
        pressure=pressure,
        pressure_true=pressure_true,
        Swater=Swater,
        Swater_true=Swater_true,
        Soil=Soil,
        Soil_true=Soil_true,
        Sgas=Sgas,
        Sgas_true=Sgas_true,
        ouut_peacemann=ouut_peacemann,
        out_fcn_true=out_fcn_true,
        cfg=cfg,
        device=device,
        num_cores=num_cores,
        oldfolder=oldfolder,
        folderr=folderr,
        N_injw=N_injw,
        N_pr=N_pr,
        N_injg=N_injg,
        injectors=injectors,
        producers=producers,
        gass=gass,
        well_names=well_names,
        inn=inn,
        min_inn_fcn=min_inn_fcn,
        max_inn_fcn=max_inn_fcn,
        target_min=target_min,
        target_max=target_max,
        minK=minK,
        maxK=maxK,
        minT=minT,
        maxT=maxT,
        minP=minP,
        maxP=maxP,
        models=models,
        min_out_fcn=min_out_fcn,
        max_out_fcn=max_out_fcn,
        Time=Time,
        effective_abi=effective_abi,
        degg=degg,
        experts=experts,
        min_out_fcn2=min_out_fcn2,
        max_out_fcn2=max_out_fcn2,
        min_inn_fcn2=min_inn_fcn2,
        max_inn_fcn2=max_inn_fcn2,
        compdat_data=compdat_data,
        output_variables=output_variables,
        well_measurements=well_measurements,
        effectiveuse=effectiveuse,
        columns=columns,
        lenwels=lenwels,
        awater=awater,
        agas=agas,
        aoil=aoil,
        aqq=aqq,
        minQ=minQ,
        maxQ=maxQ,
        minQw=minQw,
        maxQw=maxQw,
        minQg=minQg,
        maxQg=maxQg,
    )
    os.chdir((oldfolder))
    logger.info(
        "-------------------PROGRAM EXECUTED-----------------------------------"
    )


if __name__ == "__main__":
    main()
