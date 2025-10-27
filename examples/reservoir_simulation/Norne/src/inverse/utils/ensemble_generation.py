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
                    ENSEMBLE GENERATION UTILITIES MODULE
=====================================================================

This module provides ensemble generation utilities for inverse problems
in reservoir simulation. It includes model loading, ensemble creation,
and data processing utilities.

Key Features:
- Model loading and processing
- Ensemble generation and management
- Data processing utilities
- Visualization tools

@Author : Clement Etienam
"""

# 🛠 Standard Library
import os
import pickle
import time
import gzip
import shutil
import yaml
import logging
from collections import OrderedDict
from math import sqrt

# 🔧 Third-party Libraries
import numpy as np
import numpy.matlib
import pandas as pd

# import scipy.io as sio
import torch
from hydra.utils import to_absolute_path
import scipy
from physicsnemo.models.fno import FNO
from physicsnemo.models.module import Module
import matplotlib.pyplot as plt

# 📦 Local Modules
from inverse.inversion_operation_surrogate import (
    remove_rows,
)
from inverse.inversion_operation_ensemble import (
    honour2,
)


from inverse.inversion_operation_misc import (
    read_until_line,
    add_gnoise,
)


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
    return logger


logger = setup_logging()


def load_modell(model, model_path, is_distributed, device, express, namee):
    """Load model weights; handle DDP 'module.' prefixes when needed."""
    logger.info(f"🔄 Loading model from: {model_path}")
    if express == 1:
        state_dict = torch.load(model_path, map_location=device)
        if is_distributed == 1:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:] if k.startswith("module.") else k] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
    else:
        checkpoint = torch.load(model_path, map_location=device)
        if namee == "PRESSURE":
            state_dict = checkpoint["surrogate_pressure_state_dict"]
        if namee == "SWAT":
            state_dict = checkpoint["surrogate_saturation_state_dict"]

        if namee == "SOIL":
            state_dict = checkpoint["surrogate_oil_state_dict"]

        if namee == "SGAS":
            state_dict = checkpoint["surrogate_gas_state_dict"]
        if namee == "PEACEMANN":
            state_dict = checkpoint["surrogate_peacemann_state_dict"]
        # ✅ Handle Distributed Data Parallel (Remove `module.` prefix if needed)
        if is_distributed == 1:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:] if k.startswith("module.") else k] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
    return model


class FNOModel(Module):
    def __init__(
        self,
        input_dim,
        steppi,
        output_shape,
        device,
        num_layers=4,
        decoder_layers=1,
        decoder_layer_size=32,
        dimension=3,
        latent_channels=32,
        num_fno_layers=4,
        padding=8,
        num_fno_modes=16,
    ):
        super().__init__()
        self.fno = FNO(
            in_channels=input_dim,
            out_channels=output_shape * steppi,
            decoder_layers=decoder_layers,
            decoder_layer_size=decoder_layer_size,
            dimension=dimension,
            latent_channels=latent_channels,
            num_fno_layers=num_layers,
            padding=padding,
            num_fno_modes=num_fno_modes,
        ).to(torch.device(device))  # Explicit device conversion
        self.meta = type("", (), {})()  # Empty object
        self.meta.name = "fno_model"

    def forward(self, x):
        return self.fno(x)


def create_fno_model(
    input_dim,
    steppi,
    output_shape,
    device,
    num_layers=4,
    decoder_layers=1,
    decoder_layer_size=32,
    dimension=3,
    latent_channels=32,
    num_fno_layers=4,
    padding=8,
    num_fno_modes=16,
):
    """Factory for FNO-based surrogate with given IO dimensions and config."""
    if dimension not in [1, 2, 3]:
        raise ValueError(f"Invalid dimension: {dimension}. Must be 1, 2 or 3.")
    return FNOModel(
        input_dim=input_dim,
        steppi=steppi,
        output_shape=output_shape,
        device=device,
        num_layers=num_layers,
        decoder_layers=decoder_layers,
        decoder_layer_size=decoder_layer_size,
        dimension=dimension,
        latent_channels=latent_channels,
        num_fno_layers=num_fno_layers,
        padding=padding,
        num_fno_modes=num_fno_modes,
    )


def Get_Time(nx, ny, nz, steppi, steppi_indices, N):
    """Return tiled time volume and shape helpers for dataset construction."""
    with gzip.open(to_absolute_path("../data/data_train.pkl.gz"), "rb") as f2:
        mat = pickle.load(f2)
    X_data1 = mat
    # del mat
    # gc.collect()
    # mat = sio.loadmat(to_absolute_path("../data/conversions.mat"))
    Time = X_data1["Time"]  # * mat ["maxT"]
    np_array2 = np.zeros((Time.shape[1]))
    for mm in range(Time.shape[1]):
        np_array2[mm] = Time[0, mm, 0, 0, 0]
    Timee = []
    for k in range(N):
        check = np.ones((nx, ny, nz), dtype=np.float16)
        unie = []
        for zz in range(len(np_array2)):
            aa = np_array2[zz] * check
            unie.append(aa)
        Time = np.stack(unie, axis=0)
        Timee.append(Time)
    Timee = np.stack(Timee, axis=0)
    return Timee


def historydata(timestep, steppi, steppi_indices, N_pr):
    """Load and assemble historical RSM slices by category for a NORNE deck."""
    file_path = "../simulator_data/Flow.xlsx"
    df = pd.read_excel(file_path, skiprows=1)
    data_array = df.to_numpy()[:10, 1:]  # Skips first column (assuming it's time)
    WOIL1 = data_array[:, :N_pr]
    WWATER1 = data_array[:, N_pr : 2 * N_pr]
    WGAS1 = data_array[:, 2 * N_pr : 3 * N_pr]
    DATA = {"OIL": WOIL1, "WATER": WWATER1, "GAS": WGAS1}
    oil = WOIL1.reshape(-1, 1, order="F")  # Column-major reshaping
    water = WWATER1.reshape(-1, 1, order="F")
    gas = WGAS1.reshape(-1, 1, order="F")
    DATA2 = np.vstack([oil, water, gas])  # History-matching data
    new = np.hstack([WOIL1, WWATER1, WGAS1])  # Original structure but stacked
    return DATA, DATA2, new


def Plot_RSM_singleT(True_mat, timezz, N_pr, well_names):
    """Plot single time series per well with multi-indexed headers."""
    columns = well_names  # ['L1', 'L2', 'L3', 'LU1', 'LU2',
    plt.figure(figsize=(40, 40))
    for k in range(N_pr):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k], color="red", lw="2", label="model")
        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()
    plt.savefig("Oil_singleT.png")
    plt.clf()
    plt.close()
    plt.figure(figsize=(40, 40))
    for k in range(N_pr):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k + N_pr], color="red", lw="2", label="model")

        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
        # plt.ylim((0,25000))
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()
    plt.savefig("Water_singleT.png")
    plt.clf()
    plt.close()
    plt.figure(figsize=(40, 40))
    for k in range(N_pr):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k + 2 * N_pr], color="red", lw="2", label="model")
        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()
    plt.savefig("Gas_singleT.png")
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()


def scale_array(arr):
    """Scale array magnitude to ~3 digits and return scaled array and factor."""
    max_val = np.max(np.abs(arr))
    if max_val == 0:
        return arr, 1  # No scaling needed for an array of zeroes
    num_digits_before_decimal = int(np.floor(np.log10(max_val))) + 1
    if num_digits_before_decimal >= 3:
        scaling_factor = 10 ** (num_digits_before_decimal - 3)
    else:
        scaling_factor = 10 ** (3 - num_digits_before_decimal)
    if num_digits_before_decimal >= 3:
        scaled_arr = arr / scaling_factor
        bool1 = 1
    else:
        scaled_arr = arr * scaling_factor
        bool1 = 2
    return scaled_arr, scaling_factor, bool1


def setup_models_and_data(
    # Configuration
    input_variables,
    output_variables,
    TEMPLATEFILE,
    cfg,
    dist,
    Ne,
    # Model parameters
    steppi,
    N_pr,
    lenwels,
    # Device and settings
    device,
    excel,
    oldfolder,
    DEFAULT,
    # Data arrays
    perm_ensembley,
    poro_ensembley,
    fault_ensemblepy,
    # Additional parameters
    nx,
    ny,
    nz,
    steppi_indices,
    well_names,
    minK,
    maxK,
):
    """Initialise surrogates, load data, and prepare loaders for training."""
    """
    Setup FNO/PINO models and prepare data for history matching.

    Returns:
        tuple: Models, configuration, and processed data
    """
    input_keys = []
    if "PERM" in input_variables:
        input_keys.append("perm")
    if "PORO" in input_variables:
        input_keys.append("poro")
    if "PINI" in input_variables:
        input_keys.append("pini")
    if "SINI" in input_variables:
        input_keys.append("sini")
    if "FAULT" in input_variables:
        input_keys.append("fault")
    # input_keys_peacemann = ["X"]
    output_keys_peacemann = ["Y"]
    if "PRESSURE" in output_variables:
        output_keys_pressure = []
        output_keys_pressure.append("pressure")
    if "SGAS" in output_variables:
        output_keys_gas = []
        output_keys_gas.append("gas_sat")
    if "SWAT" in output_variables:
        output_keys_saturation = []
        output_keys_saturation.append("water_sat")
    if "SOIL" in output_variables:
        output_keys_oil = []
        output_keys_oil.append("oil_sat")
    # input_keys_peacemann = ["X"]
    output_keys_peacemann = ["Y"]
    if "PRESSURE" in output_variables:
        fno_supervised_pressure = create_fno_model(
            len(input_keys),
            steppi,
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
            steppi,
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
            steppi,
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
            steppi,
            len(output_keys_oil),
            device,
            num_fno_modes=16,
            latent_channels=32,
            decoder_layer_size=32,
            padding=22,
            decoder_layers=4,
            dimension=3,
        )
    if dist.rank == 0:
        logger.info(
            "*******************Load the trained Forward models*******************"
        )
    if cfg.custom.fno_type == "FNO":
        os.chdir("../MODELS/FNO")
        if cfg.custom.model_saturation == "FNO":
            logger.info(
                "|   PRESSURE MODEL = FNO;   SATUARATION MODEL = FNO; PEACEMAN MODEL = FNO |"
            )
        models = {}
        base_paths = {
            "pressure": "./checkpoints_pressure",
            "gas": "./checkpoints_gas",
            "peacemann": "./checkpoints_peacemann",
            "saturation": "./checkpoints_saturation",
            "oil": "./checkpoints_oil",
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
            if excel == 1:
                model_path = os.path.join(
                    base_paths["saturation"], "fno_saturation_forward_model.pth"
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
                "SOIL",
            )
            models["oil"] = fno_supervised_oil
    else:
        os.chdir("../MODELS/PINO")
        if cfg.custom.model_saturation == "FNO":
            logger.info(
                "|   PRESSURE MODEL = PINO;   SATUARATION MODEL = PINO; PEACEMAN MODEL = PINO |"
            )
        models = {}
        base_paths = {
            "pressure": "./checkpoints_pressure",
            "gas": "./checkpoints_gas",
            "peacemann": "./checkpoints_peacemann",
            "saturation": "./checkpoints_saturation",
            "oil": "./checkpoints_oil",
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
                    base_paths["saturation"], "fno_saturation_forward_model.pth"
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
                "SOIL",
            )
            models["oil"] = fno_supervised_oil
    os.chdir(oldfolder)
    os.chdir(oldfolder)
    if DEFAULT == "Yes":
        Trainmoe = "MoE"  # FNO #MoE
        logger.info("Inference peacemann with Mixture of Experts")
    else:
        Trainmoe = cfg.custom.INVERSE_PROBLEM.Train_Moe
    if Trainmoe == "MoE":
        TEMPLATEFILE["Peaceman modelling inference"] = (
            "Inference peacemann = Mixture of Experts"
        )
    else:
        TEMPLATEFILE["Peaceman modelling inference"] = "Inference peacemann = FNO"
    if Trainmoe == "MoE":
        if dist.rank == 0:
            logger.info(
                "------------------------------------------------------L-----------"
            )
            logger.info(
                "Using Cluster Classify Regress (CCR) for peacemann model          "
            )
            logger.info("References for CCR include: ")
            logger.info(
                "(1): David E. Bernholdt, Mark R. Cianciosa, David L. Green, Jin M. Park,\n\
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
    else:
        pred_type = 1
    degg = 3
    rho = 1.05
    aay1 = minK  # np.min(Truee)
    bby1 = maxK  # np.max(Truee)
    Low_K1, High_K1 = aay1, bby1
    perm_high = maxK  # np.asscalar(np.amax(perm,axis=0)) + 300
    perm_low = minK  # np.asscalar(np.amin(perm,axis=0))-50
    High_P, Low_P = 0.5, 0.05
    poro_high = High_P  # np.asscalar(np.amax(poro,axis=0))+0.3
    poro_low = Low_P  # np.asscalar(np.amin(poro,axis=0))-0.1
    High_K, Low_K, High_P, Low_P = perm_high, perm_low, poro_high, poro_low
    if dist.rank == 0:
        if not os.path.exists(to_absolute_path("../RESULTS/HM_RESULTS")):
            os.makedirs(to_absolute_path("../RESULTS/HM_RESULTS"), exist_ok=True)
        else:
            shutil.rmtree(to_absolute_path("../RESULTS/HM_RESULTS"))
            os.makedirs(to_absolute_path("../RESULTS/HM_RESULTS"), exist_ok=True)
    # locc = 10
    if DEFAULT == "Yes":
        BASSE = "Percentage of data value"
        if dist.rank == 0:
            logger.info(
                "Covarance data noise matrix using percentage of measured value"
            )
    else:
        BASSE = cfg.custom.INVERSE_PROBLEM.CD_matrix
    if BASSE == "Percentage of data value":
        TEMPLATEFILE["Covariance matrix generation"] = (
            "Covariance noise matrix generation = data percentage\n"
        )
    else:
        TEMPLATEFILE["Covariance matrix generation"] = (
            "Covariance noise matrix generation = constant value\n"
        )
    os.chdir(to_absolute_path(cfg.custom.file_location))
    Time = Get_Time(nx, ny, nz, steppi, steppi_indices, 1)
    Time_unie1 = np.zeros((steppi))
    for i in range(steppi):
        Time_unie1[i] = Time[0, i, 0, 0, 0]
    os.chdir(oldfolder)
    source_dir = cfg.custom.file_location
    os.chdir(source_dir)
    timestep = np.genfromtxt(to_absolute_path("../simulator_data/timestep.out"))
    timestep = timestep.astype(int)
    os.chdir(oldfolder)
    if dist.rank == 0:
        logger.info("Read Historical data")
    indii = 20
    _, True_data1, True_mat = historydata(timestep, steppi, steppi_indices, N_pr)
    True_mat[True_mat <= 0] = 0
    # True_mat = True_data1
    os.chdir(to_absolute_path("../RESULTS/HM_RESULTS"))
    if dist.rank == 0:
        Plot_RSM_singleT(True_mat, Time_unie1, N_pr, well_names)
    os.chdir(oldfolder)
    True_K = perm_ensembley[:, indii]
    # True_P = poro_ensembley[:, indii]
    # True_fault = fault_ensemblepy[:, indii]
    if dist.rank == 0:
        pass  # remove unused assignment
    os.chdir(source_dir)
    Time = Get_Time(nx, ny, nz, steppi, steppi_indices, 1)
    Time_unie1 = np.zeros((steppi))
    for i in range(steppi):
        Time_unie1[i] = Time[0, i, 0, 0, 0]
    os.chdir(oldfolder)
    _, True_data1, True_mat = historydata(timestep, steppi, steppi_indices, N_pr)
    True_mat[True_mat <= 0] = 0
    os.chdir(to_absolute_path("../RESULTS/HM_RESULTS"))
    if dist.rank == 0:
        Plot_RSM_singleT(True_mat, Time_unie1, N_pr, well_names)
    os.chdir(oldfolder)
    quant_big = {}  # ✅ Dictionary to store all wells
    for k in range(lenwels):
        quantt = {}  # ✅ Temporary dictionary for each well
        ajes, bjes, cjes = scale_array(True_mat[:, k * N_pr : (k + 1) * N_pr])
        quantt["value"] = ajes
        quantt["scale"] = bjes
        quantt["boolean"] = cjes
        quant_big[f"K_{k}"] = quantt
    jesuni = []
    for k in range(lenwels):
        quantt = quant_big[f"K_{k}"]["value"]
        jesuni.append(quantt)
    True_data = np.hstack(jesuni)
    True_data = np.reshape(True_data, (-1, 1), "F")
    rows_to_remove = np.where(True_data <= 1e-4)[0]
    True_data = remove_rows(True_data, rows_to_remove).reshape(-1, 1)
    # removed unused copy True_yet
    sdoperat = np.std(True_data, axis=0)
    sdoperat = np.reshape(sdoperat, (1, -1), "F")
    menoperat = np.mean(True_data, axis=0)
    menoperat = np.reshape(menoperat, (1, -1), "F")
    True_dataTI = True_data
    True_dataTI = np.reshape(True_dataTI, (-1, 1), "F")
    Time = Get_Time(nx, ny, nz, steppi, steppi_indices, Ne)
    return (
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
    )


def fast_gaussian(dimension, Sdev, Corr):
    """Return Gaussian random field in k-space given std and correlation."""
    dimension = np.array(dimension).flatten()
    m = dimension[0]
    if len(dimension) == 1:
        n = m
    elif len(dimension) == 2:
        n = dimension[1]
    else:
        raise ValueError(
            "FastGaussian: Wrong input, dimension should have length at most 2"
        )
    if np.max(np.size(Sdev)) > 1:  # check input
        variance = 1
    else:
        variance = Sdev
    if len(Corr) == 1:
        Corr = np.array([Corr[0], Corr[0]])
    elif len(Corr) > 2:
        raise ValueError("FastGaussian: Wrong input, Corr should have length at most 2")
    dist = np.arange(0, m) / Corr[0]
    T = scipy.linalg.toeplitz(dist)
    T = variance * np.exp(-(T**2)) + 1e-10 * np.eye(m)
    cholT = np.linalg.cholesky(T)
    if Corr[0] == Corr[1] and n == m:
        cholT2 = cholT
    else:
        dist2 = np.arange(0, n) / Corr[1]
        T2 = scipy.linalg.toeplitz(dist2)
        T2 = variance * np.exp(-(T2**2)) + 1e-10 * np.eye(n)
        cholT2 = np.linalg.cholesky(T2)
    x = np.random.randn(m * n)
    x = np.dot(cholT.T, np.dot(x.reshape(m, n), cholT2))
    x = x.flatten()
    if np.max(np.size(Sdev)) > 1:
        if np.min(np.shape(Sdev)) == 1 and len(Sdev) == len(x):
            x = Sdev * x
        else:
            raise ValueError("FastGaussian: Inconsistent dimension of Sdev")
    return x


def get_shape(t):
    """Return (nz, nx, ny) inferred shape from flattened or 3D tensor `t`."""
    shape = []
    while isinstance(t, tuple):
        shape.append(len(t))
        t = t[0]
    return tuple(shape)


def NorneInitialEnsemble(nx, ny, nz, ensembleSize=100, randomNumber=1.2345e5):
    """Create NORNE-shaped initial ensemble from random seeds."""
    np.random.seed(int(randomNumber))
    N = ensembleSize
    norne = NorneGeostat(nx, ny, nz)
    A = norne["actnum"]
    D = norne["dim"]
    N_F = D[0] * D[1] * D[2]
    M = [norne["poroMean"], norne["permxLogMean"], 0.6]
    S = [norne["poroStd"], norne["permxStd"], norne["ntgStd"]]
    A_L = [A[i : i + D[1] * D[2]] for i in range(0, len(A), D[1] * D[2])]
    A_L = np.array(A_L)
    M_MF = 0.6
    S_MF = norne["multfltStd"]
    C = [norne["poroRange"], norne["permxRange"], norne["ntgRange"]]
    C_S = 2
    R1 = norne["poroPermxCorr"]
    ensembleperm = np.zeros((N_F, N))
    ensemblefault = np.zeros((53, N))
    ensembleporo = np.zeros((N_F, N))
    indices = np.where(A == 1)
    for i in range(N):
        A_MZ = A_L[:, [0, 7, 10, 11, 14, 17]]  # Adjusted indexing to 0-based
        A_MZ = A_MZ.flatten()
        X = M_MF + S_MF * np.random.randn(53)
        ensemblefault[:, i] = X
        C = np.array(C)
        X1 = gaussian_with_variable_parameters(D, np.zeros(N_F), 1, C[0], C_S)[0]
        X1 = X1.reshape(-1, 1)
        ensembleporo[indices, i] = (M[0] + S[0] * X1[indices]).ravel()
        X2 = gaussian_with_variable_parameters(D, np.zeros(N_F), 1, C[1], C_S)[0]
        X2 = X2.reshape(-1, 1)
        X = R1 * X1 + np.sqrt(1 - R1**2) * X2
        indices = np.where(A == 1)
        ensembleperm[indices, i] = np.exp((M[1] + S[1] * X[indices]).ravel())
    return ensembleperm, ensembleporo, ensemblefault


def gaussian_with_variable_parameters(
    field_dim, mean_value, sdev, mean_corr_length, std_corr_length
):
    """Sample Gaussian field with variable variance/correlation across layers."""
    corr_length = add_gnoise(mean_corr_length, std_corr_length, 1)
    if len(field_dim) < 3:
        x = mean_value + fast_gaussian(field_dim, sdev, corr_length)
    else:
        layer_dim = np.prod(field_dim[:2])
        x = np.copy(mean_value)
        if np.isscalar(sdev):
            for i in range(field_dim[2]):
                idx_range = slice(i * layer_dim, (i + 1) * layer_dim)
                x[idx_range] = mean_value[idx_range] + fast_gaussian(
                    field_dim[:2], sdev, corr_length
                )
                # Generate new correlation length for the next layer
                corr_length = add_gnoise(mean_corr_length, std_corr_length, 1)
        else:
            for i in range(field_dim[2]):
                idx_range = slice(i * layer_dim, (i + 1) * layer_dim)
                x[idx_range] = mean_value[idx_range] + fast_gaussian(
                    field_dim[:2], sdev[idx_range], corr_length
                )
                # Generate new correlation length for the next layer
                corr_length = add_gnoise(mean_corr_length, std_corr_length, 1)
    return x, corr_length


def NorneGeostat(nx, ny, nz):
    """Compute NORNE geostatistics (means/stds and correlations per layer)."""
    norne = {}
    dim = np.array([nx, ny, nz])
    ldim = dim[0] * dim[1]
    norne["dim"] = dim
    act = read_until_line("../simulator_data/ACTNUM_0704.prop")
    act = act.T
    act = np.reshape(act, (-1,), "F")
    norne["actnum"] = act
    meanv = np.zeros(dim[2])
    stdv = np.zeros(dim[2])
    file_path = "../simulator_data/porosity.dat"
    p = read_until_line(file_path)
    p = p[act != 0]
    for nr in range(int(dim[2])):
        index_start = ldim * nr
        index_end = ldim * (nr + 1)
        values_range_start = int(np.sum(act[:index_start]))
        values_range_end = int(np.sum(act[:index_end]))
        values = p[values_range_start:values_range_end]
        meanv[nr] = np.mean(values)
        stdv[nr] = np.std(values)
    norne["poroMean"] = p
    norne["poroLayerMean"] = meanv
    norne["poroLayerStd"] = stdv
    norne["poroStd"] = 0.05
    norne["poroLB"] = 0.1
    norne["poroUB"] = 0.4
    norne["poroRange"] = 26
    k = read_until_line("../simulator_data/permx.dat")
    k = np.log(k)
    k = k[act != 0]
    meanv = np.zeros(dim[2])
    stdv = np.zeros(dim[2])
    for nr in range(int(dim[2])):
        index_start = ldim * nr
        index_end = ldim * (nr + 1)
        values_range_start = int(np.sum(act[:index_start]))
        values_range_end = int(np.sum(act[:index_end]))
        values = k[values_range_start:values_range_end]
        meanv[nr] = np.mean(values)
        stdv[nr] = np.std(values)
    norne["permxLogMean"] = k
    norne["permxLayerLnMean"] = meanv
    norne["permxLayerStd"] = stdv
    norne["permxStd"] = 1
    norne["permxLB"] = 0.1
    norne["permxUB"] = 10
    norne["permxRange"] = 26
    corr_with_next_layer = np.zeros(dim[2] - 1)
    for nr in range(dim[2] - 1):
        index_start = ldim * nr
        index_end = ldim * (nr + 1)
        index2_start = ldim * (nr + 1)
        index2_end = ldim * (nr + 2)
        act_layer1 = act[index_start:index_end]
        act_layer2 = act[index2_start:index2_end]
        active = act_layer1 * act_layer2
        values1_range_start = int(np.sum(act[:index_start]))
        values1_range_end = int(np.sum(act[:index_end]))
        values1 = np.concatenate(
            (
                k[values1_range_start:values1_range_end],
                p[values1_range_start:values1_range_end],
            )
        )
        values2_range_start = int(np.sum(act[:index2_start]))
        values2_range_end = int(np.sum(act[:index2_end]))
        values2 = np.concatenate(
            (
                k[values2_range_start:values2_range_end],
                p[values2_range_start:values2_range_end],
            )
        )
        v1 = np.concatenate((act_layer1, act_layer1))
        v1[v1 == 1] = values1.flatten()
        v2 = np.concatenate((act_layer2, act_layer2))
        v2[v2 == 1] = values2.flatten()
        active_full = np.concatenate((active, active))
        co = np.corrcoef(v1[active_full == 1], v2[active_full == 1])
        corr_with_next_layer[nr] = co[0, 1]
    norne["corr_with_next_layer"] = corr_with_next_layer.T
    norne["poroPermxCorr"] = 0.7
    norne["poroNtgCorr"] = 0.6
    norne["ntgStd"] = 0.1
    norne["ntgLB"] = 0.01
    norne["ntgUB"] = 1
    norne["ntgRange"] = 26
    norne["krwMean"] = 1.15
    norne["krwLB"] = 0.8
    norne["krwUB"] = 1.5
    norne["krgMean"] = 0.9
    norne["krgLB"] = 0.8
    norne["krgUB"] = 1
    norne["owcMean"] = np.array([2692.0, 2585.5, 2618.0, 2400.0, 2693.3])
    norne["owcLB"] = norne["owcMean"] - 10
    norne["owcUB"] = norne["owcMean"] + 10
    norne["multregtLogMean"] = np.log10(np.array([0.0008, 0.1, 0.05]))
    norne["multregtStd"] = 0.5
    norne["multregtLB"] = -5
    norne["multregtUB"] = 0
    z_means = [-2, -1.3, -2, -2, -2, -2]
    z_stds = [0.5, 0.5, 0.5, 0.5, 1, 1]
    for i, (mean_, std_) in enumerate(zip(z_means, z_stds), start=1):
        norne[f"z{i}Mean"] = mean_
        norne[f"z{i}Std"] = std_
    norne["zLB"] = -4
    norne["zUB"] = 0
    norne["multzRange"] = 26
    norne["multfltStd"] = 0.5
    norne["multfltLB"] = -5
    norne["multfltUB"] = 2
    return norne


def generate_ensemble(
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
    yet,
    device,
    noise_level,
):
    """Top-level ensemble generation pipeline orchestrating all steps."""
    if dist.rank == 0:
        logger.info("****************************************************************")
        logger.info("                     Generating ensemble                        ")
        logger.info("****************************************************************")
    if Ne == int(cfg.custom.ntrain):
        with gzip.open(to_absolute_path("../data/static.pkl.gz"), "rb") as f2:
            mat = pickle.load(f2)
        X_data1 = mat
        for key, value in X_data1.items():
            if dist.rank == 0:
                logger.info(
                    "****************************************************************"
                )
                logger.info(f"For key '{key}':")
                logger.info("\tContains inf: %s", np.isinf(value).any())
                logger.info("\tContains -inf: %s", np.isinf(-value).any())
                logger.info("\tContains NaN: %s", np.isnan(value).any())
                logger.info(
                    "****************************************************************"
                )
        perm_ensemble = X_data1["ensemble"]
        poro_ensemble = X_data1["ensemblep"]
        fault_ensemblep = X_data1["ensemblefault"]
        perm_ensemble = np.delete(perm_ensemble, indii, axis=1)
        poro_ensemble = np.delete(poro_ensemble, indii, axis=1)
        fault_ensemblep = np.delete(fault_ensemblep, indii, axis=1)
        Neuse = 1  # int(Ne-99)
        perm, poro, fault = NorneInitialEnsemble(
            nx, ny, nz, ensembleSize=Neuse, randomNumber=1.2345e5
        )
        ini_ensemble = perm
        ini_ensemblep = poro
        ini_ensemblefault = fault
        ini_ensemble = np.hstack((ini_ensemble, perm_ensemble))
        ini_ensemblep = np.hstack((ini_ensemblep, poro_ensemble))
        ini_ensemblefault = np.hstack((ini_ensemblefault, fault_ensemblep))
        outt = {}
        outt["PERM"] = ini_ensemble
        outt["PORO"] = ini_ensemblep
        outt = honour2(outt, nx, ny, nz, N_ens, High_K, Low_K, High_P, Low_P, effec)
        ini_ensemble = outt["PERM"]
        ini_ensemblep = outt["PORO"]
        os.chdir(to_absolute_path(cfg.custom.file_location))
        Time = Get_Time(nx, ny, nz, steppi, steppi_indices, Ne)
        Time_unie = np.zeros((steppi))
        for i in range(steppi):
            Time_unie[i] = Time[0, i, 0, 0, 0]
        _, True_data1, True_mat = historydata(timestep, steppi, steppi_indices, N_pr)
        True_mat[True_mat <= 0] = 0
        os.chdir(to_absolute_path("../RESULTS/HM_RESULTS"))
        Plot_RSM_singleT(True_mat, Time_unie1, N_pr, well_names)
        os.chdir(oldfolder)
        jesuni = []
        for k in range(lenwels):
            quantt = quant_big[f"K_{k}"]
            if quantt["boolean"] == 1:
                kodsval = True_mat[:, k * N_pr : (k + 1) * N_pr] / quantt["scale"]
            else:
                kodsval = True_mat[:, k * N_pr : (k + 1) * N_pr] * quantt["scale"]
            jesuni.append(kodsval)
        True_data = np.hstack(jesuni)
        True_data = np.reshape(True_data, (-1, 1), "F")
        True_data = remove_rows(True_data, rows_to_remove).reshape(-1, 1)
        # True_yet = True_data
        Nop = True_data.shape[0]
        os.chdir(oldfolder)
        dt = Time_unie
    else:
        pass
    if (Ne > int(cfg.custom.ntrain)) and (Ne < 5000):
        os.chdir(to_absolute_path(cfg.custom.file_location))
        Time = Get_Time(nx, ny, nz, steppi, steppi_indices, Ne)
        Time_unie = np.zeros((steppi))
        for i in range(steppi):
            Time_unie[i] = Time[0, i, 0, 0, 0]
        os.chdir(oldfolder)
        dt = Time_unie
        with gzip.open(to_absolute_path("../data/static.pkl.gz"), "rb") as f2:
            mat = pickle.load(f2)
        X_data1 = mat
        for key, value in X_data1.items():
            if dist.rank == 0:
                logger.info(
                    "****************************************************************"
                )
                logger.info(f"For key '{key}':")
                logger.info("\tContains inf: %s", np.isinf(value).any())
                logger.info("\tContains -inf: %s", np.isinf(-value).any())
                logger.info("\tContains NaN: %s", np.isnan(value).any())
                logger.info(
                    "****************************************************************"
                )
        perm_ensemble = X_data1["ensemble"]
        poro_ensemble = X_data1["ensemblep"]
        fault_ensemblep = X_data1["ensemblefault"]
        perm_ensemble = np.delete(perm_ensemble, indii, axis=1)
        poro_ensemble = np.delete(poro_ensemble, indii, axis=1)
        fault_ensemblep = np.delete(fault_ensemblep, indii, axis=1)
        _, True_data1, True_mat = historydata(timestep, steppi, steppi_indices, N_pr)
        True_mat[True_mat <= 0] = 0
        os.chdir(to_absolute_path("../RESULTS/HM_RESULTS"))
        Plot_RSM_singleT(True_mat, Time_unie1, N_pr, well_names)
        os.chdir(oldfolder)
        jesuni = []
        for k in range(lenwels):
            quantt = quant_big[f"K_{k}"]
            if quantt["boolean"] == 1:
                kodsval = True_mat[:, k * N_pr : (k + 1) * N_pr] / quantt["scale"]
            else:
                kodsval = True_mat[:, k * N_pr : (k + 1) * N_pr] * quantt["scale"]
            jesuni.append(kodsval)
        True_data = np.hstack(jesuni)
        True_data = np.reshape(True_data, (-1, 1), "F")
        True_data = remove_rows(True_data, rows_to_remove).reshape(-1, 1)
        # True_yet = True_data
        Nop = True_data.shape[0]
        Neuse = int(Ne - int(cfg.custom.ntrain)) + 1
        perm, poro, fault = NorneInitialEnsemble(
            nx, ny, nz, ensembleSize=Neuse, randomNumber=1.2345e5
        )
        ini_ensemble = perm
        ini_ensemblep = poro
        ini_ensemblefault = fault
        ini_ensemble = np.hstack((ini_ensemble, perm_ensemble))
        ini_ensemblep = np.hstack((ini_ensemblep, poro_ensemble))
        ini_ensemblefault = np.hstack((ini_ensemblefault, fault_ensemblep))
    else:
        pass
    if Ne < int(cfg.custom.ntrain):
        indices = np.random.choice(Ne, size=Ne, replace=False)
        with gzip.open(to_absolute_path("../data/static.pkl.gz"), "rb") as f2:
            mat = pickle.load(f2)
        X_data1 = mat
        for key, value in X_data1.items():
            if dist.rank == 0:
                logger.info(
                    "****************************************************************"
                )
                logger.info(f"For key '{key}':")
                logger.info("\tContains inf: %s", np.isinf(value).any())
                logger.info("\tContains -inf: %s", np.isinf(-value).any())
                logger.info("\tContains NaN: %s", np.isnan(value).any())
                logger.info(
                    "****************************************************************"
                )
        ini_ensemble = X_data1["ensemble"]
        ini_ensemblep = X_data1["ensemblep"]
        ini_ensemblefault = X_data1["ensemblefault"]
        ini_ensemble = np.delete(ini_ensemble, indii, axis=1)
        ini_ensemblep = np.delete(ini_ensemblep, indii, axis=1)
        ini_ensemblefault = np.delete(ini_ensemblefault, indii, axis=1)
        ini_ensemble = ini_ensemble[:, indices]
        ini_ensemblep = ini_ensemblep[:, indices]
        ini_ensemblefault = ini_ensemblefault[:, indices]  # avoid unused variable
        os.chdir(to_absolute_path(cfg.custom.file_location))
        Time = Get_Time(nx, ny, nz, steppi, steppi_indices, Ne)
        Time_unie = np.zeros((steppi))
        for i in range(steppi):
            Time_unie[i] = Time[0, i, 0, 0, 0]
        _, True_data1, True_mat = historydata(timestep, steppi, steppi_indices, N_pr)
        True_mat[True_mat <= 0] = 0
        os.chdir(to_absolute_path("../RESULTS/HM_RESULTS"))
        Plot_RSM_singleT(True_mat, Time_unie1, N_pr, well_names)
        os.chdir(oldfolder)
        jesuni = []
        for k in range(lenwels):
            quantt = quant_big[f"K_{k}"]
            if quantt["boolean"] == 1:
                kodsval = True_mat[:, k * N_pr : (k + 1) * N_pr] / quantt["scale"]
            else:
                kodsval = True_mat[:, k * N_pr : (k + 1) * N_pr] * quantt["scale"]
            jesuni.append(kodsval)
        True_data = np.hstack(jesuni)
        True_data = np.reshape(True_data, (-1, 1), "F")
        True_data = remove_rows(True_data, rows_to_remove).reshape(-1, 1)
        # True_yet = True_data
        Nop = True_data.shape[0]
        os.chdir(oldfolder)
        dt = Time_unie
    else:
        pass
    TEMPLATEFILE["Ensemble size"] = Ne
    if dist.rank == 0:
        logger.info(
            "----------------------------------------------------------------------"
        )
        logger.info(
            "              History Matching Operational conditions                 "
        )
        logger.info(
            "----------------------------------------------------------------------"
        )
        for key, value in TEMPLATEFILE.items():
            logger.info(f"{key}: {value}")
    yaml_filename = to_absolute_path(
        "../RESULTS/HM_RESULTS/History_Matching_Template_file.yaml"
    )
    if dist.rank == 0:
        with open(yaml_filename, "w") as yaml_file:
            yaml.dump(TEMPLATEFILE, yaml_file)

    start_time = time.time()
    if dist.rank == 0:
        print("----------------------------------------------------------------------")
        print(
            "----------------Starting the History matching with - ",
            str(Ne) + " Ensemble members  ",
        )
        print("****************************************************************")
    os.chdir(oldfolder)
    ensemble = ini_ensemble
    ensemble = np.nan_to_num(ensemble, copy=True, nan=Low_K1)
    ensemblep = ini_ensemblep
    ensemblef = ini_ensemblefault
    outt = {}
    outt["PERM"] = ensemble
    outt["PORO"] = ensemblep
    outt = honour2(outt, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P, effec)
    ensemble = outt["PERM"]
    ensemblep = outt["PORO"]
    ensemble = ensemble  # * effec
    ensemblep = ensemblep  # * effec
    Nop = True_data.shape[0]
    ax = np.zeros((Nop, 1))
    for iq in range(Nop):
        if (True_data[iq, :] > 0) and (True_data[iq, :] <= 1e10):
            ax[iq, :] = sqrt(noise_level * True_data[iq, :])
        else:
            ax[iq, :] = 1
    R = ax**2
    R = torch.as_tensor(R, dtype=torch.float32).to(
        device
    )  # Convert R to a tensor if not already
    CDd = torch.diag(R.view(-1))
    Cini = CDd.clone().to(device)  # Move Cini to GPU
    mean = torch.zeros(Nop, dtype=torch.float32).to(device)
    perturbations = torch.distributions.MultivariateNormal(
        mean, covariance_matrix=Cini
    ).sample((Ne,))
    perturbations = perturbations.T  # Transpose
    if yet == 0:
        perturbations = perturbations.detach().cpu().numpy()
    else:
        pass

    return (
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
    )
