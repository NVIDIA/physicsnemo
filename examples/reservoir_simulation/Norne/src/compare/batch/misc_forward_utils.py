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
                    FORWARD SIMULATION UTILITIES
=====================================================================

This module provides utility functions for forward simulation operations in batch
processing for reservoir simulation. It includes functions for data processing,
model validation, parameter setup, and simulation orchestration.

Key Features:
- Data validation and preprocessing utilities
- Simulation parameter setup and configuration
- Model performance evaluation and metrics
- File I/O operations with proper error handling
- Data transformation and normalization
- Ensemble processing and aggregation
- Machine learning model integration

Usage:
    from compare.batch.misc_forward_utils import (
        setup_simulation_parameters,
        validate_input_data,
        process_ensemble_results
    )

Inputs:
    - Configuration parameters
    - Input data arrays
    - Model specifications
    - Simulation parameters

Outputs:
    - Processed and validated data
    - Simulation results
    - Performance metrics
    - Logged status messages

@Author : Clement Etienam
"""

# 🛠 Standard Library
import os

# import re
import pickle
import logging

# 🔧 Third-party Libraries
import numpy as np
import numpy.matlib

# import pandas as pd
import torch
import xgboost as xgb
from sklearn.cluster import MiniBatchKMeans
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.utils.cholesky import psd_safe_cholesky

# 📦 Local Modules
from hydra.utils import to_absolute_path
from compare.batch.misc_gather import (
    get_dyna,
    EclBinaryParser,
)

from compare.batch.misc_forward import process_data


def setup_logging() -> logging.Logger:
    """Configure and return the main logger."""
    logger = logging.getLogger("Forward problem")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


type_dict = {
    b"INTE": "i",
    b"CHAR": "8s",
    b"REAL": "f",
    b"DOUB": "d",
    b"LOGI": "4s",
    b"MESS": "?",
}

ecl_extensions = [
    ".DATA",
    ".DBG",
    ".ECLEND",
    ".EGRID",
    ".FEGRID",
    ".FGRID",
    ".FINIT",
    ".FINSPEC",
    ".FRFT",
    ".FRSSPEC",
    ".FSMSPEC",
    ".FUNRST",
    ".FUNSMRY",
    ".GRID",
    ".INIT",
    ".INSPEC",
    ".MSG",
    ".PRT",
    ".RFT",
    ".RSM",
    ".RSSPEC",
    ".SMSPEC",
    ".UNRST",
    ".UNSMRY",
    ".dbprtx",
]

dynamic_props = [
    "SEQNUM",
    "PRESSURE",
    "SWAT",
    "SGAS",
    "SOIL",
    "RS",
    "RV",
    "RSSAT",
    "RVSAT",
    "STATES",
    "OWC",
    "OGC",
    "GWC",
    "EOWC",
    "EOGC",
    "OILAPI",
    "SDENO",
    "FIPOIL",
    "RFIPOIL",
    "FIPGAS",
    "RFIPGAS",
    "FIPWAT",
    "RFIPWAT",
    "SFIPOIL",
    "SFIPGAS",
    "SFIPWAT",
    "SFIPPLY",
    "RFIPPLY",
    "SFIPSAL",
    "RFIPSAL",
    "SFIPSOL",
    "SFIPGGI",
    "RFIPOIL",
    "RFIPGAS",
    "RFIPWAT",
    "RFIPSOL",
    "RFIPGGI",
    "OIL-POTN",
    "GAS-POTN",
    "WAT-POTN",
    "POLYMER",
    "PADS",
    "PLYTRRFA",
    "POLYMAX",
    "SALT",
    "TEMP",
    "XMF",
    "YMF",
    "ZMF",
    "SSOL",
    "PBUB",
    "PDEW",
    "SURFACT",
    "SURFADS",
    "SURFMAX",
    "SURFCNM",
    "SURFST",
    "GGI",
    "WAT-PRES",
    "GAS-PRES",
    "OIL-VISC",
    "WAT-VISC",
    "GAS-VISC",
    "OIL-DEN",
    "WAT-DEN",
    "GAS-DEN",
    "DRAINAGE",
    "DRAINMIN",
    "PCOW",
    "PCOG",
    "1OVERBO",
    "1OVERBW",
    "1OVERBG",
    "POT_CORR",
    "OILKR",
    "WATKR",
    "GASKR",
    "HYDH",
    "HYDHFW",
    "PORV",
    "RPORV",
    "FOAM",
    "FOAMADS",
    "FOAMMAX",
    "FOAMDCY",
    "FOAMCNM",
    "FOAM_HL",
    "FOAMMOB",
    "ALKALINE",
    "ALKADS",
    "ALKMAX",
    "STMALK",
    "SFADALK",
    "PLADALK",
    "PADMAX",
    "CATSURF",
    "CATROCK",
    "ESALSUR",
    "ESALPLY",
    "COALGAS",
    "COALSOLV",
    "GASSATC",
    "MLANG",
    "MLANGSLV",
    "SWMIN",
    "SWMAX",
    "ISTHW",
    "SOMAX",
    "ISTHG",
    "SGMIN",
    "SGMAX",
    "PRESROCC",
    "CNV_OIL",
    "CNV_WAT",
    "CNV_GAS",
    "CNV_PLY",
    "TRANEXX",
    "TRANEXY",
    "TRANEXZ",
    "EXCAVNUM",
    "CNV_SAL",
    "CNV_SOL",
    "CNV_GGI",
    "CNV_DPRE",
    "CNV_DWAT",
    "CNV_DGAS",
    "CNV_DPLY",
    "CNV_DSAL",
    "CNV_DSOL",
    "CNV_DGGI",
    "CONV_VBR",
    "CONV_PRU",
    "CONV_NEW",
    "FLOOILI+",
    "FLOOILJ+",
    "FLOOILK+",
    "FLOGASI+",
    "FLOGASJ+",
    "FLOGASK+",
    "FLOWATI+",
    "FLOWATJ+",
    "FLOWATK+",
    "FLROILI+",
    "FLROILJ+",
    "FLROILK+",
    "FLRGASI+",
    "FLRGASJ+",
    "FLRGASK+",
    "FLRWATI+",
    "FLRWATJ+",
    "FLRWATK+",
    "VOILI+",
    "VOILJ+",
    "VOILK+",
    "VGASI+",
    "VGASJ+",
    "VGASK+",
    "VWATI+",
    "VWATJ+",
    "VWATK+",
    "FLOOILN+",
    "FLOGASN+",
    "FLOWATN+",
    "FLOOILL+",
    "FLOGASL+",
    "FLOWATL+",
    "FLOOILA+",
    "FLOGASA+",
    "FLOWATA+",
    "FLROILN+",
    "FLRGASN+",
    "FLRWATN+",
    "FLROILL+",
    "FLRGASL+",
    "FLRWATL+",
    "FLROILA+",
    "FLRGASA+",
    "FLRWATA+",
]


ecl_vectors = [
    "COPR",
    "COPT",
    "CWFR",
    "CWIR",
    "CWPR",
    "CWPT",
    "FGIR",
    "FGIT",
    "FGLIR",
    "FGOR",
    "FGORH",
    "FGPR",
    "FGPT",
    "FLPR",
    "FLPT",
    "FMCTP",
    "FMWWO",
    "FMWWT",
    "FODEN",
    "FOE",
    "FOIP",
    "FOPR",
    "FOPRF",
    "FOPRH",
    "FOPRS",
    "FOPT",
    "FOPTH",
    "FPR",
    "FVIR",
    "FVIT",
    "FVPR",
    "FVPT",
    "FWCT",
    "FWCTH",
    "FWIP",
    "FWIR",
    "FWIT",
    "FWPR",
    "FWPT",
    "GGOR",
    "GGPR",
    "GGPT",
    "GOPR",
    "GOPT",
    "GVIR",
    "GVIT",
    "GVPR",
    "GVPT",
    "GWCT",
    "GWIR",
    "GWPR",
    "MSUMLINS",
    "RGPV",
    "RHPV",
    "ROE",
    "ROEW",
    "ROPV",
    "ROSAT",
    "RPR",
    "RRPV",
    "RWPV",
    "TCPU",
    "TIME",
    "WBHP",
    "WBHPH",
    "WBP",
    "WBP4",
    "WBP9",
    "WGIR",
    "WGIT",
    "WGLIR",
    "WGOR",
    "WGORH",
    "WGPR",
    "WGPRH",
    "WGPTH",
    "WLPR",
    "WLPRH",
    "WLPT",
    "WLPTH",
    "WMCON",
    "WMCTL",
    "WOPR",
    "WOPRH",
    "WOPT",
    "WOPTH",
    "WPI",
    "WTHP",
    "WTICIW1",
    "WTICIW2",
    "WTIRIW1",
    "WTIRIW2",
    "WTPCIW1",
    "WTPCIW2",
    "WTPRIW1",
    "WTPRIW2",
    "WWCT",
    "WWCTH",
    "WWIR",
    "WWIRH",
    "WWIT",
    "WWITH",
    "WWPR",
    "WWPRH",
    "WWPT",
    "WWPTH",
    "YEARS",
]

static_props = [
    "DEPTH",
    "DX",
    "DR",
    "DY",
    "DTHETA",
    "DZ",
    "PORO",
    "PERMX",
    "PERMR",
    "PERMI",
    "PERMY",
    "PERMTHT",
    "PERMJ",
    "PERMZ",
    "PERMK",
    "MULTX",
    "MULTR",
    "MULTI",
    "MULTY",
    "MULTTHT",
    "MULTJ",
    "MULTZ",
    "MULTK",
    "TRANX",
    "TRANR",
    "TRANI",
    "TRANY",
    "TRANTHT",
    "TRANJ",
    "TRANZ",
    "TRANK",
    "DIFFMX",
    "DIFFMR",
    "DIFFMI",
    "DIFFMY",
    "DIFFMTHT",
    "DIFFMJ",
    "DIFFMZ",
    "DIFFMK",
    "DIFFX",
    "DIFFR",
    "DIFFI",
    "DIFFY",
    "DIFFTHT",
    "DIFFJ",
    "DIFFZ",
    "DIFFK",
    "DIFFTX",
    "DIFFTR",
    "DIFFTI",
    "DIFFTY",
    "DIFFTTHT",
    "DIFFTJ",
    "DIFFTZ",
    "DIFFTK",
    "HEATTX",
    "HEATTR",
    "HEATTY",
    "HEATTTHT",
    "MLANGI",
    "GASSATC",
    "MLNGSLVI",
    "MLANG",
    "GASSATC",
    "MLANGSLV",
    "AQUIFERN",
    "DOMAINS",
    "ENDNUM",
    "EQLNUM",
    "FIPNUM",
    "FLUXNUM",
    "KRO",
    "KRORW",
    "KRW",
    "KRWR",
    "MINPVV",
    "MULTNUM",
    "MULTPV",
    "MULTX",
    "MULTX-",
    "MULTY",
    "MULTY-",
    "MULTZ",
    "MULTZ-",
    "NTG",
    "OPERNUM",
    "PCW",
    "PORV",
    "PVTNUM",
    "SATNUM",
    "SOWCR",
    "SWATINIT",
    "SWCR",
    "SWL",
    "SWLPC",
    "SWU",
    "TOPS",
    "TRANNNC",
]

SUPPORTED_DATA_TYPES = {
    "INTE": (4, "i", 1000),
    "REAL": (4, "f", 1000),
    "LOGI": (4, "i", 1000),
    "DOUB": (8, "d", 1000),
    "CHAR": (8, "8s", 105),
    "MESS": (8, "8s", 105),
    "C008": (8, "8s", 105),
}


def Get_data_FFNN(
    oldfolder,
    N,
    pressure,
    Sgas,
    Swater,
    Soil,
    perm,
    Time,
    steppi,
    steppi_indices,
    N_pr,
    producer_wells,
    unique_entries,
    filenameui,
    well_measurements,
    lenwels,
):
    """Assemble FFNN inputs/targets from summary vectors and grid tensors.

    Parameters
    ----------
    oldfolder : str
        Path to restore working directory after reading each realisation.
    N : int
        Number of realisations.
    pressure, Sgas, Swater, Soil, perm, Time : np.ndarray
        Input grid tensors with shapes matching dataset conventions.
    steppi : int
        Number of timesteps.
    steppi_indices : np.ndarray | int
        1-based indices selecting timesteps to sample.
    N_pr : int
        Number of producers per well group.
    producer_wells : list[tuple]
        Producer metadata; last element is the well name.
    unique_entries : list[tuple]
        Completion intervals for averaging grid values.
    filenameui : str
        Base path (without extension) to Eclipse summary files.
    well_measurements : list[str]
        Vector names to extract from UNSMRY (e.g., ['WOPR','WWPR','WGPR']).
    lenwels : int
        Number of well groups.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `(innn, ouut)` input and output arrays for FFNN training.
    """
    well_indices = process_data(unique_entries)
    ouut = np.zeros((N, steppi, lenwels * N_pr))
    innn = np.zeros((N, steppi, (4 * N_pr) + 2))
    producer_well_names = [well[-1] for well in producer_wells]
    for i in range(N):
        folder = to_absolute_path("../RUNS/Realisation" + str(i))
        os.chdir(folder)
        unsmry_file = filenameui
        parser = EclBinaryParser(unsmry_file)
        vectors = parser.read_vectors()
        namez = well_measurements  # ['WOPR', 'WWPR', 'WGPR']
        all_arrays = []
        for namey in namez:
            dfaa = vectors[namey]
            filtered_columns = [
                coll
                for coll in dfaa.columns
                if any(well_namee in coll for well_namee in producer_well_names)
            ]
            filtered_df = dfaa[filtered_columns]
            filtered_df = filtered_df[producer_well_names]
            start_row = find_first_numeric_row(filtered_df)
            if start_row is not None:
                numeric_df = filtered_df.iloc[start_row:]
                result_array = numeric_df.to_numpy()
                logger = setup_logging()
                logger.info(f"Numeric data from {namey} processed successfully.")
            else:
                logger = setup_logging()
                logger.info(f"No numeric rows found in the DataFrame for {namey}.")
                result_array = None
            all_arrays.append(result_array)
        final_array = np.concatenate(all_arrays, axis=1)
        final_array[final_array <= 0] = 0
        out = final_array[steppi_indices - 1, :].astype(float)
        out[out <= 0] = 0
        ouut[i, :, :] = out
        permuse = perm[i, 0, :, :, :]
        presure_use = pressure[i, :, :, :, :]
        gas_use = Sgas[i, :, :, :, :]
        water_use = Swater[i, :, :, :, :]
        oil_use = Soil[i, :, :, :, :]
        Time_use = Time[i, :, :, :, :]
        mean_big = []
        for idx, indices_list in well_indices.items():
            values = [
                permuse[i_idx, j_idx, k_idx]
                if k_idx == l_idx
                else permuse[i_idx, j_idx, k_idx : l_idx + 1]
                for i_idx, j_idx, k_idx, l_idx in indices_list
            ]
            mean_big.append(np.mean(values))
        permxx = np.tile(mean_big, (steppi, 1))
        a3 = get_dyna(steppi, well_indices, water_use[steppi_indices - 1])
        a2 = get_dyna(steppi, well_indices, gas_use[steppi_indices - 1])
        a5 = get_dyna(steppi, well_indices, oil_use[steppi_indices - 1])
        a1 = np.zeros((steppi, 1))
        a4 = np.zeros((steppi, 1))
        for k in range(steppi):
            uniep = presure_use[k, :, :, :]
            permuse = uniep
            a1[k, 0] = np.mean(permuse)
            unietime = Time_use[k, :, :, :]
            permuse = unietime
            a4[k, 0] = permuse[0, 0, 0]
        inn1 = np.hstack((permxx, a1, a5, a2, a3, a4))
        innn[i, :, :] = inn1
        os.chdir(oldfolder)
    return innn, ouut


def convert_back(rescaled_tensor, target_min, target_max, min_val, max_val):
    return rescaled_tensor * max_val


def find_first_numeric_row(df):
    """Find the first row in the DataFrame where all data is numeric."""
    for i in range(len(df)):
        if df.iloc[i].apply(np.isreal).all():
            return i
    return None


def Get_data_FFNN1(
    folder,
    oldfolder,
    N,
    pressure,
    Sgas,
    Swater,
    Soil,
    perm,
    Time,
    steppi,
    steppi_indices,
    N_pr,
    producer_wells,
    unique_entries,
    filenameui,
    well_measurements,
    lenwels,
):
    well_indices = process_data(unique_entries)
    ouut = np.zeros((N, steppi, lenwels * N_pr))
    innn = np.zeros((N, steppi, (4 * N_pr) + 2))
    producer_well_names = [well[-1] for well in producer_wells]
    for i in range(N):
        os.chdir(folder)
        unsmry_file = filenameui
        parser = EclBinaryParser(unsmry_file)
        vectors = parser.read_vectors()
        namez = well_measurements  # ['WOPR', 'WWPR', 'WGPR']
        all_arrays = []
        for namey in namez:
            dfaa = vectors[namey]
            filtered_columns = [
                coll
                for coll in dfaa.columns
                if any(well_namee in coll for well_namee in producer_well_names)
            ]
            filtered_df = dfaa[filtered_columns]
            filtered_df = filtered_df[producer_well_names]
            # Extract numeric data and convert to numpy array
            start_row = find_first_numeric_row(filtered_df)
            if start_row is not None:
                numeric_df = filtered_df.iloc[start_row:]
                result_array = numeric_df.to_numpy()
                logger = setup_logging()
                logger.info(f"Numeric data from {namey} processed successfully.")
            else:
                logger = setup_logging()
                logger.info(f"No numeric rows found in the DataFrame for {namey}.")
                result_array = None
            all_arrays.append(result_array)
        final_array = np.concatenate(all_arrays, axis=1)
        final_array[final_array <= 0] = 0
        out = final_array[steppi_indices - 1, :].astype(float)
        out[out <= 0] = 0
        ouut[i, :, :] = out
        permuse = perm[i, 0, :, :, :]
        presure_use = pressure[i, :, :, :, :]
        gas_use = Sgas[i, :, :, :, :]
        water_use = Swater[i, :, :, :, :]
        oil_use = Soil[i, :, :, :, :]
        Time_use = Time[i, :, :, :, :]
        mean_big = []
        for idx, indices_list in well_indices.items():
            values = [
                permuse[i_idx, j_idx, k_idx]
                if k_idx == l_idx
                else permuse[i_idx, j_idx, k_idx : l_idx + 1]
                for i_idx, j_idx, k_idx, l_idx in indices_list
            ]
            mean_big.append(np.mean(values))
        permxx = np.tile(mean_big, (steppi, 1))
        a3 = get_dyna(steppi, well_indices, water_use)
        a2 = get_dyna(steppi, well_indices, gas_use)
        a5 = get_dyna(steppi, well_indices, oil_use)
        a1 = np.zeros((steppi, 1))
        a4 = np.zeros((steppi, 1))
        for k in range(steppi):
            uniep = presure_use[k, :, :, :]
            permuse = uniep
            a1[k, 0] = np.mean(permuse)
            unietime = Time_use[k, :, :, :]
            permuse = unietime
            a4[k, 0] = permuse[0, 0, 0]
        inn1 = np.hstack((permxx, a1, a5, a2, a3, a4))
        innn[i, :, :] = inn1
        os.chdir(oldfolder)
    return innn, ouut


def endit(i, testt, training_master, oldfolder, pred_type, degg, big, experts, device):
    logger = setup_logging()
    logger.info("")
    logger.info(f"Starting prediction from machine {i + 1}")
    numcols = len(testt[0])
    izz = PREDICTION_CCR__MACHINE(
        i,
        big,
        testt,
        numcols,
        training_master,
        oldfolder,
        pred_type,
        degg,
        experts,
        device,
    )
    logger.info("")
    logger.info(f"Finished Prediction from machine {i + 1}")
    return izz


def predict_machine(a0, model):
    ynew = model.predict(xgb.DMatrix(a0))
    return ynew


def predict_machine3(a0, deg, model, poly):
    predicted = model.predict(poly.fit_transform(a0))
    return predicted


class SparseGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, inducing_points):
        super(SparseGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        self.variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            self.variational_distribution,
            learn_inducing_locations=True,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def _cholesky(self, A):
        """✅ Fix Cholesky decomposition issue with jitter"""
        jitter = 1e-5  # Small positive value
        eye = torch.eye(A.size(-1), device=A.device)
        return psd_safe_cholesky(A + jitter * eye)  # ✅ Safe Cholesky

    def _mean_cache(self):
        """✅ Uses safe Cholesky for covariance matrix"""
        train_train_covar = self.train_train_covar.evaluate_kernel()
        train_labels_offset = self.train_labels - self.train_mean
        jitter = 1e-5
        identity = torch.eye(
            train_train_covar.size(-1), device=train_train_covar.device
        )
        chol = self._cholesky(train_train_covar + jitter * identity)
        return torch.cholesky_solve(train_labels_offset, chol).squeeze(-1)


def fit_Gp(X, y, device, itery, percentage=50.0):
    X = torch.tensor(X, dtype=torch.float32, device=device, requires_grad=True)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    X_clone = X.clone()
    with torch.no_grad():
        X_np = X_clone.cpu().numpy()  # Now safe to convert to NumPy
        num_inducing_points = max(
            int(X_np.shape[0] * (percentage / 100)), 1
        )  # Ensure at least one inducing point
        kmeans = MiniBatchKMeans(
            n_clusters=num_inducing_points, random_state=42, n_init="auto"
        )
        kmeans.fit(X_np)  # Uses clone, keeps autograd
        inducing_points = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32, device=device
        )  # Move centroids to GPU
    likelihood = GaussianLikelihood().to(device)
    model = SparseGPModel(X, y, likelihood, inducing_points).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=2e-2, betas=(0.9, 0.999), weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99998708)

    mll = VariationalELBO(likelihood, model, num_data=y.size(0))
    model.train()
    likelihood.train()
    for epoch in range(itery):
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, y)
        loss = loss.mean()  # Ensure loss is a scalar
        loss.backward(retain_graph=True)  # Keep the graph intact
        optimizer.step()
        scheduler.step()
        del loss  # Free memory
        torch.cuda.empty_cache()
    return model


def PREDICTION_CCR__MACHINE(
    ii,
    nclusters,
    inputtest,
    numcols,
    training_master,
    oldfolder,
    pred_type,
    deg,
    experts,
    device,
):
    filenamex = "clfx_%d.asv" % ii
    filenamey = "clfy_%d.asv" % ii
    os.chdir(training_master)
    if experts == 1 or experts == 3:
        filename1 = "Classifier_%d.bin" % ii
        loaded_model = xgb.Booster({"nthread": 4})  # init model
        loaded_model.load_model(filename1)  # load data
    if experts == 2:
        filename1 = "Classifier_%d.pkl" % ii
        with open(filename1, "rb") as file:
            loaded_model = pickle.load(file)
    clfx = pickle.load(open(filenamex, "rb"))
    clfy = pickle.load(open(filenamey, "rb"))
    os.chdir(oldfolder)
    inputtest = clfx.transform(inputtest)
    if experts == 2:
        labelDA = loaded_model.predict(inputtest)
    else:
        labelDA = loaded_model.predict(xgb.DMatrix(inputtest))
        if nclusters == 2:
            labelDAX = 1 - labelDA
            labelDA = np.reshape(labelDA, (-1, 1))
            labelDAX = np.reshape(labelDAX, (-1, 1))
            labelDA = np.concatenate((labelDAX, labelDA), axis=1)
            labelDA = np.argmax(labelDA, axis=-1)
        else:
            labelDA = np.argmax(labelDA, axis=-1)
        labelDA = np.reshape(labelDA, (-1, 1), "F")
    numrowstest = len(inputtest)
    operationanswer = np.zeros((numrowstest, 1))
    labelDA = np.reshape(labelDA, (-1, 1), "F")
    for i in range(nclusters):
        logging.getLogger(__name__).info(
            "-- Predicting cluster: %s | %s", str(i + 1), str(nclusters)
        )
        if experts == 1:  # Polynomial regressor experts
            filename2 = "Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) + ".pkl"
            filename2b = "polfeat_" + str(ii) + "_Cluster_" + str(i) + ".pkl"
            os.chdir(training_master)
            with open(filename2, "rb") as file:
                model0 = pickle.load(file)
            with open(filename2b, "rb") as filex:
                poly0 = pickle.load(filex)
            os.chdir(oldfolder)
            labelDA0 = (np.asarray(np.where(labelDA == i))).T
            #    ##----------------------##------------------------##
            a00 = inputtest[labelDA0[:, 0], :]
            a00 = np.reshape(a00, (-1, numcols), "F")
            if a00.shape[0] != 0:
                operationanswer[labelDA0[:, 0], :] = np.reshape(
                    predict_machine3(a00, deg, model0, poly0), (-1, 1)
                )
        elif experts == 2:
            labelDA0 = (np.asarray(np.where(labelDA == i))).T
            #    ##----------------------##------------------------##
            a00 = inputtest[labelDA0[:, 0], :]
            a00 = np.reshape(a00, (-1, numcols), "F")
            a00 = torch.tensor(a00, dtype=torch.float32).to(device)
            filename2 = "Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) + ".pth"
            os.chdir(training_master)
            checkpoint = torch.load(filename2, map_location=device)
            num_inducing_points = checkpoint[
                "variational_strategy.inducing_points"
            ].shape[0]
            input_dim = checkpoint["variational_strategy.inducing_points"].shape[1]
            # output_dim = 1  # Assuming a single output per sample
            train_x = torch.zeros(a00.shape[0], input_dim).to(device)
            train_y = torch.zeros(a00.shape[0], 1).to(device)
            train_y = train_y.squeeze(-1)
            likelihood = GaussianLikelihood().to(device)
            inducing_points = torch.zeros(num_inducing_points, input_dim).to(device)
            model = SparseGPModel(train_x, train_y, likelihood, inducing_points).to(
                device
            )
            model.load_state_dict(checkpoint, strict=False)  # ✅ Pass strict=False here
            os.chdir(oldfolder)
            model = model.to(device)
            model.eval()
            batch_size = 1  # Adjust based on memory availability
            predictions = []
            if a00.shape[0] != 0:
                with torch.no_grad():
                    for i in range(0, a00.shape[0], batch_size):
                        batch = a00[i : i + batch_size]  # Take a batch of inputs
                        prediction = model(batch)  # Forward pass
                        pred = prediction.mean.detach().cpu().numpy()
                        predictions.append(pred)  # Store batch predictions
                operationanswer[labelDA0[:, 0], :] = np.vstack(predictions)
            torch.cuda.empty_cache()  # Free unused GPU memory
        else:  # XGBoost experts
            loaded_modelr = xgb.Booster({"nthread": 4})  # init model
            filename2 = "Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) + ".bin"
            os.chdir(training_master)
            loaded_modelr.load_model(filename2)  # load data
            os.chdir(oldfolder)
            labelDA0 = (np.asarray(np.where(labelDA == i))).T
            a00 = inputtest[labelDA0[:, 0], :]
            a00 = np.reshape(a00, (-1, numcols), "F")
            if a00.shape[0] != 0:
                operationanswer[labelDA0[:, 0], :] = np.reshape(
                    predict_machine(a00, loaded_modelr), (-1, 1)
                )
    operationanswer = clfy.inverse_transform(operationanswer)
    return operationanswer
