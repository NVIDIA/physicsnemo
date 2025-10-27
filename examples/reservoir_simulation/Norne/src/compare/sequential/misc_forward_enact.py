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
                            SEQUENTIAL FORWARD ENACTMENT
=====================================================================

This module provides forward enactment capabilities for sequential processing
of FVM surrogate model comparisons. It includes functions for ensemble
processing, model prediction, and result analysis.

Key Features:
- Forward model ensemble processing
- Machine learning model predictions
- Gaussian Process implementations
- XGBoost model handling
- Data processing and conversion utilities

Usage:
    from compare.sequential.misc_forward_enact import (
        Forward_model_ensemble,
        predict_machine11,
        Make_correct,
        fit_operation,
        convert_back,
        convert_backsin,
        process_data,
        get_dyna,
        endit,
        predict_machine,
        predict_machine3,
        SparseGPModel,
        fit_Gp,
        PREDICTION_CCR__MACHINE,
        Split_Matrix,
        convert_backs
    )

@Author : Clement Etienam
"""

# 🛠 Standard Library
import os
import pickle
import logging
# Removed unused imports

# 🔧 Third-party Libraries
import numpy as np
import numpy.matlib
import scipy.io as sio
import torch
import xgboost as xgb
from hydra.utils import to_absolute_path
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.utils.cholesky import psd_safe_cholesky


# 📦 Local Modules
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


def predict_machine11(a0, model):
    ynew = model.predict(xgb.DMatrix(a0))
    return ynew


def Make_correct(array):
    new_array = np.zeros(
        (array.shape[0], array.shape[1], array.shape[3], array.shape[4], array.shape[2])
    )
    for kk in range(array.shape[0]):
        perm_big = np.zeros(
            (array.shape[1], array.shape[3], array.shape[4], array.shape[2])
        )
        for mv in range(array.shape[1]):
            mv1 = np.zeros((array.shape[3], array.shape[4], array.shape[2]))
            for i in range(array.shape[2]):
                mv1[:, :, i] = array[kk, :, :, :, :][mv, :, :, :][i, :, :]
            perm_big[mv, :, :, :] = mv1
        new_array[kk, :, :, :, :] = perm_big
    return new_array


def fit_operation(tensor, target_min, target_max, tensor_min, tensor_max):
    rescaled_tensor = tensor / tensor_max
    return rescaled_tensor


def convert_back(rescaled_tensor, target_min, target_max, min_val, max_val):
    return rescaled_tensor * max_val


def convert_backsin(rescaled_tensor, max_val, N_pr):
    C = []
    Anow = rescaled_tensor[:, :N_pr]
    max_vall = max_val[:, 0]
    rescaled_tensorr = Anow / max_vall
    C.append(rescaled_tensorr)
    Anow = rescaled_tensor[:, N_pr : N_pr + 1]
    max_vall = max_val[:, 1]
    rescaled_tensorr = Anow / max_vall
    C.append(rescaled_tensorr)
    Anow = rescaled_tensor[:, N_pr + 1 : 2 * N_pr + 1]
    max_vall = max_val[:, 2]
    rescaled_tensorr = Anow / max_vall
    C.append(rescaled_tensorr)
    Anow = rescaled_tensor[:, 2 * N_pr + 1 : 3 * N_pr + 1]
    max_vall = max_val[:, 3]
    rescaled_tensorr = Anow / max_vall
    C.append(rescaled_tensorr)
    Anow = rescaled_tensor[:, 3 * N_pr + 1 : 4 * N_pr + 1]
    max_vall = max_val[:, 4]
    rescaled_tensorr = Anow / max_vall
    C.append(rescaled_tensorr)
    Anow = rescaled_tensor[:, 4 * N_pr + 1 : 4 * N_pr + 2]
    max_vall = max_val[:, 5]
    rescaled_tensorr = Anow / max_vall
    C.append(rescaled_tensorr)
    get_it2 = np.concatenate(C, axis=-1)
    return get_it2


def process_data(data):
    well_indices = {}
    for entry in data:
        if entry[0] not in well_indices:
            well_indices[entry[0]] = []
        well_indices[entry[0]].append(
            (int(entry[1]) - 1, int(entry[2]) - 1, int(entry[3]) - 1, int(entry[4]) - 1)
        )
    return well_indices


def get_dyna(steppi, well_indices, swatuse):
    mean_big_all = []
    for xx in range(steppi):
        mean_big = []  # Collects mean values for this particular timestep
        for idx, list1 in well_indices.items():  # Direct access to lists via .items()
            temp_perm_values = [
                swatuse[xx, i_idx, j_idx, k_idx]
                if k_idx == l_idx
                else swatuse[xx, i_idx, j_idx, k_idx : l_idx + 1]
                for i_idx, j_idx, k_idx, l_idx in list1
            ]
            mean_all = np.mean(temp_perm_values)
            mean_big.append(mean_all)
        mean_big_all.append(mean_big)
    outt2 = np.array(mean_big_all)
    return outt2


def endit(i, testt, training_master, oldfolder, pred_type, degg, big, experts, device):
    logger = setup_logging()
    logger.info("")
    logger.info("Starting prediction from machine %d" % (i + 1))
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
    logger.info("Finished Prediction from machine %d" % (i + 1))
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
        logger = setup_logging()
        logger.info("-- Predicting cluster: " + str(i + 1) + " | " + str(nclusters))
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


def Split_Matrix(matrix, sizee):
    x_split = np.split(matrix, sizee, axis=0)
    return x_split


def convert_backs(rescaled_tensor, max_val, N_pr, lenwels):
    C = []
    for k in range(lenwels):
        rescaled_tensorr = (
            rescaled_tensor[:, :, k * N_pr : (k + 1) * N_pr] * max_val[:, k]
        )
        C.append(rescaled_tensorr)
    get_it2 = np.concatenate(C, axis=-1)
    return get_it2


def Forward_model_ensemble(
    N,
    x_true,
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
    effectiveuse,
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
    producer_wells,
    unique_entries,
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
):
    #### ===================================================================== ####
    #                     RESERVOIR SIMULATOR WITH PHYSICSNEMO
    #
    #### ===================================================================== ####
    if "PRESSURE" in output_variables:
        modelP = models["pressure"]
        pressure = torch.zeros(N, steppi, nz, nx, ny).to(device, torch.float32)
    if "SWAT" in output_variables:
        output_keys_saturation = []
        modelS = models["saturation"]
        output_keys_saturation.append("water_sat")
        swater = torch.zeros(N, steppi, nz, nx, ny).to(device, torch.float32)
    if "SOIL" in output_variables:
        output_keys_oil = []
        modelO = models["oil"]
        output_keys_oil.append("oil_sat")
        soil = torch.zeros(N, steppi, nz, nx, ny).to(device, torch.float32)
    if "SGAS" in output_variables:
        output_keys_gas = []
        output_keys_gas.append("gas_sat")
        modelG = models["gas"]
        sgas = torch.zeros(N, steppi, nz, nx, ny).to(device, torch.float32)
    modelPe = models["peacemann"]

    # Initialize flow rate tensors
    Qg = torch.zeros(N, steppi, nz, nx, ny).to(device, torch.float32)
    Qw = torch.zeros(N, steppi, nz, nx, ny).to(device, torch.float32)
    Qo = torch.zeros(N, steppi, nz, nx, ny).to(device, torch.float32)
    Q = torch.zeros(N, steppi, nz, nx, ny).to(device, torch.float32)

    Qg1 = torch.zeros(N, steppi, nx, ny, nz).to(device, torch.float32)
    Qw1 = torch.zeros(N, steppi, nx, ny, nz).to(device, torch.float32)
    Qo1 = torch.zeros(N, steppi, nx, ny, nz).to(device, torch.float32)
    Q1 = torch.zeros(N, steppi, nx, ny, nz).to(device, torch.float32)

    # Convert flow rates to tensors
    agas = torch.from_numpy(agas).to(device, dtype=torch.float32)
    awater = torch.from_numpy(awater).to(device, dtype=torch.float32)
    aoil = torch.from_numpy(aoil).to(device, dtype=torch.float32)
    aqq = torch.from_numpy(aqq).to(device, dtype=torch.float32)

    # Initialize flow rates
    for i in range(N):
        Qg1[i, :, :, :, :] = agas
        Qw1[i, :, :, :, :] = awater
        Qo1[i, :, :, :, :] = aoil
        Q1[i, :, :, :, :] = aqq

    # Reshape flow rates
    for i in range(nz):
        Qw[:, :, i, :, :] = Qw1[:, :, :, :, i]
        Qg[:, :, i, :, :] = Qg1[:, :, :, :, i]
        Qo[:, :, i, :, :] = Qo1[:, :, :, :, i]
        Q[:, :, i, :, :] = Q1[:, :, :, :, i]

    del Qg1, Qw1, Qo1, Q1

    # Normalize flow rates
    if not torch.is_tensor(maxQ):
        maxQ = torch.tensor(maxQ, device=Q.device, dtype=Q.dtype)
    if not torch.is_tensor(maxQg):
        maxQg = torch.tensor(maxQg, device=Q.device, dtype=Q.dtype)
    if not torch.is_tensor(maxQw):
        maxQw = torch.tensor(maxQw, device=Q.device, dtype=Q.dtype)

    Q = Q / maxQ
    Qg = Qg / maxQg
    Qw = Qw / maxQw
    Q[Q == 0] = 1.0
    Qw[Qw == 0] = 1.0
    Qg[Qg == 0] = 1.0

    # Time setup
    Timeafter = Time
    Timebefore = np.zeros_like(Timeafter)
    Timebefore[1:] = Timeafter[:-1]
    dt = Timeafter - Timebefore
    dt = dt / maxT
    dt_full = torch.from_numpy(dt).to(device, dtype=torch.float32)

    # Sequential forwarding with chunking
    for i in range(N):
        perm_sample = x_true["perm"][i, :, :, :, :][None, :, :, :, :]
        poro_sample = x_true["poro"][i, :, :, :, :][None, :, :, :, :]
        Q_sample = Q[i, :, :, :, :][None, :, :, :, :]
        Qg_sample = Qg[i, :, :, :, :][None, :, :, :, :]
        Qw_sample = Qw[i, :, :, :, :][None, :, :, :, :]
        fault_sample = x_true["fault"][i, :, :, :, :][None, :, :, :, :]
        pbefore = x_true["pini"][i, :, :, :, :][None, :, :, :, :]
        swbefore = x_true["sini"][i, :, :, :, :][None, :, :, :, :]
        sgbefore = x_true["sgini"][i, :, :, :, :][None, :, :, :, :]
        sobefore = x_true["soini"][i, :, :, :, :][None, :, :, :, :]

        for t in range(steppi):
            dt_in = dt_full[0, t, 0, 0, 0] * torch.ones_like(perm_sample, device=device)
            temp = {
                "perm": perm_sample,
                "poro": poro_sample,
                "pini": pbefore,
                "sini": swbefore,
                "sgini": sgbefore,
                "soini": sobefore,
                "fault": fault_sample,
                "Q": Q_sample[:, t : t + 1, :, :, :],
                "Qg": Qg_sample[:, t : t + 1, :, :, :],
                "Qw": Qw_sample[:, t : t + 1, :, :, :],
                "dt": dt_in,
            }

            with torch.no_grad():
                tensors = [
                    value for value in temp.values() if isinstance(value, torch.Tensor)
                ]
                if not tensors:
                    raise ValueError("🚨 No valid input tensors found for the model!")

                input_tensor = torch.cat(tensors, dim=1)
                nz_current = input_tensor.shape[2]

                # Setup chunking - only if nz > 30
                if nz_current > cfg.custom.allowable_size:
                    chunk_size = max(1, int(nz_current * 0.1))
                    num_chunks = (nz_current + chunk_size - 1) // chunk_size
                    use_chunking = True
                else:
                    chunk_size = nz_current
                    num_chunks = 1
                    use_chunking = False

                # Initialize chunk outputs
                pafter_chunks, swafter_chunks, soafter_chunks, sgafter_chunks = (
                    [],
                    [],
                    [],
                    [],
                )

                # Process in chunks
                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, nz_current)
                    current_chunk_size = end_idx - start_idx

                    # Extract chunk
                    input_temp = input_tensor[:, :, start_idx:end_idx, :, :]

                    # Pad if needed (only when chunking)
                    if use_chunking:
                        pad_size = chunk_size - current_chunk_size
                        if pad_size > 0:
                            input_temp = torch.nn.functional.pad(
                                input_temp, (0, 0, 0, 0, 0, pad_size)
                            )
                    else:
                        pad_size = 0

                    # Model predictions
                    if "PRESSURE" in output_variables and modelP is not None:
                        pafter_chunk = modelP(input_temp)
                    if "SWAT" in output_variables and modelS is not None:
                        swafter_chunk = modelS(input_temp)
                    if "SOIL" in output_variables and modelO is not None:
                        soafter_chunk = modelO(input_temp)
                    if "SGAS" in output_variables and modelG is not None:
                        sgafter_chunk = modelG(input_temp)

                    # Remove padding if applied
                    if use_chunking and pad_size > 0:
                        if "PRESSURE" in output_variables and modelP is not None:
                            pafter_chunk = pafter_chunk[:, :, :current_chunk_size, :, :]
                        if "SWAT" in output_variables and modelS is not None:
                            swafter_chunk = swafter_chunk[
                                :, :, :current_chunk_size, :, :
                            ]
                        if "SOIL" in output_variables and modelO is not None:
                            soafter_chunk = soafter_chunk[
                                :, :, :current_chunk_size, :, :
                            ]
                        if "SGAS" in output_variables and modelG is not None:
                            sgafter_chunk = sgafter_chunk[
                                :, :, :current_chunk_size, :, :
                            ]

                    # Store chunk results
                    if "PRESSURE" in output_variables and modelP is not None:
                        pafter_chunks.append(pafter_chunk)
                    if "SWAT" in output_variables and modelS is not None:
                        swafter_chunks.append(swafter_chunk)
                    if "SOIL" in output_variables and modelO is not None:
                        soafter_chunks.append(soafter_chunk)
                    if "SGAS" in output_variables and modelG is not None:
                        sgafter_chunks.append(sgafter_chunk)

                # Combine chunk results
                if "PRESSURE" in output_variables and modelP is not None:
                    pafter = torch.cat(pafter_chunks, dim=2)
                if "SWAT" in output_variables and modelS is not None:
                    swafter = torch.cat(swafter_chunks, dim=2)
                if "SOIL" in output_variables and modelO is not None:
                    soafter = torch.cat(soafter_chunks, dim=2)
                if "SGAS" in output_variables and modelG is not None:
                    sgafter = torch.cat(sgafter_chunks, dim=2)

            # Update states and store results
            if "PRESSURE" in output_variables and modelP is not None:
                puse = pafter[0, 0, :, :, :]
                pressure[i, t, :, :, :] = puse
                pbefore = puse[None, None, :, :, :]
            if "SWAT" in output_variables and modelS is not None:
                swuse = swafter[0, 0, :, :, :]
                swater[i, t, :, :, :] = swuse
                swbefore = swuse[None, None, :, :, :]
            if "SOIL" in output_variables and modelO is not None:
                souse = soafter[0, 0, :, :, :]
                soil[i, t, :, :, :] = souse
                sobefore = souse[None, None, :, :, :]
            if "SGAS" in output_variables and modelG is not None:
                sguse = sgafter[0, 0, :, :, :]
                sgas[i, t, :, :, :] = sguse
                sgbefore = sguse[None, None, :, :, :]

            # Clean up
            for var in ["pafter", "swafter", "soafter", "sgafter"]:
                if var in locals():
                    del locals()[var]

            if (
                torch.cuda.is_available()
                and torch.cuda.memory_reserved()
                > 0.9 * torch.cuda.max_memory_allocated()
            ):
                torch.cuda.empty_cache()

            # Clean up after each sample
            del temp, input_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Convert and process final results
    if "PRESSURE" in output_variables:
        pressure = pressure.detach().cpu().numpy()
        pressure = Make_correct(pressure)
        pressure = convert_back(pressure, target_min, target_max, minP, maxP)
        pressure = np.clip(pressure, a_min=0, a_max=None)

    if "SWAT" in output_variables:
        swater = swater.detach().cpu().numpy()
        swater = Make_correct(swater)
        swater = np.clip(swater, 0, 1)

    if "SGAS" in output_variables:
        sgas = sgas.detach().cpu().numpy()
        sgas = Make_correct(sgas)
        sgas = np.clip(sgas, 0, 1)

    if "SOIL" in output_variables:
        soil = soil.detach().cpu().numpy()
        soil = Make_correct(soil)
        soil = np.clip(soil, 0, 1)

    perm = convert_back(
        x_true["perm"].detach().cpu().numpy(), target_min, target_max, minK, maxK
    )
    perm = Make_correct(perm)

    effective_abi = effective_abi[None, None, :, :, :]
    resultss = {}

    if "PRESSURE" in output_variables:
        resultss["PRESSURE"] = pressure * effective_abi
    if "SWAT" in output_variables:
        resultss["SWAT"] = swater * effective_abi
    if "SOIL" in output_variables:
        resultss["SOIL"] = soil * effective_abi
    if "SGAS" in output_variables:
        resultss["SGAS"] = sgas * effective_abi
    if Trainmoe == "FNO":
        innn = np.zeros((N, (N_pr * 4) + 2, steppi))
    else:
        innn = np.zeros((N, steppi, (N_pr * 4) + 2))
    well_indices = process_data(unique_entries)
    for i in range(N):
        permuse = perm[i, 0, :, :, :]
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
        presure_use = pressure[i, :, :, :, :]
        gas_use = sgas[i, :, :, :, :]
        water_use = swater[i, :, :, :, :]
        oil_use = soil[i, :, :, :, :]
        Time_usee = Time[i, :, :, :, :]
        a3 = get_dyna(steppi, well_indices, water_use)
        a2 = get_dyna(steppi, well_indices, gas_use)
        a5 = get_dyna(steppi, well_indices, oil_use)
        a1 = np.zeros((steppi, 1))
        a4 = np.zeros((steppi, 1))
        for k in range(steppi):
            uniep = presure_use[k, :, :, :]
            permuse = uniep
            a1[k, 0] = np.mean(permuse)
            unietime = Time_usee[k, :, :, :]
            permuse = unietime
            a4[k, 0] = permuse[0, 0, 0]
        inn1 = np.hstack((permxx, a1, a5, a2, a3, a4))

        if Trainmoe == "FNO":
            inn1 = fit_operation(inn1, target_min, target_max, min_inn_fcn, max_inn_fcn)
            innn[i, :, :] = inn1.T
        else:
            inn1 = convert_backsin(inn1, max_inn_fcn2, N_pr)
            innn[i, :, :] = inn1
    if Trainmoe == "FNO":
        innn = torch.from_numpy(innn).to(device, torch.float32)
        ouut_p = []
        for i in range(N):
            temp = innn[i, :, :][None, :, :]
            with torch.no_grad():
                ouut_p1 = modelPe(temp)
                ouut_p1 = ouut_p1.detach().cpu().numpy() * max_out_fcn
            ouut_p.append(ouut_p1)
            del temp
            torch.cuda.empty_cache()
        ouut_p = np.vstack(ouut_p)
        ouut_p = np.transpose(ouut_p, (0, 2, 1))
        ouut_p[ouut_p <= 0] = 0
    else:
        useq = lenwels * N_pr
        innn = np.vstack(innn)
        cluster_all = sio.loadmat(
            to_absolute_path("../ML_MACHINE/clustersizescost.mat")
        )["cluster"]
        cluster_all = np.reshape(cluster_all, (-1, 1), "F")

        ies = Parallel(n_jobs=num_cores, backend="loky")(
            delayed(PREDICTION_CCR__MACHINE)(
                ib,
                int(cluster_all[ib, :]),
                innn,
                innn.shape[1],
                to_absolute_path("../ML_MACHINE"),
                oldfolder,
                pred_type,
                degg,
                experts,
                device,
            )
            for ib in range(useq)
        )

        ouut_p = np.array(Split_Matrix(np.hstack(ies), N))
        ouut_p = convert_backs(ouut_p, max_out_fcn2, N_pr, lenwels)
        ouut_p[ouut_p <= 0] = 0
    sim = []
    for zz in range(ouut_p.shape[0]):
        lista = []
        for k in range(lenwels):
            rescaled_tensorr = ouut_p[:, :, k * N_pr : (k + 1) * N_pr]
            logger = setup_logging()
            logger.info(rescaled_tensorr.shape)
            lista.append(rescaled_tensorr)
        lista = np.hstack(lista)
        spit = np.reshape(lista, (-1, 1), "F")
        use = spit
        sim.append(use)
    sim = np.hstack(sim)

    resultss["sim"] = sim
    resultss["ouut_p"] = ouut_p
    return resultss
