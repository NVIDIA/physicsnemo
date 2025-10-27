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
 NVIDIA PHYSICSNEMO MOE CCR (Mixture of Experts - Cluster-based
 Conditional Regression) for Reservoir Simulation Forward Modelling
=====================================================================
@Author : Clement Etienam

This module implements a Mixture of Experts (MOE) approach using Cluster-based
Conditional Regression (CCR) for reservoir simulation forward modelling. It provides
a machine learning framework for predicting reservoir behavior using multiple
specialized expert models trained on clustered data.

Key Features:
- Cluster-based data partitioning using K-means
- Multiple expert models (Polynomial, SparseGP, XGBoost)
- Ensemble prediction with weighted averaging
- Comprehensive model evaluation and visualization

Usage:
    python Moe_ccr.py --config-path=conf --config-name=DECK_CONFIG

Inputs:
    - Configuration file with model parameters
    - Training data from reservoir simulations
    - Test data for model evaluation

Outputs:
    - Trained expert models
    - Prediction results with evaluation metrics
    - Visualization plots for model performance
"""

# -------------------- 📌 FUTURE IMPORTS -------------------------
# from __future__ import print_function

# 🛠 Standard Library
import os
import pickle
import gzip
import datetime
import multiprocessing
from copy import copy
from pathlib import Path
from typing import Tuple, Dict, Any

# 🔧 Third-party Libraries
import numpy as np
import numpy.linalg as LA
from omegaconf import DictConfig
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import rankdata, norm
from scipy import interpolate
import scipy.io as sio

# 🔥 PhyNeMo & ML Libraries
from physicsnemo.distributed import DistributedManager
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

# 📦 Hydra & Configuration
import hydra
from hydra.utils import to_absolute_path
from joblib import Parallel, delayed
import xgboost as xgb
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from kneed import KneeLocator

# 🎯 Logging
import logging
from gpytorch.utils.cholesky import psd_safe_cholesky
import warnings


def setup_logging() -> logging.Logger:
    """Configure and return the main logger."""
    logger = logging.getLogger("Mixture of Experts")
    logger.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger
    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger


logger = setup_logging()


def load_training_data(logger: logging.Logger) -> Dict[str, Any]:
    """Load training data and configuration parameters."""
    logger.info(
        "-------------------LOAD INPUT DATA-------------------------------------"
    )
    mat = sio.loadmat(to_absolute_path("../data/conversions.mat"))

    # Extract configuration parameters
    config_params = {
        "minK": mat["minK"],
        "maxK": mat["maxK"],
        "minT": mat["minT"],
        "maxT": mat["maxT"],
        "minP": mat["minP"],
        "maxP": mat["maxP"],
        "minQw": mat["minQW"],
        "maxQw": mat["maxQW"],
        "minQg": mat["minQg"],
        "maxQg": mat["maxQg"],
        "minQ": mat["minQ"],
        "maxQ": mat["maxQ"],
        "min_inn_fcn": mat["min_inn_fcn"],
        "max_inn_fcn": mat["max_inn_fcn"],
        "min_out_fcn": mat["min_out_fcn"],
        "max_out_fcn": mat["max_out_fcn"],
        "steppi": int(mat["steppi"]),
        "steppi_indices": mat["steppi_indices"],
        "N_ens": int(mat["N_ens"]),
        "N_pr": int(mat["N_pr"]),
        "lenwels": mat["lenwels"],
    }

    # Log configuration values
    logger.info("These are the values:")
    for key, value in config_params.items():
        logger.info(f"{key} value is: {value}")

    return config_params


def load_peaceman_data(logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray]:
    """Load Peaceman well model training data."""
    with gzip.open(
        to_absolute_path("../data/data_train_peaceman.pkl.gz"), "rb"
    ) as f:
        mat = pickle.load(f)

    X = np.vstack(mat["X2"])
    Y = np.vstack(mat["Y2"])

    return X, Y


warnings.filterwarnings("ignore")


def interpolatebetween(
    xtrain: np.ndarray, cdftrain: np.ndarray, xnew: np.ndarray
) -> np.ndarray:
    """
    Interpolate between training data points using linear interpolation.

    Args:
        xtrain: Training input data points
        cdftrain: Training cumulative distribution function values
        xnew: New input points for interpolation

    Returns:
        Interpolated values for the new input points
    """
    numrows1 = len(xnew)
    numcols = len(xnew[0])
    norm_cdftest2 = np.zeros((numrows1, numcols))
    for i in range(numcols):
        f = interpolate.interp1d((xtrain[:, i]), cdftrain[:, i], kind="linear")
        cdftest = f(xnew[:, i])
        norm_cdftest2[:, i] = np.ravel(cdftest)
    return norm_cdftest2


def gaussianizeit(input1: np.ndarray) -> np.ndarray:
    """
    Transform input data to Gaussian distribution using rank-based transformation.

    Args:
        input1: Input data array to be transformed

    Returns:
        Gaussianized data array
    """
    numrows1 = len(input1)
    numcols = len(input1[0])
    # Vectorized implementation
    newbig = np.zeros((numrows1, numcols))
    for i in range(numcols):
        input11 = input1[:, i]
        # Vectorized rank-based transformation
        ranks = rankdata(input11)
        normalized_ranks = ranks / (len(input11) + 1)
        newX = norm.ppf(normalized_ranks)
        newbig[:, i] = newX
    return newbig


def getoptimumk(X, i, training_master, oldfolder):
    distortions = []
    Kss = range(1, 10)

    for k in Kss:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(
            sum(np.min(cdist(X, kmeanModel.cluster_centers_, "euclidean"), axis=1))
            / X.shape[0]
        )

    myarray = np.array(distortions)

    knn = KneeLocator(
        Kss, myarray, curve="convex", direction="decreasing", interp_method="interp1d"
    )
    kuse = knn.knee

    # Plot the elbow
    plt.figure(figsize=(10, 10))
    plt.plot(Kss, distortions, "bx-")
    plt.xlabel("cluster size")
    plt.ylabel("Distortion")
    plt.title("Elbow Method showing the optimal n_clusters for machine %d" % (i))
    os.chdir(training_master)
    plt.savefig("machine_%d.jpg" % (i + 1))
    os.chdir(oldfolder)
    # plt.show()
    plt.close()
    plt.clf()
    return kuse


def getoptimumkcost(X, i, training_master, oldfolder):
    distortions = []
    Kss = range(1, 10)

    for k in Kss:
        kmeanModel = MiniBatchKMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(
            sum(np.min(cdist(X, kmeanModel.cluster_centers_, "euclidean"), axis=1))
            / X.shape[0]
        )

    myarray = np.array(distortions)

    knn = KneeLocator(
        Kss, myarray, curve="convex", direction="decreasing", interp_method="interp1d"
    )
    kuse = knn.knee

    # Plot the elbow
    plt.figure(figsize=(10, 10))
    plt.plot(Kss, distortions, "bx-")
    plt.xlabel("cluster size")
    plt.ylabel("Distortion")
    plt.title("Elbow Method showing the optimal n_clusters for machine %d" % (i))
    os.chdir(training_master)
    plt.savefig("machine_Energy__%d.jpg" % (i + 1))
    os.chdir(oldfolder)
    plt.show()
    return kuse


def best_fit(X, Y):
    xbar = sum(X) / len(X)
    ybar = sum(Y) / len(Y)
    n = len(X)  # or len(Y)
    numer = sum([xi * yi for xi, yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2
    b = numer / denum
    a = ybar - b * xbar

    setup_logging().info("best fit line:\ny = {:.2f} + {:.2f}x".format(a, b))
    return a, b


def Performance_plot_cost(CCR, Trued, stringg, training_master, oldfolder):
    CoDview = np.zeros((1, Trued.shape[1]))
    R2view = np.zeros((1, Trued.shape[1]))

    plt.figure(figsize=(40, 40))

    for machine_idx in range(Trued.shape[1]):
        setup_logging().info(
            " Compute L2 and R2 for the machine _" + str(machine_idx + 1)
        )

        predicted_output = np.reshape(CCR[:, machine_idx], (-1, 1))
        true_output = np.reshape(Trued[:, machine_idx], (-1, 1))
        # numrowstest = len(true_output)
        true_output = np.reshape(true_output, (-1, 1))
        Lerrorsparse = (
            LA.norm(true_output - predicted_output) / LA.norm(true_output)
        ) ** 0.5
        L_22 = 1 - (Lerrorsparse**2)
        # Coefficient of determination - vectorized
        outputreq = true_output - np.mean(true_output)
        CoDspa = 1 - (LA.norm(true_output - predicted_output) / LA.norm(outputreq))
        CoD2 = 1 - (1 - CoDspa) ** 2
        setup_logging().info("")

        CoDview[:, machine_idx] = CoD2
        R2view[:, machine_idx] = L_22

        machine_number = machine_idx + 1
        jk = machine_number
        plt.subplot(9, 9, jk)
        palette = copy(plt.get_cmap("inferno_r"))
        palette.set_under("white")  # 1.0 represents not transparent
        palette.set_over("black")  # 1.0 represents not transparent
        vmin = min(np.ravel(true_output))
        vmax = max(np.ravel(true_output))
        sc = plt.scatter(
            np.ravel(predicted_output),
            np.ravel(true_output),
            c=np.ravel(true_output),
            vmin=vmin,
            vmax=vmax,
            s=35,
            cmap=palette,
        )
        plt.colorbar(sc)
        plt.title("Energy_" + str(machine_idx + 1), fontsize=9)
        plt.ylabel("Machine", fontsize=9)
        plt.xlabel("True data", fontsize=9)
        a, b = best_fit(
            np.ravel(predicted_output),
            np.ravel(true_output),
        )
        yfit = [a + b * xi for xi in np.ravel(predicted_output)]
        plt.plot(np.ravel(predicted_output), yfit, color="r")
        plt.annotate(
            "R2= %.3f" % CoD2,
            (0.8, 0.2),
            xycoords="axes fraction",
            ha="center",
            va="center",
            size=9,
        )

    CoDoverall = (np.sum(CoDview, axis=1)) / Trued.shape[1]
    R2overall = (np.sum(R2view, axis=1)) / Trued.shape[1]
    os.chdir(training_master)
    plt.savefig("%s.jpg" % stringg)
    os.chdir(oldfolder)
    return CoDoverall, R2overall, CoDview, R2view


def run_model(inn, ouut, i, training_master, oldfolder, nclus):
    # model=xgb.XGBClassifier(n_estimators=4000,
    #                         objective='multi:softmax',
    #                         num_class= nclus)
    model = xgb.XGBClassifier(n_estimators=4000)
    model.fit(inn, ouut)
    filename = "Classifier_%d.bin" % i
    os.chdir(training_master)
    model.save_model(filename)
    os.chdir(oldfolder)
    return model


def startit(
    i,
    outpuut2,
    inpuut2,
    training_master,
    oldfolder,
    degg,
    use_elbow,
    gezz,
    device,
    itery,
    experts,
):
    setup_logging().info("")
    setup_logging().info("Starting CCR training machine %d" % (i + 1))
    useeo = outpuut2[:, i]
    useeo = np.reshape(useeo, (-1, 1), "F")

    usein = inpuut2
    usein = np.reshape(usein, (-1, gezz), "F")  # 9+4

    clust = CCR_Machine(
        usein,
        useeo,
        i,
        training_master,
        oldfolder,
        degg,
        use_elbow,
        device,
        itery,
        experts,
    )

    bigs = clust
    return bigs
    setup_logging().info("")
    setup_logging().info("Finished training machine %d" % (i + 1))


def endit(i, testt, training_master, oldfolder, pred_type, degg, big, experts, device):
    setup_logging().info("")
    setup_logging().info("Starting prediction from machine %d" % (i + 1))

    numcols = len(testt[0])
    clemzz = PREDICTION_CCR__MACHINE(
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

    setup_logging().info("")
    setup_logging().info("Finished Prediction from machine %d" % (i + 1))
    return clemzz


def fit_machine(a0, b0):
    model = xgb.XGBRegressor(
        n_estimators=2000, objective="reg:squarederror", learning_rate=0.1
    )
    model.fit(a0, b0)
    return model


def predict_machine(a0, model):
    ynew = model.predict(xgb.DMatrix(a0))

    return ynew


def fit_machine3(a0, b0, deg):
    polynomial_features = PolynomialFeatures(degree=deg, include_bias=False)
    x_poly = polynomial_features.fit_transform(a0)
    model = LinearRegression()
    model.fit(x_poly, b0)
    return model, polynomial_features


def predict_machine3(a0, deg, model, poly):
    predicted = model.predict(poly.fit_transform(a0))
    return predicted


def CCR_Machine(
    inpuutj,
    outputtj,
    ii,
    training_master,
    oldfolder,
    degg,
    use_elbow,
    device,
    itery,
    experts,
):
    X = inpuutj
    y = outputtj
    numruth = len(X[0])

    y_traind = y
    scaler1a = MinMaxScaler(feature_range=(0, 1))
    (scaler1a.fit(X))
    X = scaler1a.transform(X)
    scaler2a = MinMaxScaler(feature_range=(0, 1))
    (scaler2a.fit(y))
    y = scaler2a.transform(y)
    yruth = y
    os.chdir(training_master)
    filenamex = "clfx_%d.asv" % ii
    filenamey = "clfy_%d.asv" % ii
    pickle.dump(scaler1a, open(filenamex, "wb"))
    pickle.dump(scaler2a, open(filenamey, "wb"))
    os.chdir(oldfolder)
    y_traind = numruth * 10 * y
    matrix = np.concatenate((X, y_traind), axis=1)
    # matrix=y.reshape(-1,1)
    if use_elbow == 1:
        k = getoptimumk(matrix, ii, training_master, oldfolder)
        nclusters = k
    else:
        nclusters = 4
    setup_logging().info("Optimal k is: %s", nclusters)
    # kmeans = MiniBatchKMeans(n_clusters=nclusters,max_iter=2000).fit(matrix)
    kmeans = KMeans(n_clusters=nclusters).fit(matrix)
    filename = "Clustering_%d.asv" % ii
    os.chdir(training_master)
    pickle.dump(kmeans, open(filename, "wb"))
    os.chdir(oldfolder)
    dd = kmeans.labels_
    dd = dd.T
    dd = np.reshape(dd, (-1, 1))
    dd1 = dd
    # -------------------#---------------------------------#
    inputtrainclass = X
    outputtrainclass = np.reshape(dd, (-1, 1))
    if experts == 2:
        clf = RandomForestClassifier(n_estimators=500, random_state=42)
        clf.fit(inputtrainclass, outputtrainclass)
        filename1 = "Classifier_%d.pkl" % ii

        os.chdir(training_master)
        with open(filename1, "wb") as file1:
            pickle.dump(clf, file1)

        with open(filename1, "rb") as file2:
            loaded_model = pickle.load(file2)
        labelDA = loaded_model.predict(X)
        labelDA = np.reshape((labelDA), (-1, 1), "F")
        os.chdir(oldfolder)
    else:
        run_model(
            inputtrainclass, outputtrainclass, ii, training_master, oldfolder, nclusters
        )
        filename1 = "Classifier_%d.bin" % ii
        os.chdir(training_master)
        loaded_model = xgb.Booster({"nthread": 4})  # init model
        loaded_model.load_model(filename1)  # load data
        os.chdir(oldfolder)

        labelDA = loaded_model.predict(xgb.DMatrix(X))
        if nclusters == 2:
            labelDAX = 1 - labelDA
            labelDA = np.reshape(labelDA, (-1, 1))
            labelDAX = np.reshape(labelDAX, (-1, 1))
            labelDA = np.concatenate((labelDAX, labelDA), axis=1)
        else:
            labelDA = np.argmax(labelDA, axis=-1)
        labelDA = np.reshape((labelDA), (-1, 1), "F")

    # y_train = labelDA
    y_train = dd1

    X_train = X

    # -------------------Regression----------------#
    # print('Learn regression of the clusters with different labels from k-means ' )
    for i in range(nclusters):
        logger.info("-- Learning cluster: " + str(i + 1) + " | " + str(nclusters))
        label0 = (np.asarray(np.where(y_train == i))).T
        a0 = X_train[label0[:, 0], :]
        a0 = np.reshape(a0, (-1, numruth), "F")
        b0 = yruth[label0[:, 0], :]
        b0 = np.reshape(b0, (-1, 1), "F")
        if (a0.shape[0] != 0) and (b0.shape[0] != 0):
            if experts == 1:  # Polynomial regressor experts
                theta, con1 = fit_machine3(a0, b0, degg)
                filename = (
                    "Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) + ".pkl"
                )
                filename2 = "polfeat_" + str(ii) + "_Cluster_" + str(i) + ".pkl"
                os.chdir(training_master)
                # dump(theta, filename)
                # dump(con1, filename2)
                with open(filename, "wb") as file:
                    pickle.dump(theta, file)

                with open(filename2, "wb") as fileb:
                    pickle.dump(con1, fileb)

                os.chdir(oldfolder)
            elif experts == 2:
                model_out = fit_Gp(a0, b0, device, itery)
                filename = (
                    "Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) + ".pth"
                )
                os.chdir(training_master)
                torch.save(model_out.state_dict(), filename)
                os.chdir(oldfolder)
            else:  # XGBoost experts
                theta = fit_machine(a0, b0)
                filename = (
                    "Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) + ".bin"
                )
                os.chdir(training_master)
                # sio.savemat(filename, {'model0':model0})
                theta.save_model(filename)
                os.chdir(oldfolder)
    return nclusters


class SparseGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, inducing_points):
        super(SparseGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        # ✅ Use inducing points for sparse variational GP
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

        # ✅ Use the safe Cholesky function
        jitter = 1e-5
        identity = torch.eye(
            train_train_covar.size(-1), device=train_train_covar.device
        )
        chol = self._cholesky(train_train_covar + jitter * identity)

        return torch.cholesky_solve(train_labels_offset, chol).squeeze(-1)


def fit_Gp(X, y, device, itery, percentage=50.0):
    X = torch.tensor(X, dtype=torch.float32, device=device, requires_grad=True)
    y = torch.tensor(y, dtype=torch.float32, device=device)

    # ✅ Clone X but DO NOT detach it permanently
    X_clone = X.clone()

    # ✅ Temporarily disable autograd inside no_grad()
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

    # inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)

    # ✅ Initialize model and likelihood
    likelihood = GaussianLikelihood().to(device)
    model = SparseGPModel(X, y, likelihood, inducing_points).to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=2e-2, betas=(0.9, 0.999), weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99998708)

    mll = VariationalELBO(likelihood, model, num_data=y.size(0))

    model.train()
    likelihood.train()
    # ✅ Training loop
    for epoch in range(itery):
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, y)

        loss = loss.mean()  # Ensure loss is a scalar
        # print(f"Epoch {epoch + 1}/{itery}, Loss: {loss.item()}")  # ✅ Print loss
        loss.backward(retain_graph=True)  # Keep the graph intact
        optimizer.step()
        scheduler.step()

        del loss  # Free memory
        torch.cuda.empty_cache()

    return model


def fit_Gp1(X, y, device, itery, percentage=50.0):
    X = torch.tensor(X, dtype=torch.float32, device=device, requires_grad=True)
    y = torch.tensor(y, dtype=torch.float32, device=device)

    # ✅ Clone `X` before passing to KMeans (keeps computational graph intact)
    X_clone = X.clone()
    num_inducing_points = max(
        int(X.shape[0] * (percentage / 100)), 1
    )  # Ensure at least one inducing point
    kmeans = MiniBatchKMeans(
        n_clusters=num_inducing_points, random_state=42, n_init="auto"
    )
    kmeans.fit(X_clone.cpu().numpy())  # Uses clone, keeps autograd
    inducing_points = torch.tensor(
        kmeans.cluster_centers_, dtype=torch.float32, device=device
    )  # Move centroids to GPU

    # ✅ Initialize model and likelihood
    likelihood = GaussianLikelihood().to(device)
    # model = SparseGPModel(X, y, likelihood, inducing_points).to(device)
    model = SparseGPModel(X, y, likelihood, inducing_points).to(device)

    model.train()
    likelihood.train()

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99998708)

    mll = VariationalELBO(likelihood, model, num_data=y.size(0))

    # ✅ Training loop
    for epoch in range(itery):
        optimizer.zero_grad(set_to_none=True)
        output = model(X)
        loss = -mll(output, y)

        loss = loss.mean()  # Ensure loss is a scalar
        print(f"Epoch {epoch + 1}/{itery}, Loss: {loss.item()}")  # ✅ Print loss
        loss.backward()  # Keep the graph intact
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
    processanswer = np.zeros((numrowstest, 1))
    # numcols=13
    labelDA = np.reshape(labelDA, (-1, 1), "F")
    for i in range(nclusters):
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
                processanswer[labelDA0[:, 0], :] = np.reshape(
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

            # model.load_state_dict(torch.load(filename2,strict=False))
            # checkpoint = torch.load(filename2, map_location=device)  # ✅ Load checkpoint
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

                # Concatenate all predictions
                processanswer[labelDA0[:, 0], :] = np.vstack(predictions)

            del model
            torch.cuda.empty_cache()  # Free unused GPU memory

        else:  # XGBoost experts
            loaded_modelr = xgb.Booster({"nthread": 4})  # init model
            filename2 = "Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) + ".bin"

            os.chdir(training_master)
            loaded_modelr.load_model(filename2)  # load data

            os.chdir(oldfolder)

            labelDA0 = (np.asarray(np.where(labelDA == i))).T
            #    ##----------------------##------------------------##
            a00 = inputtest[labelDA0[:, 0], :]
            a00 = np.reshape(a00, (-1, numcols), "F")
            if a00.shape[0] != 0:
                processanswer[labelDA0[:, 0], :] = np.reshape(
                    predict_machine(a00, loaded_modelr), (-1, 1)
                )

    processanswer = clfy.inverse_transform(processanswer)
    return processanswer


@hydra.main(version_base="1.2", config_path="conf", config_name="DECK_CONFIG")
def main(cfg: DictConfig) -> None:
    """Main function for MOE CCR reservoir simulation forward modelling."""
    # Setup logging
    logger = setup_logging()

    # Initialize environment
    oldfolder = os.getcwd()
    cores = multiprocessing.cpu_count()
    logger.info(
        f"This computer has {cores} cores, which will all be utilised in parallel"
    )
    start_time = datetime.datetime.now()
    logger.info(f"Starting execution at: {start_time}")

    # Initialize distributed training
    DistributedManager.initialize()
    dist = DistributedManager()
    if "RANK" not in os.environ:
        os.environ["RANK"] = str(dist.rank)
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(dist.rank % torch.cuda.device_count())

    # Assign GPU or CPU
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
    logger.info(
        "-------------------LOAD INPUT DATA-------------------------------------"
    )
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
    min_inn_fcn = mat["min_inn_fcn2"]
    max_inn_fcn = mat["max_inn_fcn2"]
    min_out_fcn = mat["min_out_fcn2"]
    max_out_fcn = mat["max_out_fcn2"]
    N_pr = int(mat["N_pr"])
    lenwels = mat["lenwels"]

    target_min = 0.01
    target_max = 1
    logger.info("These are the values:")
    logger.info(f"minK value is: {minK}")
    logger.info(f"maxK value is: {maxK}")
    logger.info(f"minT value is: {minT}")
    logger.info(f"maxT value is: {maxT}")
    logger.info(f"minP value is: {minP}")
    logger.info(f"maxP value is: {maxP}")
    logger.info(f"minQw value is: {minQw}")
    logger.info(f"maxQw value is: {maxQw}")
    logger.info(f"minQg value is: {minQg}")
    logger.info(f"maxQg value is: {maxQg}")
    logger.info(f"minQ value is: {minQ}")
    logger.info(f"maxQ value is: {maxQ}")
    logger.info(f"min_inn_fcn value is: {min_inn_fcn}")
    logger.info(f"max_inn_fcn value is: {max_inn_fcn}")
    logger.info(f"min_out_fcn value is: {min_out_fcn}")
    logger.info(f"max_out_fcn value is: {max_out_fcn}")
    logger.info(f"target_min value is: {target_min}")
    logger.info(f"target_max value is: {target_max}")

    with gzip.open(
        to_absolute_path("../data/data_train_peaceman.pkl.gz"), "rb"
    ) as f:
        mat = pickle.load(f)
    X_data2 = mat

    data2 = X_data2
    well_measurements = cfg.custom.well_measurements
    lenwels = int(len(well_measurements))

    X = np.vstack(data2["X2"])
    Y = np.vstack(data2["Y2"])
    Y = Y[:, : lenwels * N_pr]
    Y[Y <= 0] = 0
    # datafind = Path(oldfolder) / "Data"
    # degg=int(input('select degree polynomial: 4-10: '))
    degg = 3
    gezz = X.shape[1]
    Machinetrue = "../ML_MACHINE"
    if not os.path.exists(to_absolute_path("../ML_MACHINE")):
        os.makedirs(to_absolute_path("../ML_MACHINE"))
    else:
        pass

    np.random.seed(5)
    trainingmaster = Path(oldfolder) / Machinetrue

    inpuutx, outpuutx = X, Y
    os.chdir(oldfolder)
    inpuutx = inpuutx.astype("float32")
    outpuutx = outpuutx.astype("float32")

    # intee_raw = inpuutx
    # pred_type=int(input('Choose: 1=Hard Prediction, 2= Soft Prediction: '))
    pred_type = 1
    # pred_type=2
    logger.info("-------------MODEL FITTING FOR PEACEMANN WELL MODEL-----------")
    logger.info("Using CCR for peacemann model fitting")
    logger.info("")
    logger.info("|-----------------------------------------------------------------|")
    logger.info("References for CCR include: ")
    print(
        " (1): David E. Bernholdt, Mark R. Cianciosa, David L. Green, Jin M. Park,\n\
    Kody J. H. Law, and Clement Etienam. Cluster, classify, regress:A general\n\
    method for learning discontinuous functions.Foundations of Data Science,\n\
    1(2639-8001-2019-4-491):491, 2019.\n"
    )
    logger.info("|-----------------------------------------------------------------|")
    logger.info("")
    print(
        "(2): Clement Etienam, Kody Law, Sara Wade. Ultra-fast Deep Mixtures of\n\
    Gaussian Process Experts. arXiv preprint arXiv:2006.13309, 2020.\n"
    )
    logger.info(
        "|----------------------------------------------------------------------|"
    )
    outpuutx2 = outpuutx
    inpuutx2 = inpuutx
    # iniguess = inpuutx2
    # inpuutx2=(scaler2a.transform(inpuutx2))
    inpuut2, X_test2, outpuut2, y_test2 = train_test_split(
        inpuutx2, outpuutx2, test_size=0.01
    )  # train_size=2000)

    logger.info(
        "--------------------- Learn the Forward model with CCR----------------"
    )
    inputsz = range(Y.shape[1])

    num_cores = multiprocessing.cpu_count()
    njobs = (num_cores // 4) - 1
    num_cores = njobs

    use_elbow = int(cfg.custom.Number_of_experts)
    experts = int(cfg.custom.Type_of_experts)
    iteryy = int(cfg.custom.iteration_experts)

    if experts == 1:
        logger.info("Use Polynomial regressor Experts")

    elif experts == 2:
        logger.info("Use SparseGP Experts")
    else:
        logger.info("Use XGboost experts")

    choice = {"expert": experts}
    sio.savemat(to_absolute_path("../data/exper.mat"), choice)

    os.chdir(trainingmaster)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bigs = Parallel(n_jobs=num_cores, backend="loky")(
            delayed(startit)(
                ib,
                outpuut2,
                inpuut2,
                trainingmaster,
                oldfolder,
                degg,
                use_elbow,
                gezz,
                device,
                iteryy,
                experts,
            )
            for ib in inputsz
        )

    big = np.vstack(bigs)

    os.chdir(trainingmaster)
    print(Y.shape[1])
    print(big.shape)
    cluster = {"cluster": big}
    sio.savemat("clustersizescost.mat", cluster)
    os.chdir(oldfolder)
    logger.info(" -------------------------Predict For Energy Machine-----------------")
    os.chdir(trainingmaster)
    cluster_all = sio.loadmat("clustersizescost.mat")["cluster"]
    cluster_all = np.reshape(cluster_all, (-1, 1), "F")
    os.chdir(oldfolder)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clemes = Parallel(n_jobs=num_cores, backend="loky")(
            delayed(PREDICTION_CCR__MACHINE)(
                ib,
                int(cluster_all[ib, :]),
                X_test2,
                X.shape[1],
                trainingmaster,
                oldfolder,
                pred_type,
                degg,
                experts,
                device,
            )
            for ib in inputsz
        )
    outputpredenergy = np.hstack(clemes)

    logger.info(" ")
    print(max_out_fcn)

    C = []
    for k in range(lenwels):
        Anow = outputpredenergy[:, k * N_pr : (k + 1) * N_pr]
        max_vall = max_out_fcn[:, k]
        rescaled_tensor = Anow * max_vall
        C.append(rescaled_tensor)
    outputpredenergy = np.concatenate(C, 1)
    outputpredenergy[outputpredenergy <= 0] = 0

    C = []
    for k in range(lenwels):
        Anow = y_test2[:, k * N_pr : (k + 1) * N_pr]
        max_vall = max_out_fcn[:, k]
        rescaled_tensor = Anow * max_vall
        C.append(rescaled_tensor)
    y_test2 = np.concatenate(C, 1)

    CoDoveralle, L_2overalle, CoDviewe, L_2viewe = Performance_plot_cost(
        outputpredenergy, y_test2, "Machine_Energy_perform", trainingmaster, oldfolder
    )
    logger.info(f"R2 of fit using the Energy machine for model is : {CoDoveralle}")
    logger.info(f"L2 of fit using the Energy machine for model is : {L_2overalle}")
    logger.info(
        "-------------------PROGRAM EXECUTED-------------------------------------"
    )


if __name__ == "__main__":
    main()
