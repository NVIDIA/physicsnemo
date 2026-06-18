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

import os
import pickle
import gzip
import datetime
import multiprocessing
from copy import copy
from pathlib import Path
from typing import Any
from sklearn.metrics import r2_score as r2_score

import numpy as np
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

from physicsnemo.distributed import DistributedManager
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

import hydra
from hydra.utils import to_absolute_path
from joblib import Parallel, delayed
import xgboost as xgb
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from kneed import KneeLocator

import logging
from gpytorch.utils.cholesky import psd_safe_cholesky
import warnings

from utils.logging_utils import setup_logging

logger = setup_logging("Mixture of Experts")


def load_training_data(logger: logging.Logger) -> dict[str, Any]:
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


def load_peaceman_data(logger: logging.Logger) -> tuple[np.ndarray, np.ndarray]:
    """Load Peaceman well model training data."""
    with gzip.open(to_absolute_path("../data/data_train_peaceman.pkl.gz"), "rb") as f:
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
    """Determine optimal KMeans cluster count using the elbow method.

    Uses KneeLocator on KMeans distortions for k in 1-9 and saves an elbow plot.

    Parameters
    ----------
    X : numpy.ndarray
        Input feature matrix of shape (n_samples, n_features).
    i : int
        Machine index used for labelling the saved plot file.
    training_master : str
        Directory path where the elbow plot image is saved.
    oldfolder : str
        Original working directory to restore after saving the plot.

    Returns
    -------
    int
        Optimal number of clusters identified at the elbow point.
    """
    distortions = []
    Kss = range(1, 10)

    for k in Kss:
        kmeanModel = KMeans(n_clusters=k)
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
    plt.title(f"Elbow Method showing the optimal n_clusters for machine {i}")
    os.chdir(training_master)
    plt.savefig(f"machine_{i + 1}.jpg")
    os.chdir(oldfolder)
    # plt.show()
    plt.close()
    plt.clf()
    return kuse


def getoptimumkcost(X, i, training_master, oldfolder):
    """Determine optimal MiniBatchKMeans cluster count using the elbow method.

    Uses KneeLocator on MiniBatchKMeans distortions for k in 1-9 and saves an elbow plot.

    Parameters
    ----------
    X : numpy.ndarray
        Input feature matrix of shape (n_samples, n_features).
    i : int
        Machine index used for labelling the saved plot file.
    training_master : str
        Directory path where the elbow plot image is saved.
    oldfolder : str
        Original working directory to restore after saving the plot.

    Returns
    -------
    int
        Optimal number of clusters identified at the elbow point.
    """
    distortions = []
    Kss = range(1, 10)

    for k in Kss:
        kmeanModel = MiniBatchKMeans(n_clusters=k)
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
    plt.title(f"Elbow Method showing the optimal n_clusters for machine {i}")
    os.chdir(training_master)
    plt.savefig(f"machine_Energy__{i + 1}.jpg")
    os.chdir(oldfolder)
    # plt.show()
    return kuse


def best_fit(X, Y):
    """Compute intercept and slope of OLS best-fit line, robust to degenerate inputs.

    Returns (0, 0) if inputs are empty, contain non-finite values, or X has zero variance.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    # Drop non-finite pairs
    mask = np.isfinite(X) & np.isfinite(Y)
    X, Y = X[mask], Y[mask]

    if X.size < 2:
        setup_logging().warning("best_fit: <2 valid points; returning (0, 0).")
        return 0.0, 0.0

    xbar = X.mean()
    ybar = Y.mean()
    denum = np.sum((X - xbar) ** 2)

    if denum == 0 or not np.isfinite(denum):
        setup_logging().warning("best_fit: zero variance in X; returning (ybar, 0).")
        return float(ybar), 0.0

    b = float(np.sum((X - xbar) * (Y - ybar)) / denum)
    a = float(ybar - b * xbar)

    setup_logging().info(f"best fit line: y = {a:.4f} + {b:.4f}x")
    return a, b


def Performance_plot_cost(CCR, Trued, stringg, training_master, oldfolder):
    """Plot scatter comparisons per machine and compute R²/L² performance metrics.

    Generates an adaptive grid (5 columns) comparing predicted vs true outputs
    per output column and saves the figure, returning overall and per-column
    coefficients of determination.

    Parameters
    ----------
    CCR : numpy.ndarray
        Predicted values array of shape (n_samples, n_machines).
    Trued : numpy.ndarray
        Ground-truth values array of shape (n_samples, n_machines).
    stringg : str
        Filename stem (without extension) for the saved plot image.
    training_master : str
        Directory where the plot image is saved.
    oldfolder : str
        Original working directory to restore after saving.

    Returns
    -------
    CoDoverall : numpy.ndarray
        Mean coefficient of determination across all machines, shape (1,).
    R2overall : numpy.ndarray
        Mean L² score across all machines, shape (1,).
    CoDview : numpy.ndarray
        Per-machine coefficient of determination, shape (1, n_machines).
    R2view : numpy.ndarray
        Per-machine L² score, shape (1, n_machines).
    """
    n_machines = Trued.shape[1]
    CoDview = np.zeros((1, n_machines))
    R2view = np.zeros((1, n_machines))

    # Adaptive grid: 5 columns, as many rows as needed
    NCOLS = 5
    nrows = int(np.ceil(n_machines / NCOLS))

    fig, axes = plt.subplots(
        nrows,
        NCOLS,
        figsize=(5.5 * NCOLS, 4.5 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    # Font sizes scaled up for readability
    TITLE_FS = 14
    LABEL_FS = 12
    TICK_FS = 10
    ANNOT_FS = 13

    for machine_idx in range(n_machines):
        setup_logging().info(
            " Compute L2 and R2 for the machine _" + str(machine_idx + 1)
        )

        predicted_output = np.reshape(CCR[:, machine_idx], (-1, 1))
        true_output = np.reshape(Trued[:, machine_idx], (-1, 1))
        r2 = r2_score(true_output, predicted_output)
        l2 = l2_error(true_output, predicted_output)

        CoDview[:, machine_idx] = r2
        R2view[:, machine_idx] = l2

        row = machine_idx // NCOLS
        col = machine_idx % NCOLS
        ax = axes[row, col]

        palette = copy(plt.get_cmap("inferno_r"))
        palette.set_under("white")
        palette.set_over("black")
        vmin = float(np.min(np.ravel(true_output)))
        vmax = float(np.max(np.ravel(true_output)))

        sc = ax.scatter(
            np.ravel(predicted_output),
            np.ravel(true_output),
            c=np.ravel(true_output),
            vmin=vmin,
            vmax=vmax,
            s=60,
            cmap=palette,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.ax.tick_params(labelsize=TICK_FS)

        ax.set_title(f"Well_{machine_idx + 1}", fontsize=TITLE_FS, fontweight="bold")
        ax.set_ylabel("Machine", fontsize=LABEL_FS)
        ax.set_xlabel("True data", fontsize=LABEL_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS)

        a, b = best_fit(
            np.ravel(predicted_output),
            np.ravel(true_output),
        )
        yfit = [a + b * xi for xi in np.ravel(predicted_output)]
        ax.plot(np.ravel(predicted_output), yfit, color="r", linewidth=2)

        ax.annotate(
            f"R²= {r2:.3f}",
            (0.05, 0.95),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=ANNOT_FS,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor="#888", alpha=0.9
            ),
        )

    # Hide unused subplots in the last row
    for extra in range(n_machines, nrows * NCOLS):
        row = extra // NCOLS
        col = extra % NCOLS
        axes[row, col].set_visible(False)

    CoDoverall = (np.sum(CoDview, axis=1)) / n_machines
    R2overall = (np.sum(R2view, axis=1)) / n_machines

    os.chdir(training_master)
    fig.savefig(f"{stringg}.jpg", bbox_inches="tight", dpi=150)
    plt.close(fig)
    os.chdir(oldfolder)
    return CoDoverall, R2overall, CoDview, R2view


def run_model(inn, ouut, i, training_master, oldfolder, nclus):
    """Train an XGBClassifier and save it to disk.

    Parameters
    ----------
    inn : numpy.ndarray
        Input feature matrix of shape (n_samples, n_features).
    ouut : numpy.ndarray
        Target cluster labels of shape (n_samples,).
    i : int
        Machine index used to name the saved model file.
    training_master : str
        Directory where the classifier binary is saved.
    oldfolder : str
        Original working directory to restore after saving.
    nclus : int
        Number of clusters (unused directly; retained for API consistency).

    Returns
    -------
    xgboost.XGBClassifier
        Fitted classifier instance.
    """
    # model=xgb.XGBClassifier(n_estimators=4000,
    #                         objective='multi:softmax',
    #                         num_class= nclus)
    model = xgb.XGBClassifier(n_estimators=20)
    model.fit(inn, ouut)
    filename = f"Classifier_{i}.bin"
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
    """Launch CCR training pipeline for a single output machine index.

    Extracts the i-th output column and full input matrix, then delegates to
    ``CCR_Machine`` to train the full cluster-based conditional regressor.

    Parameters
    ----------
    i : int
        Zero-based machine (output column) index to train.
    outpuut2 : numpy.ndarray
        Full output matrix of shape (n_samples, n_machines).
    inpuut2 : numpy.ndarray
        Full input feature matrix of shape (n_samples * gezz,) or similar.
    training_master : str
        Directory where model artefacts are saved.
    oldfolder : str
        Original working directory to restore after saving.
    degg : int
        Polynomial degree for polynomial expert models.
    use_elbow : int
        If 1, use elbow method to select cluster count; otherwise use 8 clusters.
    gezz : int
        Number of input features per sample row.
    device : torch.device
        Device used for GP model training.
    itery : int
        Number of training iterations for SparseGP models.
    experts : int
        Expert type selector (1=XGBoost, 2=RandomForest, 3=Polynomial, 4=GP).

    Returns
    -------
    int
        Optimal cluster count ``clust`` returned by ``CCR_Machine``.
    """
    setup_logging().info("")
    setup_logging().info(f"Starting CCR training machine {i + 1}")
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
    setup_logging().info("")
    setup_logging().info(f"Finished training machine {i + 1}")
    return bigs


def endit(i, testt, training_master, oldfolder, pred_type, degg, big, experts, device):
    """Run prediction from a trained CCR machine for a single output index.

    Delegates to ``PREDICTION_CCR__MACHINE`` using the saved model artefacts
    for machine ``i``.

    Parameters
    ----------
    i : int
        Zero-based machine (output column) index to predict.
    testt : numpy.ndarray
        Test input feature matrix of shape (n_test_samples, n_features).
    training_master : str
        Directory containing saved model artefacts.
    oldfolder : str
        Original working directory to restore after loading models.
    pred_type : int
        Prediction variant selector passed to ``PREDICTION_CCR__MACHINE``.
    degg : int
        Polynomial degree for polynomial expert models.
    big : int
        Number of clusters for this machine.
    experts : int
        Expert type selector (1=XGBoost, 2=RandomForest, 3=Polynomial, 4=GP).
    device : torch.device
        Device used for GP inference.

    Returns
    -------
    numpy.ndarray
        Predicted output values for the i-th machine, shape (n_test_samples,).
    """
    setup_logging().info("")
    setup_logging().info(f"Starting prediction from machine {i + 1}")

    numcols = len(testt[0])
    prediction_result = PREDICTION_CCR__MACHINE(
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
    setup_logging().info(f"Finished Prediction from machine {i + 1}")
    return prediction_result


def fit_machine(a0, b0):
    """Train an XGBRegressor on the provided feature and target arrays.

    Parameters
    ----------
    a0 : numpy.ndarray
        Input feature matrix of shape (n_samples, n_features).
    b0 : numpy.ndarray
        Target values of shape (n_samples,).

    Returns
    -------
    xgboost.XGBRegressor
        Fitted regressor instance.
    """
    model = xgb.XGBRegressor(
        n_estimators=20, objective="reg:squarederror", learning_rate=0.1
    )
    model.fit(a0, b0)
    return model


def predict_machine(a0, model):
    """Run XGBoost regression inference on a feature matrix.

    Parameters
    ----------
    a0 : numpy.ndarray
        Input feature matrix of shape (n_samples, n_features).
    model : xgboost.XGBRegressor
        Fitted XGBoost regressor produced by ``fit_machine``.

    Returns
    -------
    numpy.ndarray
        Predicted values of shape (n_samples,).
    """
    return model.predict(xgb.DMatrix(a0))


def l2_error(y_true, y_pred):
    """Compute relative L2 error between *y_pred* and *y_true*.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values.
    y_pred : np.ndarray
        Predicted values, same shape as *y_true*.

    Returns
    -------
    float
        Relative L2 error, i.e. ``||y_true - y_pred||_2 / (||y_true||_2 + 1e-12)``.
    """
    return np.sqrt(np.sum((y_true - y_pred) ** 2)) / (
        np.sqrt(np.sum(y_true**2)) + 1e-12
    )


def fit_machine3(a0, b0, deg):
    """Train a polynomial regression model of specified degree.

    Parameters
    ----------
    a0 : numpy.ndarray
        Input feature matrix of shape (n_samples, n_features).
    b0 : numpy.ndarray
        Target values of shape (n_samples,).
    deg : int
        Degree of the polynomial feature expansion.

    Returns
    -------
    model : sklearn.linear_model.LinearRegression
        Fitted linear regression model on polynomial features.
    polynomial_features : sklearn.preprocessing.PolynomialFeatures
        Fitted polynomial feature transformer.
    """
    polynomial_features = PolynomialFeatures(degree=deg, include_bias=False)
    x_poly = polynomial_features.fit_transform(a0)
    model = LinearRegression()
    model.fit(x_poly, b0)
    return model, polynomial_features


def predict_machine3(a0, deg, model, poly):
    """Run polynomial regression inference on a feature matrix.

    Parameters
    ----------
    a0 : numpy.ndarray
        Input feature matrix of shape (n_samples, n_features).
    deg : int
        Polynomial degree (unused here; retained for API consistency).
    model : sklearn.linear_model.LinearRegression
        Fitted linear model on polynomial features.
    poly : sklearn.preprocessing.PolynomialFeatures
        Fitted polynomial feature transformer.

    Returns
    -------
    numpy.ndarray
        Predicted values of shape (n_samples,).
    """
    return model.predict(poly.fit_transform(a0))


# def CCR_Machine(
# inpuutj,
# outputtj,
# ii,
# training_master,
# oldfolder,
# degg,
# use_elbow,
# device,
# itery,
# experts,
# ):
# """Train the full cluster-based conditional regressor for one output column.

# Normalises inputs/outputs, clusters the joint space with KMeans, fits a
# classifier and per-cluster expert regressors, then saves all artefacts.

# Parameters
# ----------
# inpuutj : numpy.ndarray
# Input feature matrix of shape (n_samples, n_features).
# outputtj : numpy.ndarray
# Output target column of shape (n_samples, 1).
# ii : int
# Zero-based machine index used for naming saved artefacts.
# training_master : str
# Directory where all model artefacts are saved.
# oldfolder : str
# Original working directory to restore after each save operation.
# degg : int
# Polynomial degree for polynomial expert models.
# use_elbow : int
# If 1, use elbow method to find optimal cluster count; otherwise use 8.
# device : torch.device
# Device used for SparseGP expert training.
# itery : int
# Number of training iterations for SparseGP models.
# experts : int
# Expert type (1=XGBoost, 2=RandomForest, 3=Polynomial, 4=SparseGP).

# Returns
# -------
# int
# Number of clusters used by this machine (``nclusters``).
# """
# X = inpuutj
# y = outputtj
# numruth = len(X[0])

# #y_traind = y
# scaler1a = MinMaxScaler(feature_range=(0, 1))
# (scaler1a.fit(X))
# X = scaler1a.transform(X)
# scaler2a = MinMaxScaler(feature_range=(0, 1))
# (scaler2a.fit(y))
# y = scaler2a.transform(y)
# yruth = y
# os.chdir(training_master)
# filenamex = f"clfx_{ii}.asv"
# filenamey = f"clfy_{ii}.asv"
# with open(filenamex, "wb") as fh:
# pickle.dump(scaler1a, fh)
# with open(filenamey, "wb") as fh:
# pickle.dump(scaler2a, fh)
# os.chdir(oldfolder)
# y_traind = numruth * 10 * y
# matrix = np.concatenate((X, y_traind), axis=1)
# # matrix=y.reshape(-1,1)
# if use_elbow == 1:
# k = getoptimumk(matrix, ii, training_master, oldfolder)
# nclusters = k
# else:
# nclusters = 8
# setup_logging().info("Optimal k is: %s", nclusters)
# # kmeans = MiniBatchKMeans(n_clusters=nclusters,max_iter=2000).fit(matrix)
# kmeans = KMeans(n_clusters=nclusters).fit(matrix)
# filename = f"Clustering_{ii}.asv"
# os.chdir(training_master)
# with open(filename, "wb") as fh:
# pickle.dump(kmeans, fh)
# os.chdir(oldfolder)
# dd = kmeans.labels_
# dd = dd.T
# dd = np.reshape(dd, (-1, 1))
# dd1 = dd
# # -------------------#---------------------------------#
# inputtrainclass = X
# outputtrainclass = np.reshape(dd, (-1, 1))
# if experts == 2:
# clf = RandomForestClassifier(n_estimators=20, random_state=42)
# clf.fit(inputtrainclass, outputtrainclass)
# filename1 = f"Classifier_{ii}.pkl"

# os.chdir(training_master)
# with open(filename1, "wb") as file1:
# pickle.dump(clf, file1)

# loaded_model = clf
# labelDA = loaded_model.predict(X)
# labelDA = np.reshape((labelDA), (-1, 1), "F")
# os.chdir(oldfolder)
# else:
# run_model(
# inputtrainclass, outputtrainclass, ii, training_master, oldfolder, nclusters
# )
# filename1 = f"Classifier_{ii}.bin"
# os.chdir(training_master)
# loaded_model = xgb.Booster({"nthread": 4})  # init model
# loaded_model.load_model(filename1)  # load data
# os.chdir(oldfolder)

# labelDA = loaded_model.predict(xgb.DMatrix(X))
# if nclusters == 2:
# labelDAX = 1 - labelDA
# labelDA = np.reshape(labelDA, (-1, 1))
# labelDAX = np.reshape(labelDAX, (-1, 1))
# labelDA = np.concatenate((labelDAX, labelDA), axis=1)
# else:
# labelDA = np.argmax(labelDA, axis=-1)
# labelDA = np.reshape((labelDA), (-1, 1), "F")

# # y_train = labelDA
# y_train = dd1

# X_train = X

# # -------------------Regression----------------#
# # print('Learn regression of the clusters with different labels from k-means ' )
# for i in range(nclusters):
# logger.info("-- Learning cluster: " + str(i + 1) + " | " + str(nclusters))
# label0 = (np.asarray(np.where(y_train == i))).T
# a0 = X_train[label0[:, 0], :]
# a0 = np.reshape(a0, (-1, numruth), "F")
# b0 = yruth[label0[:, 0], :]
# b0 = np.reshape(b0, (-1, 1), "F")
# if (a0.shape[0] != 0) and (b0.shape[0] != 0):
# if experts == 1:  # Polynomial regressor experts
# theta, con1 = fit_machine3(a0, b0, degg)
# filename = (
# "Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) + ".pkl"
# )
# filename2 = "polfeat_" + str(ii) + "_Cluster_" + str(i) + ".pkl"
# os.chdir(training_master)
# # dump(theta, filename)
# # dump(con1, filename2)
# with open(filename, "wb") as file:
# pickle.dump(theta, file)

# with open(filename2, "wb") as fileb:
# pickle.dump(con1, fileb)

# os.chdir(oldfolder)
# elif experts == 2:
# model_out = fit_Gp(a0, b0, device, itery)
# filename = (
# "Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) + ".pth"
# )
# os.chdir(training_master)
# torch.save(model_out.state_dict(), filename)
# os.chdir(oldfolder)
# else:  # XGBoost experts
# theta = fit_machine(a0, b0)
# filename = (
# "Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) + ".bin"
# )
# os.chdir(training_master)
# # sio.savemat(filename, {'model0':model0})
# theta.save_model(filename)
# os.chdir(oldfolder)
# return nclusters


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
    """Train the full cluster-based conditional regressor for one output column.

    Normalises inputs/outputs, clusters the joint space with KMeans, fits a
    classifier and per-cluster expert regressors, then saves all artefacts.

    Parameters
    ----------
    inpuutj : numpy.ndarray
        Input feature matrix of shape (n_samples, n_features).
    outputtj : numpy.ndarray
        Output target column of shape (n_samples, 1).
    ii : int
        Zero-based machine index used for naming saved artefacts.
    training_master : str
        Directory where all model artefacts are saved.
    oldfolder : str
        Original working directory to restore after each save operation.
    degg : int
        Polynomial degree for polynomial expert models.
    use_elbow : int
        If 1, use elbow method to find optimal cluster count; otherwise use 8.
    device : torch.device
        Device used for SparseGP expert training.
    itery : int
        Number of training iterations for SparseGP models.
    experts : int
        Expert type (1=XGBoost, 2=RandomForest, 3=Polynomial, 4=SparseGP).

    Returns
    -------
    int
        Number of clusters used by this machine (``nclusters``).
    """
    X = inpuutj
    y = outputtj
    numruth = X.shape[1]

    # ── Scale inputs and outputs ─────────────────────────────────────────────
    scaler1a = MinMaxScaler(feature_range=(0, 1))
    X = scaler1a.fit_transform(X)
    scaler2a = MinMaxScaler(feature_range=(0, 1))
    y = scaler2a.fit_transform(y)
    yruth = y

    os.chdir(training_master)
    with open(f"clfx_{ii}.asv", "wb") as fh:
        pickle.dump(scaler1a, fh)
    with open(f"clfy_{ii}.asv", "wb") as fh:
        pickle.dump(scaler2a, fh)
    os.chdir(oldfolder)

    # ── Cluster the joint (X, y) space ───────────────────────────────────────
    y_traind = numruth * 10 * y
    matrix = np.concatenate((X, y_traind), axis=1)

    if use_elbow == 1:
        nclusters = getoptimumk(matrix, ii, training_master, oldfolder)
    else:
        nclusters = 8

    # ── NEW: cap clusters so each has at least 200 samples ──────────────────
    min_samples_per_cluster = 200
    max_clusters = max(2, X.shape[0] // min_samples_per_cluster)
    if nclusters > max_clusters:
        logger.info(
            f"  Capping nclusters: {nclusters} → {max_clusters} "
            f"(need ≥{min_samples_per_cluster} samples per cluster)"
        )
        nclusters = max_clusters

    logger.info(f"  Optimal k for machine {ii}: {nclusters}")

    kmeans = KMeans(n_clusters=nclusters, n_init=10, random_state=42).fit(matrix)
    os.chdir(training_master)
    with open(f"Clustering_{ii}.asv", "wb") as fh:
        pickle.dump(kmeans, fh)
    os.chdir(oldfolder)

    dd = kmeans.labels_.reshape(-1, 1)
    dd1 = dd

    # ── Train classifier (cluster routing) ───────────────────────────────────
    inputtrainclass = X
    outputtrainclass = dd

    if experts == 2:
        clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        clf.fit(inputtrainclass, outputtrainclass.ravel())
        os.chdir(training_master)
        with open(f"Classifier_{ii}.pkl", "wb") as file1:
            pickle.dump(clf, file1)
        os.chdir(oldfolder)
    else:
        run_model(
            inputtrainclass, outputtrainclass, ii, training_master, oldfolder, nclusters
        )

    y_train = dd1
    X_train = X

    # ── Train per-cluster experts with regularization & early stopping ──────
    for i in range(nclusters):
        logger.info(f"  Cluster {i + 1}/{nclusters}")
        label0 = np.asarray(np.where(y_train == i)).T
        a0 = X_train[label0[:, 0], :].reshape(-1, numruth)
        b0 = yruth[label0[:, 0], :].reshape(-1, 1)

        if a0.shape[0] == 0 or b0.shape[0] == 0:
            logger.warning(f"  Cluster {i + 1} is empty; skipping expert.")
            continue

        if a0.shape[0] < 50:
            logger.warning(
                f"  Cluster {i + 1} has only {a0.shape[0]} samples; "
                f"this expert may overfit."
            )

        if experts == 1:
            # ── Polynomial: use Ridge for regularization, cap degree at 2 ──
            from sklearn.linear_model import Ridge

            poly_deg = min(degg, 2)  # never go above 2 for high-dim inputs
            poly = PolynomialFeatures(
                degree=poly_deg, include_bias=False, interaction_only=True
            )
            X_poly = poly.fit_transform(a0)

            # Strong Ridge regularization
            ridge = Ridge(alpha=10.0)
            ridge.fit(X_poly, b0.ravel())

            os.chdir(training_master)
            with open(f"Regressor_Machine_{ii}_Cluster_{i}.pkl", "wb") as f:
                pickle.dump(ridge, f)
            with open(f"polfeat_{ii}_Cluster_{i}.pkl", "wb") as f:
                pickle.dump(poly, f)
            os.chdir(oldfolder)

        elif experts == 2:
            model_out = fit_Gp(a0, b0, device, itery)
            os.chdir(training_master)
            torch.save(
                model_out.state_dict(), f"Regressor_Machine_{ii}_Cluster_{i}.pth"
            )
            os.chdir(oldfolder)

        else:
            # ── XGBoost: more trees, regularization, early stopping ────────
            # Hold out 15% of cluster samples for validation
            from sklearn.model_selection import train_test_split as _tts

            if a0.shape[0] >= 50:
                a_tr, a_va, b_tr, b_va = _tts(
                    a0, b0.ravel(), test_size=0.15, random_state=42
                )
                model = xgb.XGBRegressor(
                    n_estimators=500,
                    max_depth=4,  # shallower trees → less overfit
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,  # L1 regularization
                    reg_lambda=1.0,  # L2 regularization
                    early_stopping_rounds=30,
                    tree_method="hist",
                    objective="reg:squarederror",
                )
                model.fit(a_tr, b_tr, eval_set=[(a_va, b_va)], verbose=False)
            else:
                # Too few samples to hold out — just fit with strong regularization
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.05,
                    reg_alpha=0.5,
                    reg_lambda=2.0,
                    objective="reg:squarederror",
                )
                model.fit(a0, b0.ravel())

            os.chdir(training_master)
            model.save_model(f"Regressor_Machine_{ii}_Cluster_{i}.bin")
            os.chdir(oldfolder)

    return nclusters


class SparseGPModel(ExactGP):
    """Sparse G P Model."""

    def __init__(self, train_x, train_y, likelihood, inducing_points):
        """Initialise a sparse variational Gaussian Process model.

        Sets up constant mean, RBF covariance kernel, Cholesky variational
        distribution, and variational strategy with learnable inducing locations.

        Parameters
        ----------
        train_x : torch.Tensor
            Training input tensor of shape (n_samples, n_features).
        train_y : torch.Tensor
            Training target tensor of shape (n_samples,).
        likelihood : gpytorch.likelihoods.GaussianLikelihood
            Gaussian observation likelihood model.
        inducing_points : torch.Tensor
            Initial inducing point locations of shape (n_inducing, n_features).
        """
        super().__init__(train_x, train_y, likelihood)
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
        """Compute the prior GP distribution at input locations.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (n_samples, n_features).

        Returns
        -------
        gpytorch.distributions.MultivariateNormal
            Prior multivariate normal distribution at the given inputs.
        """
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
    """Train a SparseGPModel using VariationalELBO with an ExponentialLR scheduler.

    Selects inducing points via MiniBatchKMeans clustering of training inputs.

    Parameters
    ----------
    X : numpy.ndarray
        Training input feature matrix of shape (n_samples, n_features).
    y : numpy.ndarray
        Training target values of shape (n_samples,).
    device : torch.device
        Device on which to train the model (CPU or CUDA).
    itery : int
        Number of optimisation iterations.
    percentage : float, optional
        Percentage of training samples to use as inducing points (default 50.0).

    Returns
    -------
    SparseGPModel
        Trained sparse GP model in eval-ready state.
    """
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
    for _epoch in range(itery):
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, y)

        loss = loss.mean()  # Ensure loss is a scalar
        # print(f"Epoch {epoch + 1}/{itery}, Loss: {loss.item()}")  # ✅ Print loss
        loss.backward()  # Keep the graph intact
        optimizer.step()
        scheduler.step()

        del loss  # Free memory
        torch.cuda.empty_cache()

    return model


def fit_Gp1(X, y, device, itery, percentage=50.0):
    """Train a SparseGPModel using VariationalELBO with a lower learning rate variant.

    Alternative to ``fit_Gp`` using lr=2e-3 and no ``torch.no_grad`` context for
    KMeans-based inducing point selection.

    Parameters
    ----------
    X : numpy.ndarray
        Training input feature matrix of shape (n_samples, n_features).
    y : numpy.ndarray
        Training target values of shape (n_samples,).
    device : torch.device
        Device on which to train the model (CPU or CUDA).
    itery : int
        Number of optimisation iterations.
    percentage : float, optional
        Percentage of training samples to use as inducing points (default 50.0).

    Returns
    -------
    SparseGPModel
        Trained sparse GP model in eval-ready state.
    """
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
    for _epoch in range(itery):
        optimizer.zero_grad(set_to_none=True)
        output = model(X)
        loss = -mll(output, y)

        loss = loss.mean()  # Ensure loss is a scalar
        # print(f"Epoch {epoch + 1}/{itery}, Loss: {loss.item()}")  # ✅ Print loss
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
    """Run CCR inference over all clusters for a single output machine.

    Loads the saved scaler, classifier, and per-cluster expert models, then
    aggregates predictions weighted by cluster membership probabilities.

    Parameters
    ----------
    ii : int
        Zero-based machine (output column) index.
    nclusters : int
        Number of clusters used during training for this machine.
    inputtest : numpy.ndarray
        Test input feature matrix of shape (n_test_samples, n_features).
    numcols : int
        Number of features in the test input matrix.
    training_master : str
        Directory containing saved model artefacts.
    oldfolder : str
        Original working directory to restore after loading.
    pred_type : int
        Prediction variant (currently unused; reserved for future use).
    deg : int
        Polynomial degree for polynomial expert models.
    experts : int
        Expert type (1=XGBoost, 2=SparseGP, 3=Polynomial, 4=GP alternate).
    device : torch.device
        Device used for SparseGP inference.

    Returns
    -------
    numpy.ndarray
        Predicted output values for machine ii, shape (n_test_samples,).
    """
    filenamex = f"clfx_{ii}.asv"
    filenamey = f"clfy_{ii}.asv"

    os.chdir(training_master)
    if experts == 1 or experts == 3:
        filename1 = f"Classifier_{ii}.bin"
        loaded_model = xgb.Booster({"nthread": 4})  # init model
        loaded_model.load_model(filename1)  # load data
    if experts == 2:
        filename1 = f"Classifier_{ii}.pkl"
        with open(filename1, "rb") as file:
            loaded_model = pickle.load(file)
    with open(filenamex, "rb") as fh:
        clfx = pickle.load(fh)
    with open(filenamey, "rb") as fh:
        clfy = pickle.load(fh)
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
                    for batch_idx in range(0, a00.shape[0], batch_size):
                        batch = a00[batch_idx : batch_idx + batch_size]

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

    return clfy.inverse_transform(processanswer)


def plot_well_predictions(
    pred: np.ndarray,
    true: np.ndarray,
    Time_vals: np.ndarray,
    well_names: list,
    metric_name: str,
    ylabel: str,
    save_filename: str,
    trainingmaster: "str | Path",
    oldfolder: "str | Path",
    logger=None,
) -> tuple:
    """
    Universal plotting function for any well metric.

    Parameters
    ----------
    pred           : (B, T, N_wells)  predicted
    true           : (B, T, N_wells)  numerical
    Time_vals      : (T,)             time axis in days
    well_names     : list of well names
    metric_name    : 'WOPR' | 'WWPR' | 'WGPR'
    ylabel         : y-axis label
    save_filename  : filename for saved plot
    trainingmaster : subfolder name
    oldfolder      : base output folder
    logger         : logging.Logger | None

    Returns
    -------
    (r2_overall, l2_overall) : tuple of floats (fractions, not percentages)
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    time = np.asarray(Time_vals)
    B, _T, N_wells = pred.shape

    # Reflow wells into a grid (5 columns) per batch
    NCOLS = 5
    nrows_per_batch = int(np.ceil(N_wells / NCOLS))
    total_rows = B * nrows_per_batch

    fig, axes = plt.subplots(
        total_rows,
        NCOLS,
        figsize=(5 * NCOLS, 3.5 * total_rows),
        sharex=False,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1 and total_rows == 1 and NCOLS > 1:
        # already (1, NCOLS) — nothing to do
        pass

    fig.suptitle(
        f"{metric_name} — PhysicsNeMo vs Numerical", fontsize=16, fontweight="bold"
    )

    for b in range(B):
        for w in range(N_wells):
            row = b * nrows_per_batch + (w // NCOLS)
            col = w % NCOLS
            ax = axes[row, col]
            wn = well_names[w] if w < len(well_names) else f"Well {w + 1}"
            r2 = r2_score(true[b, :, w], pred[b, :, w])
            l2 = l2_error(true[b, :, w], pred[b, :, w])

            ax.plot(
                time,
                true[b, :, w],
                color="red",
                linewidth=2,
                linestyle="-",
                label="Numerical",
            )
            ax.plot(
                time,
                pred[b, :, w],
                color="blue",
                linewidth=2,
                linestyle="--",
                label="PhysicsNeMo",
            )

            ax.set_title(f"{wn}\nR²={r2 * 100:.2f}%  L²={l2 * 100:.2f}%", fontsize=10)
            ax.set_xlabel("Time (years)", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="x", rotation=45, labelsize=7)

    # Hide unused subplots in the last row of each batch
    for b in range(B):
        for extra in range(N_wells, nrows_per_batch * NCOLS):
            row = b * nrows_per_batch + (extra // NCOLS)
            col = extra % NCOLS
            axes[row, col].set_visible(False)

    r2_overall = r2_score(true.ravel(), pred.ravel())
    l2_overall = l2_error(true.ravel(), pred.ravel())

    fig.text(
        0.5,
        -0.01,
        f"Overall  R² = {r2_overall * 100:.2f}%   L² = {l2_overall * 100:.2f}%",
        ha="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"),
    )

    save_dir = Path(oldfolder) / trainingmaster
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / save_filename
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if logger:
        logger.info(f"  {metric_name} plot saved → {save_path}")
        logger.info(f"  Overall R² = {r2_overall * 100:.2f}%")
        logger.info(f"  Overall L² = {l2_overall * 100:.2f}%")

    return r2_overall, l2_overall


def restore_3d(arr, dim2, dim3):
    """Restore a 2D vstacked array back to (N, dim2, dim3)."""
    N = arr.shape[0] // dim2
    return arr.reshape(N, dim2, dim3)


@hydra.main(version_base="1.2", config_path="conf", config_name="DECK_CONFIG")
def main(cfg: DictConfig) -> None:
    """Main function for MOE CCR reservoir simulation forward modelling."""

    logger = setup_logging()
    oldfolder = os.getcwd()
    cores = multiprocessing.cpu_count()
    start_time = datetime.datetime.now()

    logger.info(f"This computer has {cores} cores — all used in parallel")
    logger.info(f"Starting execution at: {start_time}")

    # ── distributed setup ────────────────────────────────────────────────────
    DistributedManager.initialize()
    dist = DistributedManager()
    if "RANK" not in os.environ:
        os.environ["RANK"] = str(dist.rank)
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(dist.rank % torch.cuda.device_count())

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        device_id = dist.rank % gpu_count
        torch.cuda.set_device(device_id)
        logger.info(
            f"Process {dist.rank} using GPU {device_id}: {torch.cuda.get_device_name(device_id)}"
        )
    else:
        logger.info(f"Process {dist.rank} using CPU")
    device = dist.device

    # ── load conversions ─────────────────────────────────────────────────────
    logger.info("=" * 68)
    logger.info("  LOAD INPUT DATA")
    logger.info("=" * 68)
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
    N_pr = int(mat["N_pr"])
    lenwels = mat["lenwels"]

    target_min, target_max = 0.01, 1

    for k, v in [
        ("minK", minK),
        ("maxK", maxK),
        ("minT", minT),
        ("maxT", maxT),
        ("minP", minP),
        ("maxP", maxP),
        ("minQw", minQw),
        ("maxQw", maxQw),
        ("minQg", minQg),
        ("maxQg", maxQg),
        ("minQ", minQ),
        ("maxQ", maxQ),
        ("min_inn_fcn", min_inn_fcn),
        ("max_inn_fcn", max_inn_fcn),
        ("min_out_fcn", min_out_fcn),
        ("max_out_fcn", max_out_fcn),
        ("target_min", target_min),
        ("target_max", target_max),
    ]:
        logger.info(f"  {k} = {v}")

    # ── load Peacemann training data ─────────────────────────────────────────
    with gzip.open(to_absolute_path("../data/data_train_peaceman.pkl.gz"), "rb") as f:
        mat = pickle.load(f)
    X_data2 = mat
    data2 = X_data2

    well_measurements = cfg.custom.well_measurements
    lenwels = len(well_measurements)

    X = np.vstack(data2["X"])
    Y = np.vstack(data2["Y"])
    Y = Y[:, : lenwels * N_pr]
    Y[Y <= 0] = 0
    degg = 3
    gezz = X.shape[1]

    Machinetrue = "../ML_MACHINE"
    if not os.path.exists(to_absolute_path("../ML_MACHINE")):
        os.makedirs(to_absolute_path("../ML_MACHINE"))

    np.random.seed(5)
    trainingmaster = Path(oldfolder) / Machinetrue

    inpuutx, outpuutx = X, Y
    os.chdir(oldfolder)
    inpuutx = inpuutx.astype("float32")
    outpuutx = outpuutx.astype("float32")
    pred_type = 1

    # ── references ───────────────────────────────────────────────────────────
    logger.info("=" * 68)
    logger.info("  CCR REFERENCES")
    logger.info("=" * 68)
    logger.info(
        "(1) Bernholdt, Cianciosa, Green, Park, Law, Etienam. "
        "Cluster, Classify, Regress: a general method for learning discontinuous "
        "functions. Foundations of Data Science, 1(2639-8001-2019-4-491):491, 2019."
    )
    logger.info(
        "(2) Etienam, Law, Wade. Ultra-fast Deep Mixtures of Gaussian Process "
        "Experts. arXiv:2006.13309, 2020."
    )

    # ── train/test split ─────────────────────────────────────────────────────
    outpuutx2 = outpuutx
    inpuutx2 = inpuutx
    inpuut2, X_test2, outpuut2, y_test2 = train_test_split(
        inpuutx2,
        outpuutx2,
        test_size=0.01,
    )
    inputsz = range(Y.shape[1])

    # ── CCR config ───────────────────────────────────────────────────────────
    use_elbow = int(cfg.custom.Number_of_experts)
    experts = int(cfg.custom.Type_of_experts)
    iteryy = int(cfg.custom.iteration_experts)

    expert_labels = {1: "Polynomial regressor", 2: "SparseGP", 3: "XGBoost"}
    logger.info(f"Expert type: {expert_labels.get(experts, 'XGBoost')}")

    sio.savemat(to_absolute_path("../data/exper.mat"), {"expert": experts})

    num_cores = multiprocessing.cpu_count()
    njobs = max(1, (num_cores // 4) - 1)

    # ── train CCR ────────────────────────────────────────────────────────────
    logger.info("=" * 68)
    logger.info("  LEARN FORWARD MODEL WITH CCR")
    logger.info("=" * 68)
    os.chdir(trainingmaster)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bigs = Parallel(n_jobs=njobs, backend="loky")(
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
    sio.savemat("clustersizescost.mat", {"cluster": big})
    logger.info(f"  Y columns: {Y.shape[1]}   cluster matrix shape: {big.shape}")
    os.chdir(oldfolder)

    # ── predict (single inference run) ───────────────────────────────────────
    logger.info("=" * 68)
    logger.info("  PREDICT")
    logger.info("=" * 68)
    os.chdir(trainingmaster)
    cluster_all = sio.loadmat("clustersizescost.mat")["cluster"]
    cluster_all = np.reshape(cluster_all, (-1, 1), "F")
    os.chdir(oldfolder)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clemes = Parallel(n_jobs=njobs, backend="loky")(
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

    # ── per-block rescale (matches A exactly) ────────────────────────────────
    C = []
    for k in range(lenwels):
        Anow = outputpredenergy[:, k * N_pr : (k + 1) * N_pr]
        max_vall = max_out_fcn  # [:, k]
        rescaled_tensor = Anow * max_vall
        C.append(rescaled_tensor)
    outputpredenergy = np.concatenate(C, 1)
    outputpredenergy[outputpredenergy <= 0] = 0

    C = []
    for k in range(lenwels):
        Anow = y_test2[:, k * N_pr : (k + 1) * N_pr]
        max_vall = max_out_fcn  # [:, k]
        rescaled_tensor = Anow * max_vall
        C.append(rescaled_tensor)
    y_test2 = np.concatenate(C, 1)

    # ── per-phase Performance_plot_cost + plot_well_predictions ──────────────
    # Slice the rescaled output: WOPR=[0:N_pr], WWPR=[N_pr:2N_pr], WGPR=[2N_pr:3N_pr]
    logger.info("=" * 68)
    logger.info("  PER-PHASE PERFORMANCE — WOPR / WWPR / WGPR")
    logger.info("=" * 68)

    T = 10  # steppi — number of recorded time-steps
    Time_vals = np.arange(1, T + 1, dtype=np.float32)
    well_names_plot = [f"Well_{i + 1}" for i in range(N_pr)]

    phase_specs = [
        ("WOPR", 0, "Oil rate  (bbl/day)", "WOPR_PhysicsNeMo_vs_Numerical.png"),
        ("WWPR", 1, "Water rate  (bbl/day)", "WWPR_PhysicsNeMo_vs_Numerical.png"),
        ("WGPR", 2, "Gas rate  (bbl/day)", "WGPR_PhysicsNeMo_vs_Numerical.png"),
    ]

    phase_results = {}
    for tag, blk, ylabel, fname in phase_specs:
        if blk >= lenwels:
            continue
        cols = slice(blk * N_pr, (blk + 1) * N_pr)
        pred_blk = outputpredenergy[:, cols]
        true_blk = y_test2[:, cols]
        # 1. Performance_plot_cost — scatter + R²/L² per machine
        CoD, L2, _, _ = Performance_plot_cost(
            pred_blk,
            true_blk,
            f"Machine_{tag}_perform",
            trainingmaster,
            oldfolder,
        )
        CoD = float(np.asarray(CoD).flat[0])
        L2 = float(np.asarray(L2).flat[0])
        logger.info(f"  R² of fit ({tag}) = {CoD:.4f}")
        logger.info(f"  L² of fit ({tag}) = {L2:.4f}")

        # 2. plot_well_predictions — time-series per well
        r2_p, l2_p = plot_well_predictions(
            pred=restore_3d(pred_blk, T, N_pr)[0:1],
            true=restore_3d(true_blk, T, N_pr)[0:1],
            Time_vals=Time_vals,
            well_names=well_names_plot,
            metric_name=tag,
            ylabel=ylabel,
            save_filename=fname,
            trainingmaster=trainingmaster,
            oldfolder=oldfolder,
            logger=logger,
        )
        phase_results[tag] = (CoD, L2, r2_p, l2_p)

    # ── final summary ────────────────────────────────────────────────────────
    logger.info("=" * 68)
    logger.info("  FINAL SUMMARY")
    logger.info("=" * 68)
    for tag, (CoD, L2, r2_p, l2_p) in phase_results.items():
        logger.info(
            f"  {tag}   Performance R²={CoD:.4f}  L²={L2:.4f}   "
            f"|   Plot R²={r2_p:.4f}  L²={l2_p:.4f}"
        )
    logger.info("=" * 68)
    logger.info("  PROGRAM EXECUTED")
    logger.info("=" * 68)


if __name__ == "__main__":
    main()
