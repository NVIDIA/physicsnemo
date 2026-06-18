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
CCR (Composite Reservoir Characterisation) ML prediction utilities shared across all sub-modules.
@Author : Clement Etienam
"""

# ---- Standard Library ----
import os
import pickle

# ---- Third-party Libraries ----
import numpy as np
import torch
import xgboost as xgb
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.utils.cholesky import psd_safe_cholesky
from sklearn.cluster import MiniBatchKMeans
from joblib import Parallel, delayed
import scipy.io as sio
from hydra.utils import to_absolute_path
import torch.distributed as torchdist

# ---- Logging ----
from utils.logging_utils import setup_logging

logger = setup_logging("CCR Utils")


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

        # Use inducing points for sparse variational GP
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
        """Fix Cholesky decomposition issue with jitter."""
        jitter = 1e-5  # Small positive value
        eye = torch.eye(A.size(-1), device=A.device)
        return psd_safe_cholesky(A + jitter * eye)  # Safe Cholesky

    def _mean_cache(self):
        """Uses safe Cholesky for covariance matrix."""
        train_train_covar = self.train_train_covar.evaluate_kernel()
        train_labels_offset = self.train_labels - self.train_mean

        # Use the safe Cholesky function
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

    # Clone X but DO NOT detach it permanently
    X_clone = X.clone()

    # Temporarily disable autograd inside no_grad()
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

    # Initialize model and likelihood
    likelihood = GaussianLikelihood().to(device)
    model = SparseGPModel(X, y, likelihood, inducing_points).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=2e-2, betas=(0.9, 0.999), weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99998708)

    mll = VariationalELBO(likelihood, model, num_data=y.size(0))

    model.train()
    likelihood.train()
    # Training loop
    for _epoch in range(itery):
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, y)

        loss = loss.mean()  # Ensure loss is a scalar
        loss.backward()  # Keep the graph intact
        optimizer.step()
        scheduler.step()

        del loss  # Free memory
        torch.cuda.empty_cache()

    return model


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
    logger.info("")
    logger.info(f"Starting prediction from machine {i + 1}")

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

    logger.info("")
    logger.info(f"Finished Prediction from machine {i + 1}")
    return prediction_result


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
    labelDA = np.reshape(labelDA, (-1, 1), "F")
    for i in range(nclusters):
        # logger.info("-- Predicting cluster: " + str(i + 1) + " | " + str(nclusters))
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
            a00 = inputtest[labelDA0[:, 0], :]
            a00 = np.reshape(a00, (-1, numcols), "F")
            if a00.shape[0] != 0:
                processanswer[labelDA0[:, 0], :] = np.reshape(
                    predict_machine3(a00, deg, model0, poly0), (-1, 1)
                )

        elif experts == 2:
            labelDA0 = (np.asarray(np.where(labelDA == i))).T
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

            train_x = torch.zeros(a00.shape[0], input_dim).to(device)
            train_y = torch.zeros(a00.shape[0], 1).to(device)
            train_y = train_y.squeeze(-1)
            likelihood = GaussianLikelihood().to(device)

            inducing_points = torch.zeros(num_inducing_points, input_dim).to(device)
            model = SparseGPModel(train_x, train_y, likelihood, inducing_points).to(
                device
            )

            model.load_state_dict(checkpoint, strict=False)  # Pass strict=False here

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

            # del model
            # torch.cuda.empty_cache()  # Free unused GPU memory

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
                processanswer[labelDA0[:, 0], :] = np.reshape(
                    predict_machine(a00, loaded_modelr), (-1, 1)
                )

    return clfy.inverse_transform(processanswer)


def predict_machine11(a0, model):
    """Generate predictions from an XGBoost model.

    Parameters
    ----------
    a0 : np.ndarray
        2-D input feature array for prediction.
    model : xgb.Booster
        Trained XGBoost booster model.

    Returns
    -------
    np.ndarray
        Predicted values from the XGBoost model.
    """
    return model.predict(xgb.DMatrix(a0))


def convert_backs(rescaled_tensor, max_val, N_pr, lenwels):
    """Denormalize a tensor by multiplying each well-group slice by its max value.

    Parameters
    ----------
    rescaled_tensor : np.ndarray
        Normalized 3-D array, shape ``(N_ens, n_timesteps, lenwels * N_pr)``.
    max_val : np.ndarray
        2-D array of maximum values per well group, shape ``(N_ens, lenwels)``.
    N_pr : int
        Number of producer wells per well-group block.
    lenwels : int
        Number of well-measurement types (groups).

    Returns
    -------
    np.ndarray
        Denormalized array with same shape as `rescaled_tensor`.
    """
    C = []
    for k in range(lenwels):
        rescaled_tensorr = (
            rescaled_tensor[:, :, k * N_pr : (k + 1) * N_pr] * max_val[:, k]
        )
        C.append(rescaled_tensorr)
    return np.concatenate(C, axis=-1)


def convert_backsin(rescaled_tensor, max_val, N_pr):
    """Normalize a flat input tensor by dividing each segment by its column max.

    Parameters
    ----------
    rescaled_tensor : np.ndarray
        2-D input array containing concatenated well segments to normalize.
    max_val : np.ndarray
        2-D array of maximum values, shape ``(N_ens, n_segments)``.
    N_pr : int
        Number of producer wells; determines segment boundary sizes.

    Returns
    -------
    np.ndarray
        Concatenated normalized array with the same number of rows as input.
    """
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
    return np.concatenate(C, axis=-1)


def run_transolver(x, model):
    """Run a Transolver model over a batched 6-D input tensor in chunks.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape ``(B, steppi, nz, nx, ny, C)`` where C is channels.
    model : torch.nn.Module
        Transolver model accepting 5-D input ``(B*t, nz, nx, ny, C)`` and
        returning ``(B*t, nz, nx, ny, out_dim)``.

    Returns
    -------
    torch.Tensor
        Predicted output tensor of shape ``(B, steppi, nz, nx, ny)``.
    """
    B, steppi, nz, nx, ny, _C = x.shape

    batch_chunk_size = 2
    time_chunk_size = 3

    # Output: (B, steppi, nz, nx, ny)
    output = torch.zeros(B, steppi, nz, nx, ny, device=x.device)

    (
        (B + batch_chunk_size - 1)
        // batch_chunk_size
        * (steppi + time_chunk_size - 1)
        // time_chunk_size
    )
    current_chunk = 0

    for batch_start in range(0, B, batch_chunk_size):
        batch_end = min(batch_start + batch_chunk_size, B)

        for time_start in range(0, steppi, time_chunk_size):
            time_end = min(time_start + time_chunk_size, steppi)

            current_chunk += 1

            # input_chunk: (b, t, nz, nx, ny, C)
            input_chunk = x[batch_start:batch_end, time_start:time_end]
            b, t, nz_c, nx_c, ny_c, C_c = input_chunk.shape

            # Merge batch and time for TransolverModel:
            # x_5d: (b * t, nz, nx, ny, C)
            x_5d = input_chunk.view(b * t, nz_c, nx_c, ny_c, C_c)

            # model(...) returns (b * t, nz, nx, ny, out_dim)
            pred_5d = model(x_5d)

            # Remove out_dim=1 and restore (b, t, nz, nx, ny)
            pred_5d = pred_5d.squeeze(-1)
            pred_3d = pred_5d.view(b, t, nz_c, nx_c, ny_c)

            output[batch_start:batch_end, time_start:time_end] = pred_3d

            # Clean up
            del input_chunk, x_5d, pred_5d, pred_3d
            if current_chunk % 3 == 0:
                torch.cuda.empty_cache()

    return output


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
    active_cells_ensemble,
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
    active_mask_3d,
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
    """Run the full forward model ensemble producing reservoir states and well rates.

    Parameters
    ----------
    N : int
        Number of ensemble members.
    x_true : dict
        Dict of input tensors keyed by property name (``'perm'``, ``'poro'``, etc.).
    steppi : int
        Number of simulation time steps.
    min_inn_fcn : float
        Minimum normalization value for the Peacemann input features.
    max_inn_fcn : float
        Maximum normalization value for the Peacemann input features.
    target_min : float
        Target minimum for denormalization of model outputs.
    target_max : float
        Target maximum for denormalization of model outputs.
    minK : float
        Minimum permeability value for denormalization.
    maxK : float
        Maximum permeability value for denormalization.
    minT : float
        Minimum time value for normalization.
    maxT : float
        Maximum time value for normalization.
    minP : float
        Minimum pressure value for denormalization.
    maxP : float
        Maximum pressure value for denormalization.
    models : dict
        Dict of loaded ``torch.nn.Module`` objects keyed by output variable name.
    device : torch.device
        Compute device for all tensor operations.
    min_out_fcn : float
        Minimum normalization scalar for the FNO Peacemann output.
    max_out_fcn : float
        Maximum normalization scalar for the FNO Peacemann output.
    Time : np.ndarray
        Time array of shape ``(N, steppi, nz, nx, ny)``.
    active_cells_ensemble : np.ndarray
        Boolean or integer mask of active grid cells (unused in body).
    Trainmoe : str
        Model type selector; ``'FNO'`` uses the FNO Peacemann path.
    num_cores : int
        Number of parallel CPU cores for mixture-of-experts prediction.
    pred_type : int
        Prediction type passed to ``PREDICTION_CCR__MACHINE``.
    oldfolder : str
        Original working directory to restore after directory changes.
    degg : int
        Polynomial degree for polynomial-regression experts.
    experts : int
        Expert type: 1 = polynomial, 2 = sparse GP, 3 = XGBoost.
    min_out_fcn2 : np.ndarray
        Per-column minimum normalization values for MOE Peacemann output.
    max_out_fcn2 : np.ndarray
        Per-column maximum normalization values for MOE Peacemann output.
    min_inn_fcn2 : np.ndarray
        Per-column minimum normalization values for MOE Peacemann input.
    max_inn_fcn2 : np.ndarray
        Per-column maximum normalization values for MOE Peacemann input.
    producer_wells : list
        List of producer well descriptors used to build well index mapping.
    unique_entries : list of tuple
        Unique well location entries ``(name, i, j, k_start, k_end)``.
    output_variables : list of str
        Names of output variables to compute (e.g. ``['PRESSURE', 'SWAT']``).
    well_measurements : list of str
        Well measurement types (e.g. ``['WOPR', 'WWPR', 'WGPR']``).
    cfg : object
        Hydra/OmegaConf configuration object with ``cfg.custom.model_type``.
    N_pr : int
        Number of producer wells.
    lenwels : int
        Number of well-measurement types.
    active_mask_3d : np.ndarray
        3-D boolean mask of active cells, shape ``(nz, nx, ny)``.
    awater : np.ndarray
        Water injection rate field, shape ``(steppi, nx, ny, nz)``.
    agas : np.ndarray
        Gas injection rate field, shape ``(steppi, nx, ny, nz)``.
    aoil : np.ndarray
        Oil production rate field, shape ``(steppi, nx, ny, nz)``.
    aqq : np.ndarray
        Total injection rate field, shape ``(steppi, nx, ny, nz)``.
    nx : int
        Grid dimension in the x-direction.
    ny : int
        Grid dimension in the y-direction.
    nz : int
        Grid dimension in the z-direction.
    minQ : float
        Minimum total injection rate for normalization.
    maxQ : float
        Maximum total injection rate for normalization.
    minQw : float
        Minimum water injection rate for normalization.
    maxQw : float
        Maximum water injection rate for normalization.
    minQg : float
        Minimum gas injection rate for normalization.
    maxQg : float
        Maximum gas injection rate for normalization.

    Returns
    -------
    dict
        Results dictionary containing predicted fields (e.g. ``'PRESSURE'``,
        ``'SWAT'``, ``'SOIL'``, ``'SGAS'``), ``'sim'`` (flattened well rates),
        and ``'ouut_p'`` (raw Peacemann model output).
    """

    # Import helper functions locally to avoid circular imports
    from utils.array_utils import (
        Make_correct,
        convert_back,
        fit_operation,
        Split_Matrix,
    )
    from compare.sequential.misc_forward_enact import (
        process_data,
        get_dyna,
    )

    # ── Distributed setup ─────────────────────────────────────────────────────
    is_dist = torchdist.is_available() and torchdist.is_initialized()
    if is_dist:
        world_size = torchdist.get_world_size()
        rank = torchdist.get_rank()
    else:
        world_size = 1
        rank = 0

    # Strided sharding: rank r owns ensemble indices [r, r+W, r+2W, ...]
    indices_local = list(range(rank, N, world_size))
    N_local = len(indices_local)

    # ── Model setup ───────────────────────────────────────────────────────────
    if "PRESSURE" in output_variables:
        modelP = models["pressure"].eval()
    if "SWAT" in output_variables:
        modelS = models["saturation"].eval()
    if "SOIL" in output_variables:
        modelO = models["oil"].eval()
    if "SGAS" in output_variables:
        modelG = models["gas"].eval()
    modelPe = models["peacemann"].eval()

    # ── Local result tensors (only this rank's slice) ─────────────────────────
    if "PRESSURE" in output_variables:
        pressure_local = torch.zeros(
            N_local, steppi, nz, nx, ny, device=device, dtype=torch.float32
        )
    if "SWAT" in output_variables:
        swater_local = torch.zeros(
            N_local, steppi, nz, nx, ny, device=device, dtype=torch.float32
        )
    if "SOIL" in output_variables:
        soil_local = torch.zeros(
            N_local, steppi, nz, nx, ny, device=device, dtype=torch.float32
        )
    if "SGAS" in output_variables:
        sgas_local = torch.zeros(
            N_local, steppi, nz, nx, ny, device=device, dtype=torch.float32
        )

    # ── Build local Q/Qg/Qw/Qo for this rank's ensemble slice ─────────────────
    Qg_local = torch.zeros(
        N_local, steppi, nz, nx, ny, device=device, dtype=torch.float32
    )
    Qw_local = torch.zeros(
        N_local, steppi, nz, nx, ny, device=device, dtype=torch.float32
    )
    Qo_local = torch.zeros(
        N_local, steppi, nz, nx, ny, device=device, dtype=torch.float32
    )
    Q_local = torch.zeros(
        N_local, steppi, nz, nx, ny, device=device, dtype=torch.float32
    )

    Qg1 = torch.zeros(N_local, steppi, nx, ny, nz, device=device, dtype=torch.float32)
    Qw1 = torch.zeros(N_local, steppi, nx, ny, nz, device=device, dtype=torch.float32)
    Qo1 = torch.zeros(N_local, steppi, nx, ny, nz, device=device, dtype=torch.float32)
    Q1 = torch.zeros(N_local, steppi, nx, ny, nz, device=device, dtype=torch.float32)

    agas = torch.from_numpy(agas).to(device, dtype=torch.float32)
    awater = torch.from_numpy(awater).to(device, dtype=torch.float32)
    aoil = torch.from_numpy(aoil).to(device, dtype=torch.float32)
    aqq = torch.from_numpy(aqq).to(device, dtype=torch.float32)

    for i in range(N_local):
        Qg1[i] = agas
        Qw1[i] = awater
        Qo1[i] = aoil
        Q1[i] = aqq

    for i in range(nz):
        Qw_local[:, :, i, :, :] = Qw1[:, :, :, :, i]
        Qg_local[:, :, i, :, :] = Qg1[:, :, :, :, i]
        Qo_local[:, :, i, :, :] = Qo1[:, :, :, :, i]
        Q_local[:, :, i, :, :] = Q1[:, :, :, :, i]

    del Qg1, Qw1, Qo1, Q1

    # Normalize flow rates
    if not torch.is_tensor(maxQ):
        maxQ = torch.tensor(maxQ, device=device, dtype=torch.float32)
    if not torch.is_tensor(maxQg):
        maxQg = torch.tensor(maxQg, device=device, dtype=torch.float32)
    if not torch.is_tensor(maxQw):
        maxQw = torch.tensor(maxQw, device=device, dtype=torch.float32)

    Q_local = Q_local / maxQ
    Qg_local = Qg_local / maxQg
    Qw_local = Qw_local / maxQw
    Q_local[Q_local == 0] = 0.1
    Qw_local[Qw_local == 0] = 0.1
    Qg_local[Qg_local == 0] = 0.1

    # ── Time setup ────────────────────────────────────────────────────────────
    Timeafter = Time
    Timebefore = np.zeros_like(Timeafter)
    Timebefore[:, 1:, :, :, :] = Timeafter[:, :-1, :, :, :]
    dt = Timeafter - Timebefore
    dt = dt / maxT
    Time = Time / maxT
    dt_full = torch.from_numpy(dt).to(device, dtype=torch.float32)
    t_full = torch.from_numpy(Time).to(device, dtype=torch.float32)

    input_keys = [
        "perm",
        "poro",
        "pini",
        "sini",
        "sgini",
        "soini",
        "fault",
        "Q",
        "Qg",
        "Qw",
        "dt",
        "t",
    ]

    # ── Sequential autoregressive forwarding — one ensemble member at a time ──
    for local_idx, i in enumerate(indices_local):
        perm_sample = x_true["perm"][i, :, :, :, :][None, :, :, :, :]
        poro_sample = x_true["poro"][i, :, :, :, :][None, :, :, :, :]
        Q_sample = Q_local[local_idx][None, :, :, :, :]
        Qg_sample = Qg_local[local_idx][None, :, :, :, :]
        Qw_sample = Qw_local[local_idx][None, :, :, :, :]
        fault_sample = x_true["fault"][i, :, :, :, :][None, :, :, :, :]
        pbefore = x_true["pini"][i, :, :, :, :][None, :, :, :, :]
        swbefore = x_true["sini"][i, :, :, :, :][None, :, :, :, :]
        sgbefore = x_true["sgini"][i, :, :, :, :][None, :, :, :, :]
        sobefore = x_true["soini"][i, :, :, :, :][None, :, :, :, :]

        for t in range(steppi):
            dt_in = dt_full[0, t, 0, 0, 0] * torch.ones_like(perm_sample, device=device)
            t_in = t_full[0, t, 0, 0, 0] * torch.ones_like(perm_sample, device=device)
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
                "t": t_in,
            }

            with torch.no_grad():
                if cfg.custom.model_type == "FNO":
                    tensors = [v for v in temp.values() if isinstance(v, torch.Tensor)]
                    if not tensors:
                        raise ValueError("No valid input tensors found for the model!")
                    input_tensor = torch.cat(tensors, dim=1)
                else:
                    vars_for_cat = [temp[key].unsqueeze(-1) for key in input_keys]
                    input_tensor = torch.cat(vars_for_cat, dim=-1)

                if cfg.custom.model_type == "FNO":
                    if "PRESSURE" in output_variables and modelP is not None:
                        pafter = torch.clamp(modelP(input_tensor), 0.0, 1.0)
                    if "SWAT" in output_variables and modelS is not None:
                        swafter = torch.clamp(modelS(input_tensor), 0.0, 1.0)
                    if "SOIL" in output_variables and modelO is not None:
                        soafter = torch.clamp(modelO(input_tensor), 0.0, 1.0)
                    if "SGAS" in output_variables and modelG is not None:
                        sgafter = torch.clamp(modelG(input_tensor), 0.0, 1.0)
                else:
                    if "PRESSURE" in output_variables and modelP is not None:
                        pafter = torch.clamp(
                            run_transolver(input_tensor, modelP), 0.0, 1.0
                        )
                    if "SWAT" in output_variables and modelS is not None:
                        swafter = torch.clamp(
                            run_transolver(input_tensor, modelS), 0.0, 1.0
                        )
                    if "SOIL" in output_variables and modelO is not None:
                        soafter = torch.clamp(
                            run_transolver(input_tensor, modelO), 0.0, 1.0
                        )
                    if "SGAS" in output_variables and modelG is not None:
                        sgafter = torch.clamp(
                            run_transolver(input_tensor, modelG), 0.0, 1.0
                        )

            # Store this timestep into LOCAL slot (local_idx), feed back as next-step state
            if "PRESSURE" in output_variables and modelP is not None:
                puse = pafter[0, 0, :, :, :]
                pressure_local[local_idx, t, :, :, :] = puse
                pbefore = puse[None, None, :, :, :]
            if "SWAT" in output_variables and modelS is not None:
                swuse = swafter[0, 0, :, :, :]
                swater_local[local_idx, t, :, :, :] = swuse
                swbefore = swuse[None, None, :, :, :]
            if "SOIL" in output_variables and modelO is not None:
                souse = soafter[0, 0, :, :, :]
                soil_local[local_idx, t, :, :, :] = souse
                sobefore = souse[None, None, :, :, :]
            if "SGAS" in output_variables and modelG is not None:
                sguse = sgafter[0, 0, :, :, :]
                sgas_local[local_idx, t, :, :, :] = sguse
                sgbefore = sguse[None, None, :, :, :]

            # del temp, input_tensor
            # if torch.cuda.is_available():
            # torch.cuda.empty_cache()

    # ─────────────────────────────────────────────────────────────────────────
    # Hybrid per-well source mask for Trainmoe == "BOTH".
    # Length 66: 0..21 = WOPR, 22..43 = WWPR, 44..65 = WGPR.
    # Convention: 0 = CCR, 1 = FNO.
    # ─────────────────────────────────────────────────────────────────────────
    _HYBRID_SOURCE_MASK = np.array(
        [
            # WOPR (0..21)
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            0,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            1,
            1,
            0,
            # WWPR (22..43)
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            # WGPR (44..65)
            0,
            1,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            1,
            1,
            0,
            1,
            1,
            0,
        ],
        dtype=np.int32,
    )

    def _assemble_hybrid_per_well(ouut_fno, ouut_ccr):
        """Assemble a hybrid (B, T, 66) by per-well source assignment."""
        assert ouut_fno.shape == ouut_ccr.shape, (
            f"FNO/CCR shapes differ: {ouut_fno.shape} vs {ouut_ccr.shape}"
        )
        assert ouut_fno.shape[-1] == _HYBRID_SOURCE_MASK.size, (
            f"Last dim is {ouut_fno.shape[-1]} but the hybrid mask has "
            f"{_HYBRID_SOURCE_MASK.size} entries; they must match."
        )
        pick_fno = _HYBRID_SOURCE_MASK.reshape(1, 1, -1).astype(bool)
        return np.where(pick_fno, ouut_fno, ouut_ccr)

    # ── All-gather strided slices back into full (N, ...) arrays ─────────────
    def _gather_strided(local_t, full_N):
        """All-gather strided slices and reorder into the original layout."""
        if not is_dist or world_size == 1:
            return local_t

        max_local = (full_N + world_size - 1) // world_size
        pad_amount = max_local - local_t.shape[0]
        if pad_amount > 0:
            pad = torch.zeros(
                pad_amount,
                *local_t.shape[1:],
                device=local_t.device,
                dtype=local_t.dtype,
            )
            local_padded = torch.cat([local_t, pad], dim=0)
        else:
            local_padded = local_t

        local_padded = local_padded.contiguous()
        gathered = [torch.zeros_like(local_padded) for _ in range(world_size)]
        torchdist.all_gather(gathered, local_padded)

        full = torch.zeros(
            full_N, *local_t.shape[1:], device=local_t.device, dtype=local_t.dtype
        )
        for r in range(world_size):
            r_indices = list(range(r, full_N, world_size))
            for j, idx in enumerate(r_indices):
                full[idx] = gathered[r][j]
        return full

    if "PRESSURE" in output_variables:
        pressure = _gather_strided(pressure_local, N)
    if "SWAT" in output_variables:
        swater = _gather_strided(swater_local, N)
    if "SOIL" in output_variables:
        soil = _gather_strided(soil_local, N)
    if "SGAS" in output_variables:
        sgas = _gather_strided(sgas_local, N)

    if is_dist:
        torchdist.barrier()

    # ── Convert to numpy & post-process ──────────────────────────────────────
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

    active_mask_3d = active_mask_3d[None, None, :, :, :]
    resultss = {}

    if "PRESSURE" in output_variables:
        resultss["PRESSURE"] = pressure * active_mask_3d
    if "SWAT" in output_variables:
        resultss["SWAT"] = swater * active_mask_3d
    if "SOIL" in output_variables:
        resultss["SOIL"] = soil * active_mask_3d
    if "SGAS" in output_variables:
        resultss["SGAS"] = sgas * active_mask_3d

    # ── Decide which Peacemann inputs need to be built ───────────────────────
    run_fno = Trainmoe in ("FNO", "BOTH")
    run_ccr = Trainmoe in ("MoE", "BOTH")

    n_chan = (N_pr * 4) + 2
    well_indices = process_data(unique_entries)

    # Allocate full and local per-rank input tensors only for the model(s) we run.
    innn_fno_full = innn_ccr_full = None
    if run_fno:
        innn_fno_full = np.zeros((N, n_chan, steppi))
        innn_fno_local = (
            np.zeros((N_local, n_chan, steppi))
            if N_local > 0
            else np.zeros((0, n_chan, steppi))
        )
    if run_ccr:
        innn_ccr_full = np.zeros((N, steppi, n_chan))
        innn_ccr_local = (
            np.zeros((N_local, steppi, n_chan))
            if N_local > 0
            else np.zeros((0, steppi, n_chan))
        )

    # ── Build Peacemann inputs (shard across ranks) ──────────────────────────
    for local_idx, i in enumerate(indices_local):
        permuse = perm[i, 0, :, :, :]
        mean_big = []
        for indices_list in well_indices.values():
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
            a1[k, 0] = np.mean(presure_use[k, :, :, :])
            a4[k, 0] = Time_usee[k, :, :, :][0, 0, 0]
        inn1 = np.hstack((permxx, a1, a5, a2, a3, a4))  # shape (steppi, n_chan)

        # Build each model's input independently. Each uses its own normalization.
        if run_fno:
            inn1_fno = fit_operation(
                inn1, target_min, target_max, min_inn_fcn, max_inn_fcn
            )
            innn_fno_local[local_idx, :, :] = inn1_fno.T  # (n_chan, steppi)
        if run_ccr:
            inn1_ccr = fit_operation(
                inn1, target_min, target_max, min_inn_fcn, max_inn_fcn
            )
            innn_ccr_local[local_idx, :, :] = inn1_ccr  # (steppi, n_chan)

    # Gather input tensors from all ranks
    if run_fno:
        if is_dist and world_size > 1:
            t = torch.from_numpy(innn_fno_local).to(device).contiguous()
            innn_fno_full = _gather_strided(t, N).detach().cpu().numpy()
        else:
            innn_fno_full = innn_fno_local

    if run_ccr:
        if is_dist and world_size > 1:
            t = torch.from_numpy(innn_ccr_local).to(device).contiguous()
            innn_ccr_full = _gather_strided(t, N).detach().cpu().numpy()
        else:
            innn_ccr_full = innn_ccr_local

    # ── Peacemann forward (FNO, MoE, or BOTH) ────────────────────────────────
    ouut_fno = None
    ouut_ccr = None

    # ── FNO Peacemann (every rank processes its slice, then gather) ──────────
    if run_fno:
        innn_fno_t = torch.from_numpy(innn_fno_full).to(device, torch.float32)
        ouut_local = []
        for local_idx, i in enumerate(indices_local):
            temp = innn_fno_t[i, :, :][
                None, :, :
            ]  # (1, n_chan, steppi) — correct for FNO
            with torch.no_grad():
                out1 = modelPe(temp).detach().cpu().numpy() * max_out_fcn
            ouut_local.append(out1)
        if len(ouut_local) > 0:
            ouut_local_arr = np.vstack(ouut_local)
        else:
            ouut_local_arr = np.zeros((0, 3 * N_pr, steppi))

        if is_dist and world_size > 1:
            ouut_local_t = torch.from_numpy(ouut_local_arr).to(device).contiguous()
            gathered = _gather_strided(ouut_local_t, N)
            ouut_fno = gathered.detach().cpu().numpy()
        else:
            ouut_fno = ouut_local_arr

        ouut_fno = np.transpose(ouut_fno, (0, 2, 1))  # (B, T, 66)
        ouut_fno[ouut_fno <= 0] = 0

    # ── CCR/MoE Peacemann (every rank runs it locally — no broadcast needed) ─
    if run_ccr:
        useq = lenwels * N_pr
        innn_flat = np.vstack(innn_ccr_full)
        cluster_all = sio.loadmat(
            to_absolute_path("../ML_MACHINE/clustersizescost.mat")
        )["cluster"]
        cluster_all = np.reshape(cluster_all, (-1, 1), "F")
        ies = Parallel(n_jobs=num_cores, backend="loky")(
            delayed(PREDICTION_CCR__MACHINE)(
                ib,
                int(cluster_all[ib, :]),
                innn_flat,
                innn_flat.shape[1],
                to_absolute_path("../ML_MACHINE"),
                oldfolder,
                pred_type,
                degg,
                experts,
                device,
            )
            for ib in range(useq)
        )
        ouut_ccr = np.array(Split_Matrix(np.hstack(ies), N))
        ouut_ccr = ouut_ccr * max_out_fcn
        ouut_ccr[ouut_ccr <= 0] = 0
    # ── Decide the final ouut_p ───────────────────────────────────────────────
    if Trainmoe == "FNO":
        ouut_p = ouut_fno

    elif Trainmoe == "MoE":
        ouut_p = ouut_ccr

    elif Trainmoe == "BOTH":
        ouut_p = _assemble_hybrid_per_well(ouut_fno, ouut_ccr)
        if rank == 0:
            n_fno = int((_HYBRID_SOURCE_MASK == 1).sum())
            n_ccr = int((_HYBRID_SOURCE_MASK == 0).sum())
            print(
                f"  BOTH-mode (fixed per-well assignment): "
                f"FNO {n_fno}/66, CCR {n_ccr}/66"
            )

    else:
        raise ValueError(
            f"Unknown Trainmoe value: {Trainmoe!r}. Expected 'FNO', 'MoE', or 'BOTH'."
        )

    # ── Build sim ────────────────────────────────────────────────────────────
    # sim = []
    # for zz in range(ouut_p.shape[0]):
    # lista = []
    # for k in range(lenwels):
    # rescaled_tensorr = ouut_p[zz, :, k * N_pr : (k + 1) * N_pr]
    # lista.append(rescaled_tensorr)
    # lista = np.hstack(lista)
    # spit  = np.reshape(lista, (-1, 1), "F")
    # sim.append(spit)
    # sim = np.hstack(sim)

    # ── Build sim ────────────────────────────────────────────────────────────
    sim = []
    for zz in range(ouut_p.shape[0]):
        lista = ouut_p[zz, :, 2 * N_pr : 3 * N_pr]
        spit = np.reshape(lista, (-1, 1), "F")
        sim.append(spit)
    sim = np.hstack(sim)

    resultss["sim"] = sim
    resultss["ouut_p"] = ouut_p
    return resultss
