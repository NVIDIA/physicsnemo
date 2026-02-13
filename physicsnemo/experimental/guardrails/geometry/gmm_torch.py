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

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TorchGMM(nn.Module):
    r"""
    GPU-accelerated Gaussian Mixture Model using PyTorch.

    This class implements a Gaussian Mixture Model that can leverage GPU
    acceleration for faster inference on large datasets. It uses the
    Expectation-Maximization (EM) algorithm for parameter estimation.

    Parameters
    ----------
    n_components : int
        Number of Gaussian components in the mixture.
    reg_covar : float, optional
        Regularization term added to covariance diagonal for numerical stability.
        Default is 1e-6.
    max_iter : int, optional
        Maximum number of EM iterations. Default is 100.
    tol : float, optional
        Convergence threshold for log-likelihood improvement. Default is 1e-3.
    device : str or torch.device, optional
        Device to run computations on (e.g., ``"cpu"``, ``"cuda"``).
        Default is ``"cuda"`` if available, else ``"cpu"``.
    random_state : int or None, optional
        Random seed for reproducible initialization. If provided, sets the
        seed for PyTorch's random number generator. Default is 0.

    Attributes
    ----------
    weights_ : torch.Tensor
        Mixture weights of shape :math:`(K,)` where :math:`K` is n_components.
    means_ : torch.Tensor
        Component means of shape :math:`(K, D)` where :math:`D` is feature dim.
    covariances_ : torch.Tensor
        Covariance matrices of shape :math:`(K, D, D)`.
    """

    def __init__(
        self,
        n_components: int,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        tol: float = 1e-3,
        device: str | torch.device | None = None,
        random_state: int | None = 0,
    ):
        super().__init__()
        
        self.n_components = n_components
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Set random seed if provided
        if random_state is not None:
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_state)
        
        # Parameters (will be initialized during fit)
        self.weights_: torch.Tensor | None = None
        self.means_: torch.Tensor | None = None
        self.covariances_: torch.Tensor | None = None
        self.precisions_cholesky_: torch.Tensor | None = None
        
        self.converged_ = False
        self.n_iter_ = 0

    def _initialize_parameters(self, X: torch.Tensor) -> None:
        """
        Initialize GMM parameters using k-means++ style initialization.

        Parameters
        ----------
        X : torch.Tensor
            Training data of shape (n_samples, n_features).
        """
        n_samples, n_features = X.shape
        
        # Initialize weights uniformly
        self.weights_ = torch.ones(
            self.n_components, device=self.device
        ) / self.n_components
        
        # Initialize means using k-means++ style
        # Pick first mean randomly
        indices = torch.randperm(n_samples, device=self.device)[: self.n_components]
        self.means_ = X[indices].clone()
        
        # Initialize covariances as identity scaled by data variance
        data_var = torch.var(X, dim=0).mean()
        self.covariances_ = torch.eye(
            n_features, device=self.device
        ).unsqueeze(0).repeat(self.n_components, 1, 1) * data_var
        
        # Add regularization
        self.covariances_ += torch.eye(
            n_features, device=self.device
        ).unsqueeze(0) * self.reg_covar

    def _compute_precision_cholesky(self) -> torch.Tensor:
        """
        Compute Cholesky decomposition of precision matrices.

        Returns
        -------
        torch.Tensor
            Cholesky factors of shape (n_components, n_features, n_features).
        """
        n_components, n_features, _ = self.covariances_.shape
        precisions_chol = torch.zeros_like(self.covariances_)
        
        for k in range(n_components):
            cov = self.covariances_[k]
            # Compute Cholesky of covariance, then invert
            try:
                cov_chol = torch.linalg.cholesky(cov)
                precisions_chol[k] = torch.linalg.solve_triangular(
                    cov_chol, torch.eye(n_features, device=self.device), upper=False
                ).T
            except RuntimeError:
                # Fallback: add more regularization
                cov_reg = cov + torch.eye(n_features, device=self.device) * 1e-3
                cov_chol = torch.linalg.cholesky(cov_reg)
                precisions_chol[k] = torch.linalg.solve_triangular(
                    cov_chol, torch.eye(n_features, device=self.device), upper=False
                ).T
        
        return precisions_chol

    def _estimate_log_prob(self, X: torch.Tensor) -> torch.Tensor:
        """
        Estimate log probability under each Gaussian component.

        Parameters
        ----------
        X : torch.Tensor
            Data of shape (n_samples, n_features).

        Returns
        -------
        torch.Tensor
            Log probabilities of shape (n_samples, n_components).
        """
        n_samples, n_features = X.shape
        log_prob = torch.zeros(n_samples, self.n_components, device=self.device)
        
        for k in range(self.n_components):
            diff = X - self.means_[k]  # (n_samples, n_features)
            
            # Compute Mahalanobis distance using precision Cholesky
            prec_chol = self.precisions_cholesky_[k]
            y = torch.matmul(diff, prec_chol)  # (n_samples, n_features)
            maha_dist = torch.sum(y * y, dim=1)  # (n_samples,)
            
            # Log determinant from Cholesky factor
            log_det = 2 * torch.sum(torch.log(torch.diag(prec_chol)))
            
            # Log probability
            log_prob[:, k] = (
                -0.5 * (n_features * np.log(2 * np.pi) + maha_dist) + 0.5 * log_det
            )
        
        return log_prob

    def _e_step(self, X: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        E-step: compute responsibilities and log-likelihood.

        Parameters
        ----------
        X : torch.Tensor
            Data of shape (n_samples, n_features).

        Returns
        -------
        tuple[torch.Tensor, float]
            - Responsibilities of shape (n_samples, n_components)
            - Log-likelihood (scalar)
        """
        log_prob = self._estimate_log_prob(X)
        log_weights = torch.log(self.weights_)
        
        # Weighted log probabilities
        weighted_log_prob = log_prob + log_weights
        
        # Log sum exp for normalization
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1)
        
        # Responsibilities
        log_resp = weighted_log_prob - log_prob_norm.unsqueeze(1)
        resp = torch.exp(log_resp)
        
        # Log-likelihood
        log_likelihood = torch.mean(log_prob_norm)
        
        return resp, log_likelihood.item()

    def _m_step(self, X: torch.Tensor, resp: torch.Tensor) -> None:
        """
        M-step: update parameters based on responsibilities.

        Parameters
        ----------
        X : torch.Tensor
            Data of shape (n_samples, n_features).
        resp : torch.Tensor
            Responsibilities of shape (n_samples, n_components).
        """
        n_samples, n_features = X.shape
        
        # Effective number of samples per component
        nk = torch.sum(resp, dim=0) + 1e-10  # (n_components,)
        
        # Update weights
        self.weights_ = nk / n_samples
        
        # Update means
        self.means_ = torch.matmul(resp.T, X) / nk.unsqueeze(1)
        
        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means_[k]  # (n_samples, n_features)
            weighted_diff = resp[:, k].unsqueeze(1) * diff  # (n_samples, n_features)
            cov = torch.matmul(weighted_diff.T, diff) / nk[k]
            
            # Add regularization
            cov += torch.eye(n_features, device=self.device) * self.reg_covar
            self.covariances_[k] = cov

    def fit(self, X: torch.Tensor | np.ndarray) -> "TorchGMM":
        """
        Fit GMM parameters using the EM algorithm.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Training data of shape (n_samples, n_features).

        Returns
        -------
        TorchGMM
            Fitted model (self).
        """
        # Convert to torch tensor if needed
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(self.device)
        else:
            X = X.to(self.device)
        
        # Initialize parameters
        self._initialize_parameters(X)
        
        # EM iterations
        log_likelihood = -np.inf
        
        for iteration in range(self.max_iter):
            # Compute precision Cholesky factors
            self.precisions_cholesky_ = self._compute_precision_cholesky()
            
            # E-step
            resp, new_log_likelihood = self._e_step(X)
            
            # Check convergence
            if iteration > 0 and abs(new_log_likelihood - log_likelihood) < self.tol:
                self.converged_ = True
                break
            
            log_likelihood = new_log_likelihood
            
            # M-step
            self._m_step(X, resp)
        
        self.n_iter_ = iteration + 1
        
        # Final precision computation
        self.precisions_cholesky_ = self._compute_precision_cholesky()
        
        return self

    def score_samples(self, X: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Compute log-likelihood of samples under the model.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Test data of shape (n_samples, n_features).

        Returns
        -------
        torch.Tensor
            Log-likelihoods of shape (n_samples,). Returns tensor on the same device as input.
        """
        # Convert to torch tensor if needed
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(self.device)
        else:
            X = X.to(self.device)
        
        # Compute log probabilities
        log_prob = self._estimate_log_prob(X)
        log_weights = torch.log(self.weights_)
        
        # Weighted log probabilities
        weighted_log_prob = log_prob + log_weights
        
        # Log sum exp
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1)
        
        # Return as torch tensor (keep on device for efficiency)
        return log_prob_norm

    def score(self, X: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Compute anomaly scores (negative log-likelihood) for samples.

        This method computes the negative log-likelihood for each sample,
        which serves as an anomaly score: higher values indicate more
        anomalous samples (lower probability under the model).

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Test data of shape (n_samples, n_features).

        Returns
        -------
        torch.Tensor
            Anomaly scores of shape (n_samples,). Higher values indicate
            more anomalous samples. Returns tensor on the same device as input.

        Notes
        -----
        The anomaly score is computed as:

        .. math::

            s(\mathbf{x}) = -\log p(\mathbf{x} | \theta)

        where :math:`p(\mathbf{x} | \theta)` is the probability density
        under the fitted GMM.
        """
        # Return negative log-likelihood (higher = more anomalous)
        return -self.score_samples(X)

    def get_state(self) -> dict:
        r"""
        Get model state for serialization.

        Returns
        -------
        dict
            Dictionary containing model hyperparameters and fitted attributes.
        """
        return {
            "weights_": self.weights_.cpu().numpy(),
            "means_": self.means_.cpu().numpy(),
            "covariances_": self.covariances_.cpu().numpy(),
            "converged_": self.converged_,
            "n_iter_": self.n_iter_,
        }

    def set_state(self, state: dict, device: str | torch.device) -> None:
        r"""
        Set model state from dictionary (for deserialization).

        Parameters
        ----------
        state : dict
            Dictionary of parameters from :meth:`get_state`.
        device : str or torch.device
            Device to load model on. Allows loading a model trained on one
            device and using it on another device.
        """
        # Set device (runtime parameter, not part of model state)
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        self.weights_ = torch.from_numpy(state["weights_"]).float().to(self.device)
        self.means_ = torch.from_numpy(state["means_"]).float().to(self.device)
        self.covariances_ = torch.from_numpy(state["covariances_"]).float().to(self.device)
        self.converged_ = state["converged_"]
        self.n_iter_ = state["n_iter_"]
        
        # Compute precision Cholesky
        self.precisions_cholesky_ = self._compute_precision_cholesky()
