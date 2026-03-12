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

from itertools import product

import numpy as np
import torch

from physicsnemo.core.version_check import check_version_spec


class TorchPCEDensityModel:
    r"""
    Polynomial Chaos Expansion using Hermite polynomials for density-based anomaly detection.

    This model uses PCA dimensionality reduction followed by **Hermite polynomial**
    expansion to estimate the probability density of training data. Hermite polynomials
    are orthogonal with respect to the Gaussian distribution, making them the natural
    choice for normalized PCA components.

    Anomaly scores are computed based on the reconstruction error in the Hermite
    polynomial space using Mahalanobis distance.

    Supports both CPU and GPU computation via PyTorch.

    Parameters
    ----------
    n_components : int, optional
        Number of principal components to retain. If None, keeps components
        that explain 95% of variance. Default is None.
    poly_degree : int, optional
        Maximum degree of Hermite polynomial expansion (1=linear, 2=quadratic, etc.).
        Higher degrees capture more complex distributions but risk overfitting.
        Default is 2.
    interaction_only : bool, optional
        If True, only include interaction terms (no pure higher powers).
        For Hermite polynomials, this limits the maximum degree in any single dimension.
        Default is False.
    random_state : int or None, optional
        Random seed for reproducibility. Default is 0.
    device : str or torch.device, optional
        Device to run computations on (e.g., ``"cpu"``, ``"cuda"``).
        Default is ``"cpu"``.

    Attributes
    ----------
    device : torch.device
        Device being used for computation.
    hermite_degree_ : int
        Maximum degree of Hermite polynomials used.
    poly_mean_ : torch.Tensor
        Mean of Hermite polynomial features from training data.
    poly_cov_ : torch.Tensor
        Covariance matrix of Hermite polynomial features.
    poly_cov_inv_ : torch.Tensor
        Inverse covariance matrix (precomputed for efficiency).
    training_scores_ : torch.Tensor
        Anomaly scores for training data (for percentile computation).
    n_features_in_ : int
        Number of input features.
    n_pca_components_ : int
        Number of PCA components actually used.
    pca_mean_ : torch.Tensor
        Mean of training data (for standardization).
    pca_std_ : torch.Tensor
        Standard deviation of training data (for standardization).
    pca_components_ : torch.Tensor
        PCA components (eigenvectors).
    pca_explained_variance_ratio_ : torch.Tensor
        Explained variance ratio for each component.

    Examples
    --------
    >>> import numpy as np
    >>> import torch
    >>> from physicsnemo.experimental.guardrails.geometry import TorchPCEDensityModel
    >>> 
    >>> # Training data (100 samples, 22 features)
    >>> torch.manual_seed(42)
    >>> X_train = torch.randn(100, 22)
    >>> 
    >>> # Fit PCE model with Hermite polynomials
    >>> model = TorchPCEDensityModel(n_components=10, poly_degree=2, random_state=42)
    >>> result = model.fit(X_train)
    >>> isinstance(result, TorchPCEDensityModel)
    True
    >>> 
    >>> # Score new data
    >>> X_test = torch.randn(10, 22)
    >>> scores = model.score(X_test)
    >>> scores.shape
    torch.Size([10])
    >>> all(s > 0 for s in scores)  # Scores should be positive
    True
    >>> 
    >>> # Compute percentiles
    >>> percentiles = model.percentiles(scores)
    >>> percentiles.shape
    (10,)

    Notes
    -----
    **Algorithm Overview**:

    1. **PCA Dimensionality Reduction**: Projects features onto principal components
       to capture main variance directions and reduce noise. Components are automatically
       standardized (zero mean, unit variance).

    2. **Hermite Polynomial Expansion**: Generates orthogonal Hermite polynomial features
       up to specified degree. Uses **probabilist's Hermite polynomials** which are
       orthogonal with respect to the standard normal distribution :math:`\mathcal{N}(0,1)`.

    3. **Mahalanobis Distance**: Computes anomaly scores as the Mahalanobis distance
       in the Hermite polynomial feature space, which accounts for correlations.

    The probabilist's Hermite polynomials satisfy:

    .. math::

        \int_{-\infty}^{\infty} H_m(x) H_n(x) \frac{1}{\sqrt{2\pi}} e^{-x^2/2} dx = n! \delta_{mn}

    **Choosing Polynomial Degree**:

    - ``poly_degree=1``: Linear model (fastest, captures linear correlations only)
    - ``poly_degree=2``: Quadratic (good default, captures second-order effects)
    - ``poly_degree=3``: Cubic (more expressive, risk of overfitting)
    - ``poly_degree >= 4``: Use with caution (likely to overfit unless large dataset)

    **Number of Hermite Terms**:

    For :math:`d` PCA components and maximum degree :math:`p`, the number of terms is:

    .. math::

        N = \binom{d + p}{p} = \frac{(d + p)!}{d! \, p!}

    Examples: d=10, p=2 → 66 terms; d=10, p=3 → 286 terms
    """

    def __init__(
        self,
        n_components: int | None = None,
        poly_degree: int = 2,
        interaction_only: bool = False,
        random_state: int | None = 0,
        device: str | torch.device = "cpu",
    ):
        self.n_components = n_components
        self.poly_degree = poly_degree
        self.interaction_only = interaction_only
        self.random_state = random_state
        
        # Set device
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        # Set random seed if provided
        if random_state is not None:
            torch.manual_seed(random_state)
            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.manual_seed(random_state)

        # Fitted attributes (set during fit)
        self.hermite_degree_ = None
        self.n_pca_components_ = None
        self.poly_mean_ = None
        self.poly_cov_ = None
        self.poly_cov_inv_ = None
        self.training_scores_ = None
        self.n_features_in_ = None
        self.pca_mean_ = None
        self.pca_std_ = None
        self.pca_components_ = None
        self.pca_explained_variance_ratio_ = None

    def fit(self, X: torch.Tensor | np.ndarray) -> "TorchPCEDensityModel":
        r"""
        Fit PCE density model using Hermite polynomials to training data.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Training features of shape :math:`(N, D)` where :math:`N` is the
            number of samples and :math:`D` is the feature dimension.
            If numpy array, will be converted to torch tensor.

        Returns
        -------
        self : TorchPCEDensityModel
            Fitted model instance (for method chaining).

        Raises
        ------
        ValueError
            If X has insufficient samples or invalid shape.
        """
        # Convert to torch tensor if needed and move to device
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(self.device)
        elif isinstance(X, torch.Tensor):
            X = X.float().to(self.device)
        else:
            raise TypeError(f"X must be np.ndarray or torch.Tensor, got {type(X)}")

        # Validate input
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        if X.shape[0] < 10:
            raise ValueError(f"Need at least 10 samples for fitting, got {X.shape[0]}")

        self.n_features_in_ = X.shape[1]

        # Standardize features (zero mean, unit variance)
        self.pca_mean_ = X.mean(dim=0, keepdim=True)
        self.pca_std_ = X.std(dim=0, keepdim=True)
        # Avoid division by zero
        self.pca_std_ = torch.clamp(self.pca_std_, min=1e-8)
        X_scaled = (X - self.pca_mean_) / self.pca_std_

        # Apply PCA using SVD
        # Center the data (already done by standardization)
        # Compute SVD: X = U @ S @ V^T
        U, S, Vt = torch.linalg.svd(X_scaled, full_matrices=False)
        
        # Explained variance ratio
        explained_variance = (S ** 2) / (X.shape[0] - 1)
        total_variance = explained_variance.sum()
        self.pca_explained_variance_ratio_ = explained_variance / total_variance
        
        # Determine number of components
        if self.n_components is None:
            # Auto-select to explain 95% variance
            cum_var = torch.cumsum(self.pca_explained_variance_ratio_, dim=0)
            n_keep = torch.searchsorted(cum_var, torch.tensor(0.95, device=self.device)) + 1
            n_keep = min(n_keep.item(), X.shape[1])
        else:
            n_keep = min(self.n_components, X.shape[0], X.shape[1])
        
        # Store PCA components (transpose of Vt)
        self.pca_components_ = Vt[:n_keep, :].T  # Shape: (n_features, n_components)
        
        # Project to PCA space
        X_pca = X_scaled @ self.pca_components_  # Shape: (n_samples, n_components)
        
        self.n_pca_components_ = X_pca.shape[1]
        self.hermite_degree_ = self.poly_degree

        # Generate Hermite polynomial features
        X_hermite = self._generate_hermite_features(X_pca)

        # Compute statistics in Hermite polynomial space
        self.poly_mean_ = X_hermite.mean(dim=0)  # Shape: (n_terms,)
        
        # Compute covariance matrix
        X_centered = X_hermite - self.poly_mean_.unsqueeze(0)
        self.poly_cov_ = (X_centered.T @ X_centered) / (X.shape[0] - 1)  # Shape: (n_terms, n_terms)

        # Add regularization to covariance for numerical stability
        self.poly_cov_ += 1e-6 * torch.eye(self.poly_cov_.shape[0], device=self.device)

        # Precompute inverse for Mahalanobis distance
        self.poly_cov_inv_ = torch.linalg.inv(self.poly_cov_)

        # Store training scores for percentile computation
        self.training_scores_ = self._compute_scores(X_hermite)

        return self

    def _generate_hermite_features(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Generate Hermite polynomial features from PCA components.

        Uses PyTorch implementation of probabilist's Hermite polynomials which are
        orthogonal with respect to the standard normal distribution.

        Parameters
        ----------
        X : torch.Tensor
            PCA components, shape (N, d) where d is number of PCA components.

        Returns
        -------
        X_hermite : torch.Tensor
            Hermite polynomial features, shape (N, M) where M is the number
            of polynomial terms.

        Notes
        -----
        Generates all multivariate Hermite polynomial terms up to total degree
        ``poly_degree``. For interaction_only=True, limits the maximum degree
        in any single dimension.

        The probabilist's Hermite polynomials are defined recursively:
        
        .. math::

            H_0(x) = 1, \quad H_1(x) = x, \quad H_{n+1}(x) = x H_n(x) - n H_{n-1}(x)
        """
        n_samples, n_dims = X.shape
        max_degree = self.poly_degree

        # Generate all multi-indices (polynomial powers for each dimension)
        if self.interaction_only:
            # Limit max degree in any dimension to 1 (only interactions)
            indices = [idx for idx in product(range(2), repeat=n_dims) 
                      if 0 < sum(idx) <= max_degree]
        else:
            # All combinations up to total degree
            indices = [idx for idx in product(range(max_degree + 1), repeat=n_dims)
                      if 0 < sum(idx) <= max_degree]

        # Add constant term (all zeros)
        indices = [(0,) * n_dims] + indices

        n_terms = len(indices)
        X_hermite = torch.zeros((n_samples, n_terms), device=self.device, dtype=X.dtype)

        # Precompute Hermite polynomials for each dimension and degree
        hermite_cache = {}
        for dim in range(n_dims):
            hermite_cache[dim] = {}
            for deg in range(max_degree + 1):
                hermite_cache[dim][deg] = self._hermite_poly(X[:, dim], deg)

        # Evaluate each multivariate Hermite term
        for i, idx in enumerate(indices):
            term = torch.ones(n_samples, device=self.device, dtype=X.dtype)
            for dim, deg in enumerate(idx):
                if deg > 0:
                    term *= hermite_cache[dim][deg]
            X_hermite[:, i] = term

        return X_hermite

    def _hermite_poly(self, x: torch.Tensor, degree: int) -> torch.Tensor:
        r"""
        Evaluate probabilist's Hermite polynomial of given degree using PyTorch.

        Uses recursive definition:
        - H_0(x) = 1
        - H_1(x) = x
        - H_{n+1}(x) = x * H_n(x) - n * H_{n-1}(x)

        Parameters
        ----------
        x : torch.Tensor
            Input values.
        degree : int
            Polynomial degree.

        Returns
        -------
        torch.Tensor
            Hermite polynomial evaluated at x.
        """
        if degree == 0:
            return torch.ones_like(x)
        elif degree == 1:
            return x
        
        # Recursive computation
        h_prev = torch.ones_like(x)  # H_0
        h_curr = x  # H_1
        
        for n in range(1, degree):
            h_next = x * h_curr - n * h_prev
            h_prev = h_curr
            h_curr = h_next
        
        return h_curr

    def score(self, X: torch.Tensor | np.ndarray) -> torch.Tensor:
        r"""
        Compute anomaly scores for new data using Hermite polynomial expansion.

        Anomaly scores are computed as the Mahalanobis distance in the
        Hermite polynomial feature space. Higher scores indicate more anomalous samples.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Features to score, shape :math:`(N, D)`.
            If numpy array, will be converted to torch tensor.

        Returns
        -------
        scores : torch.Tensor
            Anomaly scores, shape :math:`(N,)`. Higher values are more anomalous.

        Notes
        -----
        The score is computed as:

        .. math::

            s(\mathbf{x}) = \sqrt{(\mathbf{h} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{h} - \boldsymbol{\mu})}

        where :math:`\mathbf{h}` is the Hermite polynomial feature vector,
        :math:`\boldsymbol{\mu}` is the mean, and :math:`\boldsymbol{\Sigma}` is
        the covariance matrix from training data.

        This is the Mahalanobis distance in Hermite polynomial space, which accounts
        for correlations and scales appropriately with variance.
        """
        if self.pca_components_ is None:
            raise RuntimeError("Model must be fitted before scoring")

        # Convert to torch tensor if needed and move to device
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(self.device)
        elif isinstance(X, torch.Tensor):
            X = X.float().to(self.device)
        else:
            raise TypeError(f"X must be np.ndarray or torch.Tensor, got {type(X)}")

        # Standardize
        X_scaled = (X - self.pca_mean_) / self.pca_std_

        # Project to PCA space
        X_pca = X_scaled @ self.pca_components_  # Shape: (n_samples, n_components)

        # Trim to match training components (should already match, but be safe)
        if X_pca.shape[1] > self.n_pca_components_:
            X_pca = X_pca[:, :self.n_pca_components_]

        X_hermite = self._generate_hermite_features(X_pca)

        # Compute Mahalanobis distance
        return self._compute_scores(X_hermite)

    def _compute_scores(self, X_poly: torch.Tensor) -> torch.Tensor:
        """Compute Mahalanobis distance for polynomial features."""
        # Center the data
        X_centered = X_poly - self.poly_mean_.unsqueeze(0)

        # Mahalanobis distance: sqrt((x - mu)^T Sigma^-1 (x - mu))
        # Using einsum for efficiency: (x-mu) @ Sigma^-1 @ (x-mu)^T
        mahal_dist = torch.sqrt(
            torch.sum(X_centered @ self.poly_cov_inv_ * X_centered, dim=1)
        )

        return mahal_dist

    def percentiles(self, scores: torch.Tensor | np.ndarray) -> np.ndarray:
        r"""
        Convert anomaly scores to empirical percentiles.

        Percentiles are computed relative to the training data distribution.
        A percentile of 95.0 means the sample is more anomalous than 95% of
        training samples.

        Parameters
        ----------
        scores : torch.Tensor or np.ndarray
            Anomaly scores from :meth:`score`.

        Returns
        -------
        percentiles : np.ndarray
            Percentiles in range [0, 100].
        """
        if self.training_scores_ is None:
            raise RuntimeError("Model must be fitted before computing percentiles")

        # Convert to torch tensor if needed
        if isinstance(scores, np.ndarray):
            scores = torch.from_numpy(scores).to(self.device)
        elif isinstance(scores, torch.Tensor):
            scores = scores.to(self.device)
        else:
            raise TypeError(f"scores must be np.ndarray or torch.Tensor, got {type(scores)}")

        # Validate training_scores_ is not empty
        if len(self.training_scores_) == 0:
            raise RuntimeError("Training scores are empty. Model may not have been fitted correctly.")
        
        # Compute percentile of each score relative to training distribution
        # Using broadcasting for efficiency
        scores_expanded = scores.unsqueeze(1)  # (n_scores, 1)
        training_expanded = self.training_scores_.unsqueeze(0)  # (1, n_training)
        percentiles = 100.0 * (training_expanded <= scores_expanded).sum(dim=1).float() / len(self.training_scores_)

        # Convert to numpy for return
        return percentiles.cpu().numpy()

    def get_state(self) -> dict:
        r"""
        Get model state for serialization.

        Returns
        -------
        dict
            Dictionary containing model hyperparameters and fitted attributes.
        """
        return {
            "n_components": self.n_components,
            "poly_degree": self.poly_degree,
            "interaction_only": self.interaction_only,
            "hermite_degree_": self.hermite_degree_,
            "n_pca_components_": self.n_pca_components_,
            "poly_mean_": self.poly_mean_.cpu().numpy() if self.poly_mean_ is not None else None,
            "poly_cov_": self.poly_cov_.cpu().numpy() if self.poly_cov_ is not None else None,
            "poly_cov_inv_": self.poly_cov_inv_.cpu().numpy() if self.poly_cov_inv_ is not None else None,
            "training_scores_": self.training_scores_.cpu().numpy() if self.training_scores_ is not None else None,
            "n_features_in_": self.n_features_in_,
            "pca_mean_": self.pca_mean_.cpu().numpy() if self.pca_mean_ is not None else None,
            "pca_std_": self.pca_std_.cpu().numpy() if self.pca_std_ is not None else None,
            "pca_components_": self.pca_components_.cpu().numpy() if self.pca_components_ is not None else None,
            "pca_explained_variance_ratio_": self.pca_explained_variance_ratio_.cpu().numpy() if self.pca_explained_variance_ratio_ is not None else None,
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
        self.n_components = state["n_components"]
        self.poly_degree = state["poly_degree"]
        self.interaction_only = state["interaction_only"]
        
        # Set device (runtime parameter, not part of model state)
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        self.hermite_degree_ = state["hermite_degree_"]
        self.n_pca_components_ = state["n_pca_components_"]
        self.n_features_in_ = state["n_features_in_"]
        
        # Convert numpy arrays back to torch tensors on device
        # Use .get() to handle missing keys (which were None and excluded during save)
        if state.get("poly_mean_") is not None:
            self.poly_mean_ = torch.from_numpy(state["poly_mean_"]).float().to(self.device)
        else:
            self.poly_mean_ = None
        if state.get("poly_cov_") is not None:
            self.poly_cov_ = torch.from_numpy(state["poly_cov_"]).float().to(self.device)
        else:
            self.poly_cov_ = None
        if state.get("poly_cov_inv_") is not None:
            self.poly_cov_inv_ = torch.from_numpy(state["poly_cov_inv_"]).float().to(self.device)
        else:
            self.poly_cov_inv_ = None
        if state.get("training_scores_") is not None:
            self.training_scores_ = torch.from_numpy(state["training_scores_"]).float().to(self.device)
        else:
            self.training_scores_ = None
        if state.get("pca_mean_") is not None:
            self.pca_mean_ = torch.from_numpy(state["pca_mean_"]).float().to(self.device)
        else:
            self.pca_mean_ = None
        if state.get("pca_std_") is not None:
            self.pca_std_ = torch.from_numpy(state["pca_std_"]).float().to(self.device)
        else:
            self.pca_std_ = None
        if state.get("pca_components_") is not None:
            self.pca_components_ = torch.from_numpy(state["pca_components_"]).float().to(self.device)
        else:
            self.pca_components_ = None
        if state.get("pca_explained_variance_ratio_") is not None:
            self.pca_explained_variance_ratio_ = torch.from_numpy(state["pca_explained_variance_ratio_"]).float().to(self.device)
        else:
            self.pca_explained_variance_ratio_ = None