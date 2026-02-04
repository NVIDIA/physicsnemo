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

from physicsnemo.core.version_check import require_version_spec


@require_version_spec("scikit-learn", "1.0.0")
class PCEDensityModel:
    r"""
    Polynomial Chaos Expansion using Hermite polynomials for density-based anomaly detection.

    This model uses PCA dimensionality reduction followed by **Hermite polynomial**
    expansion to estimate the probability density of training data. Hermite polynomials
    are orthogonal with respect to the Gaussian distribution, making them the natural
    choice for normalized PCA components.

    Anomaly scores are computed based on the reconstruction error in the Hermite
    polynomial space using Mahalanobis distance.
    
    .. note::
        
        **CPU-only implementation**: This method currently supports CPU computation only.
        For GPU acceleration, use GMM (``method="gmm"``) instead.

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
        Random seed for reproducibility. Default is None.

    Attributes
    ----------
    pca_ : sklearn.decomposition.PCA
        Fitted PCA transformer.
    hermite_degree_ : int
        Maximum degree of Hermite polynomials used.
    poly_mean_ : np.ndarray
        Mean of Hermite polynomial features from training data.
    poly_cov_ : np.ndarray
        Covariance matrix of Hermite polynomial features.
    training_scores_ : np.ndarray
        Anomaly scores for training data (for percentile computation).
    n_features_in_ : int
        Number of input features.
    n_pca_components_ : int
        Number of PCA components actually used.

    Examples
    --------
    >>> import numpy as np
    >>> from physicsnemo.experimental.guardrails.geometry import PCEDensityModel
    >>> 
    >>> # Training data (100 samples, 22 features)
    >>> X_train = np.random.randn(100, 22)
    >>> 
    >>> # Fit PCE model with Hermite polynomials
    >>> model = PCEDensityModel(n_components=10, poly_degree=2)
    >>> model.fit(X_train)
    >>> 
    >>> # Score new data
    >>> X_test = np.random.randn(10, 22)
    >>> scores = model.score(X_test)
    >>> percentiles = model.percentiles(scores)
    >>> print(f"Anomaly percentiles: {percentiles}")

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
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.poly_degree = poly_degree
        self.interaction_only = interaction_only
        self.random_state = random_state

        # Fitted attributes (set during fit)
        self.pca_ = None
        self.scaler_ = None
        self.hermite_degree_ = None
        self.n_pca_components_ = None
        self.poly_mean_ = None
        self.poly_cov_ = None
        self.poly_cov_inv_ = None
        self.training_scores_ = None
        self.n_features_in_ = None

    def fit(self, X: np.ndarray) -> PCEDensityModel:
        r"""
        Fit PCE density model using Hermite polynomials to training data.

        Parameters
        ----------
        X : np.ndarray
            Training features of shape :math:`(N, D)` where :math:`N` is the
            number of samples and :math:`D` is the feature dimension.

        Returns
        -------
        self : PCEDensityModel
            Fitted model instance (for method chaining).

        Raises
        ------
        ValueError
            If X has insufficient samples or invalid shape.
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # Validate input
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        if X.shape[0] < 10:
            raise ValueError(f"Need at least 10 samples for fitting, got {X.shape[0]}")

        self.n_features_in_ = X.shape[1]

        # Standardize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Apply PCA
        if self.n_components is None:
            # Auto-select to explain 95% variance
            n_comp = min(X.shape[0], X.shape[1])
        else:
            n_comp = min(self.n_components, X.shape[0], X.shape[1])

        self.pca_ = PCA(n_components=n_comp, random_state=self.random_state)
        X_pca = self.pca_.fit_transform(X_scaled)

        # If n_components was None, trim to 95% variance
        if self.n_components is None:
            cum_var = np.cumsum(self.pca_.explained_variance_ratio_)
            n_keep = np.searchsorted(cum_var, 0.95) + 1
            X_pca = X_pca[:, :n_keep]
        
        self.n_pca_components_ = X_pca.shape[1]
        self.hermite_degree_ = self.poly_degree

        # Generate Hermite polynomial features
        X_hermite = self._generate_hermite_features(X_pca)

        # Compute statistics in Hermite polynomial space
        self.poly_mean_ = np.mean(X_hermite, axis=0)
        self.poly_cov_ = np.cov(X_hermite, rowvar=False)

        # Add regularization to covariance for numerical stability
        self.poly_cov_ += 1e-6 * np.eye(self.poly_cov_.shape[0])

        # Precompute inverse for Mahalanobis distance
        self.poly_cov_inv_ = np.linalg.inv(self.poly_cov_)

        # Store training scores for percentile computation
        self.training_scores_ = self._compute_scores(X_hermite)

        return self

    def _generate_hermite_features(self, X: np.ndarray) -> np.ndarray:
        r"""
        Generate Hermite polynomial features from PCA components.

        Uses NumPy's probabilist's Hermite polynomials which are orthogonal
        with respect to the standard normal distribution.

        Parameters
        ----------
        X : np.ndarray
            PCA components, shape (N, d) where d is number of PCA components.

        Returns
        -------
        X_hermite : np.ndarray
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
        from itertools import product

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
        X_hermite = np.zeros((n_samples, n_terms))

        # Precompute Hermite polynomials for each dimension and degree
        hermite_cache = {}
        for dim in range(n_dims):
            hermite_cache[dim] = {}
            for deg in range(max_degree + 1):
                hermite_cache[dim][deg] = self._hermite_poly(X[:, dim], deg)

        # Evaluate each multivariate Hermite term
        for i, idx in enumerate(indices):
            term = np.ones(n_samples)
            for dim, deg in enumerate(idx):
                if deg > 0:
                    term *= hermite_cache[dim][deg]
            X_hermite[:, i] = term

        return X_hermite

    def _hermite_poly(self, x: np.ndarray, degree: int) -> np.ndarray:
        r"""
        Evaluate probabilist's Hermite polynomial of given degree.

        Uses NumPy's hermite polynomial implementation (numpy.polynomial.hermite_e).

        Parameters
        ----------
        x : np.ndarray
            Input values.
        degree : int
            Polynomial degree.

        Returns
        -------
        np.ndarray
            Hermite polynomial evaluated at x.
        """
        from numpy.polynomial.hermite_e import hermeval

        # Create coefficient array (all zeros except for the degree term)
        coef = np.zeros(degree + 1)
        coef[degree] = 1.0

        return hermeval(x, coef)

    def score(self, X: np.ndarray) -> np.ndarray:
        r"""
        Compute anomaly scores for new data using Hermite polynomial expansion.

        Anomaly scores are computed as the Mahalanobis distance in the
        Hermite polynomial feature space. Higher scores indicate more anomalous samples.

        Parameters
        ----------
        X : np.ndarray
            Features to score, shape :math:`(N, D)`.

        Returns
        -------
        scores : np.ndarray
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
        if self.pca_ is None:
            raise RuntimeError("Model must be fitted before scoring")

        # Transform to Hermite polynomial space
        X_scaled = self.scaler_.transform(X)
        X_pca = self.pca_.transform(X_scaled)

        # Trim to match training components
        if X_pca.shape[1] > self.n_pca_components_:
            X_pca = X_pca[:, :self.n_pca_components_]

        X_hermite = self._generate_hermite_features(X_pca)

        # Compute Mahalanobis distance
        return self._compute_scores(X_hermite)

    def _compute_scores(self, X_poly: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distance for polynomial features."""
        # Center the data
        X_centered = X_poly - self.poly_mean_

        # Mahalanobis distance: sqrt((x - mu)^T Sigma^-1 (x - mu))
        mahal_dist = np.sqrt(
            np.sum(X_centered @ self.poly_cov_inv_ * X_centered, axis=1)
        )

        return mahal_dist

    def percentiles(self, scores: np.ndarray) -> np.ndarray:
        r"""
        Convert anomaly scores to empirical percentiles.

        Percentiles are computed relative to the training data distribution.
        A percentile of 95.0 means the sample is more anomalous than 95% of
        training samples.

        Parameters
        ----------
        scores : np.ndarray
            Anomaly scores from :meth:`score`.

        Returns
        -------
        percentiles : np.ndarray
            Percentiles in range [0, 100].
        """
        if self.training_scores_ is None:
            raise RuntimeError("Model must be fitted before computing percentiles")

        # Compute percentile of each score relative to training distribution
        percentiles = np.array(
            [
                100.0 * (self.training_scores_ < score).sum() / len(self.training_scores_)
                for score in scores
            ]
        )

        return percentiles

    def get_params(self) -> dict:
        r"""
        Get model parameters for serialization.

        Returns
        -------
        dict
            Dictionary containing model hyperparameters and fitted attributes.
        """
        return {
            "n_components": self.n_components,
            "poly_degree": self.poly_degree,
            "interaction_only": self.interaction_only,
            "random_state": self.random_state,
            "pca_": self.pca_,
            "scaler_": self.scaler_,
            "hermite_degree_": self.hermite_degree_,
            "n_pca_components_": self.n_pca_components_,
            "poly_mean_": self.poly_mean_,
            "poly_cov_": self.poly_cov_,
            "poly_cov_inv_": self.poly_cov_inv_,
            "training_scores_": self.training_scores_,
            "n_features_in_": self.n_features_in_,
        }

    def set_params(self, params: dict) -> None:
        r"""
        Set model parameters from dictionary (for deserialization).

        Parameters
        ----------
        params : dict
            Dictionary of parameters from :meth:`get_params`.
        """
        self.n_components = params["n_components"]
        self.poly_degree = params["poly_degree"]
        self.interaction_only = params["interaction_only"]
        self.random_state = params["random_state"]
        self.pca_ = params["pca_"]
        self.scaler_ = params["scaler_"]
        self.hermite_degree_ = params["hermite_degree_"]
        self.n_pca_components_ = params["n_pca_components_"]
        self.poly_mean_ = params["poly_mean_"]
        self.poly_cov_ = params["poly_cov_"]
        self.poly_cov_inv_ = params["poly_cov_inv_"]
        self.training_scores_ = params["training_scores_"]
        self.n_features_in_ = params["n_features_in_"]
