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

r"""
Density estimation for geometry guardrails.

This module implements multiple density estimation methods for detecting
out-of-distribution geometric configurations:

- **Gaussian Mixture Models (GMM)**: Flexible, supports multi-modal distributions
- **Polynomial Chaos Expansion (PCE)**: Physics-informed, better for high-dimensional correlated data

Supports both CPU (scikit-learn) and GPU (PyTorch for GMM) backends for
performance optimization on large datasets.
"""

from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture


class GeometryDensityModel:
    r"""
    Density model for anomaly detection with multiple backend options.

    This class provides a unified interface for density estimation supporting
    multiple methods:

    - **GMM (Gaussian Mixture Model)**: Flexible, supports multi-modal distributions
    - **PCE (Polynomial Chaos Expansion)**: Physics-informed, better for correlated features

    Parameters
    ----------
    method : str, optional
        Density estimation method. Options:
        - ``"gmm"``: Gaussian Mixture Model (default)
        - ``"pce"``: Polynomial Chaos Expansion
        Default is ``"gmm"``.
    n_components : int, optional
        For GMM: Number of Gaussian components. For PCE: Number of PCA components
        (None = auto-select). Default is 1.
    covariance_type : str, optional
        For GMM only: Covariance type (``"full"``, ``"tied"``, ``"diag"``, ``"spherical"``).
        Default is ``"full"``.
    poly_degree : int, optional
        For PCE only: Polynomial degree for expansion. Default is 2.
    interaction_only : bool, optional
        For PCE only: If True, only include interaction terms. Default is False.
    random_state : int or None, optional
        Random seed for reproducible initialization. Default is 0.
    device : str, optional
        Device to use for computation. Options:
        - ``"cpu"``: Use scikit-learn on CPU (default)
        - ``"cuda"``: Use PyTorch GMM on GPU (GMM only)
        - ``"cuda:0"``, ``"cuda:1"``, etc.: Specific GPU device (GMM only)
        Default is ``"cpu"``.

    Attributes
    ----------
    model : GaussianMixture, TorchGMM, or PCEDensityModel
        The underlying density estimation model.
    ref_scores : np.ndarray or None
        Reference anomaly scores from training data for percentile computation.
    device : str
        Device being used for computation.
    backend : str
        Backend being used: ``"sklearn"``, ``"torch"``, or ``"pce"``.
    method : str
        Density estimation method: ``"gmm"`` or ``"pce"``.

    Examples
    --------
    GMM on CPU (default):

    >>> import numpy as np
    >>> from physicsnemo.experimental.guardrails.geometry import GeometryDensityModel
    >>> 
    >>> # Training data
    >>> rng = np.random.RandomState(42)
    >>> X_train = rng.randn(100, 22)
    >>> 
    >>> # Fit GMM model
    >>> model = GeometryDensityModel(method="gmm", n_components=1)
    >>> model.fit(X_train)
    >>> 
    >>> # Score new samples
    >>> X_test = rng.randn(10, 22)
    >>> scores = model.score(X_test)
    >>> percentiles = model.percentiles(scores)

    PCE for physics-informed detection:

    >>> # Fit PCE model (better for correlated features)
    >>> model_pce = GeometryDensityModel(
    ...     method="pce",
    ...     n_components=10,  # PCA components
    ...     poly_degree=2,     # Quadratic expansion
    ... )
    >>> model_pce.fit(X_train)
    >>> scores_pce = model_pce.score(X_test)

    GPU-accelerated GMM:

    >>> # Requires PyTorch
    >>> model_gpu = GeometryDensityModel(method="gmm", n_components=2, device="cuda")
    >>> model_gpu.fit(X_train)
    >>> scores_gpu = model_gpu.score(X_test)

    Notes
    -----
    **Choosing a Method**:

    Use **GMM** when:
    
    - Data has multi-modal distributions (multiple sub-populations)
    - Features are relatively independent
    - You need GPU acceleration for large datasets
    - Default choice for most applications

    Use **PCE** when:
    
    - Features have strong correlations (common in physics)
    - You want interpretable polynomial coefficients
    - Data is high-dimensional with smooth distributions
    - You prefer avoiding GMM hyperparameter tuning

    **Performance Comparison**:

    =====================  =========  =========  ==============  ================
    Method                 Multi-modal Correlated GPU Support    Interpretability
    =====================  =========  =========  ==============  ================
    GMM (n_comp=1)         ✗          Medium     ✓               Medium
    GMM (n_comp>1)         ✓          Medium     ✓               Low
    PCE                    ✗          ✓          ✗               ✓
    =====================  =========  =========  ==============  ================

    **Device and Backend Selection**:

    - CPU + GMM → scikit-learn (always available)
    - GPU + GMM → PyTorch (requires torch, 2-10x faster)
    - CPU + PCE → NumPy + scikit-learn (GPU not supported for PCE)

    See Also
    --------
    :class:`GeometryGuardrail` : Main API that uses this density model.
    :class:`PCEDensityModel` : Standalone PCE implementation.
    """

    def __init__(
        self,
        method: str = "gmm",
        n_components: int = 1,
        covariance_type: str = "full",
        poly_degree: int = 2,
        interaction_only: bool = False,
        random_state: int | None = 0,
        device: str = "cpu",
    ):
        self.method = method.lower()
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.poly_degree = poly_degree
        self.interaction_only = interaction_only
        self.random_state = random_state
        self.device = device
        self.ref_scores = None

        # Validate method
        if self.method not in ["gmm", "pce"]:
            raise ValueError(f"method must be 'gmm' or 'pce', got '{self.method}'")

        # Validate device for PCE
        if self.method == "pce" and device != "cpu":
            raise ValueError("PCE method only supports device='cpu'")

        # Initialize model based on method
        if self.method == "gmm":
            self._init_gmm()
        elif self.method == "pce":
            self._init_pce()

    def _init_gmm(self):
        """Initialize GMM model (sklearn or torch)."""
        if self.device == "cpu":
            # Use sklearn GMM (wrapped for consistent API)
            from sklearn.mixture import GaussianMixture as SklearnGMM
            
            class SklearnGMMWrapper:
                """Wrapper to add .score() method to sklearn GMM."""
                def __init__(self, gmm):
                    self.gmm = gmm
                
                def fit(self, X):
                    return self.gmm.fit(X)
                
                def score(self, X):
                    """Return negative log-likelihood (anomaly scores)."""
                    return -self.gmm.score_samples(X)
                
                def score_samples(self, X):
                    """Return log-likelihood (for compatibility)."""
                    return self.gmm.score_samples(X)
            
            gmm_base = SklearnGMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
            )
            self.model = SklearnGMMWrapper(gmm_base)
            self.backend = "sklearn"
        else:
            # Use PyTorch GMM
            from physicsnemo.core.version_check import check_version_spec

            check_version_spec("torch", "2.0.0", hard_fail=True)

            from .gmm_torch import TorchGMM

            self.model = TorchGMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                device=self.device,
                random_state=self.random_state,
            )
            self.backend = "torch"

    def _init_pce(self):
        """Initialize PCE model."""
        from .density_pce import PCEDensityModel

        self.model = PCEDensityModel(
            n_components=self.n_components,
            poly_degree=self.poly_degree,
            interaction_only=self.interaction_only,
            random_state=self.random_state,
        )
        self.backend = "pce"

    def fit(self, X: np.ndarray) -> None:
        r"""
        Fit the density model and store reference scores.

        This method trains the underlying model (GMM or PCE) on the provided feature
        array and computes anomaly scores for all training samples to establish
        a reference distribution. Automatically uses the appropriate backend based
        on the method and device settings.

        Parameters
        ----------
        X : np.ndarray
            Training feature array of shape :math:`(N, D)` where :math:`N` is
            the number of samples and :math:`D` is the feature dimensionality.

        Examples
        --------
        >>> import numpy as np
        >>> from physicsnemo.experimental.guardrails.geometry import GeometryDensityModel
        >>> 
        >>> rng = np.random.RandomState(42)
        >>> X_train = rng.randn(100, 22)
        >>> 
        >>> # GMM on CPU
        >>> model = GeometryDensityModel(method="gmm", n_components=2, device="cpu")
        >>> model.fit(X_train)
        >>> print(f"Backend: {model.backend}")
        Backend: sklearn
        >>> 
        >>> # GMM on GPU (requires PyTorch and CUDA)
        >>> model_gpu = GeometryDensityModel(method="gmm", n_components=2, device="cuda")
        >>> model_gpu.fit(X_train)
        >>> print(f"Backend: {model_gpu.backend}")
        Backend: torch
        >>> 
        >>> # PCE on CPU
        >>> model_pce = GeometryDensityModel(method="pce", n_components=10)
        >>> model_pce.fit(X_train)
        >>> print(f"Backend: {model_pce.backend}")
        Backend: pce

        Notes
        -----
        The reference scores are essential for converting raw anomaly scores to
        empirical percentiles. They represent the expected distribution of scores
        for in-distribution samples.

        **Performance**: GPU fitting (GMM only) provides speedup for N > 1000 samples.
        For smaller datasets, CPU may be faster due to transfer overhead.
        """
        self.model.fit(X)
        # Compute reference scores for training data
        self.ref_scores = self.model.score(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        r"""
        Compute anomaly scores for samples.

        Parameters
        ----------
        X : np.ndarray
            Feature array of shape :math:`(N, D)` where :math:`N` is the
            number of samples and :math:`D` is the feature dimensionality.

        Returns
        -------
        np.ndarray
            Anomaly scores of shape :math:`(N,)`. Higher scores indicate
            more anomalous samples.

        Examples
        --------
        >>> import numpy as np
        >>> from physicsnemo.experimental.guardrails.geometry import GeometryDensityModel
        >>> 
        >>> rng = np.random.RandomState(42)
        >>> X_train = rng.randn(100, 22)
        >>> X_test = rng.randn(10, 22)
        >>> 
        >>> model = GeometryDensityModel(method="gmm")
        >>> model.fit(X_train)
        >>> scores = model.score(X_test)
        >>> print(f"Score range: [{scores.min():.2f}, {scores.max():.2f}]")

        Notes
        -----
        The anomaly score depends on the method:

        - **GMM**: Negative log-likelihood :math:`-\log p(\mathbf{x} | \theta)`
        - **PCE**: Mahalanobis distance in Hermite polynomial space

        Higher scores always indicate more anomalous samples.
        """
        return self.model.score(X)

    def percentiles(self, scores: np.ndarray) -> np.ndarray:
        r"""
        Convert anomaly scores to empirical percentiles.

        This method converts raw anomaly scores to percentiles relative to
        the reference distribution established during training. Percentiles
        provide an intuitive interpretation: a percentile of 95 means the
        sample is more anomalous than 95% of the training data.

        Parameters
        ----------
        scores : np.ndarray
            Anomaly scores of shape :math:`(N,)` as returned by :meth:`score`.

        Returns
        -------
        np.ndarray
            Empirical percentiles of shape :math:`(N,)`, ranging from 0 to 100.

        Raises
        ------
        RuntimeError
            If the density model has not been fitted yet (i.e., :attr:`ref_scores`
            is ``None``).

        Examples
        --------
        >>> import numpy as np
        >>> from physicsnemo.experimental.guardrails.geometry import GeometryDensityModel
        >>> 
        >>> rng = np.random.RandomState(42)
        >>> X_train = rng.randn(100, 22)
        >>> X_test = rng.randn(10, 22)
        >>> 
        >>> model = GeometryDensityModel(method="gmm")
        >>> model.fit(X_train)
        >>> scores = model.score(X_test)
        >>> pcts = model.percentiles(scores)
        >>> print(f"Percentiles: {pcts}")

        Notes
        -----
        The percentile for score :math:`s` is computed as:

        .. math::

            \text{percentile}(s) = 100 \times \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[s_i^{\text{ref}} \leq s]

        where :math:`\{s_1^{\text{ref}}, \ldots, s_N^{\text{ref}}\}` are the
        reference scores from training data.
        """
        if self.ref_scores is None:
            raise RuntimeError(
                "Density model not fitted. Call fit() before computing percentiles."
            )

        return np.array([100.0 * np.mean(self.ref_scores <= s) for s in scores])

    def get_state(self) -> dict:
        """
        Get model state for serialization.

        Returns
        -------
        dict
            Dictionary containing all necessary state for reconstruction.
        """
        state = {
            "method": self.method,
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "poly_degree": self.poly_degree,
            "interaction_only": self.interaction_only,
            "random_state": self.random_state,
            "device": self.device,
            "backend": self.backend,
            "ref_scores": self.ref_scores,
        }

        # Serialize model-specific parameters
        if self.backend == "sklearn":
            # For sklearn GMM, save the underlying GMM parameters
            state["model_params"] = {
                "weights_": self.model.gmm.weights_,
                "means_": self.model.gmm.means_,
                "covariances_": self.model.gmm.covariances_,
                "converged_": self.model.gmm.converged_,
                "n_iter_": self.model.gmm.n_iter_,
            }
        elif self.backend == "torch":
            # For TorchGMM, convert to sklearn-compatible dict
            state["model_params"] = self.model.to_sklearn_dict()
        elif self.backend == "pce":
            # For PCE, use its get_params method
            state["model_params"] = self.model.get_params()
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

        return state

    def set_state(self, state: dict) -> None:
        """
        Restore model state from serialized data.

        Parameters
        ----------
        state : dict
            State dictionary as returned by :meth:`get_state`.
        """
        # Restore basic attributes
        self.method = state["method"]
        self.n_components = state["n_components"]
        self.covariance_type = state["covariance_type"]
        self.poly_degree = state["poly_degree"]
        self.interaction_only = state["interaction_only"]
        self.random_state = state["random_state"]
        self.device = state["device"]
        self.backend = state["backend"]
        self.ref_scores = state["ref_scores"]

        # Restore model
        if self.backend == "sklearn":
            from sklearn.mixture import GaussianMixture as SklearnGMM
            
            class SklearnGMMWrapper:
                """Wrapper to add .score() method to sklearn GMM."""
                def __init__(self, gmm):
                    self.gmm = gmm
                
                def fit(self, X):
                    return self.gmm.fit(X)
                
                def score(self, X):
                    """Return negative log-likelihood (anomaly scores)."""
                    return -self.gmm.score_samples(X)
                
                def score_samples(self, X):
                    """Return log-likelihood (for compatibility)."""
                    return self.gmm.score_samples(X)
            
            gmm_base = SklearnGMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
            )
            # Restore fitted parameters
            params = state["model_params"]
            gmm_base.weights_ = params["weights_"]
            gmm_base.means_ = params["means_"]
            gmm_base.covariances_ = params["covariances_"]
            gmm_base.converged_ = params["converged_"]
            gmm_base.n_iter_ = params["n_iter_"]
            gmm_base.precisions_cholesky_ = np.linalg.cholesky(
                np.linalg.inv(params["covariances_"])
            )
            
            self.model = SklearnGMMWrapper(gmm_base)
        elif self.backend == "torch":
            from .gmm_torch import TorchGMM
            import torch
            
            self.model = TorchGMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                device=self.device,
                random_state=self.random_state,
            )
            # Restore fitted parameters
            params = state["model_params"]
            self.model.weights_ = torch.from_numpy(params["weights_"]).float().to(self.device)
            self.model.means_ = torch.from_numpy(params["means_"]).float().to(self.device)
            self.model.covariances_ = torch.from_numpy(params["covariances_"]).float().to(self.device)
            self.model.converged_ = params["converged_"]
            self.model.n_iter_ = params["n_iter_"]
            # Recompute precisions_cholesky_
            self.model.precisions_cholesky_ = self.model._compute_precision_cholesky()
        elif self.backend == "pce":
            from .density_pce import PCEDensityModel
            
            self.model = PCEDensityModel(
                n_components=self.n_components,
                poly_degree=self.poly_degree,
                interaction_only=self.interaction_only,
                random_state=self.random_state,
            )
            self.model.set_params(state["model_params"])
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")