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

Both methods use PyTorch and support CPU and GPU acceleration.
"""

from __future__ import annotations

import numpy as np
import torch

class GeometryDensityModel:
    r"""
    Density model for anomaly detection with multiple method options.

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
        For GMM only: Covariance type. Currently only ``"full"`` is supported.
        Default is ``"full"``. This parameter is stored for serialization but
        TorchGMM always uses full covariance matrices.
    poly_degree : int, optional
        For PCE only: Polynomial degree for expansion. Default is 2.
    interaction_only : bool, optional
        For PCE only: If True, only include interaction terms. Default is False.
    random_state : int or None, optional
        Random seed for reproducible initialization. Default is 0.
    device : str or torch.device, optional
        Device to use for computation. Options:
        - ``"cpu"``: Use PyTorch on CPU (default)
        - ``"cuda"``: Use PyTorch on GPU (both GMM and PCE supported)
        - ``"cuda:0"``, ``"cuda:1"``, etc.: Specific GPU device
        Default is ``"cpu"``. Both GMM and PCE support GPU acceleration.

    Attributes
    ----------
    model : TorchGMM or PCEDensityModel
        The underlying density estimation model (both use PyTorch).
    ref_scores : torch.Tensor or None
        Reference anomaly scores from training data for percentile computation.
        Stored on the same device as the model (GPU-first resident).
    device : torch.device
        Device being used for computation.
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
        self.ref_scores = None

        # Validate method
        if self.method not in ["gmm", "pce"]:
            raise ValueError(f"method must be 'gmm' or 'pce', got '{self.method}'")

        # Parse device
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Initialize model based on method
        if self.method == "gmm":
            self._init_gmm()
        elif self.method == "pce":
            self._init_pce()

    def _init_gmm(self):
        """Initialize GMM model using PyTorch (works on both CPU and GPU)."""
        from .gmm_torch import TorchGMM

        self.model = TorchGMM(
            n_components=self.n_components,
            device=self.device,
            random_state=self.random_state,
        )

    def _init_pce(self):
        """Initialize PCE model."""
        from .density_pce import PCEDensityModel

        self.model = PCEDensityModel(
            n_components=self.n_components,
            poly_degree=self.poly_degree,
            interaction_only=self.interaction_only,
            random_state=self.random_state,
            device=self.device,
        )

    def fit(self, X: np.ndarray | torch.Tensor) -> None:
        r"""
        Fit the density model and store reference scores.

        This method trains the underlying model (GMM or PCE) on the provided feature
        array and computes anomaly scores for all training samples to establish
        a reference distribution.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Training feature array of shape :math:`(N, D)` where :math:`N` is
            the number of samples and :math:`D` is the feature dimensionality.
            If numpy array, will be converted to torch tensor and moved to device.

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
        >>> print(f"Method: {model.method}")
        Method: gmm
        >>> 
        >>> # GMM on GPU
        >>> model_gpu = GeometryDensityModel(method="gmm", n_components=2, device="cuda")
        >>> model_gpu.fit(X_train)
        >>> 
        >>> model_pce = GeometryDensityModel(method="pce", n_components=10, device="cuda")
        >>> model_pce.fit(X_train)

        Notes
        -----
        The reference scores are essential for converting raw anomaly scores to
        empirical percentiles. They represent the expected distribution of scores
        for in-distribution samples.
        """
        # Convert to torch tensor if needed and move to device
        if isinstance(X, np.ndarray):
            X_torch = torch.from_numpy(X).float().to(self.device)
        elif isinstance(X, torch.Tensor):
            X_torch = X.float().to(self.device)
        else:
            raise TypeError(f"X must be np.ndarray or torch.Tensor, got {type(X)}")
        
        self.model.fit(X_torch)
        # Compute reference scores for training data (returns torch tensor)
        scores = self.model.score(X_torch)
        # Store as tensor on device (GPU-first resident)
        self.ref_scores = scores  # Already a torch tensor on device

    def score(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        r"""
        Compute anomaly scores for samples.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Feature array of shape :math:`(N, D)` where :math:`N` is the
            number of samples and :math:`D` is the feature dimensionality.
            If numpy array, will be converted to torch tensor and moved to device.

        Returns
        -------
        torch.Tensor
            Anomaly scores of shape :math:`(N,)`. Higher scores indicate
            more anomalous samples. Returns tensor on the same device as input (GPU-first resident).

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
        # Convert to torch tensor if needed and move to device
        if isinstance(X, np.ndarray):
            X_torch = torch.from_numpy(X).float().to(self.device)
        elif isinstance(X, torch.Tensor):
            X_torch = X.float().to(self.device)
        else:
            raise TypeError(f"X must be np.ndarray or torch.Tensor, got {type(X)}")
        
        scores = self.model.score(X_torch)
        # Keep as tensor on device (GPU-first resident)
        return scores

    def percentiles(self, scores: np.ndarray | torch.Tensor) -> np.ndarray:
        r"""
        Convert anomaly scores to empirical percentiles.

        This method converts raw anomaly scores to percentiles relative to
        the reference distribution established during training. Percentiles
        provide an intuitive interpretation: a percentile of 95 means the
        sample is more anomalous than 95% of the training data.

        Parameters
        ----------
        scores : np.ndarray or torch.Tensor
            Anomaly scores of shape :math:`(N,)` as returned by :meth:`score`.
            If torch tensor, will be moved to same device as ref_scores.

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

        # Convert to torch tensor if needed
        if isinstance(scores, np.ndarray):
            scores_torch = torch.from_numpy(scores).to(self.device)
        elif isinstance(scores, torch.Tensor):
            scores_torch = scores.to(self.device)
        else:
            raise TypeError(f"scores must be np.ndarray or torch.Tensor, got {type(scores)}")
        
        # Ensure ref_scores is on same device (always torch.Tensor after fit)
        ref_scores_torch = self.ref_scores.to(self.device)
        
        # Validate ref_scores is not empty
        if len(ref_scores_torch) == 0:
            raise RuntimeError("Reference scores are empty. Model may not have been fitted correctly.")
        
        # Compute percentiles using broadcasting (efficient on GPU)
        scores_expanded = scores_torch.unsqueeze(1)  # (n_scores, 1)
        ref_expanded = ref_scores_torch.unsqueeze(0)  # (1, n_ref)
        percentiles = 100.0 * (ref_expanded <= scores_expanded).sum(dim=1).float() / len(ref_scores_torch)
        
        # Return as numpy array
        return percentiles.cpu().numpy()

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
            "ref_scores": self.ref_scores.cpu().numpy(),
        }

        # Serialize model-specific parameters based on method
        if self.method == "gmm":
            # For GMM (TorchGMM), convert to sklearn-compatible dict
            state["model_params"] = self.model.to_sklearn_dict()
        elif self.method == "pce":
            # For PCE, use its get_state method
            state["model_params"] = self.model.get_state()
        else:
            raise RuntimeError(f"Unknown method: {self.method}")

        return state

    def set_state(self, state: dict, device: str | torch.device) -> None:
        """
        Restore model state from serialized data.

        Parameters
        ----------
        state : dict
            State dictionary as returned by :meth:`get_state`.
        device : str or torch.device
            Device to load the model on.
        """
        # Restore basic attributes
        self.method = state["method"]
        self.n_components = state["n_components"]
        self.covariance_type = state["covariance_type"]
        self.poly_degree = state["poly_degree"]
        self.interaction_only = state["interaction_only"]
        self.random_state = state["random_state"]
        
        # Set device (runtime parameter, not part of model state)
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        # Convert ref_scores back to torch tensor on device
        ref_scores_data = state["ref_scores"]
        self.ref_scores = torch.from_numpy(ref_scores_data).float().to(self.device)

        # Restore model based on method
        if self.method == "gmm":
            from .gmm_torch import TorchGMM
            
            self.model = TorchGMM(
                n_components=self.n_components,
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
        elif self.method == "pce":
            from .density_pce import PCEDensityModel
            
            self.model = PCEDensityModel(
                n_components=self.n_components,
                poly_degree=self.poly_degree,
                interaction_only=self.interaction_only,
                random_state=self.random_state,
                device=self.device,
            )
            self.model.set_state(state["model_params"], device=self.device)
        else:
            raise RuntimeError(f"Unknown method: {self.method}")