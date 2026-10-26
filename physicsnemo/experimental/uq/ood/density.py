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


class DensityScorer:
    r"""
    Density model for anomaly detection with multiple method options.

    This class provides a unified interface for density estimation supporting
    multiple methods:

    - GMM (Gaussian Mixture Model)
    - PCE (Polynomial Chaos Expansion)

    Parameters
    ----------
    method : str, optional
        Density estimation method. Options:
        - ``"gmm"``: Gaussian Mixture Model (default)
        - ``"pce"``: Polynomial Chaos Expansion
    gmm_components : int, optional
        For GMM only: Number of Gaussian mixture components. Default is 1.
    pce_components : int or None, optional
        For PCE only: Number of PCA components (None = auto-select to 95% variance).
        Default is None.
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
    model : object
        The fitted PyTorch density backend.
    ref_scores : torch.Tensor or None
        Reference anomaly scores from training data for percentile computation.
    device : torch.device
        Device being used for computation.
    method : str
        Density estimation method: ``"gmm"`` or ``"pce"``.

    Notes
    -----
    Features may come from model embeddings, geometry descriptors, or another
    application. Descriptor extraction and accept/reject thresholds belong to
    the caller. Fitted GMM/PCE transformations and reference scores are preserved
    in :meth:`get_state` / :meth:`set_state`. Scores are tensors;
    :meth:`percentiles` returns a NumPy array.
    """

    def __init__(
        self,
        method: str = "gmm",
        gmm_components: int = 1,
        pce_components: int | None = None,
        poly_degree: int = 2,
        interaction_only: bool = False,
        random_state: int | None = 0,
        device: str = "cpu",
    ):
        self.method = method.lower()
        self.gmm_components = gmm_components
        self.pce_components = pce_components
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
        from ._gmm import TorchGMM

        self.model = TorchGMM(
            n_components=self.gmm_components,
            device=self.device,
            random_state=self.random_state,
        )

    def _init_pce(self):
        """Initialize PCE model."""
        from ._pce import TorchPCEDensityModel

        self.model = TorchPCEDensityModel(
            n_components=self.pce_components,
            poly_degree=self.poly_degree,
            interaction_only=self.interaction_only,
            random_state=self.random_state,
            device=self.device,
        )

    @torch.no_grad()
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
        """
        # Convert to torch tensor if needed and move to device
        if isinstance(X, np.ndarray):
            X_torch = torch.from_numpy(X).float().to(self.device)
        elif isinstance(X, torch.Tensor):
            X_torch = X.detach().float().to(self.device)
        else:
            raise TypeError(f"X must be np.ndarray or torch.Tensor, got {type(X)}")

        if X_torch.ndim != 2 or X_torch.shape[0] < 2 or X_torch.shape[1] == 0:
            raise ValueError(
                "Expected at least two reference samples with shape (N, D)"
            )
        if not torch.isfinite(X_torch).all():
            raise ValueError("Reference features must be finite")
        # Backend fitting mutates its parameters. If it fails, the old reference
        # scores must not remain paired with a partially replaced model.
        self.ref_scores = None
        self.model.fit(X_torch)
        # Compute reference scores for training data (returns torch tensor)
        scores = self.model.score(X_torch)
        # Store as tensor on device
        if not torch.isfinite(scores).all():
            raise RuntimeError("Density fitting produced non-finite reference scores")
        self.ref_scores = scores

    @torch.no_grad()
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
            more anomalous samples. Returns tensor on the configured scorer device.
        """
        # Convert to torch tensor if needed and move to device
        if isinstance(X, np.ndarray):
            X_torch = torch.from_numpy(X).float().to(self.device)
        elif isinstance(X, torch.Tensor):
            X_torch = X.detach().float().to(self.device)
        else:
            raise TypeError(f"X must be np.ndarray or torch.Tensor, got {type(X)}")

        if self.ref_scores is None:
            raise RuntimeError("Call fit() before score()")
        if X_torch.ndim != 2 or not torch.isfinite(X_torch).all():
            raise ValueError("Expected finite features with shape (N, D)")
        # The old geometry wrapper enforced a fixed descriptor schema. A generic
        # scorer must reject width mismatches itself: a singleton feature axis
        # would otherwise broadcast against the fitted means/standardization.
        expected_features = self.model.n_features_in_
        if X_torch.shape[1] != expected_features:
            raise ValueError(
                f"Query feature dimension must match the fitted reference ({expected_features})"
            )
        scores = self.model.score(X_torch)
        if not torch.isfinite(scores).all():
            raise RuntimeError(
                "Density scoring produced non-finite values; check feature scaling and fitted state"
            )
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
            raise TypeError(
                f"scores must be np.ndarray or torch.Tensor, got {type(scores)}"
            )

        if scores_torch.ndim != 1 or self.ref_scores.ndim != 1:
            raise ValueError("Percentiles require one-dimensional score vectors")

        # Ensure ref_scores is on same device (always torch.Tensor after fit)
        ref_scores_torch = self.ref_scores.to(self.device)

        # Validate ref_scores is not empty
        if len(ref_scores_torch) == 0:
            raise RuntimeError(
                "Reference scores are empty. Model may not have been fitted correctly."
            )

        if (
            not torch.isfinite(scores_torch).all()
            or not torch.isfinite(ref_scores_torch).all()
        ):
            raise ValueError("Percentiles require finite query and reference scores")

        # Compute percentiles using broadcasting (efficient on GPU)
        scores_expanded = scores_torch.unsqueeze(1)  # (n_scores, 1)
        ref_expanded = ref_scores_torch.unsqueeze(0)  # (1, n_ref)
        percentiles = (
            100.0
            * (ref_expanded <= scores_expanded).sum(dim=1).float()
            / len(ref_scores_torch)
        )

        # Return as numpy array
        return percentiles.cpu().numpy()

    def get_state(self) -> dict:
        """
        Get model state for serialization.

        Returns
        -------
        dict
            Dictionary containing all necessary state for reconstruction.
            Method-specific parameters are specified with method prefix (gmm_ or pce_).
        """
        if self.ref_scores is None:
            raise RuntimeError("Call fit() before serializing the scorer")
        state = {
            "method": self.method,
            "gmm_components": self.gmm_components,
            "pce_components": self.pce_components,
            "poly_degree": self.poly_degree,
            "interaction_only": self.interaction_only,
            "random_state": self.random_state,
            "ref_scores": self.ref_scores.cpu().numpy(),
        }

        # Flatten method-specific parameters with method prefix to avoid nested dicts
        model_state = self.model.get_state()
        prefix = f"{self.method}_"
        for key, value in model_state.items():
            state[f"{prefix}{key}"] = value

        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in state.items()
        }

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
        # Own restored arrays even when torch.from_numpy can share CPU storage.
        state = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in state.items()
        }
        # Build a complete replacement before touching this fitted scorer. A bad
        # checkpoint must not destroy an otherwise usable model.
        candidate = DensityScorer(
            method=state["method"],
            gmm_components=state["gmm_components"],
            pce_components=state.get("pce_components"),
            poly_degree=state["poly_degree"],
            interaction_only=state["interaction_only"],
            random_state=state["random_state"],
            device=device,
        )
        reference = torch.from_numpy(state["ref_scores"]).float().to(candidate.device)
        if (
            reference.ndim != 1
            or not reference.numel()
            or not torch.isfinite(reference).all()
        ):
            raise ValueError("Saved reference scores must be a nonempty finite vector")
        prefix = f"{candidate.method}_"
        model_state = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
            and key not in ("gmm_components", "pce_components")
        }
        candidate.model.set_state(model_state, device=candidate.device)
        candidate.ref_scores = reference
        self.__dict__.update(candidate.__dict__)
