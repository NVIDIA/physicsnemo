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

from pathlib import Path

import numpy as np
import torch

from physicsnemo.core.version_check import check_version_spec
from physicsnemo.mesh import Mesh as PhysicsNeMoMesh

# Check for required dependencies
check_version_spec("pyvista", "0.40.0", hard_fail=True)

from .density_model import GeometryDensityModel
from .feature_extraction import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    extract_features,
    feature_hash,
)
from .mesh_io import load_features_from_dir
from .feature_schema import FeatureSchema


class GeometryGuardrail:
    r"""
    Geometry out-of-distribution guardrail based on density estimation.

    This class provides a complete pipeline for detecting anomalous geometric
    configurations. It extracts non-invariant geometric features, fits a
    probabilistic density model, and classifies new geometries as OK, WARN,
    or REJECT based on configurable percentile thresholds.

    Supports multiple density estimation methods: Gaussian Mixture Model (GMM) and
    Polynomial Chaos Expansion (PCE).

    Parameters
    ----------
    method : str, optional
        Density estimation method. Options:
        - ``"gmm"``: Gaussian Mixture Model
        - ``"pce"``: Polynomial Chaos Expansion
        Default is ``"gmm"``.
    gmm_components : int, optional
        For GMM only: Number of Gaussian mixture components (1=unimodal, >1=multimodal).
        Default is 1.
    pce_components : int or None, optional
        For PCE only: Number of PCA components (None=auto-select to 95% variance).
        Default is None.
    warn_pct : float, optional
        Percentile threshold for issuing warnings. Geometries with anomaly scores
        above this percentile will be flagged as WARN. Must be in range [0, 100].
        Default is 99.0.
    reject_pct : float, optional
        Percentile threshold for rejection. Geometries with anomaly scores above
        this percentile will be flagged as REJECT. Must be in range [0, 100] and
        should be >= ``warn_pct``. Default is 99.9.
    poly_degree : int, optional
        For PCE only: Polynomial degree for expansion (1=linear, 2=quadratic, etc.).
        Default is 2.
    interaction_only : bool, optional
        For PCE only: If True, only include interaction terms (no pure higher powers).
        Default is False.
    random_state : int or None, optional
        Random seed for reproducible initialization. If None, uses non-deterministic
        behavior. Default is None.
    device : str or torch.device, optional
        Device to use for density model computations and feature tensors. Options:
        - ``"cpu"``: Use CPU (default)
        - ``"cuda"``: Use GPU
        - ``"cuda:0"``, ``"cuda:1"``, etc.: Specific GPU device
        Default is ``"cpu"``.
        Mesh operations are always performed on CPU in worker processes, and only the resulting
        feature tensors are moved to the specified device.

    Attributes
    ----------
    warn_pct : float
        Warning percentile threshold.
    reject_pct : float
        Rejection percentile threshold.
    density : GeometryDensityModel
        Underlying density estimation model.
    feature_names : list[str]
        List of feature names used by this guardrail.
    feature_version : str
        Feature schema version identifier.
    feature_hash : str
        Cryptographic hash of the feature schema for compatibility checking.
    device : str or torch.device
        Device being used for all computations.

    Examples
    --------
    CPU-based (default):

    >>> import pyvista as pv
    >>> from pathlib import Path
    >>> from physicsnemo.mesh.io import from_pyvista
    >>> from physicsnemo.experimental.guardrails import GeometryGuardrail
    >>> 
    >>> # Create and fit guardrail from training meshes (CPU)
    >>> train_meshes = [from_pyvista(pv.Cube()) for _ in range(100)]
    >>> guardrail = GeometryGuardrail(gmm_components=1, device="cpu", random_state=42)
    >>> guardrail.fit(train_meshes)
    >>> 
    >>> # Query new geometries
    >>> test_meshes = [from_pyvista(pv.Sphere()), from_pyvista(pv.Cylinder())]
    >>> results = guardrail.query(test_meshes)
    >>> len(results)
    2
    >>> results[0]["status"] in ["OK", "WARN", "REJECT"]
    True
    >>> 0 <= results[0]["percentile"] <= 100
    True

    Notes
    -----
    **Feature Extraction**:

    The guardrail extracts 22 geometric features from each mesh, including:
    - Centroid position (3D)
    - Principal component axes and eigenvalues
    - Bounding box extents
    - Second moments of inertia
    - Total and projected surface areas

    These features are intentionally **not** invariant to translation, rotation,
    or scale. This allows the guardrail to detect geometries that differ in
    absolute position, orientation, or size from the training distribution.

    **Density Modeling**:

    The guardrail uses Gaussian Mixture Models (GMMs) to learn a probabilistic
    density :math:`p(\mathbf{x})` over the feature space. For a new geometry with
    features :math:`\mathbf{x}`, the anomaly score is:

    .. math::

        s(\mathbf{x}) = -\log p(\mathbf{x} | \theta)

    where :math:`\theta` are the fitted GMM parameters. Higher scores indicate
    lower likelihood (more anomalous).

    **Classification Logic**:

    Given anomaly score :math:`s` and its percentile :math:`p` relative to the
    training distribution:

    - **OK**: :math:`p < \text{warn\_pct}` (typical geometry)
    - **WARN**: :math:`\text{warn\_pct} \leq p < \text{reject\_pct}` (unusual geometry)
    - **REJECT**: :math:`p \geq \text{reject\_pct}` (highly anomalous geometry)

    See Also
    --------
    :class:`GeometryDensityModel` : Underlying density estimation model.
    :func:`extract_features` : Feature extraction function.
    :class:`FeatureSchema` : Feature schema definition and validation.
    """

    def __init__(
        self,
        method: str = "gmm",
        gmm_components: int = 1,
        pce_components: int | None = None,
        warn_pct: float = 99.0,
        reject_pct: float = 99.9,
        poly_degree: int = 2,
        interaction_only: bool = False,
        random_state: int | None = 0,
        device: str = "cpu",
    ):
        # Validate thresholds
        if not 0 <= warn_pct <= 100:
            raise ValueError(f"warn_pct must be in [0, 100], got {warn_pct}")
        if not 0 <= reject_pct <= 100:
            raise ValueError(f"reject_pct must be in [0, 100], got {reject_pct}")
        if warn_pct > reject_pct:
            raise ValueError(
                f"warn_pct ({warn_pct}) must be <= reject_pct ({reject_pct})"
            )

        self.warn_pct = warn_pct
        self.reject_pct = reject_pct
        
        # Parse device
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        # Store method parameters for serialization
        self.method = method
        self.gmm_components = gmm_components
        self.pce_components = pce_components
        self.poly_degree = poly_degree
        self.interaction_only = interaction_only
        self.random_state = random_state

        self.density = GeometryDensityModel(
            method=method,
            gmm_components=gmm_components,
            pce_components=pce_components,
            poly_degree=poly_degree,
            interaction_only=interaction_only,
            random_state=random_state,
            device=self.device,
        )

        self.feature_names = FEATURE_NAMES
        self.feature_version = FEATURE_VERSION
        self.feature_hash = feature_hash(FEATURE_NAMES)

    # -------------------------------------------------------------------------
    # Fitting
    # -------------------------------------------------------------------------

    def fit(self, meshes: list[PhysicsNeMoMesh]) -> None:
        r"""
        Fit guardrail from a list of physicsnemo.mesh.Mesh objects.

        This method extracts features from all provided meshes and trains the
        density model to learn the distribution of in-distribution geometries.

        Parameters
        ----------
        meshes : list[physicsnemo.mesh.Mesh]
            List of training meshes representing the in-distribution geometry space.
            All meshes must pass validation checks.

        Raises
        ------
        ValueError
            If any mesh fails validation (see :func:`validate_mesh`).
        ValueError
            If feature extraction fails for any mesh.
        """
        # Validate input
        if not meshes:
            raise ValueError("Cannot fit on empty list of meshes")
        
        # Move meshes to device and extract features as tensors
        features_list = []
        
        for mesh in meshes:
            # Move mesh to device if not already there
            mesh_on_device = mesh.to(self.device)
            # Extract features as tensor (stays on device)
            feat = extract_features(mesh_on_device, return_tensor=True)
            features_list.append(feat)
        
        # Stack features into tensor
        X = torch.stack(features_list)  # Shape: (N, 22)
        
        # Validate feature array (convert to numpy for validation)
        X_np = X.cpu().numpy()
        FeatureSchema.validate_array(X_np)
        
        # Fit density model (accepts torch tensor)
        self.density.fit(X)

    def fit_from_dir(self, stl_dir: Path, **loader_kwargs) -> None:
        r"""
        Fit guardrail from a directory of STL files.

        This method provides a convenient interface for training on large
        datasets stored as STL files. It uses parallel processing for efficiency.

        Parameters
        ----------
        stl_dir : Path
            Directory containing STL files to use as training data. Only files
            with ``.stl`` extension are processed.
        **loader_kwargs
            Additional keyword arguments passed to :func:`load_features_from_dir`.
            Common options include ``n_workers``.

        Raises
        ------
        RuntimeError
            If no valid STL files are found in the directory.

        Notes
        -----
        This method is equivalent to:

        .. code-block:: python

            features, _ = load_features_from_dir(stl_dir, **loader_kwargs)
            guardrail.density.fit(np.vstack(features))

        Invalid or corrupted STL files are automatically skipped with a warning.

        See Also
        --------
        :func:`load_features_from_dir` : Parallel STL loading and feature extraction.
        """
        # Load features (always on CPU in workers)
        feats, _ = load_features_from_dir(stl_dir, **loader_kwargs)
        
        # Validate that we have features
        if not feats:
            raise ValueError("No valid features extracted from STL files")
        
        # Convert to torch tensor and move to device
        X = torch.from_numpy(np.vstack(feats)).float().to(self.device)
        self.density.fit(X)

    # -------------------------------------------------------------------------
    # Querying
    # -------------------------------------------------------------------------

    def query(self, meshes: list[PhysicsNeMoMesh]) -> list[dict]:
        r"""
        Query guardrail for a list of meshes.

        This method extracts features from the provided meshes, computes anomaly
        scores, converts them to percentiles, and classifies each geometry as
        OK, WARN, or REJECT.

        Parameters
        ----------
        meshes : list[physicsnemo.mesh.Mesh]
            List of query meshes to evaluate.

        Returns
        -------
        list[dict]
            List of result dictionaries, one per input mesh. Each dictionary contains:
            - ``"percentile"`` (float): Empirical percentile relative to training data
            - ``"status"`` (str): Classification as ``"OK"``, ``"WARN"``, or ``"REJECT"``

        Raises
        ------
        ValueError
            If any mesh fails validation (see :func:`validate_mesh`).
        RuntimeError
            If the guardrail has not been fitted yet.

        Notes
        -----
        The query process:

        1. Extract features :math:`\mathbf{x}_i` from each mesh
        2. Compute anomaly scores :math:`s_i = -\log p(\mathbf{x}_i | \theta)`
        3. Convert to percentiles :math:`p_i` relative to training distribution
        4. Classify based on thresholds

        See Also
        --------
        :meth:`query_from_dir` : Query geometries from a directory of STL files.
        """
        # Validate input
        if not meshes:
            raise ValueError("Cannot query empty list of meshes")
        
        # Move meshes to device and extract features as tensors
        features_list = []
        
        for mesh in meshes:
            # Move mesh to device if not already there
            mesh_on_device = mesh.to(self.device)
            # Extract features as tensor (stays on device)
            feat = extract_features(mesh_on_device, return_tensor=True)
            features_list.append(feat)
        
        # Stack features into tensor
        X = torch.stack(features_list)  # Shape: (N, 22)
        
        # Validate feature array (convert to numpy for validation)
        X_np = X.cpu().numpy()
        FeatureSchema.validate_array(X_np)
        
        # Compute scores and percentiles
        scores = self.density.score(X)
        pcts = self.density.percentiles(scores)

        # Classify each geometry
        return [
            {
                "percentile": float(p),
                "status": self._classify(p),
            }
            for p in pcts
        ]

    def query_from_dir(self, stl_dir: Path, **loader_kwargs) -> list[dict]:
        r"""
        Query guardrail for all STL files in a directory.

        This method provides a convenient interface for evaluating large datasets
        stored as STL files. It uses parallel processing for efficiency.

        Parameters
        ----------
        stl_dir : Path
            Directory containing STL files to query. Only files with ``.stl``
            extension are processed.
        **loader_kwargs
            Additional keyword arguments passed to :func:`load_features_from_dir`.
            Common options include ``n_workers``.

        Returns
        -------
        list[dict]
            List of result dictionaries, one per valid STL file. Each dictionary contains:
            - ``"name"`` (str): Filename of the STL file
            - ``"percentile"`` (float): Empirical percentile relative to training data
            - ``"status"`` (str): Classification as ``"OK"``, ``"WARN"``, or ``"REJECT"``

        Raises
        ------
        RuntimeError
            If no valid STL files are found in the directory.
        RuntimeError
            If the guardrail has not been fitted yet.

        Notes
        -----
        Invalid or corrupted STL files are automatically skipped with a warning.

        See Also
        --------
        :func:`load_features_from_dir` : Parallel STL loading and feature extraction.
        :meth:`query` : Query individual mesh objects.
        """
        # Load features (always on CPU in workers)
        feats, names = load_features_from_dir(stl_dir, **loader_kwargs)
        
        # Validate that we have features
        if not feats:
            raise ValueError("No valid features extracted from STL files")
        
        # Convert to torch tensor and move to device
        X = torch.from_numpy(np.vstack(feats)).float().to(self.device)

        # Compute scores and percentiles
        scores = self.density.score(X)
        pcts = self.density.percentiles(scores)

        # Classify each geometry
        return [
            {
                "name": name,
                "percentile": float(p),
                "status": self._classify(p),
            }
            for name, p in zip(names, pcts)
        ]

    def _classify(self, pct: float) -> str:
        """
        Classify a percentile value into OK/WARN/REJECT categories.

        Parameters
        ----------
        pct : float
            Percentile value in range [0, 100].

        Returns
        -------
        str
            Classification: "REJECT", "WARN", or "OK".
        """
        if pct >= self.reject_pct:
            return "REJECT"
        elif pct >= self.warn_pct:
            return "WARN"
        return "OK"

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self, path: Path) -> None:
        r"""
        Serialize the fitted guardrail to disk.

        This method saves all necessary state to a compressed NumPy archive,
        including the fitted GMM, reference scores, thresholds, and feature
        schema metadata for compatibility checking.

        Parameters
        ----------
        path : Path
            Output file path. Conventionally uses ``.npz`` extension.

        Raises
        ------
        RuntimeError
            If the guardrail has not been fitted yet.

        Notes
        -----
        The saved file contains:

        - ``density_state``: Fitted density model state (GMM or PCE)
        - ``ref_scores``: Reference anomaly scores from training data
        - ``warn_pct``: Warning percentile threshold
        - ``reject_pct``: Rejection percentile threshold
        - ``feature_names``: List of feature names
        - ``feature_version``: Feature schema version
        - ``feature_hash``: Cryptographic hash of feature schema

        The feature metadata enables compatibility checking when loading to
        ensure the saved model uses the same feature extraction as the current
        code version.

        See Also
        --------
        :meth:`load` : Load a saved guardrail from disk.
        """
        if self.density.ref_scores is None:
            raise RuntimeError("Guardrail not fitted. Call fit() before saving.")

        # Get density model state
        density_state = self.density.get_state()

        # Filter out None values to avoid object arrays that require pickle
        # None values will be restored as None during loading
        density_state_clean = {k: v for k, v in density_state.items() if v is not None}

        # Unpack density_state directly into npz to avoid nested dicts and pickle
        # This saves each key-value pair as a separate array in the npz file
        # Save feature_names as Unicode string array (not object array) to avoid pickle
        np.savez(
            path,
            **density_state_clean,  # Unpack all density state keys (None values excluded)
            warn_pct=self.warn_pct,
            reject_pct=self.reject_pct,
            feature_names=np.array(self.feature_names, dtype='U100'),  # Unicode string array, no pickle needed
            feature_version=self.feature_version,
            feature_hash=self.feature_hash,
        )

    @classmethod
    def load(cls, path: Path, device: str | torch.device = "cpu") -> "GeometryGuardrail":
        r"""
        Load a serialized guardrail from disk.

        This class method reconstructs a fitted guardrail from a saved file,
        with automatic compatibility checking to ensure feature schema consistency.

        Parameters
        ----------
        path : Path
            Path to the saved guardrail file (typically ``.npz`` extension).
        device : str, optional
            Device to use for loaded model. Options:
            - ``"cpu"``: Use CPU (default)
            - ``"cuda"``: Use GPU (requires PyTorch)
            - ``"cuda:0"``, ``"cuda:1"``, etc.: Specific GPU device
            Default is ``"cpu"``.

        Returns
        -------
        GeometryGuardrail
            Loaded guardrail instance, ready for querying.

        Raises
        ------
        RuntimeError
            If the feature version does not match the current code version.
        RuntimeError
            If the feature names do not match the current schema.
        RuntimeError
            If the feature hash does not match (indicates schema modification).

        Notes
        -----
        **Compatibility Checking**:

        The loading process performs three levels of schema validation:

        1. **Version check**: Ensures ``feature_version`` matches current code
        2. **Name check**: Ensures ``feature_names`` list is identical
        3. **Hash check**: Ensures cryptographic hash matches (detects tampering)

        If any check fails, a :exc:`RuntimeError` is raised with a descriptive
        error message. This prevents silent failures from schema mismatches.

        **Device Selection**:

        The saved model does not store device information. You can load the same
        model on different devices as needed. This is useful for:

        - Training on GPU, deploying on CPU
        - Sharing models across different hardware configurations

        See Also
        --------
        :meth:`save` : Save a fitted guardrail to disk.
        """
        data = np.load(path, allow_pickle=False)

        # Check feature version compatibility
        if data["feature_version"] != FEATURE_VERSION:
            raise RuntimeError(
                f"Feature version mismatch: saved model uses {data['feature_version']}, "
                f"but current code expects {FEATURE_VERSION}"
            )

        # Check feature names match
        # Normalize to str to handle NumPy str_ objects
        loaded_feature_names = [str(name) for name in data["feature_names"]]
        if loaded_feature_names != FEATURE_NAMES:
            raise RuntimeError(
                f"Feature schema mismatch: saved model uses different feature names"
            )

        # Check feature hash for additional safety
        if data["feature_hash"] != feature_hash(FEATURE_NAMES):
            raise RuntimeError(
                f"Feature hash mismatch: saved model may have been corrupted or "
                f"uses a different feature extraction implementation"
            )

        # Reconstruct density state dict from flattened npz keys
        # All density state keys are saved directly in the npz file
        # Required base keys (must be present)
        required_keys = ["method", "ref_scores"]
        # Optional base keys (may be None and excluded during save)
        optional_keys = ["gmm_components", "pce_components", "poly_degree", "interaction_only", "random_state"]
        
        # Extract method first (required)
        if "method" not in data:
            raise RuntimeError("Missing required key 'method' in saved file")
        method = str(data["method"])
        prefix = f"{method}_"
        
        # Reconstruct density_state dict
        density_state = {}
        # Add required keys (must be present)
        for key in required_keys:
            if key not in data:
                raise RuntimeError(f"Missing required key '{key}' in saved file")
            value = data[key]
            # Handle scalar values that might be numpy scalars
            if hasattr(value, 'item') and value.ndim == 0:
                value = value.item()
            density_state[key] = value
        
        # Add optional keys (missing keys were None and excluded during save)
        for key in optional_keys:
            if key in data:
                value = data[key]
                # Handle scalar values that might be numpy scalars
                if hasattr(value, 'item') and value.ndim == 0:
                    value = value.item()
                density_state[key] = value
            else:
                # Key was missing, it was None and excluded during save
                density_state[key] = None
        
        # Add method-prefixed keys (all keys starting with "gmm_" or "pce_")
        # Missing keys in PCE state (like pce_poly_mean_ if None) are handled in set_state
        for key in list(data.keys()):  # Convert to list to avoid modification during iteration
            if key.startswith(prefix):
                value = data[key]
                # Handle scalar values that might be numpy scalars
                if hasattr(value, 'item') and value.ndim == 0:
                    value = value.item()
                density_state[key] = value
        
        # Extract parameters for guardrail constructor
        # Handle None values that were excluded during saving
        method = str(density_state["method"])
        gmm_components_raw = density_state.get("gmm_components", 1)
        gmm_components = 1 if gmm_components_raw is None else int(gmm_components_raw)
        pce_components_raw = density_state.get("pce_components", None)
        pce_components = None if pce_components_raw is None else int(pce_components_raw)
        poly_degree_raw = density_state.get("poly_degree", 2)
        poly_degree = 2 if poly_degree_raw is None else int(poly_degree_raw)
        interaction_only_raw = density_state.get("interaction_only", False)
        interaction_only = False if interaction_only_raw is None else bool(interaction_only_raw)
        random_state = density_state.get("random_state", 0)
        if random_state is not None:
            random_state = int(random_state)
        
        obj = cls(
            method=method,
            gmm_components=gmm_components,
            pce_components=pce_components,
            warn_pct=float(data["warn_pct"]),
            reject_pct=float(data["reject_pct"]),
            poly_degree=poly_degree,
            interaction_only=interaction_only,
            random_state=random_state,
            device=device,
        )

        # Restore density model state (now flattened, no nested dicts)
        obj.density.set_state(density_state, device=obj.device)

        return obj
