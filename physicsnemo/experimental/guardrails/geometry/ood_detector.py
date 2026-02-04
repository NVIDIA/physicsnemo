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
Geometry out-of-distribution detection guardrail.

This module provides the main user-facing API for detecting anomalous
geometric configurations using density-based methods. The guardrail learns
the distribution of in-distribution geometries and flags novel or unusual
shapes at inference time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from physicsnemo.core.version_check import check_version_spec

# Check for required dependencies
check_version_spec("trimesh", "3.0.0", hard_fail=True)
check_version_spec("scikit-learn", "1.0.0", hard_fail=True)

import trimesh

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

    Supports multiple density estimation methods (GMM, PCE) and both CPU and
    GPU acceleration for improved performance on large datasets.

    Parameters
    ----------
    method : str, optional
        Density estimation method. Options:
        - ``"gmm"``: Gaussian Mixture Model (default, flexible, multi-modal)
        - ``"pce"``: Polynomial Chaos Expansion (physics-informed, correlated features)
        Default is ``"gmm"``.
    n_components : int, optional
        For GMM: Number of Gaussian components (1=unimodal, >1=multimodal).
        For PCE: Number of PCA components (None=auto-select to 95% variance).
        Default is 1.
    warn_pct : float, optional
        Percentile threshold for issuing warnings. Geometries with anomaly scores
        above this percentile will be flagged as WARN. Must be in range [0, 100].
        Default is 95.0.
    reject_pct : float, optional
        Percentile threshold for rejection. Geometries with anomaly scores above
        this percentile will be flagged as REJECT. Must be in range [0, 100] and
        should be >= ``warn_pct``. Default is 99.0.
    covariance_type : str, optional
        For GMM only: Type of covariance matrix. Options: ``"full"``, ``"tied"``,
        ``"diag"``, ``"spherical"``. Default is ``"full"``.
    poly_degree : int, optional
        For PCE only: Polynomial degree for expansion (1=linear, 2=quadratic, etc.).
        Default is 2.
    interaction_only : bool, optional
        For PCE only: If True, only include interaction terms (no pure higher powers).
        Default is False.
    random_state : int or None, optional
        Random seed for reproducible initialization. Use ``None`` for
        non-deterministic behavior. Default is 0.
    device : str, optional
        Device to use for density model computation. Options:
        - ``"cpu"``: Use scikit-learn on CPU (default, always available)
        - ``"cuda"``: Use PyTorch GMM on GPU (requires PyTorch and CUDA, GMM only)
        - ``"cuda:0"``, ``"cuda:1"``, etc.: Specific GPU device
        Default is ``"cpu"``.

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
    device : str
        Device being used for density model computation.

    Examples
    --------
    CPU-based (default):

    >>> import trimesh
    >>> from pathlib import Path
    >>> from physicsnemo.experimental.guardrails import GeometryGuardrail
    >>> 
    >>> # Create and fit guardrail from training meshes (CPU)
    >>> train_meshes = [trimesh.creation.box() for _ in range(100)]
    >>> guardrail = GeometryGuardrail(n_components=1, device="cpu")
    >>> guardrail.fit(train_meshes)
    >>> 
    >>> # Query new geometries
    >>> test_meshes = [trimesh.creation.sphere(), trimesh.creation.cylinder()]
    >>> results = guardrail.query(test_meshes)
    >>> for res in results:
    ...     print(f"Status: {res['status']}, Percentile: {res['percentile']:.1f}")

    GPU-accelerated:

    >>> # Use GPU for faster inference on large batches
    >>> guardrail_gpu = GeometryGuardrail(
    ...     n_components=2,
    ...     warn_pct=95.0,
    ...     reject_pct=99.0,
    ...     device="cuda"  # Requires PyTorch and CUDA
    ... )
    >>> guardrail_gpu.fit(train_meshes)
    >>> 
    >>> # Fast batch inference on GPU
    >>> results_gpu = guardrail_gpu.query(test_meshes)

    Saving and loading:

    >>> # Save fitted guardrail (device info is NOT saved, specify on load)
    >>> guardrail.save(Path("guardrail.npz"))
    >>> 
    >>> # Load and specify device
    >>> loaded = GeometryGuardrail.load(Path("guardrail.npz"), device="cuda")

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

    **GPU Acceleration**:

    GPU acceleration is most beneficial for:

    - Batch inference on 100+ geometries
    - Latency-critical applications
    - Iterative refinement workflows

    For small batches (<100 geometries), CPU may be faster due to transfer overhead.

    .. important::

        This guardrail requires the optional dependencies ``trimesh`` and
        ``scikit-learn``. For GPU support, ``torch`` is also required. Install with:

        .. code-block:: bash

            pip install trimesh scikit-learn torch

    See Also
    --------
    :class:`GeometryDensityModel` : Underlying density estimation model.
    :func:`extract_features` : Feature extraction function.
    :class:`FeatureSchema` : Feature schema definition and validation.
    """

    def __init__(
        self,
        method: str = "gmm",
        n_components: int = 1,
        warn_pct: float = 95.0,
        reject_pct: float = 99.0,
        covariance_type: str = "full",
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
        self.device = device
        
        # Store method parameters for serialization
        self.method = method
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.poly_degree = poly_degree
        self.interaction_only = interaction_only
        self.random_state = random_state

        self.density = GeometryDensityModel(
            method=method,
            n_components=n_components,
            covariance_type=covariance_type,
            poly_degree=poly_degree,
            interaction_only=interaction_only,
            random_state=random_state,
            device=device,
        )

        self.feature_names = FEATURE_NAMES
        self.feature_version = FEATURE_VERSION
        self.feature_hash = feature_hash(FEATURE_NAMES)

    # -------------------------------------------------------------------------
    # Fitting
    # -------------------------------------------------------------------------

    def fit(self, meshes: list[trimesh.Trimesh]) -> None:
        r"""
        Fit guardrail from a list of Trimesh objects.

        This method extracts features from all provided meshes and trains the
        density model to learn the distribution of in-distribution geometries.

        Parameters
        ----------
        meshes : list[trimesh.Trimesh]
            List of training meshes representing the in-distribution geometry space.
            All meshes must pass validation checks.

        Raises
        ------
        ValueError
            If any mesh fails validation (see :func:`validate_mesh`).
        ValueError
            If feature extraction fails for any mesh.

        Examples
        --------
        >>> import trimesh
        >>> from physicsnemo.experimental.guardrails import GeometryGuardrail
        >>> 
        >>> # Generate training data
        >>> train_meshes = [
        ...     trimesh.creation.box(extents=[1, 1, 1]),
        ...     trimesh.creation.box(extents=[2, 2, 2]),
        ...     trimesh.creation.box(extents=[0.5, 0.5, 0.5]),
        ... ]
        >>> 
        >>> # Fit guardrail
        >>> guardrail = GeometryGuardrail(n_components=1)
        >>> guardrail.fit(train_meshes)

        Notes
        -----
        The fitting process involves:

        1. Extracting features from each mesh (see :func:`extract_features`)
        2. Stacking features into a matrix :math:`\mathbf{X} \in \mathbb{R}^{N \times D}`
        3. Validating feature schema (see :class:`FeatureSchema`)
        4. Fitting the GMM and computing reference scores

        After fitting, the guardrail is ready to query new geometries.
        """
        # Extract features from all meshes
        X = np.vstack([extract_features(m) for m in meshes])
        
        # Validate feature array
        FeatureSchema.validate_array(X)
        
        # Fit density model
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
            Common options include ``n_workers`` and ``chunksize``.

        Raises
        ------
        RuntimeError
            If no valid STL files are found in the directory.

        Examples
        --------
        >>> from pathlib import Path
        >>> from physicsnemo.experimental.guardrails import GeometryGuardrail
        >>> 
        >>> # Fit from directory with parallel processing
        >>> stl_dir = Path("/path/to/training/stl/files")
        >>> guardrail = GeometryGuardrail(n_components=2)
        >>> guardrail.fit_from_dir(stl_dir, n_workers=8, chunksize=16)

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
        feats, _ = load_features_from_dir(stl_dir, **loader_kwargs)
        self.density.fit(np.vstack(feats))

    # -------------------------------------------------------------------------
    # Querying
    # -------------------------------------------------------------------------

    def query(self, meshes: list[trimesh.Trimesh]) -> list[dict]:
        r"""
        Query guardrail for a list of meshes.

        This method extracts features from the provided meshes, computes anomaly
        scores, converts them to percentiles, and classifies each geometry as
        OK, WARN, or REJECT.

        Parameters
        ----------
        meshes : list[trimesh.Trimesh]
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

        Examples
        --------
        >>> import trimesh
        >>> from physicsnemo.experimental.guardrails import GeometryGuardrail
        >>> 
        >>> # Fit guardrail (assumes training data is available)
        >>> guardrail = GeometryGuardrail()
        >>> guardrail.fit(train_meshes)  # doctest: +SKIP
        >>> 
        >>> # Query new geometries
        >>> test_meshes = [
        ...     trimesh.creation.box(),
        ...     trimesh.creation.sphere(radius=100),  # Very different
        ... ]
        >>> results = guardrail.query(test_meshes)
        >>> for i, res in enumerate(results):
        ...     print(f"Mesh {i}: {res['status']} (p={res['percentile']:.1f})")
        Mesh 0: OK (p=45.2)
        Mesh 1: REJECT (p=99.8)

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
        # Extract features
        X = np.vstack([extract_features(m) for m in meshes])
        
        # Validate feature array
        FeatureSchema.validate_array(X)
        
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
            Common options include ``n_workers`` and ``chunksize``.

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

        Examples
        --------
        >>> from pathlib import Path
        >>> from physicsnemo.experimental.guardrails import GeometryGuardrail
        >>> 
        >>> # Query directory
        >>> guardrail = GeometryGuardrail.load(Path("guardrail.npz"))
        >>> results = guardrail.query_from_dir(
        ...     Path("/path/to/test/stl/files"),
        ...     n_workers=8
        ... )
        >>> 
        >>> # Filter for warnings and rejections
        >>> flagged = [r for r in results if r["status"] != "OK"]
        >>> print(f"Flagged {len(flagged)} / {len(results)} geometries")

        Notes
        -----
        Invalid or corrupted STL files are automatically skipped with a warning.

        See Also
        --------
        :func:`load_features_from_dir` : Parallel STL loading and feature extraction.
        :meth:`query` : Query individual mesh objects.
        """
        # Load features from directory
        feats, names = load_features_from_dir(stl_dir, **loader_kwargs)
        X = np.vstack(feats)

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

        Examples
        --------
        >>> from pathlib import Path
        >>> from physicsnemo.experimental.guardrails import GeometryGuardrail
        >>> 
        >>> guardrail = GeometryGuardrail()
        >>> guardrail.fit(train_meshes)  # doctest: +SKIP
        >>> guardrail.save(Path("my_guardrail.npz"))

        Notes
        -----
        The saved file contains:

        - ``gmm``: Fitted :class:`sklearn.mixture.GaussianMixture` object
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

        np.savez(
            path,
            density_state=density_state,
            warn_pct=self.warn_pct,
            reject_pct=self.reject_pct,
            feature_names=np.array(self.feature_names, dtype=object),
            feature_version=self.feature_version,
            feature_hash=self.feature_hash,
            method=self.method,
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            poly_degree=self.poly_degree,
            interaction_only=self.interaction_only,
            random_state=self.random_state,
        )

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "GeometryGuardrail":
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

        Examples
        --------
        >>> from pathlib import Path
        >>> from physicsnemo.experimental.guardrails import GeometryGuardrail
        >>> 
        >>> # Load on CPU
        >>> guardrail_cpu = GeometryGuardrail.load(Path("guardrail.npz"), device="cpu")
        >>> results = guardrail_cpu.query(test_meshes)
        >>> 
        >>> # Load on GPU for faster inference
        >>> guardrail_gpu = GeometryGuardrail.load(Path("guardrail.npz"), device="cuda")
        >>> results = guardrail_gpu.query(test_meshes)  # Faster on large batches

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

        - Training on CPU, deploying on GPU
        - Sharing models across different hardware configurations

        See Also
        --------
        :meth:`save` : Save a fitted guardrail to disk.
        """
        data = np.load(path, allow_pickle=True)

        # Check feature version compatibility
        if data["feature_version"] != FEATURE_VERSION:
            raise RuntimeError(
                f"Feature version mismatch: saved model uses {data['feature_version']}, "
                f"but current code expects {FEATURE_VERSION}"
            )

        # Check feature names match
        if list(data["feature_names"]) != FEATURE_NAMES:
            raise RuntimeError(
                f"Feature schema mismatch: saved model uses different feature names"
            )

        # Check feature hash for additional safety
        if data["feature_hash"] != feature_hash(FEATURE_NAMES):
            raise RuntimeError(
                f"Feature hash mismatch: saved model may have been corrupted or "
                f"uses a different feature extraction implementation"
            )

        # Reconstruct guardrail
        # Extract all parameters (with defaults for backward compatibility)
        method = str(data.get("method", "gmm"))
        n_components = int(data.get("n_components", 1))
        covariance_type = str(data.get("covariance_type", "full"))
        poly_degree = int(data.get("poly_degree", 2))
        interaction_only = bool(data.get("interaction_only", False))
        random_state = data.get("random_state", 0)
        if random_state is not None:
            random_state = int(random_state)
        
        obj = cls(
            method=method,
            n_components=n_components,
            warn_pct=float(data["warn_pct"]),
            reject_pct=float(data["reject_pct"]),
            covariance_type=covariance_type,
            poly_degree=poly_degree,
            interaction_only=interaction_only,
            random_state=random_state,
            device=device,
        )

        # Restore density model state
        density_state = data["density_state"].item()
        obj.density.set_state(density_state)

        return obj
