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
Feature schema validation for geometry guardrails.

This module provides immutable schema definitions and validation utilities
for ensuring feature array compatibility across different versions of the
guardrail system.
"""

from __future__ import annotations

import numpy as np

from .feature_extraction import FEATURE_NAMES, FEATURE_VERSION, feature_hash


class FeatureSchema:
    r"""
    Immutable feature schema for geometry guardrails.

    This class provides a centralized definition of the feature schema,
    including feature names, version, dimensionality, and a cryptographic
    hash for compatibility checking. All attributes are class-level and
    immutable.

    Attributes
    ----------
    names : list[str]
        Ordered list of feature names as defined in :data:`FEATURE_NAMES`.
    version : str
        Feature schema version identifier (e.g., ``"v1.0"``).
    hash : str
        SHA-256 hash of the feature names for compatibility checking.
    dim : int
        Feature vector dimensionality (number of features).

    Examples
    --------
    >>> from physicsnemo.experimental.guardrails.geometry import FeatureSchema
    >>> 
    >>> # Access schema properties
    >>> print(f"Feature dimension: {FeatureSchema.dim}")
    Feature dimension: 22
    >>> print(f"Schema version: {FeatureSchema.version}")
    Schema version: v1.0
    >>> 
    >>> # Validate a feature array
    >>> import numpy as np
    >>> features = np.random.randn(10, 22)
    >>> FeatureSchema.validate_array(features)  # Should pass
    >>> 
    >>> # Invalid array will raise error
    >>> invalid_features = np.random.randn(10, 20)
    >>> try:
    ...     FeatureSchema.validate_array(invalid_features)
    ... except ValueError as e:
    ...     print(f"Validation failed: {e}")
    Validation failed: Feature dimension mismatch: expected 22, got 20

    Notes
    -----
    This class uses class-level attributes and methods to enforce immutability
    and provide a single source of truth for the feature schema across the
    entire guardrail system.

    See Also
    --------
    :data:`FEATURE_NAMES` : Feature name definitions.
    :data:`FEATURE_VERSION` : Current schema version.
    :func:`feature_hash` : Hash computation function.
    """

    #: Ordered list of feature names
    names = FEATURE_NAMES

    #: Schema version identifier
    version = FEATURE_VERSION

    #: Cryptographic hash of feature names
    hash = feature_hash(FEATURE_NAMES)

    #: Feature vector dimensionality
    dim = len(FEATURE_NAMES)

    @classmethod
    def validate_array(cls, X: np.ndarray) -> None:
        r"""
        Validate that a feature array conforms to the schema.

        This method checks that the input array has the correct shape
        (2D with the expected number of features per sample).

        Parameters
        ----------
        X : np.ndarray
            Feature array to validate. Expected shape is :math:`(N, D)` where
            :math:`N` is the number of samples and :math:`D` is the feature
            dimensionality defined by :attr:`dim`.

        Raises
        ------
        ValueError
            If ``X`` is not a 2D array.
        ValueError
            If the feature dimension (second axis) does not match :attr:`dim`.

        Examples
        --------
        >>> import numpy as np
        >>> from physicsnemo.experimental.guardrails.geometry import FeatureSchema
        >>> 
        >>> # Valid array
        >>> X = np.random.randn(100, FeatureSchema.dim)
        >>> FeatureSchema.validate_array(X)  # Should pass
        >>> 
        >>> # Invalid: wrong number of features
        >>> X_bad = np.random.randn(100, 10)
        >>> try:
        ...     FeatureSchema.validate_array(X_bad)
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        Error: Feature dimension mismatch: expected 22, got 10
        >>> 
        >>> # Invalid: not 2D
        >>> X_1d = np.random.randn(22)
        >>> try:
        ...     FeatureSchema.validate_array(X_1d)
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        Error: Feature array must be 2D
        """
        if X.ndim != 2:
            raise ValueError(
                f"Feature array must be 2D, got {X.ndim}D array with shape {X.shape}"
            )

        if X.shape[1] != cls.dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {cls.dim}, got {X.shape[1]}"
            )
