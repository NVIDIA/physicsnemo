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
        """
        if X.ndim != 2:
            raise ValueError(
                f"Feature array must be 2D, got {X.ndim}D array with shape {X.shape}"
            )

        if X.shape[1] != cls.dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {cls.dim}, got {X.shape[1]}"
            )
