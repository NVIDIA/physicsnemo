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
Geometry guardrails for out-of-distribution detection.

This package provides tools for detecting anomalous geometric configurations
using density-based methods. The primary interface is :class:`GeometryGuardrail`,
which learns the distribution of in-distribution geometries and classifies new
shapes as OK, WARN, or REJECT based on their anomaly scores.

.. important::

    This module requires the optional dependencies ``trimesh`` and ``scikit-learn``.
    Install with:

    .. code-block:: bash

        pip install trimesh scikit-learn

Examples
--------
Basic usage with mesh objects:

>>> import trimesh
>>> from physicsnemo.experimental.guardrails.geometry import GeometryGuardrail
>>> 
>>> # Create training data
>>> train_meshes = [trimesh.creation.box() for _ in range(100)]
>>> 
>>> # Fit guardrail
>>> guardrail = GeometryGuardrail(n_components=1, warn_pct=95.0, reject_pct=99.0)
>>> guardrail.fit(train_meshes)
>>> 
>>> # Query new geometries
>>> test_meshes = [trimesh.creation.sphere(), trimesh.creation.cylinder()]
>>> results = guardrail.query(test_meshes)
>>> for res in results:
...     print(f"{res['status']}: {res['percentile']:.1f}%")

Using STL files from directories:

>>> from pathlib import Path
>>> from physicsnemo.experimental.guardrails.geometry import GeometryGuardrail
>>> 
>>> # Fit from directory
>>> guardrail = GeometryGuardrail()
>>> guardrail.fit_from_dir(Path("/path/to/training/stl"), n_workers=8)
>>> 
>>> # Query from directory
>>> results = guardrail.query_from_dir(Path("/path/to/test/stl"), n_workers=8)
>>> flagged = [r for r in results if r["status"] != "OK"]
>>> print(f"Flagged {len(flagged)} / {len(results)} geometries")

Saving and loading guardrails:

>>> from pathlib import Path
>>> from physicsnemo.experimental.guardrails.geometry import GeometryGuardrail
>>> 
>>> # Save fitted guardrail
>>> guardrail.save(Path("guardrail.npz"))
>>> 
>>> # Load for inference
>>> loaded = GeometryGuardrail.load(Path("guardrail.npz"))
>>> results = loaded.query(test_meshes)
"""

from .density_model import GeometryDensityModel
from .density_pce import PCEDensityModel
from .fast_stl import FastMesh, is_fast_reader_available, load_stl_fast
from .feature_extraction import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    extract_features,
    feature_hash,
)
from .feature_schema import FeatureSchema
from .mesh_io import load_features_from_dir
from .mesh_validation import validate_mesh
from .ood_detector import GeometryGuardrail

__all__ = [
    "GeometryGuardrail",
    "GeometryDensityModel",
    "PCEDensityModel",
    "FeatureSchema",
    "FastMesh",
    "extract_features",
    "validate_mesh",
    "load_features_from_dir",
    "load_stl_fast",
    "is_fast_reader_available",
    "feature_hash",
    "FEATURE_NAMES",
    "FEATURE_VERSION",
]
