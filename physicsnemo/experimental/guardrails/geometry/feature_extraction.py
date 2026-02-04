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
Feature extraction for geometry guardrails.

This module defines the feature schema and extracts non-invariant geometric
descriptors from triangular surface meshes. Features are designed to capture
spatial orientation, scale, and shape characteristics that distinguish different
geometric configurations.
"""

from __future__ import annotations

import hashlib

import numpy as np
import trimesh

from .mesh_validation import validate_mesh

#: Feature schema version identifier
FEATURE_VERSION = "v1.0"

#: Ordered list of feature names defining the schema
FEATURE_NAMES = [
    "centroid_x",
    "centroid_y",
    "centroid_z",
    "pca_axis1_x",
    "pca_axis1_y",
    "pca_axis1_z",
    "pca_axis2_x",
    "pca_axis2_y",
    "pca_axis2_z",
    "pca_eig1",
    "pca_eig2",
    "pca_eig3",
    "extent_x",
    "extent_y",
    "extent_z",
    "moment_x",
    "moment_y",
    "moment_z",
    "total_area",
    "area_xy",
    "area_xz",
    "area_yz",
]


def feature_hash(names: list[str]) -> str:
    r"""
    Generate a stable cryptographic hash of the feature schema.

    This hash is used for version control and compatibility checking when
    loading serialized guardrail models.

    Parameters
    ----------
    names : list[str]
        Ordered list of feature names.

    Returns
    -------
    str
        SHA-256 hash digest as a hexadecimal string.

    Examples
    --------
    >>> from physicsnemo.experimental.guardrails.geometry import feature_hash
    >>> names = ["feature_1", "feature_2", "feature_3"]
    >>> hash_value = feature_hash(names)
    >>> len(hash_value)
    64
    >>> hash_value == feature_hash(names)  # Deterministic
    True
    """
    h = hashlib.sha256()
    for n in names:
        h.update(n.encode("utf-8"))
    return h.hexdigest()


def extract_features(mesh: trimesh.Trimesh) -> np.ndarray:
    r"""
    Extract non-invariant geometric descriptors from a triangular mesh.

    This function computes a comprehensive set of geometric features that
    intentionally capture translation, rotation, and scale. Features include
    centroid position, principal component axes, eigenvalues, bounding box
    extents, second moments, and projected surface areas.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input triangular surface mesh. Must pass validation checks.

    Returns
    -------
    np.ndarray
        1D feature vector of shape :math:`(22,)` containing all geometric
        descriptors in the order specified by :data:`FEATURE_NAMES`.

    Raises
    ------
    ValueError
        If the mesh fails validation checks (see :func:`validate_mesh`).
    ValueError
        If the mesh has insufficient vertices for PCA (< 10 vertices).
    RuntimeError
        If the computed feature vector length does not match the schema.

    Examples
    --------
    >>> import trimesh
    >>> import numpy as np
    >>> from physicsnemo.experimental.guardrails.geometry import extract_features
    >>> 
    >>> # Create a unit cube
    >>> mesh = trimesh.creation.box(extents=[1, 1, 1])
    >>> features = extract_features(mesh)
    >>> print(f"Feature vector shape: {features.shape}")
    Feature vector shape: (22,)
    >>> 
    >>> # Centroid should be near origin
    >>> centroid = features[:3]
    >>> np.allclose(centroid, [0, 0, 0], atol=1e-6)
    True

    Notes
    -----
    The feature extraction process includes several key steps:

    1. **Validation**: Checks mesh integrity (see :func:`validate_mesh`)
    2. **Centroid**: Mean position of all vertices :math:`\mathbf{c} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{v}_i`
    3. **PCA**: Principal component analysis via SVD on centered vertices
    4. **Eigenvalues**: Variance along principal axes :math:`\lambda_1 \geq \lambda_2 \geq \lambda_3`
    5. **Bounding Box**: Axis-aligned extents :math:`[\Delta x, \Delta y, \Delta z]`
    6. **Second Moments**: Variance per axis :math:`\sigma^2_x, \sigma^2_y, \sigma^2_z`
    7. **Projected Areas**: Surface area projections onto coordinate planes

    **Important**: Features are intentionally **not** invariant to transformations.
    This allows the guardrail to detect geometric configurations based on their
    absolute position and orientation in space.

    See Also
    --------
    :data:`FEATURE_NAMES` : Complete list of feature names in order.
    :data:`FEATURE_VERSION` : Current feature schema version.
    :func:`validate_mesh` : Mesh validation checks.
    """
    # Validate mesh integrity
    validate_mesh(mesh)

    verts = mesh.vertices
    centroid = verts.mean(axis=0)  # Shape: (3,)
    X = verts - centroid  # Center vertices

    if X.shape[0] < 10:
        raise ValueError("Insufficient points for PCA (need at least 10)")

    # Compute PCA via singular value decomposition
    # U: left singular vectors, S: singular values, Vt: right singular vectors transposed
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Convert singular values to eigenvalues (variance)
    eigvals = (S**2) / (X.shape[0] - 1)  # Shape: (3,)
    eigvecs = Vt.T  # Shape: (3, 3)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Enforce deterministic axis orientation (flip if first component is negative)
    for i in range(3):
        if eigvecs[0, i] < 0:
            eigvecs[:, i] *= -1

    # Extract first two principal axes (6 components total)
    pca_axes = eigvecs[:, :2].reshape(-1)  # Shape: (6,)
    pca_vals = eigvals[:3]  # Shape: (3,)

    # Compute axis-aligned bounding box extents
    bbox_min = verts.min(axis=0)  # Shape: (3,)
    bbox_max = verts.max(axis=0)  # Shape: (3,)
    extents = bbox_max - bbox_min  # Shape: (3,)

    # Compute second moments (variance per axis)
    second_moments = ((verts - centroid) ** 2).mean(axis=0)  # Shape: (3,)

    # Compute projected surface areas onto coordinate planes
    normals = mesh.face_normals  # Shape: (n_faces, 3)
    areas = mesh.area_faces  # Shape: (n_faces,)

    # Project area onto each coordinate plane
    A_xy = np.sum(areas * np.abs(normals[:, 2]))  # Area weighted by |z-component|
    A_xz = np.sum(areas * np.abs(normals[:, 1]))  # Area weighted by |y-component|
    A_yz = np.sum(areas * np.abs(normals[:, 0]))  # Area weighted by |x-component|

    # Concatenate all features into a single vector
    feats = np.concatenate(
        [
            centroid,  # 3 components
            pca_axes,  # 6 components
            pca_vals,  # 3 components
            extents,  # 3 components
            second_moments,  # 3 components
            [mesh.area],  # 1 component (total surface area)
            [A_xy, A_xz, A_yz],  # 3 components
        ]
    )  # Total: 22 components

    # Sanity check: verify feature vector length matches schema
    if feats.shape[0] != len(FEATURE_NAMES):
        raise RuntimeError(
            f"Feature length mismatch: expected {len(FEATURE_NAMES)}, "
            f"got {feats.shape[0]}"
        )

    return feats
