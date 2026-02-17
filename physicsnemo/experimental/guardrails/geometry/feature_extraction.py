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

import hashlib

import numpy as np
import torch

from physicsnemo.mesh import Mesh as PhysicsNeMoMesh

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


def extract_features(mesh: PhysicsNeMoMesh, return_tensor: bool = False, skip_validation: bool = False) -> np.ndarray | torch.Tensor:
    r"""
    Extract non-invariant geometric descriptors from a triangular mesh.

    This function computes a comprehensive set of geometric features that
    intentionally capture translation, rotation, and scale. Features include
    centroid position, principal component axes, eigenvalues, bounding box
    extents, second moments, and projected surface areas.

    Parameters
    ----------
    mesh : physicsnemo.mesh.Mesh
        Input triangular surface mesh. Must pass validation checks.
    return_tensor : bool, optional
        If True, return PyTorch tensor on the same device as mesh.
        If False, return numpy array. Default is False.
    skip_validation : bool, optional
        If True, skip mesh validation (assumes validation already done).
        Useful when validation is performed earlier in the pipeline.
        Default is False.

    Returns
    -------
    np.ndarray or torch.Tensor
        1D feature vector of shape :math:`(22,)` containing all geometric
        descriptors in the order specified by :data:`FEATURE_NAMES`.
        Returns tensor if ``return_tensor=True``, otherwise numpy array.

    Raises
    ------
    ValueError
        If the mesh fails validation checks (see :func:`validate_mesh`).
        This is only checked if ``skip_validation=False``.
    ValueError
        If the mesh has insufficient vertices for PCA (< 4 vertices).
    RuntimeError
        If the computed feature vector length does not match the schema.

    Examples
    --------
    >>> import numpy as np
    >>> import pyvista as pv
    >>> from physicsnemo.mesh.io import from_pyvista
    >>> from physicsnemo.experimental.guardrails.geometry import extract_features
    >>> 
    >>> # Create a unit cube using PyVista
    >>> pv_mesh = pv.Cube()
    >>> mesh = from_pyvista(pv_mesh)
    >>> features = extract_features(mesh)
    >>> print(f"Feature vector shape: {features.shape}")
    Feature vector shape: (22,)

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

    **GPU Support**: If the mesh is on GPU, all computations are performed on GPU
    for efficiency. Set ``return_tensor=True`` to keep features on GPU.
    """
    # Validate mesh integrity (unless already validated)
    if not skip_validation:
        validate_mesh(mesh)

    # Get device from mesh
    device = mesh.points.device
    
    # Work with torch tensors directly (no conversion to numpy)
    verts = mesh.points  # Shape: (N, 3) - already on device
    centroid = verts.mean(dim=0)  # Shape: (3,)
    X = verts - centroid.unsqueeze(0)  # Center vertices

    if X.shape[0] < 4:
        raise ValueError("Insufficient points for PCA (need at least 4)")

    # Compute PCA via singular value decomposition
    # U: left singular vectors, S: singular values, Vt: right singular vectors transposed
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    
    # Convert singular values to eigenvalues (variance)
    eigvals = (S ** 2) / (X.shape[0] - 1)  # Shape: (3,)
    eigvecs = Vt.T  # Shape: (3, 3)

    # Sort eigenvalues and eigenvectors in descending order
    idx = torch.argsort(eigvals, descending=True)
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
    bbox_min = verts.min(dim=0)[0]  # Shape: (3,)
    bbox_max = verts.max(dim=0)[0]  # Shape: (3,)
    extents = bbox_max - bbox_min  # Shape: (3,)

    # Compute second moments (variance per axis)
    second_moments = ((verts - centroid.unsqueeze(0)) ** 2).mean(dim=0)  # Shape: (3,)

    # Compute projected surface areas onto coordinate planes
    # physicsnemo.mesh provides cell_normals and cell_areas
    normals = mesh.cell_normals  # Shape: (n_faces, 3) - already on device
    areas = mesh.cell_areas  # Shape: (n_faces,) - already on device

    # Project area onto each coordinate plane
    A_xy = (areas * torch.abs(normals[:, 2])).sum()  # Area weighted by |z-component|
    A_xz = (areas * torch.abs(normals[:, 1])).sum()  # Area weighted by |y-component|
    A_yz = (areas * torch.abs(normals[:, 0])).sum()  # Area weighted by |x-component|

    # Total surface area
    total_area = mesh.cell_areas.sum()

    # Concatenate all features into a single vector
    feats = torch.cat(
        [
            centroid,  # 3 components
            pca_axes,  # 6 components
            pca_vals,  # 3 components
            extents,  # 3 components
            second_moments,  # 3 components
            total_area.unsqueeze(0),  # 1 component (total surface area)
            torch.stack([A_xy, A_xz, A_yz]),  # 3 components
        ]
    )  # Total: 22 components

    # Sanity check: verify feature vector length matches schema
    if feats.shape[0] != len(FEATURE_NAMES):
        raise RuntimeError(
            f"Feature length mismatch: expected {len(FEATURE_NAMES)}, "
            f"got {feats.shape[0]}"
        )

    # Return tensor or numpy array based on flag
    if return_tensor:
        return feats
    else:
        return feats.cpu().numpy()
