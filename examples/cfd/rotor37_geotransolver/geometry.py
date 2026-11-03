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

"""Whitened PCA features of corresponding blade surfaces."""

import numpy as np


def principal_directions(matrix, rank):
    """Return the leading orthonormal directions of centered training rows.

    Directions come from the float64 eigendecomposition of the sample Gram
    matrix. Each direction is oriented so that its largest-magnitude entry is
    positive.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix @ matrix.T)
    order = np.argsort(eigenvalues)[::-1][:rank]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    if not eigenvalues[-1] > eigenvalues[0] * 1e-12:
        raise ValueError("Training geometry has fewer independent modes than the rank")
    directions = (eigenvectors.T @ matrix) / np.sqrt(eigenvalues)[:, None]
    pivots = np.abs(directions).argmax(axis=1)
    directions *= np.sign(directions[np.arange(rank), pivots])[:, None]
    return directions, eigenvalues


def fit_geometry_encodings(points, normals, rank=32):
    """Fit whitened coordinate and normal PCA features on training surfaces.

    ``points`` and ``normals`` have shape ``[cases, vertices, 3]`` with the
    same vertex order in every case. Coordinates are displacements from the
    training mean surface divided by their per-axis training standard
    deviation. Normals are centered on the training mean normals. Each channel
    is flattened in vertex-major order and projected onto its leading
    principal directions, and each projection is standardized with its
    training mean and standard deviation.

    Returns the arrays used by :func:`encode_geometry` and fit metadata.
    """
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 3 or points.shape != normals.shape:
        raise ValueError("Points and normals must share shape [cases, vertices, 3]")
    if not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise ValueError("Training geometry must be finite")
    if not 0 < rank < len(points):
        raise ValueError("The PCA rank must be positive and below the case count")
    mean_points = points.mean(axis=0)
    displacement_scale = np.sqrt(np.mean((points - mean_points) ** 2, axis=(0, 1)))
    if not np.all(displacement_scale > 0):
        raise ValueError("Training coordinates must vary along every axis")
    arrays = {
        "mean_points": mean_points,
        "displacement_scale": displacement_scale,
        "mean_normals": normals.mean(axis=0),
    }
    metadata = {"rank": rank}
    channels = (
        ("coordinate", (points - mean_points) / displacement_scale),
        ("normal", normals - arrays["mean_normals"]),
    )
    for name, centered in channels:
        matrix = centered.reshape(len(centered), -1)
        directions, eigenvalues = principal_directions(matrix, rank)
        directions = directions.astype(np.float32)
        projections = matrix @ directions.T.astype(np.float64)
        arrays[f"{name}_directions"] = directions
        arrays[f"{name}_mean"] = projections.mean(axis=0)
        arrays[f"{name}_scale"] = projections.std(axis=0)
        metadata[name] = {
            "retained_variance": float(eigenvalues.sum() / np.sum(matrix**2)),
            "variance_ratios": (eigenvalues / np.sum(matrix**2)).tolist(),
        }
    return arrays, metadata


def encode_geometry(points, normals, arrays):
    """Return the whitened coordinate and normal PCA features of one surface."""
    coordinates = (points - arrays["mean_points"]) / arrays["displacement_scale"]
    normals = normals - arrays["mean_normals"]
    features = []
    for name, values in (("coordinate", coordinates), ("normal", normals)):
        projection = arrays[f"{name}_directions"] @ values.reshape(-1)
        features.append((projection - arrays[f"{name}_mean"]) / arrays[f"{name}_scale"])
    return np.concatenate(features)
