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
Fast STL loading using Rust-based reader.

This module provides an optional high-performance STL reader implemented in Rust.
It is significantly faster than trimesh for large batches of STL files, especially
when combined with multiprocessing.

The Rust reader is optional and requires the ``stlreader`` package to be installed.
If not available, the module gracefully falls back to trimesh.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class FastMesh:
    r"""
    Lightweight mesh object compatible with feature extraction.

    This class provides a minimal interface compatible with trimesh.Trimesh
    but uses precomputed face normals and areas from the Rust reader for
    maximum performance.

    Parameters
    ----------
    vertices : np.ndarray
        Vertex coordinates of shape :math:`(N, 3)`.
    faces : np.ndarray
        Face indices of shape :math:`(M, 3)`.
    face_normals : np.ndarray
        Precomputed face normals of shape :math:`(M, 3)`.
    face_areas : np.ndarray
        Precomputed face areas of shape :math:`(M,)`.

    Attributes
    ----------
    vertices : np.ndarray
        Vertex coordinates.
    faces : np.ndarray
        Face indices.
    face_normals : np.ndarray
        Face normal vectors (unit vectors).
    area_faces : np.ndarray
        Individual face areas.
    area : float
        Total surface area (sum of all face areas).

    Examples
    --------
    >>> import numpy as np
    >>> from physicsnemo.experimental.guardrails.geometry import FastMesh
    >>> 
    >>> # Create from arrays
    >>> vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    >>> faces = np.array([[0, 1, 2]], dtype=np.int32)
    >>> normals = np.array([[0, 0, 1]], dtype=np.float32)
    >>> areas = np.array([0.5], dtype=np.float32)
    >>> 
    >>> mesh = FastMesh(vertices, faces, normals, areas)
    >>> print(f"Total area: {mesh.area:.2f}")
    Total area: 0.50

    Notes
    -----
    This class is designed to work seamlessly with the feature extraction
    pipeline while avoiding the overhead of trimesh's full mesh processing.
    It provides only the attributes needed for feature extraction:

    - ``vertices``: For centroid, PCA, bounding box, moments
    - ``face_normals``: For projected area calculations
    - ``area_faces``: For area-weighted projections
    - ``area``: For total surface area feature

    The Rust reader precomputes normals and areas during parsing, eliminating
    the need for recomputation in Python.

    See Also
    --------
    :func:`load_stl_fast` : Load STL file using Rust reader.
    :func:`extract_features` : Feature extraction (compatible with FastMesh).
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        face_normals: np.ndarray,
        face_areas: np.ndarray,
    ):
        self.vertices = vertices
        self.faces = faces
        self.face_normals = face_normals
        self.area_faces = face_areas
        self.area = float(np.sum(face_areas))

    def __repr__(self) -> str:
        return (
            f"FastMesh(vertices={self.vertices.shape[0]}, "
            f"faces={self.faces.shape[0]}, area={self.area:.2f})"
        )


def load_stl_fast(path: Path) -> FastMesh:
    r"""
    Load STL file using fast Rust-based reader.

    This function uses the optional ``stlreader`` Rust extension for high-performance
    STL parsing. It is significantly faster than trimesh, especially for large files
    or batch processing.

    Parameters
    ----------
    path : Path
        Path to STL file to load.

    Returns
    -------
    FastMesh
        Lightweight mesh object with precomputed normals and areas.

    Raises
    ------
    ImportError
        If ``stlreader`` package is not installed.
    ValueError
        If the STL file is invalid or cannot be parsed.

    Examples
    --------
    >>> from pathlib import Path
    >>> from physicsnemo.experimental.guardrails.geometry import load_stl_fast
    >>> 
    >>> # Load STL file (fast)
    >>> mesh = load_stl_fast(Path("part.stl"))
    >>> print(mesh)
    FastMesh(vertices=1523, faces=3042, area=45.23)
    >>> 
    >>> # Use with feature extraction
    >>> from physicsnemo.experimental.guardrails.geometry import extract_features
    >>> features = extract_features(mesh)
    >>> print(features.shape)
    (22,)

    Notes
    -----
    **Performance Comparison**:

    For a typical CAD part STL file (~5MB, 10K faces):

    - ``trimesh.load()``: ~50ms
    - ``load_stl_fast()``: ~5ms (10x faster)

    Speedup increases with file size and is especially beneficial when
    loading hundreds of files in parallel.

    **Installation**:

    The Rust reader requires building from source:

    .. code-block:: bash

        # Install Rust toolchain
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

        # Build and install stlreader
        cd /path/to/stlreader
        pip install maturin
        maturin develop --release

    **Compatibility**:

    The returned :class:`FastMesh` object is compatible with
    :func:`extract_features` and all guardrail functionality. It provides
    the same interface as ``trimesh.Trimesh`` for the attributes we use.

    See Also
    --------
    :class:`FastMesh` : Lightweight mesh object returned by this function.
    :func:`load_features_from_dir` : Batch loading with optional fast reader.
    """
    try:
        import stlreader
    except ImportError as e:
        raise ImportError(
            "Fast STL reader requires 'stlreader' package. "
            "Install from source:\n"
            "  git clone <repo-url>\n"
            "  cd stlreader\n"
            "  pip install maturin\n"
            "  maturin develop --release\n"
            "Or use trimesh (slower): pip install trimesh"
        ) from e

    # Parse with Rust reader (pass path as string)
    try:
        vertices, faces, normals, areas = stlreader.load_stl(str(path))
    except Exception as e:
        raise ValueError(f"Failed to parse STL file {path}: {e}") from e

    # Create FastMesh
    return FastMesh(vertices, faces, normals, areas)


def is_fast_reader_available() -> bool:
    r"""
    Check if the fast Rust STL reader is available.

    Returns
    -------
    bool
        True if ``stlreader`` package is installed and functional.

    Examples
    --------
    >>> from physicsnemo.experimental.guardrails.geometry import is_fast_reader_available
    >>> 
    >>> if is_fast_reader_available():
    ...     print("Using fast Rust reader")
    ... else:
    ...     print("Using trimesh (install stlreader for faster loading)")
    """
    try:
        import stlreader  # noqa: F401

        return True
    except ImportError:
        return False
