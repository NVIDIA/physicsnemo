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
I/O utilities for geometry guardrails.

This module provides efficient parallel loading and feature extraction for
STL mesh files. It uses multiprocessing to handle large directories of
geometric data with graceful error handling for corrupted files.

Supports optional fast Rust-based STL reader for 5-10x speedup on I/O.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
import trimesh

from .feature_extraction import extract_features


def _process_stl(path_str: str, use_fast_reader: bool = False) -> tuple[str, np.ndarray | None, str | None]:
    r"""
    Load and extract features from a single STL file.

    This is a worker function designed for use with multiprocessing.Pool.
    It handles all exceptions internally to prevent pool crashes.

    Parameters
    ----------
    path_str : str
        String path to the STL file.
    use_fast_reader : bool, optional
        If True, attempt to use fast Rust reader. Falls back to trimesh
        if not available. Default is False.

    Returns
    -------
    tuple[str, np.ndarray or None, str or None]
        A 3-tuple containing:
        - Filename (str)
        - Feature array (np.ndarray) if successful, None if failed
        - Error message (str) if failed, None if successful
    """
    path = Path(path_str)
    try:
        # Try fast reader first if requested
        if use_fast_reader:
            try:
                from .fast_stl import load_stl_fast

                mesh = load_stl_fast(path)
            except (ImportError, Exception) as e:
                # Fall back to trimesh on any error
                if not isinstance(e, ImportError):
                    # Log non-import errors (parsing failures, etc.)
                    pass  # Silent fallback for now
                mesh = trimesh.load(path, force="mesh")
        else:
            mesh = trimesh.load(path, force="mesh")

        # Handle Scene objects (convert to single mesh)
        if isinstance(mesh, trimesh.Scene):
            # Concatenate all geometries in the scene
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

        feat = extract_features(mesh)
        return path.name, feat, None
    except Exception as e:
        return path.name, None, str(e)


def load_features_from_dir(
    stl_dir: Path,
    n_workers: int | None = None,
    chunksize: int = 8,
    use_fast_reader: bool = False,
) -> tuple[list[np.ndarray], list[str]]:
    r"""
    Load and featurize all STL files in a directory using multiprocessing.

    This function parallelizes feature extraction across multiple CPU cores
    for efficient processing of large mesh datasets. It automatically handles
    errors and skips invalid files while reporting statistics.

    Parameters
    ----------
    stl_dir : Path
        Directory containing STL files to process. Only files with ``.stl``
        extension are processed.
    n_workers : int or None, optional
        Number of worker processes to use. If ``None``, defaults to
        ``cpu_count() - 1`` to leave one core available. Default is ``None``.
    chunksize : int, optional
        Number of files to process per worker task. Larger values reduce
        communication overhead but may cause load imbalance. Default is 8.
    use_fast_reader : bool, optional
        If True, use fast Rust-based STL reader (requires ``stlreader``).
        Provides 5-10x speedup for I/O. Falls back to trimesh if not available.
        Default is False.

    Returns
    -------
    tuple[list[np.ndarray], list[str]]
        A 2-tuple containing:
        - List of feature arrays, one per valid STL file
        - List of corresponding filenames (in same order as features)

    Raises
    ------
    RuntimeError
        If no valid STL files are found in the directory.

    Examples
    --------
    >>> from pathlib import Path
    >>> from physicsnemo.experimental.guardrails.geometry import load_features_from_dir
    >>> 
    >>> # Standard loading with trimesh
    >>> stl_dir = Path("/path/to/stl/files")
    >>> features, names = load_features_from_dir(stl_dir, n_workers=4)
    >>> print(f"Loaded {len(features)} geometries")
    Loaded 150 geometries
    >>> 
    >>> # Fast loading with Rust reader (5-10x faster I/O)
    >>> features_fast, names_fast = load_features_from_dir(
    ...     stl_dir,
    ...     n_workers=8,
    ...     use_fast_reader=True  # Requires stlreader package
    ... )
    >>> print(f"Loaded {len(features_fast)} geometries (fast)")
    Loaded 150 geometries (fast)

    Notes
    -----
    **Multiprocessing Strategy**:

    This function uses the ``"spawn"`` start method for multiprocessing to
    ensure compatibility across platforms and avoid issues with CUDA contexts.
    The ``imap_unordered`` method is used for memory-efficient streaming of
    results as they complete.

    **Error Handling**:

    Individual file errors (corrupted STL, invalid geometry, etc.) are caught
    and logged, but do not stop processing of remaining files. A summary of
    skipped files is printed if any errors occur.

    **Fast Reader Performance**:

    The optional Rust-based STL reader provides significant speedup:

    - **5-10x faster** file parsing compared to trimesh
    - Precomputes normals and areas during parsing
    - Best for large batches (100+ files)
    - Automatically falls back to trimesh if not available

    To install the fast reader:

    .. code-block:: bash

        # Install Rust toolchain
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

        # Build stlreader
        cd /path/to/stlreader
        pip install maturin
        maturin develop --release

    **Performance Considerations**:

    - For small datasets (< 50 files), multiprocessing overhead may outweigh
      benefits. Consider using single-threaded processing.
    - The ``chunksize`` parameter controls the granularity of work distribution.
      Typical values range from 1-16 depending on file complexity.
    - Memory usage scales with ``n_workers × chunksize``.
    - Fast reader is most beneficial when I/O is the bottleneck.

    See Also
    --------
    :func:`extract_features` : Feature extraction for individual meshes.
    :func:`load_stl_fast` : Fast Rust-based STL loader.
    :class:`GeometryGuardrail` : High-level API that uses this function.
    """
    # Find all STL files in the directory
    paths = sorted(p.as_posix() for p in stl_dir.glob("*.stl"))

    feats: list[np.ndarray] = []
    names: list[str] = []
    errors: list[str] = []

    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    # Create list of (path, use_fast_reader) tuples for starmap
    # This avoids pickling issues with local functions
    tasks = [(path, use_fast_reader) for path in paths]

    # Use spawn context for cross-platform compatibility
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        # Process files in parallel with unordered results
        # Use starmap to pass multiple arguments
        for name, feat, err in pool.starmap(_process_stl, tasks, chunksize=chunksize):
            if err is None:
                feats.append(feat)
                names.append(name)
            else:
                errors.append(err)

    # Check if any valid files were processed
    if not feats:
        error_msg = f"No valid STL files found in {stl_dir}"
        if errors:
            error_msg += f"\n{len(errors)} files failed to load. First error: {errors[0]}"
        raise RuntimeError(error_msg)

    # Report skipped files if any
    if errors:
        print(f"[geometry guardrail] Skipped {len(errors)} invalid geometries")

    return feats, names
