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

import multiprocessing as mp
from pathlib import Path

import numpy as np

from physicsnemo.core.version_check import OptionalImport
from physicsnemo.mesh.io.io_pyvista import from_pyvista

from .feature_extraction import extract_features
from .mesh_validation import validate_mesh

# Lazy import of pyvista
_pyvista = OptionalImport("pyvista")


def _process_stl(path_str: str) -> tuple[str, np.ndarray | None, str | None]:
    r"""
    Load and extract features from a single STL file.

    This is a worker function designed for use with multiprocessing.Pool.
    It handles all exceptions internally to prevent pool crashes.
    All processing is done on CPU to avoid OOM issues in multiprocessing.

    Parameters
    ----------
    path_str : str
        String path to the STL file

    Returns
    -------
    tuple[str, np.ndarray or None, str or None]
        A 3-tuple containing:
        - Filename (str)
        - Feature array (np.ndarray) if successful, None if failed
        - Error message (str) if failed, None if successful
    """
    pv = _pyvista

    path = Path(path_str)
    
    try:
        # Load STL file using PyVista
        # PyVista merges all solids in STL files into a single PolyData mesh
        pv_mesh = pv.read(str(path))
        
        # STL files should always return PolyData, not MultiBlock
        # If we get MultiBlock, something unexpected happened
        if isinstance(pv_mesh, pv.MultiBlock):
            raise ValueError(
                f"Unexpected MultiBlock returned from STL file {path.name}. "
                f"STL files should return PolyData. This may indicate a file format issue."
            )
        
        # Check if mesh has any points before conversion
        if pv_mesh.n_points == 0:
            raise ValueError("Mesh has no points")
        
        # Check if mesh has cells (faces)
        if hasattr(pv_mesh, 'n_cells') and pv_mesh.n_cells == 0:
            raise ValueError("Mesh has no cells")
        
        # Convert PyVista mesh to physicsnemo.mesh.Mesh
        # STL files are surface meshes (2D manifolds in 3D space)
        mesh = from_pyvista(pv_mesh, manifold_dim=2)
        
        # Verify mesh has cells after conversion
        if mesh.n_cells == 0:
            raise ValueError(f"Mesh has no cells after conversion (n_points={mesh.n_points})")
        
        # Validate mesh integrity (comprehensive checks)
        validate_mesh(mesh)
        
        # Extract features on CPU (mesh is already on CPU from from_pyvista)
        # Returns numpy array for multiprocessing compatibility
        feat = extract_features(mesh, return_tensor=False, skip_validation=True)
        
        # Clean up to free memory
        del mesh, pv_mesh
        
        return path.name, feat, None
    except Exception as e:
        # Include filename in error message for better debugging
        error_msg = f"{path.name}: {type(e).__name__}: {str(e)}"
        return path.name, None, error_msg


def load_features_from_dir(
    stl_dir: Path,
    n_workers: int | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    r"""
    Load and featurize all STL files in a directory using multiprocessing.

    This function parallelizes feature extraction across multiple CPU cores
    for efficient processing of large mesh datasets. It automatically handles
    errors and skips invalid files while reporting statistics.
    
    All processing is done on CPU in worker processes to avoid OOM issues.
    Features are returned as numpy arrays and can be moved to GPU in the
    main process if needed.

    Parameters
    ----------
    stl_dir : Path
        Directory containing STL files to process. Only files with ``.stl``
        extension are processed.
    n_workers : int or None, optional
        Number of worker processes to use. If ``None``, defaults to
        ``cpu_count() - 1`` to leave one core available. Default is ``None``.

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
    """
    # Find all STL files in the directory
    paths = sorted(p.as_posix() for p in stl_dir.glob("*.stl"))

    feats: list[np.ndarray] = []
    names: list[str] = []
    errors: list[str] = []

    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    # Prepare arguments for worker function
    tasks = paths

    # Use spawn context for cross-platform compatibility
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        # Process files in parallel with unordered results
        for name, feat, err in pool.map(_process_stl, tasks):
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
        # Print first few errors with filenames for debugging
        for err in errors[:5]:
            print(f"[geometry guardrail]   {err}")
        if len(errors) > 5:
            print(f"[geometry guardrail]   ... and {len(errors) - 5} more")

    return feats, names
