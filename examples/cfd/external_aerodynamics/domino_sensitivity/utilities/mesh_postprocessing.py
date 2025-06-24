# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import pyvista as pv
import numpy as np
from tqdm import tqdm
import numba
from typing import Literal


@numba.njit
def edges_to_adjacency(
    sorted_bidirectional_edges: np.ndarray, n_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a sorted bidirectional edge list to a compressed adjacency list representation.

    This function implements a compressed sparse row (CSR) format for storing graph adjacency
    information. It takes a sorted list of bidirectional edges and converts it into two arrays:
    offsets and indices, which together efficiently represent the adjacency relationships.

    Parameters
    ----------
    sorted_bidirectional_edges : np.ndarray
        A 2D array of shape (n_edges, 2) where each row contains the start and end indices
        of an edge. Edges must be sorted by increasing start index, then increasing end index.
        Each edge is listed twice, once in each direction (e.g., [0,1] and [1,0]).

    n_points : int
        The number of points in the mesh. Usually equivalent to
        np.max(sorted_bidirectional_edges) + 1, though only true if there are
        no unconnected points. This determines the size of the offsets array.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - offsets: A 1D array of shape (n_points + 1,) where offsets[i] gives the starting
          index in the indices array for the neighbors of point i. The last element
          offsets[n_points] equals the total number of edges.
        - indices: A 1D array containing the neighbor indices for all points. For point i,
          its neighbors are stored in indices[offsets[i]:offsets[i+1]].

    Notes
    -----
    This implementation uses a compressed sparse row (CSR) format, which is memory efficient
    for sparse graphs. The total memory required is O(n_points + n_edges) instead of
    O(n_points^2) for a dense adjacency matrix.

    Examples
    --------
    >>> # Create a simple graph with 4 points and 3 bidirectional edges
    >>> edges = np.array([
    ...     [0, 1],  # Edge 0->1
    ...     [1, 0],  # Edge 1->0
    ...     [1, 2],  # Edge 1->2
    ...     [2, 1],  # Edge 2->1
    ...     [2, 3],  # Edge 2->3
    ...     [3, 2],  # Edge 3->2
    ... ])
    >>> offsets, indices = edges_to_adjacency(edges, 4)
    >>> # Point 0's neighbors are in indices[offsets[0]:offsets[1]] = [1]
    >>> # Point 1's neighbors are in indices[offsets[1]:offsets[2]] = [0, 2]
    >>> # Point 2's neighbors are in indices[offsets[2]:offsets[3]] = [1, 3]
    >>> # Point 3's neighbors are in indices[offsets[3]:offsets[4]] = [2]
    """
    n_edges = len(sorted_bidirectional_edges)
    offsets = np.zeros(n_points + 1, dtype=sorted_bidirectional_edges.dtype)
    indices = np.zeros(n_edges, dtype=sorted_bidirectional_edges.dtype)

    edge_idx = 0
    for adj_index in range(n_points):
        start_offset = offsets[adj_index]
        while edge_idx < n_edges:
            start_idx = sorted_bidirectional_edges[edge_idx, 0]
            if start_idx == adj_index:
                indices[start_offset] = sorted_bidirectional_edges[edge_idx, 1]
                start_offset += 1
            elif start_idx > adj_index:
                break
            edge_idx += 1
        offsets[adj_index + 1] = start_offset

    return offsets, indices


edges_to_adjacency(np.zeros((0, 2), dtype=np.int64), 0)  # Does the precompilation


def unique_axis0(array):
    """
    A faster version of np.unique(array, axis=0) for 2D arrays.

    Returns
    -------
    np.ndarray
        The unique rows of the input array.

    Notes
    -----
    ~25x faster than np.unique(array, axis=0) on PyVista brain mesh non-unique edge array.
    """
    idxs = np.lexsort(array.T[::-1])
    array = array[idxs]
    unique_idxs = np.empty(len(array), dtype=np.bool_)
    unique_idxs[0] = True
    unique_idxs[1:] = np.any(array[:-1, :] != array[1:, :], axis=-1)
    return array[unique_idxs]


def get_edges(mesh: pv.DataSet) -> np.ndarray:
    """
    Given a mesh, returns a 2D array of shape (n_edges, 2) where each row contains the start
    and end indices of an edge. Edges are sorted by increasing start index, then increasing end index.
    Each edge is listed twice, once in each direction.

    Parameters
    ----------
    mesh : pv.DataSet
        The input mesh.

    Returns
    -------
    np.ndarray
        A 2D array of shape (n_edges, 2) with the start and end indices of each edge.

    Examples
    --------
    These should be identical:

    >>> edges = get_edges(mesh)
    >>> order = np.lexsort(edges.T[::-1])
    >>> edges = edges[order]

    and

    >>> edges = mesh.extract_all_edges(use_all_points=True, clear_data=True).lines.reshape(-1, 3)[:, 1:]
    """
    edges_from_all_cell_types: list[np.ndarray] = []

    cells_dict = mesh.cells_dict

    for cell_type, cells in cells_dict.items():
        ### Determine the canonical edges for this particular cell type
        # First, create a canonical cell (i.e., a mesh with a single cell of this same type)
        # The purpose is to dynamically determine edge connectivity for this cell type, which
        # we will then vectorize onto all cells of this type in the mesh
        n_vertices_per_cell = cells.shape[1]
        canonical_cell = pv.UnstructuredGrid(
            np.concatenate(
                [np.array([n_vertices_per_cell]), np.arange(n_vertices_per_cell)]
            ),
            [cell_type],
            np.zeros((n_vertices_per_cell, 3), dtype=float),
        )
        canonical_edges = canonical_cell.extract_all_edges(
            use_all_points=True, clear_data=True
        ).lines.reshape(-1, 3)[:, 1:]

        ### Now, map this onto all cells of this type in the mesh
        edges_from_this_cell_type = np.empty(
            (len(canonical_edges) * len(cells), 2), dtype=np.int64
        )
        for i, edge in enumerate(canonical_edges):
            edges_from_this_cell_type[i * len(cells) : (i + 1) * len(cells)] = cells[
                :, edge
            ]

        edges_from_all_cell_types.append(edges_from_this_cell_type)

    if len(edges_from_all_cell_types) == 1:
        # No need to make a memory copy in this case (which np.concatenate forces)
        edges = edges_from_all_cell_types[0]

    else:
        edges = np.concatenate(edges_from_all_cell_types, axis=0)

    ### Now, eliminate duplicate edges
    # Identical to np.sort(edges, axis=1) for Nx2 arrays, but faster
    edges = np.where(np.diff(edges, axis=1) >= 0, edges, edges[:, ::-1])

    edges = unique_axis0(edges)

    return edges


def build_point_adjacency(
    mesh: pv.DataSet, progress_bar=False
) -> tuple[np.ndarray, np.ndarray]:
    """Build a compressed adjacency list representation for the points in a mesh.

    This function extracts the edge connectivity from a mesh and converts it into a compressed
    sparse row (CSR) format adjacency list. It first extracts all edges from the mesh, including
    both directions of each edge, then sorts them and converts them to the CSR format.

    Parameters
    ----------
    mesh : pv.DataSet
        The input mesh. Can be any PyVista mesh type that supports edge extraction.

    progress_bar : bool, optional
        Whether to display a progress bar during edge computation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - offsets: A 1D array of shape (n_points + 1,) where offsets[i] gives the starting
          index in the indices array for the neighbors of point i. The last element
          offsets[n_points] equals the total number of edges.
        - indices: A 1D array containing the neighbor indices for all points. For point i,
          its neighbors are stored in indices[offsets[i]:offsets[i+1]].

    Notes
    -----
    This function:
    1. Extracts all edges from the mesh using PyVista's edge extraction
    2. Creates bidirectional edges by duplicating each edge in reverse
    3. Sorts the edges by start point index, then end point index
    4. Converts the sorted edges to CSR format using edges_to_adjacency

    The resulting adjacency list is memory efficient and allows for fast neighbor lookups.
    The total memory required is O(n_points + n_edges) instead of O(n_points^2) for a
    dense adjacency matrix.

    Examples
    --------
    >>> import pyvista as pv
    >>> # Create a simple mesh (a cube)
    >>> mesh = pv.Cube()
    >>> offsets, indices = build_point_adjacency(mesh)
    >>> # For each point i, its neighbors are in indices[offsets[i]:offsets[i+1]]
    >>> # For example, to get all neighbors of point 0:
    >>> point_0_neighbors = indices[offsets[0]:offsets[1]]
    """
    # edges = get_edges(mesh)

    ### This is the old way of getting edges (slower) + consistency check
    edge_mesh = mesh.extract_all_edges(
        use_all_points=True, clear_data=True, progress_bar=progress_bar
    )
    edges = edge_mesh.lines.reshape(-1, 3)[:, 1:]
    # order = np.lexsort(edges.T[::-1])
    # edges = edges[order]
    # assert np.all(edges == edges_old)

    # Includes not only edge [a, b] but also edge [b, a]
    bidirectional_edges = np.concatenate((edges, edges[:, ::-1]), axis=0)

    # Puts edges in order of increasing start point index, then increasing end point index
    order = np.lexsort((bidirectional_edges[:, 1], bidirectional_edges[:, 0]))
    sorted_bidirectional_edges = bidirectional_edges[order]

    adjacency = edges_to_adjacency(sorted_bidirectional_edges, mesh.n_points)

    return adjacency


@numba.njit(parallel=True)
def _smooth(
    values: np.ndarray,
    offsets: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Performs a single iteration of Laplacian smoothing on a set of values.

    For each value in the input array, computes the average of that value and its neighbors,
    and returns the smoothed values. This is a helper function used by the main laplacian_smoothing
    function to perform the actual smoothing computation.

    Args:
        values: Array of values to smooth. Can be any numeric type.
        offsets: Array of offsets, where offsets[i] gives the starting index in the indices array
            for the neighbors of point i. The last element offsets[n_points] equals the total number
            of edges.
        indices: Array containing the neighbor indices for all points. For point i, its neighbors
            are stored in indices[offsets[i]:offsets[i+1]].

    Returns:
        Array of smoothed values with the same shape and type as the input values array.
        For values with no neighbors, the original value is preserved.
    """
    new_values = np.empty_like(values)
    # for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
    for i in numba.prange(len(values)):
        start = offsets[i]
        end = offsets[i + 1]
        nbrs = indices[start:end]
        if len(nbrs) == 0:
            new_values[i] = values[i]
        else:
            total = values[i] + np.sum(values[nbrs])
            count = len(nbrs) + 1
            new_values[i] = total / count
    return new_values


def laplacian_smoothing(
    mesh: pv.PolyData,
    values: np.ndarray,
    location: Literal["points", "cells"] = "points",
    iterations: int = 10,
) -> np.ndarray:
    """
    Perform Laplacian smoothing of an array on a mesh.

    This array is a scalar or vector field defined on the surface of the mesh.

    Args:
        mesh: a PyVista mesh representing a surface. Note that the function only cares about the mesh's topology,
            not its geometry.

        array: An array of values that represent some quantity defined on the surface.
            Can be either a scalar array (shape: (N,)) or a vector array (shape: (N, 3)).
            Can be either defined on points or cells, depending on the `location` argument; the length of
            the array `N` must match the number of points or cells in the mesh, depending on the `location` argument.
            Any combination of scalar/vector fields and point/cell locations is supported.

        location: Whether the array is defined on points or cells.

        n_iterations: The number of iterations of Laplacian smoothing to perform.
            In each iteration, we perform the following steps:
            - For each (point/cell) i, compute the average value of the quantity of its 1-ring neighbors.
            - Update the value of the quantity for point/cell i to the average value computed in the previous step.

    Returns:
        The array after Laplacian smoothing.

    Example:

    >>> # Create a simple mesh
    >>> mesh = pv.Sphere(center=(0, 0, 0), radius=1)
    >>>
    >>> # Define a scalar field on the mesh (points)
    >>> scalar_field = np.random.rand(mesh.n_points)
    >>>
    >>> # Smooth the scalar field
    >>> smoothed_scalar_field = laplacian_smoothing(mesh, scalar_field, location="points", n_iterations=10)
    >>> # Returns an array with shape (n_points,)
    >>>
    >>> # Define a vector field on the mesh (cells)
    >>> vector_field = np.random.rand(mesh.n_cells, 3)
    >>>
    >>> # Smooth the vector field
    >>> smoothed_vector_field = laplacian_smoothing(mesh, vector_field, location="cells", n_iterations=10)
    >>> # Returns an array with shape (n_cells, 3)
    """
    # Ensure numpy array
    values = np.asarray(values)

    # Determine expected size
    if location == "points":
        n = mesh.n_points
    elif location == "cells":
        n = mesh.n_cells
    else:
        raise ValueError("`location` must be 'points' or 'cells'")

    # Check array shape
    if values.ndim == 1:
        if values.shape[0] != n:
            raise ValueError("Length of values must match number of mesh %s" % location)
    elif values.ndim == 2 and values.shape[1] == 3:
        if values.shape[0] != n:
            raise ValueError(
                "Number of vectors must match number of mesh %s" % location
            )
    else:
        raise ValueError("`values` must be a (N,) scalar array or (N,3) vector array")

    # Build adjacency list of neighbors
    if location == "points":
        offsets, indices = build_point_adjacency(mesh)
    else:
        raise NotImplementedError("Cell-based smoothing not implemented yet")

    # Convert to float for computation
    smoothed = values.astype(float, copy=True)
    # Iteratively apply Laplacian smoothing
    for _ in tqdm(range(iterations), desc="Laplacian smoothing"):
        smoothed = _smooth(smoothed, offsets, indices)
    return smoothed


if __name__ == "__main__":
    import pyvista as pv

    # Example: Smooth a random scalar on a sphere (point data)
    sphere = pv.Sphere(theta_resolution=200, phi_resolution=200)
    np.random.seed(0)
    vals = np.random.rand(sphere.n_points)  # random values per point
    s_vals = laplacian_smoothing(sphere, vals, location="points", iterations=20)
    print("Sphere scalar field: min,max before =", vals.min(), vals.max())
    print("After 5 iters: min,max =", s_vals.min(), s_vals.max())
    sphere["vals"] = vals
    sphere["s_vals"] = s_vals
    sphere.plot(scalars="vals", cmap="turbo")
    sphere.plot(scalars="s_vals", cmap="turbo")
