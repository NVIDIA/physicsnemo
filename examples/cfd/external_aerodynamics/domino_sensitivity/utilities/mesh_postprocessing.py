import pyvista as pv
import numpy as np
import numba
from typing import Literal
from tqdm import tqdm


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
    neighbors = []

    for i in tqdm(range(n), desc="Building neighbors list"):
        if location == "points":
            neighbors.append(mesh.point_neighbors(i))
        else:
            neighbors.append(mesh.cell_neighbors(i, "edges"))

    # Convert to float for computation
    smoothed = values.astype(float, copy=True)
    # Iteratively apply Laplacian smoothing
    for _ in tqdm(range(iterations), desc="Laplacian smoothing"):
        new_vals = smoothed.copy()
        for i in range(n):
            nbrs = neighbors[i]
            if not nbrs:
                continue  # isolated point or cell has no change
            if smoothed.ndim == 1:
                # Scalar case
                total = smoothed[i] + smoothed[nbrs].sum()
                count = len(nbrs) + 1
                new_vals[i] = total / count
            else:
                # Vector case
                total = smoothed[i] + smoothed[nbrs].sum(axis=0)
                count = len(nbrs) + 1
                new_vals[i] = total / count
        smoothed = new_vals
    return smoothed


if __name__ == "__main__":
    import pyvista as pv

    # Example 1: Smooth a random scalar on a sphere (point data)
    sphere = pv.Sphere(theta_resolution=20, phi_resolution=20)
    np.random.seed(0)
    vals = np.random.rand(sphere.n_points)  # random values per point
    s_vals = laplacian_smoothing(sphere, vals, location="points", iterations=20)
    print("Sphere scalar field: min,max before =", vals.min(), vals.max())
    print("After 5 iters: min,max =", s_vals.min(), s_vals.max())
    sphere["vals"] = vals
    sphere["s_vals"] = s_vals
    sphere.plot(scalars="vals", cmap="turbo")
    sphere.plot(scalars="s_vals", cmap="turbo")

    # # Example 2: Smooth the point coordinates (vector data) of a slightly perturbed plane
    # plane = pv.Plane(i_resolution=10, j_resolution=10)
    # # Create a noisy plane by displacing z-coords randomly
    # pts = plane.points.copy()
    # pts[:, 2] += 0.2 * np.random.randn(pts.shape[0])
    # plane.points = pts
    # # Use the coordinates as a vector field to be smoothed
    # coords = plane.points.copy()
    # smoothed_coords = laplacian_smoothing(plane, coords, location='points', iterations=10)
    # print("\nPlane coords sample before (first 3 pts):\n", coords[:3])
    # print("After smoothing (first 3 pts):\n", smoothed_coords[:3])

    # # Example 3: Smooth a per-cell scalar on the sphere (cell data)
    # cell_centers = sphere.cell_centers().points
    # # Define a simple scalar: distance of cell-center from origin (should be nearly constant for sphere)
    # cell_vals = np.linalg.norm(cell_centers, axis=1)
    # smoothed_cells = laplacian_smoothing(sphere, cell_vals, location='cells', iterations=3)
    # print("\nSphere cell-values (radius) before smoothing: {:.3f} ± {:.3f}".format(
    #     cell_vals.mean(), cell_vals.std()))
    # print("After 3 iters: {:.3f} ± {:.3f}".format(
    #     smoothed_cells.mean(), smoothed_cells.std()))
