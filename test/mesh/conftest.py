"""Pytest configuration and shared fixtures for physicsnemo.mesh tests.

This module provides common test fixtures, utilities, and parametrization helpers
for exhaustive testing across spatial dimensions, manifold dimensions, and backends.

All functions and fixtures defined here are automatically available to all test files
without explicit imports.
"""

import pytest
import torch

### Pytest Hooks ###


def pytest_collection_modifyitems(config, items):
    """Skip tests marked with 'cuda' if CUDA is not available.

    This hook runs during test collection phase and adds skip markers to CUDA tests
    when CUDA is unavailable. This is the idiomatic pytest approach for conditional
    skipping based on markers.
    """
    if torch.cuda.is_available():
        return  # CUDA available, run all tests

    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)


### Device Management ###


def get_available_devices() -> list[str]:
    """Get list of available compute devices for testing.

    Returns both 'cpu' and 'cuda' (if available). Tests marked with 'cuda'
    will be automatically skipped if CUDA is not available via pytest_collection_modifyitems.
    """
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


### Dimension Configurations ###


# Common dimensional configurations: (n_spatial_dims, n_manifold_dims)
DIMENSION_CONFIGS_2D = [
    (2, 0),  # Points in 2D
    (2, 1),  # Edges in 2D
    (2, 2),  # Triangles in 2D
]

DIMENSION_CONFIGS_3D = [
    (3, 0),  # Points in 3D
    (3, 1),  # Edges in 3D
    (3, 2),  # Triangles in 3D (surfaces)
    (3, 3),  # Tetrahedra in 3D (volumes)
]

DIMENSION_CONFIGS_ALL = DIMENSION_CONFIGS_2D + DIMENSION_CONFIGS_3D

DIMENSION_CONFIGS_CODIM1 = [
    (2, 1),  # Edges in 2D
    (3, 2),  # Surfaces in 3D
]


### Mesh Generators (Standalone Functions) ###


def create_simple_mesh(
    n_spatial_dims: int,
    n_manifold_dims: int,
    device: str = "cpu",
):
    """Create a simple mesh for testing.

    Args:
        n_spatial_dims: Dimension of embedding space (2 or 3)
        n_manifold_dims: Dimension of manifold (0, 1, 2, or 3)
        device: Compute device ('cpu' or 'cuda')

    Returns:
        A simple Mesh instance appropriate for the given dimensions
    """
    from physicsnemo.mesh.mesh import Mesh

    if n_manifold_dims > n_spatial_dims:
        raise ValueError(
            f"Manifold dimension {n_manifold_dims} cannot exceed spatial dimension {n_spatial_dims}"
        )

    if n_manifold_dims == 0:
        # Point cloud
        if n_spatial_dims == 2:
            points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], device=device)
        elif n_spatial_dims == 3:
            points = torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]], device=device
            )
        else:
            raise ValueError(f"Unsupported {n_spatial_dims=}")
        cells = torch.arange(len(points), device=device, dtype=torch.int64).unsqueeze(1)

    elif n_manifold_dims == 1:
        # Polyline
        if n_spatial_dims == 2:
            points = torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [1.5, 1.0], [0.5, 1.5]], device=device
            )
        elif n_spatial_dims == 3:
            points = torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
                device=device,
            )
        else:
            raise ValueError(f"Unsupported {n_spatial_dims=}")
        cells = torch.tensor([[0, 1], [1, 2], [2, 3]], device=device, dtype=torch.int64)

    elif n_manifold_dims == 2:
        # Triangular mesh
        if n_spatial_dims == 2:
            points = torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 0.5]], device=device
            )
        elif n_spatial_dims == 3:
            points = torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [1.5, 0.5, 0.5]],
                device=device,
            )
        else:
            raise ValueError(f"Unsupported {n_spatial_dims=}")
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]], device=device, dtype=torch.int64)

    elif n_manifold_dims == 3:
        # Tetrahedral mesh
        if n_spatial_dims == 3:
            points = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
                device=device,
            )
            cells = torch.tensor(
                [[0, 1, 2, 3], [1, 2, 3, 4]], device=device, dtype=torch.int64
            )
        else:
            raise ValueError("3-simplices require 3D embedding space")
    else:
        raise ValueError(f"Unsupported {n_manifold_dims=}")

    return Mesh(points=points, cells=cells)


def create_single_cell_mesh(
    n_spatial_dims: int,
    n_manifold_dims: int,
    device: str = "cpu",
):
    """Create a mesh with a single cell."""
    from physicsnemo.mesh.mesh import Mesh

    if n_manifold_dims > n_spatial_dims:
        raise ValueError(
            f"Manifold dimension {n_manifold_dims} cannot exceed spatial dimension {n_spatial_dims}"
        )

    if n_manifold_dims == 0:
        if n_spatial_dims == 2:
            points = torch.tensor([[0.5, 0.5]], device=device)
        elif n_spatial_dims == 3:
            points = torch.tensor([[0.5, 0.5, 0.5]], device=device)
        else:
            raise ValueError(f"Unsupported {n_spatial_dims=}")
        cells = torch.tensor([[0]], device=device, dtype=torch.int64)

    elif n_manifold_dims == 1:
        if n_spatial_dims == 2:
            points = torch.tensor([[0.0, 0.0], [1.0, 0.0]], device=device)
        elif n_spatial_dims == 3:
            points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=device)
        else:
            raise ValueError(f"Unsupported {n_spatial_dims=}")
        cells = torch.tensor([[0, 1]], device=device, dtype=torch.int64)

    elif n_manifold_dims == 2:
        if n_spatial_dims == 2:
            points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], device=device)
        elif n_spatial_dims == 3:
            points = torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device
            )
        else:
            raise ValueError(f"Unsupported {n_spatial_dims=}")
        cells = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int64)

    elif n_manifold_dims == 3:
        if n_spatial_dims == 3:
            points = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                device=device,
            )
            cells = torch.tensor([[0, 1, 2, 3]], device=device, dtype=torch.int64)
        else:
            raise ValueError("3-simplices require 3D embedding space")
    else:
        raise ValueError(f"Unsupported {n_manifold_dims=}")

    return Mesh(points=points, cells=cells)


### Assertion Helpers ###


def assert_mesh_valid(mesh, strict: bool = True) -> None:
    """Assert that a mesh is valid and well-formed."""
    assert mesh.n_points > 0, "Mesh must have at least one point"
    assert mesh.points.ndim == 2, f"Points must be 2D tensor, got {mesh.points.ndim=}"
    assert mesh.points.shape[1] == mesh.n_spatial_dims, (
        f"Points shape mismatch: {mesh.points.shape[1]=} != {mesh.n_spatial_dims=}"
    )

    if mesh.n_cells > 0:
        assert mesh.cells.ndim == 2, f"Cells must be 2D tensor, got {mesh.cells.ndim=}"
        assert mesh.cells.shape[1] == mesh.n_manifold_dims + 1, (
            f"Cells shape mismatch: {mesh.cells.shape[1]=} != {mesh.n_manifold_dims + 1=}"
        )
        assert torch.all(mesh.cells >= 0), "Cell indices must be non-negative"
        assert torch.all(mesh.cells < mesh.n_points), (
            f"Cell indices out of bounds: max={mesh.cells.max()}, n_points={mesh.n_points}"
        )

    assert mesh.points.dtype in [
        torch.float32,
        torch.float64,
    ], f"Points must be float type, got {mesh.points.dtype=}"
    assert mesh.cells.dtype == torch.int64, (
        f"Cells must be int64, got {mesh.cells.dtype=}"
    )
    assert mesh.points.device == mesh.cells.device, (
        f"Device mismatch: {mesh.points.device=} != {mesh.cells.device=}"
    )

    if strict and mesh.n_cells > 0:
        for i in range(mesh.n_cells):
            cell_verts = mesh.cells[i]
            unique_verts = torch.unique(cell_verts)
            assert len(unique_verts) == len(cell_verts), (
                f"Cell {i} has duplicate vertices: {cell_verts.tolist()}"
            )


def assert_on_device(tensor: torch.Tensor, expected_device: str) -> None:
    """Assert tensor is on expected device."""
    actual_device = tensor.device.type
    assert actual_device == expected_device, (
        f"Device mismatch: tensor is on {actual_device!r}, expected {expected_device!r}"
    )


### Pytest Fixtures ###


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
    ]
)
def device(request):
    """Parametrize tests over all available devices (CPU, CUDA).

    CUDA tests are automatically skipped if CUDA is not available via
    the pytest_collection_modifyitems hook.
    """
    return request.param


@pytest.fixture(params=DIMENSION_CONFIGS_2D)
def dims_2d(request):
    """Parametrize over 2D dimension configurations."""
    return request.param


@pytest.fixture(params=DIMENSION_CONFIGS_3D)
def dims_3d(request):
    """Parametrize over 3D dimension configurations."""
    return request.param


@pytest.fixture(params=DIMENSION_CONFIGS_ALL)
def dims_all(request):
    """Parametrize over all dimension combinations."""
    return request.param


@pytest.fixture(params=DIMENSION_CONFIGS_CODIM1)
def dims_codim1(request):
    """Parametrize over codimension-1 configurations."""
    return request.param
