# ruff: noqa: S101
import pytest
import torch

from physicsnemo.models.globe.boundary_mesh import BoundaryMesh


def make_tetrahedron_mesh() -> BoundaryMesh:
    """Create a small non-trivial 3D boundary mesh with multiple triangular faces.

    This constructs a tetrahedron (4 points, 4 triangular faces) in 3D, which is
    simple yet exercises normal and area computations beyond a single face.

    Args:
        device: Target device for the returned tensors.

    Returns:
        BoundaryMesh: A tetrahedral surface mesh with boundary condition type "no_slip".
    """
    # Four corners of a tetrahedron
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    # Four triangular faces (each is a triplet of vertex indices)
    faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=torch.long,
    )

    return BoundaryMesh(points=points, faces=faces, boundary_condition_type="no_slip")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_boundary_mesh_properties(device: str) -> None:
    """Validate basic geometric properties and shapes for the tetrahedral mesh."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    bm = make_tetrahedron_mesh().to(device)

    assert bm.n_points == 4
    assert bm.n_faces == 4
    assert bm.n_spatial_dims == 3

    assert bm.face_centers.shape == (4, 3)
    assert bm.face_normals.shape == (4, 3)
    assert bm.face_areas.shape == (4,)

    # Normals should be unit length (within tolerance)
    lengths = torch.norm(bm.face_normals, dim=-1)
    assert torch.allclose(lengths, torch.ones_like(lengths), atol=1e-5)

    # Areas should be positive and non-trivial
    assert torch.all(bm.face_areas > 0)

    # Verify tensors are on the correct device
    assert bm.points.device.type == device
    assert bm.faces.device.type == device
    assert bm.face_centers.device.type == device
    assert bm.face_normals.device.type == device
    assert bm.face_areas.device.type == device


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_boundary_mesh_merge(device: str) -> None:
    """Merging two meshes should concatenate points/faces and preserve bc type."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    bm1 = make_tetrahedron_mesh().to(device)
    # Translate a copy to avoid overlapping vertices
    bm2 = make_tetrahedron_mesh().to(device)
    bm2.points = bm2.points + torch.tensor([[2.0, 0.0, 0.0]], device=device)

    merged = BoundaryMesh.merge([bm1, bm2])

    assert merged.n_points == bm1.n_points + bm2.n_points
    assert merged.n_faces == bm1.n_faces + bm2.n_faces
    assert (
        merged.boundary_condition_type
        == bm1.boundary_condition_type
        == bm2.boundary_condition_type
    )

    # Basic geometric sanity
    assert merged.face_centers.shape == (merged.n_faces, 3)
    assert torch.all(merged.face_areas > 0)

    # Verify tensors are on the correct device
    assert merged.points.device.type == device
    assert merged.faces.device.type == device
    assert merged.face_centers.device.type == device
    assert merged.face_normals.device.type == device
    assert merged.face_areas.device.type == device


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("alpha", [0.25, 1.0, 4.0])
def test_sample_random_points_on_faces(device: str, alpha: float) -> None:
    """Test that sample_random_points_on_faces runs without errors for various alpha values."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    bm = make_tetrahedron_mesh().to(device)

    # Sample random points on faces
    random_points = bm.sample_random_points_on_faces(alpha=alpha)

    # Check output shape
    assert random_points.shape == (bm.n_faces, bm.n_spatial_dims)

    # Check that points are on the correct device
    assert random_points.device.type == device

    # Check that all points are finite (no NaNs or Infs)
    assert torch.all(torch.isfinite(random_points))

    # For each face, verify that the sampled point lies within the face's bounding box
    # (a necessary but not sufficient condition for being inside the simplex)
    for i in range(bm.n_faces):
        face_vertices = bm.points[bm.faces[i]]
        sampled_point = random_points[i]

        # Check that point is within the bounding box of face vertices
        min_coords = face_vertices.min(dim=0).values
        max_coords = face_vertices.max(dim=0).values

        assert torch.all(
            sampled_point >= min_coords - 1e-5
        )  # Small tolerance for FP errors
        assert torch.all(sampled_point <= max_coords + 1e-5)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("power", [1.5, 2.0])
def test_pad_to_next_power_surface_area_unchanged(device: str, power: float) -> None:
    """Test that total surface area remains unchanged after padding (degenerate faces have zero area)."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    bm = make_tetrahedron_mesh().to(device)
    original_total_area = bm.face_areas.sum()

    padded = bm.pad_to_next_power(power=power)
    padded_total_area = padded.face_areas.sum()

    # Degenerate padding faces should have zero area, so total area should be unchanged
    assert torch.allclose(original_total_area, padded_total_area, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("power", [1.5, 2.0])
def test_pad_to_next_power_reversibility(device: str, power: float) -> None:
    """Test that padding is losslessly reversible - we can extract the original mesh exactly."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    bm = make_tetrahedron_mesh().to(device)

    # Add some face data to test that it's also reversible
    bm.face_data["pressure"] = torch.randn(bm.n_faces, device=device)
    bm.face_data["velocity"] = torch.randn(bm.n_faces, 3, device=device)

    original_n_points = bm.n_points
    original_n_faces = bm.n_faces

    padded = bm.pad_to_next_power(power=power)

    # Extract the original mesh by slicing
    recovered_points = padded.points[:original_n_points]
    recovered_faces = padded.faces[:original_n_faces]
    recovered_face_data = padded.face_data[:original_n_faces]

    # Check exact equality
    assert torch.equal(bm.points, recovered_points)
    assert torch.equal(bm.faces, recovered_faces)
    assert torch.equal(bm.face_data["pressure"], recovered_face_data["pressure"])
    assert torch.equal(bm.face_data["velocity"], recovered_face_data["velocity"])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pad_to_next_power_only_points_padded(device: str) -> None:
    """Test case where only the points array needs padding, not faces."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create a mesh with many points but few faces
    # e.g., 100 points, but only 4 faces
    n_points = 100
    points = torch.randn(n_points, 3, device=device)

    # Only use 4 faces (referencing first 12 points in groups of 3)
    faces = torch.tensor(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=torch.long, device=device
    )

    bm = BoundaryMesh(points=points, faces=faces, boundary_condition_type="no_slip")

    original_n_points = bm.n_points
    original_n_faces = bm.n_faces

    padded = bm.pad_to_next_power(power=1.5)

    # Points should be padded (100 -> next power of 1.5)
    assert padded.n_points > original_n_points

    # Faces might or might not be padded depending on whether 4 is already
    # at a power of 1.5, but in any case should be >= original
    assert padded.n_faces >= original_n_faces

    # Original points should be unchanged
    assert torch.equal(bm.points, padded.points[:original_n_points])

    # Padding points should be copies of the last existing point
    if padded.n_points > original_n_points:
        padding_points = padded.points[original_n_points:]
        last_point = bm.points[-1]
        for i in range(len(padding_points)):
            assert torch.allclose(padding_points[i], last_point)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pad_to_next_power_only_faces_padded(device: str) -> None:
    """Test case where only the faces array needs padding, not points."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create a mesh with few points but many faces
    # e.g., 4 points forming many triangles
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )

    # Create 100 faces (many will be duplicates, but that's fine for testing)
    n_faces = 100
    faces = torch.stack(
        [
            torch.zeros(n_faces, dtype=torch.long, device=device),
            torch.ones(n_faces, dtype=torch.long, device=device),
            torch.full((n_faces,), 2, dtype=torch.long, device=device),
        ],
        dim=1,
    )

    bm = BoundaryMesh(points=points, faces=faces, boundary_condition_type="no_slip")

    original_n_points = bm.n_points
    original_n_faces = bm.n_faces

    padded = bm.pad_to_next_power(power=1.5)

    # Points might or might not be padded, but should be >= original
    assert padded.n_points >= original_n_points

    # Faces should be padded (100 -> next power of 1.5)
    assert padded.n_faces > original_n_faces

    # Original faces should be unchanged
    assert torch.equal(bm.faces, padded.faces[:original_n_faces])

    # Padding faces should all reference the last existing point (degenerate)
    if padded.n_faces > original_n_faces:
        padding_faces = padded.faces[original_n_faces:]
        expected_index = original_n_points - 1
        assert torch.all(padding_faces == expected_index)

        # These degenerate faces should have zero area
        padding_face_areas = padded.face_areas[original_n_faces:]
        assert torch.allclose(
            padding_face_areas, torch.zeros_like(padding_face_areas), atol=1e-6
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pad_to_next_power_face_data_padding(device: str) -> None:
    """Test that face_data is padded correctly with zeros."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    bm = make_tetrahedron_mesh().to(device)

    # Add various types of face data
    bm.face_data["scalar_field"] = torch.randn(bm.n_faces, device=device)
    bm.face_data["vector_field"] = torch.randn(bm.n_faces, 3, device=device)
    bm.face_data["tensor_field"] = torch.randn(bm.n_faces, 3, 3, device=device)

    original_n_faces = bm.n_faces

    padded = bm.pad_to_next_power(power=1.5)

    # Check that all face_data keys are preserved
    assert set(padded.face_data.keys()) == set(bm.face_data.keys())

    # Check that original face data is unchanged
    for key in bm.face_data.keys():
        assert torch.equal(bm.face_data[key], padded.face_data[key][:original_n_faces])

    # Check that padding face data is zeros
    if padded.n_faces > original_n_faces:
        for key, original_value in bm.face_data.items():
            padding_data = padded.face_data[key][original_n_faces:]
            expected_zeros = torch.zeros(
                (padded.n_faces - original_n_faces, *original_value.shape[1:]),
                dtype=original_value.dtype,
                device=device,
            )
            assert torch.equal(padding_data, expected_zeros)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pad_to_next_power_cached_properties(device: str) -> None:
    """Test that cached properties are correctly computed for the padded mesh."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    bm = make_tetrahedron_mesh().to(device)

    # Access cached properties on original mesh
    _ = bm.face_centers
    _ = bm.face_normals
    _ = bm.face_areas

    padded = bm.pad_to_next_power(power=1.5)

    # Cached properties should exist and have correct shapes
    assert padded.face_centers.shape == (padded.n_faces, 3)
    assert padded.face_normals.shape == (padded.n_faces, 3)
    assert padded.face_areas.shape == (padded.n_faces,)

    # Original faces should have unchanged properties
    original_n_faces = bm.n_faces
    assert torch.allclose(
        bm.face_centers, padded.face_centers[:original_n_faces], atol=1e-6
    )
    assert torch.allclose(
        bm.face_normals, padded.face_normals[:original_n_faces], atol=1e-6
    )
    assert torch.allclose(
        bm.face_areas, padded.face_areas[:original_n_faces], atol=1e-6
    )

    # Padding faces should have zero area
    if padded.n_faces > original_n_faces:
        padding_areas = padded.face_areas[original_n_faces:]
        assert torch.allclose(padding_areas, torch.zeros_like(padding_areas), atol=1e-6)

    # All normals should still be unit length (or NaN for degenerate faces with zero cross product)
    for i in range(padded.n_faces):
        normal_length = torch.norm(padded.face_normals[i])
        # Either unit length, zero, or NaN (for degenerate faces where normalization fails)
        is_valid = (
            torch.isclose(normal_length, torch.tensor(1.0, device=device), atol=1e-5)
            or torch.isclose(normal_length, torch.tensor(0.0, device=device), atol=1e-6)
            or torch.isnan(normal_length)
        )
        assert is_valid


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("power", [1.5, 2.0, 3.0])
def test_pad_to_next_power_sizes(device: str, power: float) -> None:
    """Test that padding produces the correct target sizes."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    bm = make_tetrahedron_mesh().to(device)

    padded = bm.pad_to_next_power(power=power)

    # Verify that padded sizes are >= original sizes
    assert padded.n_points >= bm.n_points
    assert padded.n_faces >= bm.n_faces

    # Check that the sizes are floor(power^n) for some integer n
    # by verifying that floor(power^ceil(log_power(size))) == size
    import math

    if padded.n_points > 1:
        n_points_exponent = math.ceil(math.log(padded.n_points) / math.log(power))
        expected_n_points = int(math.floor(power**n_points_exponent))
        assert padded.n_points == expected_n_points

    if padded.n_faces > 1:
        n_faces_exponent = math.ceil(math.log(padded.n_faces) / math.log(power))
        expected_n_faces = int(math.floor(power**n_faces_exponent))
        assert padded.n_faces == expected_n_faces


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pad_to_next_power_invalid_power(device: str) -> None:
    """Test that invalid power values raise appropriate errors."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    bm = make_tetrahedron_mesh().to(device)

    # power must be > 1
    with pytest.raises(ValueError, match="power must be > 1"):
        bm.pad_to_next_power(power=1.0)

    with pytest.raises(ValueError, match="power must be > 1"):
        bm.pad_to_next_power(power=0.5)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pad_explicit_sizes(device: str) -> None:
    """Test the low-level .pad() method with explicit target sizes."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    bm = make_tetrahedron_mesh().to(device)

    # Pad both points and faces
    padded = bm.pad(target_n_points=10, target_n_faces=10)
    assert padded.n_points == 10
    assert padded.n_faces == 10

    # Verify original data is unchanged
    assert torch.equal(bm.points, padded.points[: bm.n_points])
    assert torch.equal(bm.faces, padded.faces[: bm.n_faces])

    # Pad only points
    padded_points_only = bm.pad(target_n_points=8)
    assert padded_points_only.n_points == 8
    assert padded_points_only.n_faces == bm.n_faces

    # Pad only faces
    padded_faces_only = bm.pad(target_n_faces=8)
    assert padded_faces_only.n_points == bm.n_points
    assert padded_faces_only.n_faces == 8

    # No padding (should return self)
    no_padding = bm.pad()
    assert no_padding is bm


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pad_invalid_sizes(device: str) -> None:
    """Test that .pad() raises errors for invalid target sizes."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    bm = make_tetrahedron_mesh().to(device)

    # Target smaller than current should raise error
    with pytest.raises(ValueError, match="target_n_points"):
        bm.pad(target_n_points=2)  # bm has 4 points

    with pytest.raises(ValueError, match="target_n_faces"):
        bm.pad(target_n_faces=2)  # bm has 4 faces


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pad_to_next_power_no_nans_in_normals(device: str) -> None:
    """Test that padding doesn't produce NaN values in face normals (regression test)."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    bm = make_tetrahedron_mesh().to(device)
    padded = bm.pad_to_next_power(power=1.5)

    # Check that face normals contain no NaN values
    assert not torch.any(torch.isnan(padded.face_normals)), (
        "Padded mesh face_normals contain NaN values"
    )

    # Check that face centers contain no NaN values
    assert not torch.any(torch.isnan(padded.face_centers)), (
        "Padded mesh face_centers contain NaN values"
    )

    # Check that face areas contain no NaN values
    assert not torch.any(torch.isnan(padded.face_areas)), (
        "Padded mesh face_areas contain NaN values"
    )

    # All values should be finite
    assert torch.all(torch.isfinite(padded.face_normals))
    assert torch.all(torch.isfinite(padded.face_centers))
    assert torch.all(torch.isfinite(padded.face_areas))
