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

import numpy as np
import pytest
import torch

from physicsnemo.nn.functional import mesh_poisson_disk_sample
from physicsnemo.nn.functional.geometry import MeshPoissonDiskSample
from test.conftest import requires_module


# Build a deterministic watertight mesh for sampling tests.
def _build_case(device: str, subdivisions: int = 2):
    from physicsnemo.mesh.primitives.procedural.lumpy_sphere import (
        load as load_lumpy_sphere,
    )

    mesh = load_lumpy_sphere(subdivisions=subdivisions, device=device)
    mesh_vertices = mesh.points.to(torch.float32).contiguous()
    mesh_indices_2d = mesh.cells.to(torch.int32).contiguous()
    return mesh_vertices, mesh_indices_2d


# Sort point rows lexicographically for order-invariant comparisons.
def _sorted_points(points: torch.Tensor) -> torch.Tensor:
    if points.numel() == 0:
        return points
    points_np = points.detach().cpu().numpy()
    sort_idx = np.lexsort((points_np[:, 2], points_np[:, 1], points_np[:, 0]))
    sorted_np = points_np[sort_idx]
    return torch.from_numpy(sorted_np).to(device=points.device, dtype=points.dtype)


# Compute minimum non-diagonal pairwise distance.
def _minimum_pairwise_distance(points: torch.Tensor) -> float:
    if points.shape[0] < 2:
        return float("inf")
    distance_matrix = torch.cdist(points, points)
    distance_matrix.fill_diagonal_(float("inf"))
    return float(distance_matrix.min().item())


# Validate warp implementation behavior and minimum-distance enforcement.
@requires_module("warp")
def test_mesh_poisson_disk_sample_warp(device: str):
    mesh_vertices, mesh_indices_2d = _build_case(device=device, subdivisions=2)
    min_distance = 0.08
    kwargs = {
        "min_distance": min_distance,
        "batch_size": 4096,
        "max_points": 2048,
        "max_iterations": 10,
        "random_seed": 1234,
        "hash_grid_resolution": 64,
    }

    # Explicitly target the warp implementation for this single-backend functional.
    output_warp = mesh_poisson_disk_sample(
        mesh_vertices,
        mesh_indices_2d,
        implementation="warp",
        **kwargs,
    )

    assert output_warp.ndim == 2
    assert output_warp.shape[1] == 3
    assert output_warp.dtype == torch.float32
    assert output_warp.shape[0] <= kwargs["max_points"]
    assert output_warp.shape[0] > 0

    # Numerical tolerance allows tiny floating-point drift.
    min_pair_distance = _minimum_pairwise_distance(output_warp)
    assert min_pair_distance >= 0.9 * min_distance


# Validate equivalent outputs for flattened and (n_faces, 3) index layouts.
@requires_module("warp")
def test_mesh_poisson_disk_sample_index_layout_compatibility(device: str):
    mesh_vertices, mesh_indices_2d = _build_case(device=device, subdivisions=2)
    kwargs = {
        "min_distance": 0.08,
        "batch_size": 4096,
        "max_points": 2048,
        "max_iterations": 8,
        "random_seed": 2026,
    }

    output_faces = mesh_poisson_disk_sample(
        mesh_vertices,
        mesh_indices_2d,
        implementation="warp",
        **kwargs,
    )
    output_flat = mesh_poisson_disk_sample(
        mesh_vertices,
        mesh_indices_2d.reshape(-1),
        implementation="warp",
        **kwargs,
    )
    if device == "cpu":
        torch.testing.assert_close(
            _sorted_points(output_faces), _sorted_points(output_flat)
        )
    else:
        # GPU launches are not strictly deterministic due parallel conflict resolution.
        count_delta = abs(output_faces.shape[0] - output_flat.shape[0])
        allowed_delta = int(0.15 * max(output_faces.shape[0], output_flat.shape[0])) + 4
        assert count_delta <= allowed_delta
        torch.testing.assert_close(
            output_faces.mean(dim=0),
            output_flat.mean(dim=0),
            atol=5e-2,
            rtol=5e-2,
        )


# Validate adaptive-radius sampling path with per-vertex radii.
@requires_module("warp")
def test_mesh_poisson_disk_sample_adaptive_radius(device: str):
    mesh_vertices, mesh_indices_2d = _build_case(device=device, subdivisions=3)
    z = mesh_vertices[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min()).clamp_min(1.0e-6)
    per_vertex_radius = 0.06 + 0.05 * z_norm

    output = mesh_poisson_disk_sample(
        mesh_vertices,
        mesh_indices_2d,
        min_distance=0.06,
        per_vertex_radius=per_vertex_radius,
        batch_size=4096,
        max_points=2048,
        max_iterations=8,
        random_seed=4242,
        hash_grid_resolution=(64, 64, 64),
        implementation="warp",
    )
    assert output.ndim == 2
    assert output.shape[1] == 3
    assert output.shape[0] > 0
    assert output.shape[0] <= 2048

    # Adaptive mode still enforces meaningful separation.
    min_pair_distance = _minimum_pairwise_distance(output)
    assert min_pair_distance >= 0.05


# Validate weighted sample elimination mode with deterministic output size.
@requires_module("warp")
def test_mesh_poisson_disk_sample_weighted_sample_elimination(device: str):
    mesh_vertices, mesh_indices_2d = _build_case(device=device, subdivisions=3)
    output = mesh_poisson_disk_sample(
        mesh_vertices,
        mesh_indices_2d,
        min_distance=0.05,
        batch_size=4096,
        max_points=2048,
        target_num_points=512,
        mode="weighted_sample_elimination",
        random_seed=1337,
        implementation="warp",
    )
    assert output.ndim == 2
    assert output.shape == (512, 3)
    assert output.dtype == torch.float32

    # Elimination mode should still produce non-degenerate point sets.
    min_pair_distance = _minimum_pairwise_distance(output)
    assert min_pair_distance > 0.0


# Validate weighted mode output quality with simple geometric sanity checks.
@requires_module("warp")
def test_mesh_poisson_disk_sample_weighted_quality_sanity(device: str):
    mesh_vertices, mesh_indices_2d = _build_case(device=device, subdivisions=3)
    target_num_points = 1024
    random_seed = 1337

    output = mesh_poisson_disk_sample(
        mesh_vertices,
        mesh_indices_2d,
        min_distance=0.02,
        batch_size=4096,
        max_points=2048,
        target_num_points=target_num_points,
        mode="weighted_sample_elimination",
        random_seed=random_seed,
        implementation="warp",
    )
    assert output.shape == (target_num_points, 3)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()

    # Points should lie within mesh bounds (up to small floating-point tolerance).
    bbox_min = mesh_vertices.min(dim=0).values - 1.0e-5
    bbox_max = mesh_vertices.max(dim=0).values + 1.0e-5
    assert bool((output >= bbox_min).all())
    assert bool((output <= bbox_max).all())

    # Point cloud should not collapse to a lower-dimensional or degenerate set.
    assert float(output.std(dim=0).min().item()) > 1.0e-4

    # Avoid pathological duplicate-heavy outputs.
    unique_count = torch.unique(output, dim=0).shape[0]
    assert unique_count >= int(0.95 * target_num_points)

    # Enforce non-zero minimum separation for weighted elimination output.
    assert _minimum_pairwise_distance(output) > 0.0


# Validate weighted mode ignores per-vertex radius with a user warning.
@requires_module("warp")
def test_mesh_poisson_disk_sample_weighted_per_vertex_radius_warning(device: str):
    mesh_vertices, mesh_indices_2d = _build_case(device=device, subdivisions=3)
    per_vertex_radius = torch.full(
        (mesh_vertices.shape[0],),
        0.04,
        device=mesh_vertices.device,
        dtype=torch.float32,
    )

    with pytest.warns(UserWarning, match="per_vertex_radius is ignored"):
        output = mesh_poisson_disk_sample(
            mesh_vertices,
            mesh_indices_2d,
            min_distance=0.02,
            per_vertex_radius=per_vertex_radius,
            target_num_points=256,
            max_points=1024,
            mode="weighted_sample_elimination",
            implementation="warp",
        )
    assert output.shape[0] == 256


# Validate input/error handling paths.
@requires_module("warp")
def test_mesh_poisson_disk_sample_error_handling(device: str):
    mesh_vertices, mesh_indices_2d = _build_case(device=device, subdivisions=2)

    with pytest.raises(ValueError, match=r"shape \(n_vertices, 3\)"):
        mesh_poisson_disk_sample(
            torch.zeros(4, 2, device=mesh_vertices.device, dtype=torch.float32),
            mesh_indices_2d,
            implementation="warp",
        )

    with pytest.raises(ValueError, match=r"shape \(n_faces, 3\)"):
        mesh_poisson_disk_sample(
            mesh_vertices,
            torch.zeros(4, 4, device=mesh_vertices.device, dtype=torch.int32),
            implementation="warp",
        )

    with pytest.raises(TypeError, match="integer dtype"):
        mesh_poisson_disk_sample(
            mesh_vertices,
            mesh_indices_2d.to(torch.float32),
            implementation="warp",
        )

    with pytest.raises(ValueError, match="0 <= index < n_vertices"):
        mesh_bad = mesh_indices_2d.clone()
        mesh_bad[0, 0] = mesh_vertices.shape[0]
        mesh_poisson_disk_sample(
            mesh_vertices,
            mesh_bad,
            implementation="warp",
        )

    with pytest.raises(ValueError, match="strictly positive"):
        mesh_poisson_disk_sample(
            mesh_vertices,
            mesh_indices_2d,
            min_distance=0.0,
            implementation="warp",
        )

    with pytest.raises(ValueError, match="strictly positive"):
        mesh_poisson_disk_sample(
            mesh_vertices,
            mesh_indices_2d,
            batch_size=0,
            implementation="warp",
        )

    with pytest.raises(ValueError, match="strictly positive"):
        mesh_poisson_disk_sample(
            mesh_vertices,
            mesh_indices_2d,
            max_points=0,
            implementation="warp",
        )

    with pytest.raises(ValueError, match="strictly positive"):
        mesh_poisson_disk_sample(
            mesh_vertices,
            mesh_indices_2d,
            max_iterations=0,
            implementation="warp",
        )

    with pytest.raises(ValueError, match="per_vertex_radius must have shape"):
        mesh_poisson_disk_sample(
            mesh_vertices,
            mesh_indices_2d,
            per_vertex_radius=torch.ones(5, device=mesh_vertices.device),
            implementation="warp",
        )

    with pytest.raises(TypeError, match="floating dtype"):
        mesh_poisson_disk_sample(
            mesh_vertices,
            mesh_indices_2d,
            per_vertex_radius=torch.ones(
                mesh_vertices.shape[0], device=mesh_vertices.device, dtype=torch.int32
            ),
            implementation="warp",
        )

    with pytest.raises(ValueError, match="strictly positive"):
        mesh_poisson_disk_sample(
            mesh_vertices,
            mesh_indices_2d,
            per_vertex_radius=torch.zeros(
                mesh_vertices.shape[0], device=mesh_vertices.device, dtype=torch.float32
            ),
            implementation="warp",
        )

    with pytest.raises(ValueError, match="exactly 3 values"):
        mesh_poisson_disk_sample(
            mesh_vertices,
            mesh_indices_2d,
            hash_grid_resolution=(64, 64),
            implementation="warp",
        )

    with pytest.raises(ValueError, match="strictly positive"):
        mesh_poisson_disk_sample(
            mesh_vertices,
            mesh_indices_2d,
            hash_grid_resolution=(64, 0, 64),
            implementation="warp",
        )

    with pytest.raises(ValueError, match="mode must be one of"):
        mesh_poisson_disk_sample(
            mesh_vertices,
            mesh_indices_2d,
            mode="not_a_mode",
            implementation="warp",
        )

    with pytest.raises(ValueError, match="target_num_points must be strictly positive"):
        mesh_poisson_disk_sample(
            mesh_vertices,
            mesh_indices_2d,
            mode="weighted_sample_elimination",
            target_num_points=0,
            implementation="warp",
        )


# Validate benchmark input generation contract for this FunctionSpec.
@requires_module("warp")
def test_mesh_poisson_disk_sample_make_inputs_forward(device: str):
    cases = list(MeshPoissonDiskSample.make_inputs_forward(device=device))
    assert len(cases) == len(MeshPoissonDiskSample._BENCHMARK_CASES)

    labels = [case[0] for case in cases]
    assert labels == [case[0] for case in MeshPoissonDiskSample._BENCHMARK_CASES]

    label, args, kwargs = cases[0]
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    output = MeshPoissonDiskSample.dispatch(*args, implementation="warp", **kwargs)
    assert output.ndim == 2
    assert output.shape[1] == 3
    assert output.dtype == torch.float32
