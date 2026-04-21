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

import pytest
import torch

from physicsnemo.nn.functional import mesh_to_voxel_fraction
from physicsnemo.nn.functional.geometry import MeshToVoxelFraction
from test.conftest import requires_module


# Build a deterministic watertight mesh and voxel-grid setup for tests.
def _build_case(device: str, subdivisions: int = 2, grid_n: int = 32):
    from physicsnemo.mesh.primitives.procedural.lumpy_sphere import (
        load as load_lumpy_sphere,
    )

    torch_device = torch.device(device)
    mesh = load_lumpy_sphere(subdivisions=subdivisions, device=str(torch_device))

    mesh_vertices = mesh.points.to(torch.float32).contiguous()
    mesh_indices_2d = mesh.cells.to(torch.int32).contiguous()

    # Build a padded cubic grid around the mesh extent.
    bbox_min = mesh_vertices.min(dim=0).values
    bbox_max = mesh_vertices.max(dim=0).values
    extent = (bbox_max - bbox_min).amax().detach().cpu().item()
    extent = extent if extent > 0.0 else 1.0
    padding = 0.15 * extent

    origin = (bbox_min - padding).to(torch.float32).contiguous()
    voxel_size = float((extent + 2.0 * padding) / float(grid_n))
    grid_dims = (grid_n, grid_n, grid_n)
    return mesh_vertices, mesh_indices_2d, origin, voxel_size, grid_dims


# Validate core warp output properties for mesh-to-voxel conversion.
@requires_module("warp")
def test_mesh_to_voxel_fraction_warp(device: str):
    mesh_vertices, mesh_indices_2d, origin, voxel_size, grid_dims = _build_case(device)

    # Explicitly dispatch to warp for this single-backend functional.
    output_warp = mesh_to_voxel_fraction(
        mesh_vertices,
        mesh_indices_2d,
        origin,
        voxel_size,
        grid_dims,
        n_samples=32,
        seed=1234,
        implementation="warp",
    )

    assert output_warp.shape == (grid_dims[2], grid_dims[1], grid_dims[0])
    assert output_warp.dtype == torch.float32
    assert float(output_warp.min()) >= 0.0
    assert float(output_warp.max()) <= 1.0
    assert 0.0 < float(output_warp.sum()) < float(output_warp.numel())


# Validate equivalent results for flattened and (n_faces, 3) index layouts.
@requires_module("warp")
def test_mesh_to_voxel_fraction_index_layout_compatibility(device: str):
    mesh_vertices, mesh_indices_2d, origin, voxel_size, grid_dims = _build_case(device)

    output_faces = mesh_to_voxel_fraction(
        mesh_vertices,
        mesh_indices_2d,
        origin,
        voxel_size,
        grid_dims,
        n_samples=24,
        seed=2026,
        implementation="warp",
    )
    output_flat = mesh_to_voxel_fraction(
        mesh_vertices,
        mesh_indices_2d.reshape(-1),
        origin,
        voxel_size,
        grid_dims,
        n_samples=24,
        seed=2026,
        implementation="warp",
    )
    torch.testing.assert_close(output_faces, output_flat)


# Validate the open-mesh winding-number path against the closed-mesh sign-normal path.
@requires_module("warp")
def test_mesh_to_voxel_fraction_open_mesh_path(device: str):
    mesh_vertices, mesh_indices_2d, origin, voxel_size, grid_dims = _build_case(device)

    output_closed = mesh_to_voxel_fraction(
        mesh_vertices,
        mesh_indices_2d,
        origin,
        voxel_size,
        grid_dims,
        n_samples=24,
        seed=9001,
        open_mesh=False,
        implementation="warp",
    )
    output_open = mesh_to_voxel_fraction(
        mesh_vertices,
        mesh_indices_2d,
        origin,
        voxel_size,
        grid_dims,
        n_samples=24,
        seed=9001,
        open_mesh=True,
        winding_number_threshold=0.5,
        winding_number_accuracy=2.0,
        implementation="warp",
    )

    assert output_open.shape == output_closed.shape
    assert float(output_open.min()) >= 0.0
    assert float(output_open.max()) <= 1.0
    assert abs(float(output_open.mean() - output_closed.mean())) < 0.1


# Validate argument and shape error handling paths.
@requires_module("warp")
def test_mesh_to_voxel_fraction_error_handling(device: str):
    mesh_vertices, mesh_indices_2d, origin, voxel_size, grid_dims = _build_case(device)

    # Invalid mesh index shape.
    with pytest.raises(ValueError, match=r"shape \(n_faces, 3\)"):
        mesh_to_voxel_fraction(
            mesh_vertices,
            torch.zeros(4, 4, device=mesh_vertices.device, dtype=torch.int32),
            origin,
            voxel_size,
            grid_dims,
            implementation="warp",
        )

    # Non-integer mesh indices are rejected.
    with pytest.raises(TypeError, match="integer dtype"):
        mesh_to_voxel_fraction(
            mesh_vertices,
            mesh_indices_2d.to(torch.float32),
            origin,
            voxel_size,
            grid_dims,
            implementation="warp",
        )

    # Out-of-bounds mesh indices are rejected.
    mesh_indices_oob = mesh_indices_2d.clone()
    mesh_indices_oob[0, 0] = mesh_vertices.shape[0]
    with pytest.raises(ValueError, match=r"0 <= index < n_vertices"):
        mesh_to_voxel_fraction(
            mesh_vertices,
            mesh_indices_oob,
            origin,
            voxel_size,
            grid_dims,
            implementation="warp",
        )

    # Invalid origin shape.
    with pytest.raises(ValueError, match="origin must be a length-3 vector"):
        mesh_to_voxel_fraction(
            mesh_vertices,
            mesh_indices_2d,
            torch.zeros(2, device=mesh_vertices.device, dtype=torch.float32),
            voxel_size,
            grid_dims,
            implementation="warp",
        )

    # Invalid grid dimensions.
    with pytest.raises(ValueError, match="grid_dims must contain exactly three values"):
        mesh_to_voxel_fraction(
            mesh_vertices,
            mesh_indices_2d,
            origin,
            voxel_size,
            (32, 32),
            implementation="warp",
        )

    # Invalid voxel size.
    with pytest.raises(ValueError, match="voxel_size must be strictly positive"):
        mesh_to_voxel_fraction(
            mesh_vertices,
            mesh_indices_2d,
            origin,
            0.0,
            grid_dims,
            implementation="warp",
        )

    # Invalid number of samples.
    with pytest.raises(ValueError, match="n_samples must be strictly positive"):
        mesh_to_voxel_fraction(
            mesh_vertices,
            mesh_indices_2d,
            origin,
            voxel_size,
            grid_dims,
            n_samples=0,
            implementation="warp",
        )


# Validate benchmark input generation contract for this FunctionSpec.
@requires_module("warp")
def test_mesh_to_voxel_fraction_make_inputs_forward(device: str):
    cases = list(MeshToVoxelFraction.make_inputs_forward(device=device))
    assert len(cases) == len(MeshToVoxelFraction._BENCHMARK_CASES)

    labels = [case[0] for case in cases]
    assert labels == [case[0] for case in MeshToVoxelFraction._BENCHMARK_CASES]

    label, args, kwargs = cases[0]
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    output = MeshToVoxelFraction.dispatch(
        *args,
        implementation="warp",
        **kwargs,
    )
    assert output.ndim == 3
    assert output.dtype == torch.float32
