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

"""Tests for the ACVD-based ``remesh`` entry point."""

import pytest
import torch

from physicsnemo.mesh import Mesh

pytest.importorskip("pyacvd")

from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral  # noqa: E402
from physicsnemo.mesh.remeshing import remesh  # noqa: E402


def test_remesh_basic_sphere():
    """remesh round-trips a triangle surface and yields a valid 2D-in-3D mesh."""
    mesh = sphere_icosahedral.load(subdivisions=3)
    out = remesh(mesh, n_clusters=100)

    assert isinstance(out, Mesh)
    assert out.n_cells > 0
    assert out.n_manifold_dims == 2 and out.n_spatial_dims == 3
    assert out.points.device == mesh.points.device
    assert not torch.is_floating_point(out.cells)  # cells stay integer


def test_remesh_preserves_dtype():
    """remesh restores the input floating dtype even though pyvista round-trips
    through float32."""
    base = sphere_icosahedral.load(subdivisions=3)
    mesh = Mesh(points=base.points.double(), cells=base.cells)  # float64
    out = remesh(mesh, n_clusters=80)
    assert out.points.dtype == torch.float64
    assert not torch.is_floating_point(out.cells)


def test_remesh_rejects_non_surface():
    """remesh guards against non-2D-in-3D inputs with a clear NotImplementedError
    instead of a confusing downstream pyacvd failure."""
    pts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    cells = torch.tensor([[0, 1], [1, 2]])  # 1D curve in 3D
    mesh = Mesh(points=pts, cells=cells)
    with pytest.raises(NotImplementedError, match="2D triangle surface"):
        remesh(mesh, n_clusters=2)
