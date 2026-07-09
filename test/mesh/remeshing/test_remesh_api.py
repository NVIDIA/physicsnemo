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

"""Device-independent tests for the public Mesh remeshing API."""

import inspect

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.mesh.remeshing import remesh


def test_remesh_public_signatures():
    assert tuple(inspect.signature(remesh).parameters) == (
        "mesh",
        "n_clusters",
        "max_iterations",
        "warp_options",
    )
    assert "warp_options" in inspect.signature(Mesh.remesh).parameters


def test_remesh_rejects_wrong_options_type_without_cuda():
    source = sphere_icosahedral.load(subdivisions=2)

    with pytest.raises(TypeError, match="WarpRemeshOptions instance.*dict"):
        remesh(
            source,
            48,
            warp_options={"hash_grid_resolution": 64},
        )


def test_remesh_rejects_cpu_mesh():
    source = sphere_icosahedral.load(subdivisions=2)

    with pytest.raises(ValueError, match="requires a CUDA mesh"):
        remesh(source, 48)


def test_remesh_rejects_non_surface_mesh():
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    cells = torch.tensor([[0, 1], [1, 2]])
    source = Mesh(points=points, cells=cells)

    with pytest.raises(NotImplementedError, match="2D triangle surface"):
        remesh(source, 3)
