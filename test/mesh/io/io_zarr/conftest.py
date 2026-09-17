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

"""Pytest configuration for mesh.io.io_zarr tests.

All tests in this directory require ``zarr >= 3`` and a tensordict release
carrying the zarr storage backend (``TensorDict.from_zarr``).
"""

import pytest
import torch

zarr = pytest.importorskip("zarr", minversion="3.0")

from tensordict import TensorDict  # noqa: E402

if not hasattr(TensorDict, "from_zarr"):
    pytest.skip("tensordict build has no zarr storage backend", allow_module_level=True)

from physicsnemo.mesh import DomainMesh, Mesh  # noqa: E402


def make_mesh(n_points: int = 40, n_cells: int = 25, seed: int = 7) -> Mesh:
    """Deterministic triangle mesh with nested field trees."""
    g = torch.Generator().manual_seed(seed)
    return Mesh(
        points=torch.rand(n_points, 3, generator=g),
        cells=torch.randint(0, n_points, (n_cells, 3), generator=g),
        point_data={
            "temp": torch.rand(n_points, generator=g),
            # nested TensorDict: must round-trip (regression for review issue)
            "flow": TensorDict(
                {
                    "velocity": torch.rand(n_points, 3, generator=g),
                    "moments": TensorDict(
                        {"second": torch.rand(n_points, 2, generator=g)},
                        batch_size=[],
                    ),
                },
                batch_size=[],
            ),
        },
        cell_data={"pressure": torch.rand(n_cells, generator=g)},
        global_data={"U_inf": torch.tensor([38.889, 0.0, 0.0])},
    )


def make_point_cloud(n_points: int = 64, seed: int = 11) -> Mesh:
    """Deterministic point cloud (no cells)."""
    g = torch.Generator().manual_seed(seed)
    return Mesh(
        points=torch.rand(n_points, 3, generator=g),
        point_data={"p": torch.rand(n_points, generator=g)},
    )


def make_domain_mesh(seed: int = 3) -> DomainMesh:
    """Deterministic DomainMesh: point-cloud interior + two boundaries."""
    return DomainMesh(
        interior=make_point_cloud(seed=seed),
        boundaries={
            "wall": make_mesh(seed=seed + 1),
            "inlet": make_mesh(n_points=12, n_cells=6, seed=seed + 2),
        },
        global_data={"rho_inf": torch.tensor(1.205)},
    )


def assert_meshes_equal(a: Mesh, b: Mesh) -> None:
    """Bitwise comparison of two meshes, including nested field trees."""
    assert torch.equal(a.points, b.points)
    assert a.n_cells == b.n_cells
    if a.n_cells > 0:
        assert torch.equal(a.cells, b.cells)
    for tree in ("point_data", "cell_data", "global_data"):
        ta, tb = getattr(a, tree), getattr(b, tree)
        keys_a = set(ta.keys(include_nested=True, leaves_only=True))
        keys_b = set(tb.keys(include_nested=True, leaves_only=True))
        assert keys_a == keys_b, f"{tree}: {keys_a} != {keys_b}"
        for k in keys_a:
            assert torch.equal(ta[k], tb[k]), f"{tree}[{k}] differs"
