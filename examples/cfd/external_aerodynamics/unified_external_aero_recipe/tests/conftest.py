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

"""Shared pytest fixtures and path setup for the unified external aero recipe tests.

The recipe's `src/` modules are imported by their bare names (e.g.
``from collate import build_collate_fn``); the production entry point
(`src/train.py`) gets this for free because `src/datasets.py` runs
``sys.path.insert(0, str(Path(__file__).resolve().parent))`` at import
time. For tests, we make the same insertion explicit here so each test
file can simply ``from collate import ...``.

Importing :mod:`physicsnemo.datapipes` also registers the ``${dp:...}``
OmegaConf resolver, and the recipe-local :mod:`nondim` and :mod:`sdf`
modules register their custom transforms into the global datapipe
registry. Without those side-effect imports the dataset YAMLs cannot be
instantiated.

The fixtures below ( ``surface_domain_mesh`` / ``volume_domain_mesh`` )
mirror the structure of post-pipeline ``DomainMesh`` outputs that the
collate, loss, metric, and forward_kwargs tests all consume. Test files
that need a custom layout still build their own; these are the common
shapes.
"""

from __future__ import annotations

import sys
from pathlib import Path

_RECIPE_ROOT = Path(__file__).resolve().parent.parent
_SRC = _RECIPE_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

### Side-effect imports: register the ${dp:...} resolver plus the
### recipe-local NonDimensionalizeByMetadata / ComputeSDFFromBoundary /
### DropBoundary transforms.
import physicsnemo.datapipes  # noqa: E402, F401
import nondim  # noqa: E402, F401
import sdf  # noqa: E402, F401

import pytest  # noqa: E402
import torch  # noqa: E402

from physicsnemo.mesh import DomainMesh, Mesh  # noqa: E402

from utils import FieldType  # noqa: E402


### ---------------------------------------------------------------------------
### Shared DomainMesh factories
### ---------------------------------------------------------------------------


def make_surface_domain_mesh(
    target_config: dict[str, FieldType] | None = None,
    *,
    n_cells: int = 6,
    n_pts_factor: int = 2,
) -> DomainMesh:
    """Build a synthetic post-pipeline surface ``DomainMesh``.

    Mirrors what a surface dataset YAML's pipeline + ``MeshToDomainMesh``
    produces: an interior point cloud at cell centroids carrying targets
    in ``point_data``, and a ``vehicle`` boundary with precomputed
    ``cell_data["normals"]``. ``target_config`` defaults to a small
    pressure (scalar) + wss (vector) shape used by the surface configs;
    pass a custom mapping to drive other field layouts.
    """
    if target_config is None:
        target_config = {"pressure": "scalar", "wss": "vector"}

    interior_point_data: dict[str, torch.Tensor] = {}
    for name, ftype in target_config.items():
        if ftype == "scalar":
            interior_point_data[name] = torch.randn(n_cells)
        elif ftype == "vector":
            interior_point_data[name] = torch.randn(n_cells, 3)
        else:
            raise ValueError(f"Unknown field type {ftype!r}")
    interior = Mesh(
        points=torch.randn(n_cells, 3),
        point_data=interior_point_data,
    )

    n_pts = max(n_cells * n_pts_factor, 8)
    vehicle = Mesh(
        points=torch.randn(n_pts, 3) * 2,
        cells=torch.randint(0, n_pts, (n_cells, 3)),
        cell_data={"normals": torch.randn(n_cells, 3)},
    )
    return DomainMesh(
        interior=interior,
        boundaries={"vehicle": vehicle},
        global_data={
            "U_inf": torch.tensor([30.0, 0.0, 0.0]),
            "p_inf": torch.tensor(0.0),
            "rho_inf": torch.tensor(1.225),
            "L_ref": torch.tensor(5.0),
        },
    )


def make_volume_domain_mesh(
    target_config: dict[str, FieldType] | None = None,
    *,
    n_pts: int = 200,
) -> DomainMesh:
    """Build a synthetic post-pipeline volume ``DomainMesh``.

    Mirrors a volume dataset YAML's native ``DomainMesh``: interior
    volume point cloud carrying ``sdf`` / ``sdf_normals`` plus targets,
    and a ``vehicle`` boundary with no precomputed cell features (volume
    YAMLs do not run ``ComputeSurfaceNormals``).
    """
    if target_config is None:
        target_config = {
            "velocity": "vector",
            "pressure": "scalar",
            "nut": "scalar",
        }

    interior_point_data: dict[str, torch.Tensor] = {
        "sdf": torch.randn(n_pts),
        "sdf_normals": torch.randn(n_pts, 3),
    }
    for name, ftype in target_config.items():
        if ftype == "scalar":
            interior_point_data[name] = torch.randn(n_pts)
        elif ftype == "vector":
            interior_point_data[name] = torch.randn(n_pts, 3)
        else:
            raise ValueError(f"Unknown field type {ftype!r}")
    interior = Mesh(
        points=torch.randn(n_pts, 3),
        point_data=interior_point_data,
    )

    n_vehicle_pts = max(n_pts // 4, 8)
    n_vehicle_cells = max(n_pts // 4, 8)
    vehicle = Mesh(
        points=torch.randn(n_vehicle_pts, 3) * 2,
        cells=torch.randint(0, n_vehicle_pts, (n_vehicle_cells, 3)),
    )
    return DomainMesh(
        interior=interior,
        boundaries={"vehicle": vehicle},
        global_data={
            "U_inf": torch.tensor([30.0, 0.0, 0.0]),
            "p_inf": torch.tensor(0.0),
            "rho_inf": torch.tensor(1.225),
            "L_ref": torch.tensor(5.0),
        },
    )


@pytest.fixture
def surface_domain_mesh() -> DomainMesh:
    """Default surface DomainMesh fixture (pressure scalar + wss vector).

    For non-default layouts, call :func:`make_surface_domain_mesh`
    directly with a custom ``target_config``.
    """
    return make_surface_domain_mesh()


@pytest.fixture
def volume_domain_mesh() -> DomainMesh:
    """Default volume DomainMesh fixture (velocity / pressure / nut)."""
    return make_volume_domain_mesh()
