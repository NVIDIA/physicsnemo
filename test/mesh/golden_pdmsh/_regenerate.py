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

"""Build and write the canonical ``.pdmsh`` golden fixture.

``legacy_decorator_square.pdmsh`` is immutable data written by the
decorator-based ``DomainMesh`` and ``Mesh`` implementations. The companion test
checks that it loads exactly and that a fresh save reproduces its directory and
metadata layout.

If the writer layout intentionally changes, keep this fixture for backward
reads and write a new one beside it:

.. code-block:: bash

    uv run --no-sync python -m test.mesh.golden_pdmsh._regenerate <new_fixture_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.primitives.basic import two_triangles_2d

LEGACY_FIXTURE_DIR: Path = (
    Path(__file__).parent / "legacy_decorator_square.pdmsh"
).resolve()


def _build_boundary(points: torch.Tensor, boundary_id: int) -> Mesh:
    """Build one deterministic edge boundary with data at every level."""
    return Mesh(
        points=points,
        cells=torch.tensor([[0, 1]], dtype=torch.int64),
        point_data={
            "distance": torch.tensor([0.0, 1.0], dtype=torch.float32),
        },
        cell_data={
            "boundary_id": torch.tensor([boundary_id], dtype=torch.int64),
        },
        global_data={
            "reference_value": torch.tensor(
                boundary_id * 10.0,
                dtype=torch.float32,
            ),
        },
    )


def build_canonical_domain_mesh() -> DomainMesh:
    """Build a deterministic domain with an interior and two boundaries."""
    interior = two_triangles_2d.load()
    interior.point_data["temperature"] = torch.tensor(
        [300.0, 310.0, 320.0, 330.0],
        dtype=torch.float32,
    )
    interior.cell_data["material"] = torch.tensor([1, 2], dtype=torch.int64)
    interior.global_data["case_id"] = torch.tensor(17, dtype=torch.int64)

    wall = _build_boundary(interior.points[[0, 1]], boundary_id=1)
    inlet = _build_boundary(interior.points[[2, 3]], boundary_id=2)
    return DomainMesh(
        interior=interior,
        boundaries={"wall": wall, "inlet": inlet},
        global_data={
            "reynolds": torch.tensor(1.0e6, dtype=torch.float32),
            "time": torch.tensor(0.25, dtype=torch.float32),
        },
    )


def regenerate(fixture_dir: Path) -> None:
    """Write the canonical domain to a new fixture directory."""
    if fixture_dir.exists():
        raise FileExistsError(f"{fixture_dir} exists; committed fixtures are immutable")
    build_canonical_domain_mesh().save(fixture_dir)
    print(f"Wrote {fixture_dir}")


if __name__ == "__main__":
    regenerate(Path(sys.argv[1]))
