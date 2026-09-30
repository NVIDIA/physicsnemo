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

"""Build and write the canonical ``.pmsh`` golden fixture.

``v2.0_two_triangles.pmsh`` is immutable data written by the decorator-based
``Mesh`` implementation. The companion test checks that it loads exactly and
that a fresh save reproduces its directory and metadata layout.

If the writer layout intentionally changes, keep this fixture for backward
reads and write a new one beside it:

.. code-block:: bash

    uv run --no-sync python -m test.mesh.mesh.golden_pmsh._regenerate <new_fixture_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.primitives.basic import two_triangles_2d

### Fixture identity #########################################################

LEGACY_FIXTURE_DIR: Path = (Path(__file__).parent / "v2.0_two_triangles.pmsh").resolve()


def build_canonical_mesh() -> Mesh:
    """Build the canonical golden mesh.

    A 2-triangle 2D mesh (4 points, 2 cells) decorated with deterministic
    integer-valued tensors on every data container, so equality comparisons
    in the test can use ``torch.equal`` rather than tolerant ``allclose``.

    The exact contents are:

    - ``points``: from :func:`two_triangles_2d.load`,
      shape ``(4, 2)``, dtype ``float32``.
    - ``cells``: from :func:`two_triangles_2d.load`,
      shape ``(2, 3)``, dtype ``int64``.
    - ``point_data["p_scalar"]``: ``arange(4, dtype=float32)``
    - ``point_data["p_vector"]``: ``arange(12, dtype=float32).reshape(4, 3)``
    - ``cell_data["c_scalar"]``: ``arange(2, dtype=float32)``
    - ``cell_data["c_vector"]``: ``arange(6, dtype=float32).reshape(2, 3)``
    - ``global_data["g_scalar"]``: ``tensor(42.0, dtype=float32)``
    - ``global_data["g_vector"]``: ``tensor([1.0, 2.0, 3.0], dtype=float32)``
    """
    mesh = two_triangles_2d.load()
    mesh.point_data["p_scalar"] = torch.arange(mesh.n_points, dtype=torch.float32)
    mesh.point_data["p_vector"] = torch.arange(
        mesh.n_points * 3, dtype=torch.float32
    ).reshape(mesh.n_points, 3)
    mesh.cell_data["c_scalar"] = torch.arange(mesh.n_cells, dtype=torch.float32)
    mesh.cell_data["c_vector"] = torch.arange(
        mesh.n_cells * 3, dtype=torch.float32
    ).reshape(mesh.n_cells, 3)
    mesh.global_data["g_scalar"] = torch.tensor(42.0, dtype=torch.float32)
    mesh.global_data["g_vector"] = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    return mesh


def regenerate(fixture_dir: Path) -> None:
    """Write the canonical mesh to a new fixture directory."""
    if fixture_dir.exists():
        raise FileExistsError(f"{fixture_dir} exists; committed fixtures are immutable")
    build_canonical_mesh().save(fixture_dir)
    print(f"Wrote {fixture_dir}")


if __name__ == "__main__":
    regenerate(Path(sys.argv[1]))
