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

"""Regenerate the committed ``.pmsh`` golden fixture used to lock in the
on-disk format of :class:`physicsnemo.mesh.Mesh`.

The companion test :mod:`test.mesh.mesh.test_pmsh_golden` loads this fixture
and asserts every field round-trips intact, so any future change that
quietly alters the on-disk layout (renaming a tensorclass field, dropping
``shadow=True``, swapping the decorator for inheritance, etc.) will fail
the test.

Run this script only when the ``.pmsh`` format intentionally changes:

.. code-block:: bash

    uv run --no-sync python test/mesh/mesh/golden_pmsh/_regenerate.py

Then commit the resulting ``v2.0_two_triangles.pmsh/`` directory tree.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import torch

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.primitives.basic import two_triangles_2d

### Fixture identity #########################################################

# Bumping this name (e.g. ``v2.1_...``) lets us keep historical fixtures
# alongside new ones if we want to test multiple format generations at once.
FIXTURE_DIR: Path = (Path(__file__).parent / "v2.0_two_triangles.pmsh").resolve()


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


def regenerate(fixture_dir: Path = FIXTURE_DIR) -> None:
    """Rebuild the on-disk fixture, replacing any prior copy.

    Memmap save refuses to overwrite an existing directory, so the prior
    fixture is wiped first.
    """
    if fixture_dir.exists():
        shutil.rmtree(fixture_dir)
    fixture_dir.parent.mkdir(parents=True, exist_ok=True)
    build_canonical_mesh().save(fixture_dir)
    n_files = sum(1 for p in fixture_dir.rglob("*") if p.is_file())
    n_bytes = sum(p.stat().st_size for p in fixture_dir.rglob("*") if p.is_file())
    print(f"Wrote {fixture_dir.relative_to(Path.cwd())} ({n_files} files, {n_bytes} B)")


if __name__ == "__main__":
    regenerate()
