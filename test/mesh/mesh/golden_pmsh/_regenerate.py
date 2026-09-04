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

"""Regenerate the current ``.pmsh`` layout manifest.

The companion test keeps two compatibility records:

- ``v2.0_two_triangles.pmsh`` is immutable legacy data written by the
  decorator-based ``Mesh`` implementation. It protects backward reads.
- ``current_manifest.json`` is a compact snapshot of the current writer layout.

Run this script when the current ``.pmsh`` writer intentionally changes:

.. code-block:: bash

    uv run --no-sync python -m test.mesh.mesh.golden_pmsh._regenerate

Then commit the updated manifest. Never regenerate the legacy fixture with
current code.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.primitives.basic import two_triangles_2d
from test.mesh._serialization_manifest import serialization_manifest

### Fixture identity #########################################################

CURRENT_MANIFEST_PATH: Path = (
    Path(__file__).parent / "current_manifest.json"
).resolve()
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


def regenerate(manifest_path: Path = CURRENT_MANIFEST_PATH) -> None:
    """Write a snapshot of the current writer's directory and metadata layout."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fixture_dir = Path(tmp_dir) / "current.pmsh"
        build_canonical_mesh().save(fixture_dir)
        manifest = serialization_manifest(fixture_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {manifest_path.relative_to(Path.cwd())} ({len(manifest)} entries)")


if __name__ == "__main__":
    regenerate()
