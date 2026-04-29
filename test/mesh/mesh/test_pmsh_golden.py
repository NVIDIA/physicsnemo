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

"""Cross-version compatibility test for the ``.pmsh`` (memmap) format.

The committed fixture under ``golden_pmsh/`` was written by an earlier
revision of :class:`~physicsnemo.mesh.Mesh`. This test loads it with the
*current* code and asserts that every field round-trips intact. Any change
that quietly alters the on-disk layout (renaming a tensorclass field,
dropping ``shadow=True``, swapping the decorator for ``TensorClass``
inheritance, changing the underlying ``tensordict`` memmap convention, etc.)
will fail this test.

To intentionally update the format, run
``test/mesh/mesh/golden_pmsh/_regenerate.py`` and commit the new fixture
(or a fresh ``v<N.M>_...`` sibling, to keep older fixtures around).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from physicsnemo.mesh.mesh import Mesh

### Locate the regeneration helper without requiring `golden_pmsh` to be a package.
### Skip the whole module gracefully if the helper has been moved or renamed,
### so that a missing file produces a clean skip rather than a collection error
### (`spec_from_file_location` returns a valid spec for non-existent paths,
### so the failure would otherwise surface as a `FileNotFoundError` from
### `exec_module` at module import time, before `pytestmark` can take effect).
_REGEN_PATH = Path(__file__).parent / "golden_pmsh" / "_regenerate.py"
if not _REGEN_PATH.exists():
    pytest.skip(
        f"Golden .pmsh regeneration helper not found at {_REGEN_PATH}",
        allow_module_level=True,
    )
_spec = importlib.util.spec_from_file_location("_pmsh_golden_regen", _REGEN_PATH)
assert _spec is not None and _spec.loader is not None
_regen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_regen)
build_canonical_mesh = _regen.build_canonical_mesh
FIXTURE_DIR: Path = _regen.FIXTURE_DIR


pytestmark = pytest.mark.skipif(
    not FIXTURE_DIR.exists(),
    reason=(
        f"Golden .pmsh fixture not found at {FIXTURE_DIR}; "
        f"run {_REGEN_PATH} to (re)generate it."
    ),
)


class TestPmshGoldenFixture:
    """Load a committed ``.pmsh`` fixture and verify field-by-field equality."""

    def test_loads_without_error(self):
        """The fixture deserializes into a `Mesh` instance."""
        loaded = Mesh.load(FIXTURE_DIR)
        assert isinstance(loaded, Mesh)

    def test_geometry_matches(self):
        """`points` and `cells` round-trip exactly."""
        loaded = Mesh.load(FIXTURE_DIR)
        expected = build_canonical_mesh()
        assert loaded.n_points == expected.n_points
        assert loaded.n_cells == expected.n_cells
        assert loaded.n_spatial_dims == expected.n_spatial_dims
        assert loaded.n_manifold_dims == expected.n_manifold_dims
        assert torch.equal(loaded.points, expected.points)
        assert torch.equal(loaded.cells, expected.cells)

    def test_data_fields_match(self):
        """Every key in `point_data`, `cell_data`, `global_data` round-trips exactly."""
        loaded = Mesh.load(FIXTURE_DIR)
        expected = build_canonical_mesh()
        for field in ("point_data", "cell_data", "global_data"):
            loaded_td = getattr(loaded, field)
            expected_td = getattr(expected, field)
            assert set(loaded_td.keys()) == set(expected_td.keys()), (
                f"{field} key mismatch: "
                f"loaded={sorted(loaded_td.keys())}, "
                f"expected={sorted(expected_td.keys())}"
            )
            for key in expected_td.keys():
                assert torch.equal(loaded_td[key], expected_td[key]), (
                    f"{field}[{key!r}] value mismatch after load"
                )
