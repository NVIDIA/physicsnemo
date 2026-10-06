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

"""Writer-layout and backward-read tests for the ``.pmsh`` memmap format.

``golden_pmsh/`` contains an immutable fixture written by the decorator-based
:class:`~physicsnemo.mesh.Mesh`. It must reconstruct an exact ``Mesh``, and a
fresh save must reproduce its directory and metadata layout. Current files are
also written and round-tripped at runtime.

If the writer layout intentionally changes, keep this fixture for backward
reads and add a new one with ``python -m test.mesh.mesh.golden_pmsh._regenerate``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.mesh.mesh import Mesh
from test.mesh._serialization_manifest import serialization_manifest
from test.mesh.mesh.golden_pmsh._regenerate import (
    LEGACY_FIXTURE_DIR,
    build_canonical_mesh,
)


@pytest.fixture(
    params=("current", "legacy"),
    ids=("current-tensorclass", "legacy-decorator"),
)
def fixture_dir(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    """Return a fresh current file or the immutable legacy fixture."""
    if request.param == "current":
        path = tmp_path / "current.pmsh"
        build_canonical_mesh().save(path)
        return path
    assert LEGACY_FIXTURE_DIR.is_dir(), (
        f"Missing committed .pmsh fixture: {LEGACY_FIXTURE_DIR}"
    )
    return LEGACY_FIXTURE_DIR


class TestPmshGoldenFixture:
    """Verify current round trips and decorator-era backward compatibility."""

    def test_reconstructs_exact_mesh_type(self, fixture_dir: Path):
        """Both layouts reconstruct the structured type, not a TensorDict."""
        assert type(Mesh.load(fixture_dir)) is Mesh
        assert type(TensorDict.load(fixture_dir)) is Mesh

    def test_geometry_matches(self, fixture_dir: Path):
        """`points` and `cells` round-trip exactly."""
        loaded = Mesh.load(fixture_dir)
        expected = build_canonical_mesh()
        assert loaded.n_points == expected.n_points
        assert loaded.n_cells == expected.n_cells
        assert loaded.n_spatial_dims == expected.n_spatial_dims
        assert loaded.n_manifold_dims == expected.n_manifold_dims
        assert torch.equal(loaded.points, expected.points)
        assert torch.equal(loaded.cells, expected.cells)

    def test_data_fields_match(self, fixture_dir: Path):
        """Every key in `point_data`, `cell_data`, `global_data` round-trips exactly."""
        loaded = Mesh.load(fixture_dir)
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

    def test_current_writer_layout_matches_fixture(self, tmp_path: Path):
        """A fresh save reproduces the fixture's directory and metadata layout."""
        written = tmp_path / "current.pmsh"
        build_canonical_mesh().save(written)
        assert serialization_manifest(written) == serialization_manifest(
            LEGACY_FIXTURE_DIR
        )

    @pytest.mark.cuda
    def test_load_honors_device(self, fixture_dir: Path):
        """``device=`` applies to both layouts, not just the current one."""
        loaded = Mesh.load(fixture_dir, device="cuda")
        assert loaded.points.device.type == "cuda"
        assert loaded.cells.device.type == "cuda"
        assert loaded.point_data.device.type == "cuda"
