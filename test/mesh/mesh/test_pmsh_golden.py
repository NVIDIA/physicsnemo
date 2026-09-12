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

``golden_pmsh/`` contains an immutable decorator-era fixture and a compact
manifest of the current ``TensorClass`` writer layout. The legacy fixture must
reconstruct an exact :class:`~physicsnemo.mesh.Mesh`; current files are written
and round-tripped at runtime.

To intentionally update the current writer layout, run
``python -m test.mesh.mesh.golden_pmsh._regenerate`` and commit the new
manifest without replacing the legacy fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from physicsnemo.mesh.mesh import Mesh
from test.mesh._serialization_manifest import serialization_manifest
from test.mesh.mesh.golden_pmsh._regenerate import (
    CURRENT_MANIFEST_PATH,
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
        loaded = Mesh.load(fixture_dir)
        assert type(loaded) is Mesh

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

    def test_current_writer_layout_matches_manifest(self, tmp_path: Path):
        """A fresh save has the committed current directory and metadata layout."""
        written = tmp_path / "current.pmsh"
        build_canonical_mesh().save(written)
        expected = json.loads(CURRENT_MANIFEST_PATH.read_text())
        assert serialization_manifest(written) == expected

    def test_current_and_legacy_layouts_record_the_mesh_type(self):
        """Both writer generations retain a root-level type discriminator."""
        expected_type = "<class 'physicsnemo.mesh.mesh.Mesh'>"
        legacy_metadata = json.loads((LEGACY_FIXTURE_DIR / "meta.json").read_text())
        current_manifest = json.loads(CURRENT_MANIFEST_PATH.read_text())

        assert legacy_metadata == {"_type": expected_type}
        assert current_manifest["meta.json"] == {"_type": expected_type}
        assert (LEGACY_FIXTURE_DIR / "_tensordict").is_dir()
        assert any(path.startswith("_tensordict/") for path in current_manifest)

    def test_out_fills_preallocated_tensors(self, fixture_dir: Path):
        """Both layouts fill a caller-provided TensorDict payload."""
        expected = build_canonical_mesh()
        out = expected._tensordict.apply(torch.zeros_like)

        loaded = Mesh.load(fixture_dir, out=out)

        assert loaded._tensordict is out
        assert torch.equal(loaded.points, expected.points)
        assert torch.equal(loaded.cells, expected.cells)
        for field in ("point_data", "cell_data", "global_data"):
            for key in getattr(expected, field).keys():
                assert torch.equal(
                    getattr(loaded, field)[key], getattr(expected, field)[key]
                )

    @pytest.mark.cuda
    def test_load_honors_device(self, fixture_dir: Path):
        """``device=`` applies to both layouts, not just the current one."""
        loaded = Mesh.load(fixture_dir, device="cuda")
        assert loaded.points.device.type == "cuda"
        assert loaded.cells.device.type == "cuda"
        assert loaded.point_data.device.type == "cuda"

    @pytest.mark.cuda
    def test_load_rejects_device_that_conflicts_with_out(self, fixture_dir: Path):
        """A CPU output cannot silently override or mix with a CUDA request."""
        out = build_canonical_mesh()._tensordict.apply(torch.zeros_like)

        with pytest.raises(ValueError, match=r"device=.*conflicts with `out`"):
            Mesh.load(fixture_dir, device="cuda", out=out)
