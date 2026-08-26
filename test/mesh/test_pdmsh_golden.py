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

"""Runtime round-trip and backward-read tests for the ``.pdmsh`` memmap format."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from tensordict import TensorDictBase

from physicsnemo.mesh import DomainMesh, Mesh
from test.mesh._serialization_manifest import serialization_manifest
from test.mesh.golden_pdmsh._regenerate import (
    CURRENT_MANIFEST_PATH,
    LEGACY_FIXTURE_DIR,
    build_canonical_domain_mesh,
)


def _assert_tensordict_equal(
    loaded: TensorDictBase,
    expected: TensorDictBase,
    field: str,
) -> None:
    """Compare a tensor-only data container recursively."""
    assert loaded.batch_size == expected.batch_size, f"{field} batch-size mismatch"
    assert set(loaded.keys()) == set(expected.keys()), f"{field} key mismatch"
    for key in expected.keys():
        loaded_value = loaded[key]
        expected_value = expected[key]
        if isinstance(expected_value, TensorDictBase):
            assert isinstance(loaded_value, TensorDictBase)
            _assert_tensordict_equal(loaded_value, expected_value, f"{field}.{key}")
        else:
            assert torch.equal(loaded_value, expected_value), (
                f"{field}[{key!r}] value mismatch"
            )


def _assert_mesh_equal(loaded: Mesh, expected: Mesh, field: str) -> None:
    """Compare one reconstructed mesh, including its exact structured type."""
    assert type(loaded) is Mesh
    assert torch.equal(loaded.points, expected.points), f"{field}.points mismatch"
    assert torch.equal(loaded.cells, expected.cells), f"{field}.cells mismatch"
    for data_field in ("point_data", "cell_data", "global_data"):
        _assert_tensordict_equal(
            getattr(loaded, data_field),
            getattr(expected, data_field),
            f"{field}.{data_field}",
        )


@pytest.fixture(
    params=("current", "legacy"),
    ids=("current-tensorclass", "legacy-decorator"),
)
def fixture_dir(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    """Return a fresh current file or the immutable legacy fixture."""
    if request.param == "current":
        path = tmp_path / "current.pdmsh"
        build_canonical_domain_mesh().save(path)
        return path
    assert LEGACY_FIXTURE_DIR.is_dir(), (
        f"Missing committed .pdmsh fixture: {LEGACY_FIXTURE_DIR}"
    )
    return LEGACY_FIXTURE_DIR


class TestPdmshGoldenFixture:
    """Verify exact reconstruction from current and decorator-era layouts."""

    def test_reconstructs_exact_nested_types(self, fixture_dir: Path):
        loaded = DomainMesh.load(fixture_dir)
        assert type(loaded) is DomainMesh
        assert type(loaded.interior) is Mesh
        assert set(loaded.boundaries.keys()) == {"wall", "inlet"}
        for boundary in loaded.boundaries.values():
            assert type(boundary) is Mesh

    def test_all_fields_match(self, fixture_dir: Path):
        loaded = DomainMesh.load(fixture_dir)
        expected = build_canonical_domain_mesh()

        _assert_mesh_equal(loaded.interior, expected.interior, "interior")
        assert set(loaded.boundaries.keys()) == set(expected.boundaries.keys())
        for name in expected.boundaries.keys():
            _assert_mesh_equal(
                loaded.boundaries[name],
                expected.boundaries[name],
                f"boundaries.{name}",
            )
        _assert_tensordict_equal(
            loaded.global_data,
            expected.global_data,
            "global_data",
        )

    def test_current_writer_layout_matches_manifest(self, tmp_path: Path):
        written = tmp_path / "current.pdmsh"
        build_canonical_domain_mesh().save(written)
        expected = json.loads(CURRENT_MANIFEST_PATH.read_text())
        assert serialization_manifest(written) == expected

    def test_current_and_legacy_layouts_record_the_domain_type(self):
        """Both writer generations retain a root-level type discriminator."""
        expected_type = "<class 'physicsnemo.mesh.domain_mesh.DomainMesh'>"
        legacy_metadata = json.loads((LEGACY_FIXTURE_DIR / "meta.json").read_text())
        current_manifest = json.loads(CURRENT_MANIFEST_PATH.read_text())

        assert legacy_metadata == {"_type": expected_type}
        assert current_manifest["meta.json"] == {"_type": expected_type}
        assert (LEGACY_FIXTURE_DIR / "_tensordict").is_dir()
        assert any(path.startswith("_tensordict/") for path in current_manifest)

    def test_out_fills_nested_meshes(self, fixture_dir: Path):
        """Both layouts fill preallocated interior and boundary tensors."""
        expected = build_canonical_domain_mesh()
        out = expected._tensordict.apply(torch.zeros_like)

        loaded = DomainMesh.load(fixture_dir, out=out)

        assert loaded._tensordict is out
        _assert_mesh_equal(loaded.interior, expected.interior, "interior")
        for name in expected.boundaries.keys():
            _assert_mesh_equal(
                loaded.boundaries[name],
                expected.boundaries[name],
                f"boundaries.{name}",
            )

    @pytest.mark.cuda
    def test_load_honors_device(self, fixture_dir: Path):
        """``device=`` reaches the interior and boundaries in both layouts."""
        loaded = DomainMesh.load(fixture_dir, device="cuda")
        assert loaded.interior.points.device.type == "cuda"
        assert loaded.global_data.device.type == "cuda"
        for boundary in loaded.boundaries.values():
            assert boundary.points.device.type == "cuda"

    @pytest.mark.cuda
    def test_load_rejects_device_that_conflicts_with_out(self, fixture_dir: Path):
        """A CPU output cannot silently override or mix with a CUDA request."""
        out = build_canonical_domain_mesh()._tensordict.apply(torch.zeros_like)

        with pytest.raises(ValueError, match=r"device=.*conflicts with `out`"):
            DomainMesh.load(fixture_dir, device="cuda", out=out)
