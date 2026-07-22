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

"""Writer-layout and backward-read tests for the ``.pdmsh`` memmap format."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from tensordict import TensorDictBase

from physicsnemo.mesh import DomainMesh, Mesh

_REGEN_PATH = Path(__file__).parent / "golden_pdmsh" / "_regenerate.py"
assert _REGEN_PATH.is_file(), f"Missing .pdmsh regeneration helper: {_REGEN_PATH}"
_spec = importlib.util.spec_from_file_location("_pdmsh_golden_regen", _REGEN_PATH)
assert _spec is not None and _spec.loader is not None
_regen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_regen)
build_canonical_domain_mesh = _regen.build_canonical_domain_mesh
CURRENT_FIXTURE_DIR: Path = _regen.CURRENT_FIXTURE_DIR
LEGACY_FIXTURE_DIR: Path = _regen.LEGACY_FIXTURE_DIR


def _serialization_manifest(root: Path) -> dict[str, object]:
    """Describe paths and metadata while ignoring tensor payload bytes."""
    manifest: dict[str, object] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            manifest[f"{relative}/"] = None
        elif path.name == "meta.json":
            manifest[relative] = json.loads(path.read_text())
        else:
            manifest[relative] = path.stat().st_size
    return manifest


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
    params=(CURRENT_FIXTURE_DIR, LEGACY_FIXTURE_DIR),
    ids=("current-tensorclass", "legacy-decorator"),
)
def fixture_dir(request: pytest.FixtureRequest) -> Path:
    """Return each committed layout, failing clearly if one was removed."""
    path: Path = request.param
    assert path.is_dir(), f"Missing committed .pdmsh fixture: {path}"
    return path


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

    def test_current_writer_layout_matches_fixture(self, tmp_path: Path):
        written = tmp_path / "current.pdmsh"
        build_canonical_domain_mesh().save(written)
        assert _serialization_manifest(written) == _serialization_manifest(
            CURRENT_FIXTURE_DIR
        )
