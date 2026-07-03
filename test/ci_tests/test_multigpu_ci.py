# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the multi-GPU CI selection utility."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/multigpu_ci.py"
SPEC = importlib.util.spec_from_file_location("multigpu_ci", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
multigpu_ci = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = multigpu_ci
SPEC.loader.exec_module(multigpu_ci)

NOW = datetime(2026, 7, 2, 20, 0, tzinfo=timezone.utc)
BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40
_UNSET = object()
GIT = shutil.which("git")


def _run_git(repo: Path, *args: str, capture_output: bool = False):
    if GIT is None:
        pytest.skip("git is required for baseline-diff integration tests")
    return subprocess.run(  # noqa: S603 - fixed git executable and test-only argv
        [GIT, *args],
        cwd=repo,
        check=True,
        capture_output=capture_output,
        text=capture_output,
    )


def _manifest(
    stream: str,
    files: list[str],
    *,
    generated_at: datetime = NOW,
) -> multigpu_ci.ManifestInput:
    test_file = (
        "test/distributed/test_manager.py"
        if stream == "dynamic"
        else "test/domain_parallel/test_initialization.py"
    )
    all_files = sorted(set(files).union({test_file}))
    executed_files = all_files
    return multigpu_ci.ManifestInput(
        data={
            "schema_version": 1,
            "stream": stream,
            "complete": True,
            "source_sha": BASE_SHA,
            "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
            "coverage_version": "7.14.1",
            "coverage_shards": 3,
            "nproc": 2,
            "runner_profile": "linux-amd64-gpu-h100-latest-2",
            "test_definition_count": 1,
            "files": all_files,
            "test_files": [test_file],
            "executed_files": executed_files,
            "support_files": [],
            "junit_files": 1 if stream == "dynamic" else 2,
            "junit_tests": 2,
            "junit_skipped": 0,
            "junit_skip_reasons": [],
        }
    )


def _manifests() -> dict[str, multigpu_ci.ManifestInput]:
    return {
        "dynamic": _manifest("dynamic", ["physicsnemo/distributed/manager.py"]),
        "static": _manifest("static", ["physicsnemo/domain_parallel/shard_tensor.py"]),
    }


def _pr(
    number: int = 42,
    count: int = 1,
    labels: list | None = None,
    *,
    head_sha: str = HEAD_SHA,
    base_sha: str = BASE_SHA,
) -> dict:
    return {
        "number": number,
        "changed_files": count,
        "labels": labels or [],
        "head": {"sha": head_sha},
        "base": {"sha": base_sha},
    }


def _files(*paths: str) -> list[dict[str, str]]:
    return [{"filename": path, "status": "modified"} for path in paths]


def _evaluate(
    tmp_path: Path,
    *,
    event: str = "push",
    ref: str = "pull-request/42",
    dispatch_stream: str = "both",
    pr_data: object = _UNSET,
    pr_after_data: object = _UNSET,
    files: object = _UNSET,
    manifests: dict[str, multigpu_ci.ManifestInput] | None = None,
    baseline_change_sets: dict[str, list[multigpu_ci.FileChange]] | None = None,
):
    before = _pr() if pr_data is _UNSET else pr_data
    after = before if pr_after_data is _UNSET else pr_after_data
    inventories = {
        stream: (list(manifest.data["test_files"]), 1)
        for stream, manifest in _manifests().items()
    }
    return multigpu_ci.evaluate_gate(
        repo_root=tmp_path,
        event_name=event,
        ref_name=ref,
        dispatch_stream=dispatch_stream,
        pr_data=before,
        pr_after_data=after,
        files_payload=_files("README.md") if files is _UNSET else files,
        manifests=_manifests() if manifests is None else manifests,
        now=NOW,
        max_age_hours=72,
        expected_head_sha=HEAD_SHA,
        expected_nprocs={"dynamic": 2, "static": 2},
        expected_runner_profiles={
            "dynamic": "linux-amd64-gpu-h100-latest-2",
            "static": "linux-amd64-gpu-h100-latest-2",
        },
        test_inventories=inventories,
        baseline_change_sets=baseline_change_sets,
    )


@pytest.mark.parametrize(
    ("stream", "dynamic", "static"),
    [("both", True, True), ("dynamic", True, False), ("static", False, True)],
)
def test_dispatch_selects_requested_stream(
    tmp_path: Path, stream: str, dynamic: bool, static: bool
):
    decisions = _evaluate(
        tmp_path,
        event="workflow_dispatch",
        dispatch_stream=stream,
        pr_data=None,
        files=None,
        manifests={},
    )
    assert decisions["dynamic"].run is dynamic
    assert decisions["static"].run is static


def test_schedule_runs_both_without_gate_inputs(tmp_path: Path):
    decisions = _evaluate(
        tmp_path, event="schedule", pr_data=None, files=None, manifests={}
    )
    assert decisions["dynamic"].run
    assert decisions["static"].run


def test_gate_outputs_are_lowercase_and_complete(tmp_path: Path):
    decisions = _evaluate(tmp_path, event="schedule", pr_data=None, files=None)
    output = tmp_path / "github-output"
    summary = tmp_path / "summary.md"
    multigpu_ci._write_gate_outputs(decisions, output, summary)
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values["run_dynamic"] == "true"
    assert values["run_static"] == "true"
    assert "Multi-GPU PR selection" in summary.read_text()


def test_manifest_matches_streams_independently(tmp_path: Path):
    decisions = _evaluate(
        tmp_path,
        pr_data=_pr(count=2),
        files=_files(
            "physicsnemo/distributed/manager.py",
            "physicsnemo/domain_parallel/shard_tensor.py",
        ),
    )
    assert decisions["dynamic"].run
    assert decisions["static"].run
    assert "manifest-match" in decisions["dynamic"].reasons
    assert "manifest-match" in decisions["static"].reasons


def test_unrelated_docs_change_skips_both(tmp_path: Path):
    decisions = _evaluate(tmp_path)
    assert not decisions["dynamic"].run
    assert not decisions["static"].run
    assert decisions["dynamic"].reasons == ["no-impact"]


def test_force_label_runs_both(tmp_path: Path):
    decisions = _evaluate(
        tmp_path,
        pr_data=_pr(labels=[{"name": "ci:multi-gpu"}]),
    )
    assert decisions["dynamic"].run
    assert decisions["static"].run
    assert decisions["dynamic"].reasons == ["force-label"]


def test_missing_manifest_forces_only_that_stream(tmp_path: Path):
    manifests = _manifests()
    manifests["dynamic"] = multigpu_ci.ManifestInput(error_reason="manifest-missing")
    decisions = _evaluate(tmp_path, manifests=manifests)
    assert decisions["dynamic"].run
    assert not decisions["static"].run
    assert "manifest-missing" in decisions["dynamic"].reasons


def test_stale_manifest_forces_affected_stream(tmp_path: Path):
    manifests = _manifests()
    manifests["static"] = _manifest(
        "static",
        ["physicsnemo/domain_parallel/shard_tensor.py"],
        generated_at=NOW - timedelta(hours=73),
    )
    decisions = _evaluate(tmp_path, manifests=manifests)
    assert decisions["static"].run
    assert "manifest-stale" in decisions["static"].reasons
    assert not decisions["dynamic"].run


def test_marker_inventory_mismatch_forces_affected_stream(tmp_path: Path):
    manifests = _manifests()
    dynamic = dict(manifests["dynamic"].data or {})
    dynamic["test_definition_count"] = 2
    manifests["dynamic"] = multigpu_ci.ManifestInput(data=dynamic)
    decisions = _evaluate(tmp_path, manifests=manifests)
    assert decisions["dynamic"].run
    assert not decisions["static"].run
    assert "manifest-inventory-mismatch" in decisions["dynamic"].reasons


def test_manifest_rank_mismatch_forces_only_affected_stream(tmp_path: Path):
    inventories = {
        stream: (list(manifest.data["test_files"]), 1)
        for stream, manifest in _manifests().items()
    }
    decisions = multigpu_ci.evaluate_gate(
        repo_root=tmp_path,
        event_name="push",
        ref_name="pull-request/42",
        dispatch_stream="both",
        pr_data=_pr(),
        pr_after_data=_pr(),
        files_payload=_files("README.md"),
        manifests=_manifests(),
        now=NOW,
        max_age_hours=72,
        expected_head_sha=HEAD_SHA,
        expected_nprocs={"dynamic": 2, "static": 4},
        expected_runner_profiles={
            "dynamic": "linux-amd64-gpu-h100-latest-2",
            "static": "linux-amd64-gpu-h100-latest-2",
        },
        test_inventories=inventories,
    )
    assert not decisions["dynamic"].run
    assert decisions["static"].run
    assert "manifest-rank-mismatch" in decisions["static"].reasons


def test_manifest_runner_profile_mismatch_is_rejected():
    data = _manifest("dynamic", ["physicsnemo/distributed/manager.py"]).data
    assert data is not None
    with pytest.raises(multigpu_ci.ManifestError, match="runner profile changed"):
        multigpu_ci.validate_manifest_data(
            data,
            "dynamic",
            now=NOW,
            max_age_hours=72,
            expected_runner_profile="linux-amd64-gpu-a100-latest-2",
        )


@pytest.mark.parametrize(
    ("pr_data", "files", "reason"),
    [
        ({}, [], "pr-metadata-unavailable"),
        (_pr(count=2), _files("README.md"), "changed-files-incomplete"),
        (_pr(number=41), _files("README.md"), "pr-metadata-unavailable"),
        (_pr(count=3000), [], "changed-files-incomplete"),
    ],
)
def test_api_uncertainty_runs_both(
    tmp_path: Path, pr_data: object, files: object, reason: str
):
    decisions = _evaluate(tmp_path, pr_data=pr_data, files=files)
    assert decisions["dynamic"].run
    assert decisions["static"].run
    assert reason in decisions["dynamic"].reasons


@pytest.mark.parametrize(
    ("pr_data", "pr_after", "reason"),
    [
        (None, None, "pr-metadata-unavailable"),
        (_pr(head_sha="c" * 40), _pr(head_sha="c" * 40), "pr-snapshot-raced"),
        (_pr(), _pr(count=2), "pr-snapshot-raced"),
        (_pr(), None, "pr-metadata-unavailable"),
    ],
)
def test_missing_or_raced_pr_snapshot_runs_both(
    tmp_path: Path, pr_data: object, pr_after: object, reason: str
):
    decisions = _evaluate(
        tmp_path,
        pr_data=pr_data,
        pr_after_data=pr_after,
        files=_files("README.md"),
    )
    assert decisions["dynamic"].run
    assert decisions["static"].run
    assert reason in decisions["dynamic"].reasons


@pytest.mark.parametrize(
    "pr_data",
    [
        {**_pr(), "number": 42.0},
        {**_pr(), "number": True},
        {**_pr(), "changed_files": True},
        {**_pr(), "changed_files": 1.0},
    ],
)
def test_non_integer_pr_scalars_run_both(tmp_path: Path, pr_data: dict):
    decisions = _evaluate(tmp_path, pr_data=pr_data)
    assert decisions["dynamic"].run
    assert decisions["static"].run
    assert "pr-metadata-unavailable" in decisions["dynamic"].reasons


def test_nested_paginated_file_payload_is_flattened(tmp_path: Path):
    decisions = _evaluate(
        tmp_path,
        pr_data=_pr(count=2),
        files=[
            _files("README.md"),
            [],
            _files("physicsnemo/domain_parallel/shard_tensor.py"),
        ],
    )
    assert not decisions["dynamic"].run
    assert decisions["static"].run


def test_truncated_nested_file_payload_runs_both(tmp_path: Path):
    decisions = _evaluate(
        tmp_path,
        pr_data=_pr(count=2),
        files=[_files("README.md")],
    )
    assert decisions["dynamic"].run
    assert decisions["static"].run
    assert "changed-files-incomplete" in decisions["dynamic"].reasons


def test_git_baseline_changes_require_ancestry_and_preserve_renames(
    tmp_path: Path,
):
    _run_git(tmp_path, "init", "-q")
    _run_git(tmp_path, "config", "user.email", "ci@example.com")
    _run_git(tmp_path, "config", "user.name", "CI")
    old_path = tmp_path / "physicsnemo/old.py"
    old_path.parent.mkdir()
    old_path.write_text("VALUE = 1\n")
    _run_git(tmp_path, "add", ".")
    _run_git(tmp_path, "commit", "-qm", "base")
    source_sha = _run_git(
        tmp_path, "rev-parse", "HEAD", capture_output=True
    ).stdout.strip()
    new_path = tmp_path / "physicsnemo/new.py"
    old_path.rename(new_path)
    _run_git(tmp_path, "add", "-A")
    _run_git(tmp_path, "commit", "-qm", "rename")
    base_sha = _run_git(
        tmp_path, "rev-parse", "HEAD", capture_output=True
    ).stdout.strip()

    changes = multigpu_ci._git_baseline_changes(tmp_path, source_sha, base_sha)
    assert changes == [
        multigpu_ci.FileChange("physicsnemo/new.py", "renamed", "physicsnemo/old.py")
    ]


def test_rename_matches_previous_manifest_path(tmp_path: Path):
    manifests = _manifests()
    manifests["dynamic"] = _manifest(
        "dynamic", ["physicsnemo/models/distributed_graph.py"]
    )
    files = [
        {
            "filename": "physicsnemo/models/renamed_graph.py",
            "previous_filename": "physicsnemo/models/distributed_graph.py",
            "status": "renamed",
        }
    ]
    decisions = _evaluate(tmp_path, files=files, manifests=manifests)
    assert decisions["dynamic"].run
    assert "manifest-match" in decisions["dynamic"].reasons


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("ci:multi-gpu-dynamic", "dynamic"),
        ("ci:multi-gpu-static", "static"),
    ],
)
def test_stream_force_labels_are_independent(tmp_path: Path, label: str, expected: str):
    decisions = _evaluate(tmp_path, pr_data=_pr(labels=[{"name": label}]))
    other = "static" if expected == "dynamic" else "dynamic"
    assert decisions[expected].run
    assert not decisions[other].run
    assert "force-label" in decisions[expected].reasons


def test_unavailable_manifest_base_diff_forces_affected_stream(tmp_path: Path):
    manifests = _manifests()
    dynamic = dict(manifests["dynamic"].data or {})
    dynamic["source_sha"] = "c" * 40
    manifests["dynamic"] = multigpu_ci.ManifestInput(data=dynamic)
    decisions = _evaluate(tmp_path, manifests=manifests)
    assert decisions["dynamic"].run
    assert not decisions["static"].run
    assert "manifest-base-unavailable" in decisions["dynamic"].reasons


def test_unrelated_manifest_base_drift_can_still_skip(tmp_path: Path):
    manifests = _manifests()
    for stream in ("dynamic", "static"):
        data = dict(manifests[stream].data or {})
        data["source_sha"] = "c" * 40
        manifests[stream] = multigpu_ci.ManifestInput(data=data)
    baseline_changes = {
        stream: [multigpu_ci.FileChange("README.md", "modified")]
        for stream in ("dynamic", "static")
    }
    decisions = _evaluate(
        tmp_path,
        manifests=manifests,
        baseline_change_sets=baseline_changes,
    )
    assert not decisions["dynamic"].run
    assert not decisions["static"].run


def test_relevant_manifest_base_drift_runs_only_affected_stream(tmp_path: Path):
    manifests = _manifests()
    dynamic = dict(manifests["dynamic"].data or {})
    dynamic["source_sha"] = "c" * 40
    manifests["dynamic"] = multigpu_ci.ManifestInput(data=dynamic)
    baseline_changes = {
        "dynamic": [
            multigpu_ci.FileChange("physicsnemo/distributed/manager.py", "modified")
        ],
        "static": [],
    }
    decisions = _evaluate(
        tmp_path,
        manifests=manifests,
        baseline_change_sets=baseline_changes,
    )
    assert decisions["dynamic"].run
    assert not decisions["static"].run
    assert "manifest-base-drift" in decisions["dynamic"].reasons


def test_global_path_and_stream_prefixes(tmp_path: Path):
    decisions = _evaluate(
        tmp_path,
        pr_data=_pr(count=2),
        files=_files("uv.lock", "physicsnemo/domain_parallel/new_op.py"),
    )
    assert "global-path" in decisions["dynamic"].reasons
    assert "stream-prefix" in decisions["static"].reasons


@pytest.mark.parametrize(
    ("marker", "expected_stream"),
    [("multigpu_dynamic", "dynamic"), ("multigpu_static", "static")],
)
def test_new_marker_file_selects_stream(
    tmp_path: Path, marker: str, expected_stream: str
):
    test_file = tmp_path / "test/new/test_new.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        f"import pytest\n\n@pytest.mark.{marker}\ndef test_new():\n    pass\n"
    )
    decisions = _evaluate(
        tmp_path,
        files=_files("test/new/test_new.py"),
    )
    assert decisions[expected_stream].run
    assert "marker-file" in decisions[expected_stream].reasons


def test_non_python_asset_beneath_marker_directory_selects_stream(tmp_path: Path):
    decisions = _evaluate(
        tmp_path,
        files=_files("test/domain_parallel/data/golden.pth"),
    )
    assert decisions["static"].run
    assert decisions["dynamic"].run
    assert "test-asset" in decisions["static"].reasons
    assert "runtime-asset" in decisions["dynamic"].reasons


def test_shared_test_asset_runs_both_streams(tmp_path: Path):
    decisions = _evaluate(tmp_path, files=_files("test/shared-data/golden.bin"))
    assert decisions["dynamic"].run
    assert decisions["static"].run
    assert "runtime-asset" in decisions["dynamic"].reasons


def test_python_helper_beneath_marker_directory_selects_stream(tmp_path: Path):
    helper = tmp_path / "test/domain_parallel/helpers/new_mesh.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("VALUE = 1\n")
    decisions = _evaluate(
        tmp_path,
        files=_files("test/domain_parallel/helpers/new_mesh.py"),
    )
    assert decisions["static"].run
    assert not decisions["dynamic"].run
    assert "test-asset" in decisions["static"].reasons


def test_new_production_file_runs_both(tmp_path: Path):
    decisions = _evaluate(
        tmp_path,
        files=[{"filename": "physicsnemo/new_feature.py", "status": "added"}],
    )
    assert decisions["dynamic"].run
    assert decisions["static"].run
    assert "new-production-file" in decisions["dynamic"].reasons


def test_rename_into_production_runs_both(tmp_path: Path):
    decisions = _evaluate(
        tmp_path,
        files=[
            {
                "filename": "physicsnemo/new_feature.py",
                "previous_filename": "examples/new_feature.py",
                "status": "renamed",
            }
        ],
    )
    assert decisions["dynamic"].run
    assert decisions["static"].run
    assert "new-production-file" in decisions["static"].reasons


def test_ast_discovery_ignores_comments_and_finds_module_marker(tmp_path: Path):
    tests = tmp_path / "test"
    tests.mkdir()
    (tests / "test_dynamic.py").write_text(
        "import pytest\npytestmark = pytest.mark.multigpu_dynamic\n"
        "def test_value():\n    assert True\n"
    )
    (tests / "test_comment.py").write_text(
        '"""pytest.mark.multigpu_dynamic is only documentation."""\n'
        "def test_value():\n    assert True\n"
    )
    files, _ = multigpu_ci.discover_test_files(tmp_path, "dynamic")
    assert files == ["test/test_dynamic.py"]


def test_ast_discovery_ignores_ci_policy_tests(tmp_path: Path):
    ci_tests = tmp_path / "test/ci_tests"
    ci_tests.mkdir(parents=True)
    (ci_tests / "test_policy.py").write_text('MARKER_UNDER_TEST = "multigpu_dynamic"\n')
    real_test = tmp_path / "test/distributed/test_real.py"
    real_test.parent.mkdir(parents=True)
    real_test.write_text(
        "import pytest\n"
        "@pytest.mark.multigpu_dynamic\n"
        "def test_value():\n    assert True\n"
    )
    files, count = multigpu_ci.discover_test_files(tmp_path, "dynamic")
    assert files == ["test/distributed/test_real.py"]
    assert count == 1


def test_ast_discovery_finds_imported_marker_name(tmp_path: Path):
    tests = tmp_path / "test"
    tests.mkdir()
    (tests / "test_dynamic.py").write_text(
        "from helpers import multigpu_dynamic\n"
        "@multigpu_dynamic\n"
        "def test_value():\n    assert True\n"
    )
    files, count = multigpu_ci.discover_test_files(tmp_path, "dynamic")
    assert files == ["test/test_dynamic.py"]
    assert count == 1


def test_ast_discovery_finds_aliased_marker_import(tmp_path: Path):
    tests = tmp_path / "test"
    tests.mkdir()
    (tests / "test_static.py").write_text(
        "from helpers import multigpu_static as multi_gpu\n"
        "@multi_gpu\n"
        "def test_value():\n    assert True\n"
    )
    files, count = multigpu_ci.discover_test_files(tmp_path, "static")
    assert files == ["test/test_static.py"]
    assert count == 1


def test_static_junit_paths_are_rank_safe(tmp_path: Path):
    rank_zero = multigpu_ci.junit_path(tmp_path, "static", {"RANK": "0"})
    rank_one = multigpu_ci.junit_path(tmp_path, "static", {"RANK": "1"})
    assert rank_zero != rank_one
    assert rank_zero.name == "multigpu-static-rank-0.xml"


@pytest.mark.parametrize("stream", ["dynamic", "static"])
def test_build_manifest_validates_process_and_rank_shards(tmp_path: Path, stream: str):
    coverage = pytest.importorskip("coverage")
    source = tmp_path / "physicsnemo/module.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n")
    test_file = tmp_path / f"test/test_{stream}.py"
    test_file.parent.mkdir()
    test_file.write_text("def test_value():\n    assert True\n")

    test_list = tmp_path / "test-list.json"
    test_list.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stream": stream,
                "marker": multigpu_ci.MARKERS[stream],
                "test_definition_count": 1,
                "test_files": [test_file.relative_to(tmp_path).as_posix()],
            }
        )
    )
    junit_dir = tmp_path / "junit"
    junit_dir.mkdir()
    ranks = range(2) if stream == "static" else range(1)
    for rank in ranks:
        (junit_dir / f"multigpu-{stream}-rank-{rank}.xml").write_text(
            '<testsuites><testsuite tests="1" failures="0" errors="0" '
            'skipped="0"/></testsuites>\n'
        )

    coverage_rc = tmp_path / "coverage.rc"
    coverage_rc.write_text("[run]\nrelative_files = True\n")
    coverage_base = tmp_path / f".coverage.multigpu-{stream}"
    shard_names = (
        [f"{coverage_base.name}.rank-0.a", f"{coverage_base.name}.rank-1.b"]
        if stream == "static"
        else [f"{coverage_base.name}.{suffix}" for suffix in ("a", "b", "c")]
    )
    for shard_name in shard_names:
        data = coverage.CoverageData(basename=str(tmp_path / shard_name))
        data.add_lines(
            {
                "physicsnemo/module.py": {1},
                f"test/test_{stream}.py": {1, 2},
            }
        )
        data.write()

    output = tmp_path / "manifest.json"
    multigpu_ci.build_manifest(
        tmp_path,
        stream,
        test_list,
        coverage_base,
        coverage_rc,
        junit_dir,
        output,
        BASE_SHA,
        2,
        "linux-amd64-gpu-h100-latest-2",
    )
    manifest = json.loads(output.read_text())
    assert manifest["complete"] is True
    assert manifest["coverage_shards"] == len(shard_names)
    assert manifest["junit_files"] == len(list(ranks))
    assert manifest["files"] == [
        "physicsnemo/module.py",
        f"test/test_{stream}.py",
    ]


def test_junit_summary_rejects_missing_static_rank(tmp_path: Path):
    (tmp_path / "multigpu-static-rank-0.xml").write_text(
        '<testsuite tests="1" failures="0" errors="0" skipped="0"/>\n'
    )
    with pytest.raises(RuntimeError, match="incomplete JUnit ranks"):
        multigpu_ci._junit_summary(tmp_path, "static", 2)


@pytest.mark.parametrize(
    ("reason", "accepted"),
    [
        ("Need at least 4 ranks (divisible by 2) for 2-D mesh test", True),
        ("natten is not installed", False),
    ],
)
def test_junit_summary_allows_only_expected_skips(
    tmp_path: Path, reason: str, accepted: bool
):
    xml = (
        '<testsuite tests="2" failures="0" errors="0" skipped="1">'
        f'<testcase name="skipped"><skipped message="{reason}"/></testcase>'
        '<testcase name="passed"/>'
        "</testsuite>\n"
    )
    if accepted:
        for rank in range(2):
            (tmp_path / f"multigpu-static-rank-{rank}.xml").write_text(xml)
        summary = multigpu_ci._junit_summary(tmp_path, "static", 2)
        assert summary.skipped == 2
    else:
        (tmp_path / "multigpu-dynamic-rank-0.xml").write_text(xml)
        with pytest.raises(RuntimeError, match="unexpected skips"):
            multigpu_ci._junit_summary(tmp_path, "dynamic", 2)


def test_manifest_rejects_duplicate_paths():
    data = _manifest("dynamic", ["physicsnemo/distributed/manager.py"]).data
    assert data is not None
    broken = dict(data)
    broken["files"] = [
        "physicsnemo/distributed/manager.py",
        "physicsnemo/distributed/manager.py",
        "test/distributed/test_manager.py",
    ]
    with pytest.raises(multigpu_ci.ManifestError, match="sorted and unique"):
        multigpu_ci.validate_manifest_data(broken, "dynamic", now=NOW, max_age_hours=72)


def test_manifest_rejects_truncated_but_well_formed_payload():
    data = _manifest("dynamic", ["physicsnemo/distributed/manager.py"]).data
    assert data is not None
    broken = dict(data)
    broken.pop("executed_files")
    with pytest.raises(multigpu_ci.ManifestError, match="executed_files"):
        multigpu_ci.validate_manifest_data(broken, "dynamic", now=NOW, max_age_hours=72)


@pytest.mark.parametrize("schema_version", [True, 1.0, "1"])
def test_manifest_rejects_non_integer_schema(schema_version: object):
    data = _manifest("dynamic", ["physicsnemo/distributed/manager.py"]).data
    assert data is not None
    broken = dict(data)
    broken["schema_version"] = schema_version
    with pytest.raises(multigpu_ci.ManifestError, match="schema"):
        multigpu_ci.validate_manifest_data(broken, "dynamic", now=NOW, max_age_hours=72)
