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

"""Utilities for selecting and running PhysicsNeMo multi-GPU CI streams."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
STREAMS = ("dynamic", "static")
MARKERS = {
    "dynamic": "multigpu_dynamic",
    "static": "multigpu_static",
}
PYTEST_FLAGS = {
    "dynamic": "--multigpu-dynamic",
    "static": "--multigpu-static",
}
PR_REF_PATTERN = re.compile(r"^pull-request/([1-9][0-9]*)$")
SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40,64}$")
VALID_CHANGE_STATUSES = {
    "added",
    "changed",
    "copied",
    "modified",
    "removed",
    "renamed",
    "unchanged",
}
FORCE_LABEL = "ci:multi-gpu"
STREAM_FORCE_LABELS = {
    "dynamic": "ci:multi-gpu-dynamic",
    "static": "ci:multi-gpu-static",
}
GLOBAL_TRIGGER_PATHS = {
    ".github/ci-requirements.lock",
    ".github/ci-requirements.txt",
    ".github/scripts/multigpu_ci.py",
    ".github/workflows/github-multigpu.yml",
    "pyproject.toml",
    "test/__init__.py",
    "test/conftest.py",
    "test/coverage.multigpu.rc",
    "test/pytest_utils.py",
    "uv.lock",
}
GLOBAL_TRIGGER_PREFIXES = (
    ".github/actions/",
    "test/plugins/",
)
MAX_DIAGNOSTIC_PATHS = 20
# Keep these exact: a new skip reason must block manifest publication so PR
# selection fails open and continues running the affected multi-GPU stream.
ALLOWED_SKIP_REASONS: Mapping[str, frozenset[str]] = {
    "dynamic": frozenset(
        {
            "Skip SongUNetPosLtEmbd AMP/agnostic tests on cpu",
        }
    ),
    "static": frozenset(
        {
            "Combined ddp+domain needs >= 4 GPUs divisible by 2 (have 2)",
            "Conv1d with stride > 1 and kernel size != stride is expected to fail",
            "Conv2d with stride > 1 and kernel size != stride is expected to fail",
            "Conv3d with 2D mesh requires at least 4 GPUs",
            "Conv3d with stride > 1 and kernel size != stride is expected to fail",
            "DTensor with 2D mesh and backwards fails currently upstream",
            "Even Kernels only supported for stride = kernel size and padding = 0",
            "LayerNorm with affine=True is currently failing tests",
            "Need at least 4 ranks (divisible by 2) for 2-D mesh test",
            "Odd Kernels not yet supported for transposed convolutions",
            "Pooling requires stride == K",
            "Requires exactly 4 ranks for the (ddp=2, domain=2) mesh",
            "Skip SongUNetPosLtEmbd AMP/agnostic tests on cpu",
            "Skip tests on cpu",
            "use_orig_params=True + ShardTensor under FSDP NO_SHARD is unsupported: "
            "FSDP writeback fails when local parameter shape changes",
        }
    ),
}


def _is_allowed_skip(stream: str, reason: str) -> bool:
    """Return whether a stream's known test matrix intentionally skips a case."""

    return reason in ALLOWED_SKIP_REASONS.get(stream, frozenset())


class ManifestError(ValueError):
    """A manifest is unusable for a specific fail-open reason."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class ManifestInput:
    """A loaded manifest or an error recorded while loading it."""

    data: Mapping[str, Any] | None = None
    error_reason: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class ValidatedManifest:
    """Fields needed by the PR selector from a validated manifest."""

    files: frozenset[str]
    test_files: frozenset[str]
    source_sha: str


@dataclass(frozen=True)
class PRSnapshot:
    """Fields that must stay stable while GitHub paginates a PR diff."""

    number: int
    changed_files: int
    labels: frozenset[str]
    head_sha: str
    base_sha: str


@dataclass(frozen=True)
class JUnitSummary:
    """Validated per-rank JUnit totals for a completed stream."""

    file_count: int
    tests: int
    skipped: int
    failures: int
    errors: int
    skip_reasons: tuple[str, ...]


@dataclass(frozen=True)
class FileChange:
    """A normalized GitHub pull-request file record."""

    filename: str
    status: str
    previous_filename: str | None = None

    @property
    def paths(self) -> tuple[str, ...]:
        """Return current and previous paths for matching rename impact."""

        if self.previous_filename is None:
            return (self.filename,)
        return (self.filename, self.previous_filename)


@dataclass
class StreamDecision:
    """Decision and bounded diagnostics for one multi-GPU stream."""

    run: bool = False
    reasons: list[str] = field(default_factory=list)
    matched_files: set[str] = field(default_factory=set)

    def trigger(self, reason: str, path: str | None = None) -> None:
        """Mark the stream runnable and record a stable reason token."""

        self.run = True
        if reason not in self.reasons:
            self.reasons.append(reason)
        if path is not None:
            self.matched_files.add(path)

    def finish(self) -> None:
        """Record an explicit reason when the stream is safely skipped."""

        if not self.run and "no-impact" not in self.reasons:
            self.reasons.append("no-impact")

    def as_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-safe diagnostics."""

        matches = sorted(self.matched_files)
        return {
            "run": self.run,
            "reasons": self.reasons,
            "matched_files": matches[:MAX_DIAGNOSTIC_PATHS],
            "matched_count": len(matches),
        }


def _stream(value: str) -> str:
    if value not in STREAMS:
        raise argparse.ArgumentTypeError(f"expected one of: {', '.join(STREAMS)}")
    return value


def _normalize_repo_path(value: Any) -> str:
    """Validate and normalize a repository-relative POSIX path."""

    if not isinstance(value, str) or not value:
        raise ValueError("path must be a non-empty string")
    if "\\" in value or value.startswith("/"):
        raise ValueError(f"path is not repository-relative POSIX: {value!r}")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"path contains a control character: {value!r}")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError(f"path contains an unsafe component: {value!r}")
    normalized = PurePosixPath(value).as_posix()
    if normalized != value:
        raise ValueError(f"path is not normalized: {value!r}")
    return normalized


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("generated_at must be a string")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(candidate)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("generated_at must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _sorted_unique_paths(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list")
    normalized = [_normalize_repo_path(path) for path in value]
    if normalized != sorted(set(normalized)):
        raise ValueError(f"{field_name} must be sorted and unique")
    return normalized


def _strict_nonnegative_int(data: Mapping[str, Any], field_name: str) -> int:
    value = data.get(field_name)
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def validate_manifest_data(
    data: Mapping[str, Any],
    stream: str,
    *,
    now: datetime,
    max_age_hours: float,
    expected_source_sha: str | None = None,
    expected_test_files: Sequence[str] | None = None,
    expected_test_definition_count: int | None = None,
    expected_nproc: int | None = None,
    expected_runner_profile: str | None = None,
) -> ValidatedManifest:
    """Validate a cached impact manifest for PR selection."""

    if (
        type(data.get("schema_version")) is not int
        or data["schema_version"] != SCHEMA_VERSION
    ):
        raise ManifestError("manifest-invalid", "unsupported manifest schema")
    if data.get("stream") != stream:
        raise ManifestError("manifest-invalid", "manifest stream mismatch")
    if data.get("complete") is not True:
        raise ManifestError("manifest-invalid", "manifest is not complete")
    source_sha = data.get("source_sha")
    if not isinstance(source_sha, str) or not SHA_PATTERN.fullmatch(source_sha):
        raise ManifestError("manifest-invalid", "invalid source_sha")
    if (
        expected_source_sha is not None
        and source_sha.lower() != expected_source_sha.lower()
    ):
        raise ManifestError("manifest-base-mismatch", "manifest does not match PR base")
    try:
        generated_at = _parse_timestamp(data.get("generated_at"))
    except (TypeError, ValueError) as error:
        raise ManifestError("manifest-invalid", str(error)) from error
    now_utc = now.astimezone(timezone.utc)
    if generated_at - now_utc > timedelta(minutes=10):
        raise ManifestError("manifest-invalid", "manifest timestamp is in the future")
    if now_utc - generated_at > timedelta(hours=max_age_hours):
        raise ManifestError("manifest-stale", "manifest is stale")
    try:
        files = _sorted_unique_paths(data.get("files"), "files")
        test_files = _sorted_unique_paths(data.get("test_files"), "test_files")
        executed_files = _sorted_unique_paths(
            data.get("executed_files"), "executed_files"
        )
        support_files = data.get("support_files")
        if not isinstance(support_files, list):
            raise ValueError("support_files must be a list")
        normalized_support = [_normalize_repo_path(path) for path in support_files]
        if normalized_support != sorted(set(normalized_support)):
            raise ValueError("support_files must be sorted and unique")
        coverage_version = data.get("coverage_version")
        if not isinstance(coverage_version, str) or not coverage_version.strip():
            raise ValueError("coverage_version must be a non-empty string")
        nproc = _strict_nonnegative_int(data, "nproc")
        coverage_shards = _strict_nonnegative_int(data, "coverage_shards")
        definition_count = _strict_nonnegative_int(data, "test_definition_count")
        junit_files = _strict_nonnegative_int(data, "junit_files")
        junit_tests = _strict_nonnegative_int(data, "junit_tests")
        junit_skipped = _strict_nonnegative_int(data, "junit_skipped")
        raw_skip_reasons = data.get("junit_skip_reasons")
        if not isinstance(raw_skip_reasons, list) or not all(
            isinstance(reason, str) and reason for reason in raw_skip_reasons
        ):
            raise ValueError("junit_skip_reasons must contain non-empty strings")
        if raw_skip_reasons != sorted(set(raw_skip_reasons)):
            raise ValueError("junit_skip_reasons must be sorted and unique")
        runner_profile = data.get("runner_profile")
        if not isinstance(runner_profile, str) or not runner_profile.strip():
            raise ValueError("runner_profile must be a non-empty string")
    except ValueError as error:
        raise ManifestError("manifest-invalid", str(error)) from error
    if nproc < 2:
        raise ManifestError("manifest-invalid", "nproc must be at least two")
    if expected_nproc is not None and nproc != expected_nproc:
        raise ManifestError("manifest-rank-mismatch", "configured rank count changed")
    if (
        expected_runner_profile is not None
        and runner_profile != expected_runner_profile
    ):
        raise ManifestError(
            "manifest-runner-mismatch", "configured runner profile changed"
        )
    minimum_shards = nproc if stream == "static" else nproc + 1
    if coverage_shards < minimum_shards:
        raise ManifestError("manifest-invalid", "coverage shard count is incomplete")
    expected_junit_files = nproc if stream == "static" else 1
    if junit_files != expected_junit_files or junit_tests <= 0:
        raise ManifestError("manifest-invalid", "JUnit totals are incomplete")
    if junit_skipped > junit_tests:
        raise ManifestError("manifest-invalid", "JUnit skipped count is invalid")
    if bool(junit_skipped) != bool(raw_skip_reasons):
        raise ManifestError("manifest-invalid", "JUnit skip reasons are incomplete")
    if any(not _is_allowed_skip(stream, reason) for reason in raw_skip_reasons):
        raise ManifestError(
            "manifest-invalid", f"{stream} manifest has unexpected skips"
        )
    if definition_count <= 0:
        raise ManifestError(
            "manifest-invalid", "test_definition_count must be positive"
        )
    if not all(
        path.startswith("test/") and path.endswith(".py") for path in test_files
    ):
        raise ManifestError("manifest-invalid", "test_files contains an invalid path")
    if not all(path.startswith(("physicsnemo/", "test/")) for path in executed_files):
        raise ManifestError(
            "manifest-invalid", "executed_files contains an invalid path"
        )
    if not any(path.startswith("physicsnemo/") for path in executed_files):
        raise ManifestError("manifest-invalid", "no physicsnemo file was executed")
    if not any(path.startswith("test/") for path in executed_files):
        raise ManifestError("manifest-invalid", "no test file was executed")
    if not set(test_files).issubset(executed_files):
        raise ManifestError("manifest-invalid", "not every marker test file executed")
    if not all(path.startswith("test/") for path in normalized_support):
        raise ManifestError(
            "manifest-invalid", "support_files contains an invalid path"
        )
    exact_files = sorted(set(executed_files).union(test_files, normalized_support))
    if files != exact_files:
        raise ManifestError("manifest-invalid", "files is not the exact manifest union")
    if expected_test_files is not None and test_files != list(expected_test_files):
        raise ManifestError(
            "manifest-inventory-mismatch", "marker test inventory changed"
        )
    if (
        expected_test_definition_count is not None
        and definition_count != expected_test_definition_count
    ):
        raise ManifestError(
            "manifest-inventory-mismatch", "marker definition count changed"
        )
    return ValidatedManifest(
        files=frozenset(files),
        test_files=frozenset(test_files),
        source_sha=source_sha.lower(),
    )


def _marker_present(tree: ast.AST, marker: str) -> bool:
    return any(
        (isinstance(node, ast.Attribute) and node.attr == marker)
        or (isinstance(node, ast.Name) and node.id == marker)
        or (isinstance(node, ast.alias) and node.name == marker)
        or (isinstance(node, ast.Constant) and node.value == marker)
        for node in ast.walk(tree)
    )


def _test_definition_count(tree: ast.AST, marker: str) -> int:
    marked_count = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if not node.name.startswith("test") and not node.name.startswith("Test"):
            continue
        if any(_marker_present(decorator, marker) for decorator in node.decorator_list):
            marked_count += 1
    if marked_count:
        return marked_count
    # Module-level markers and imported aliases do not necessarily leave the
    # marker name in each decorator. In a marker-bearing file, count its test
    # definitions as a stable inventory fingerprint instead of returning zero.
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and (node.name.startswith("test") or node.name.startswith("Test"))
    )


def discover_test_files(repo_root: Path, stream: str) -> tuple[list[str], int]:
    """Discover marker-bearing test files without importing the test suite."""

    marker = MARKERS[stream]
    candidates = set(repo_root.glob("test/**/test_*.py"))
    candidates.update(repo_root.glob("test/**/*_test.py"))
    files: list[str] = []
    definition_count = 0
    for path in sorted(candidates):
        relative = path.relative_to(repo_root)
        if relative.parts[:2] == ("test", "ci_tests"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as error:
            raise RuntimeError(f"cannot parse {path}: {error}") from error
        if _marker_present(tree, marker):
            files.append(relative.as_posix())
            definition_count += _test_definition_count(tree, marker)
    if not files:
        raise RuntimeError(f"no files found for pytest marker {marker}")
    return files, definition_count


def write_test_list(repo_root: Path, stream: str, output: Path) -> None:
    """Write deterministic marker discovery results for a later test run."""

    files, definition_count = discover_test_files(repo_root, stream)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stream": stream,
        "marker": MARKERS[stream],
        "test_definition_count": definition_count,
        "test_files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_test_list(path: Path, stream: str) -> Mapping[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        type(data.get("schema_version")) is not int
        or data["schema_version"] != SCHEMA_VERSION
        or data.get("stream") != stream
    ):
        raise ValueError("test-list schema or stream mismatch")
    if data.get("marker") != MARKERS[stream]:
        raise ValueError("test-list marker mismatch")
    _sorted_unique_paths(data.get("test_files"), "test_files")
    if (
        type(data.get("test_definition_count")) is not int
        or data["test_definition_count"] <= 0
    ):
        raise ValueError("test_definition_count must be a positive integer")
    return data


def junit_path(junit_dir: Path, stream: str, environ: Mapping[str, str]) -> Path:
    """Build a rank-safe JUnit path for a stream test process."""

    if stream == "dynamic":
        rank = 0
    else:
        raw_rank = environ.get("RANK")
        if raw_rank is None or not raw_rank.isdigit():
            raise RuntimeError("static tests must run under torchrun with numeric RANK")
        rank = int(raw_rank)
    return junit_dir / f"multigpu-{stream}-rank-{rank}.xml"


def exec_tests(
    repo_root: Path,
    stream: str,
    test_list: Path,
    junit_dir: Path,
    coverage_file: Path | None,
    coverage_rc: Path | None,
) -> None:
    """Replace this process with pytest, optionally under coverage."""

    data = _load_test_list(test_list, stream)
    junit_dir.mkdir(parents=True, exist_ok=True)
    xml_path = junit_path(junit_dir, stream, os.environ)
    pytest_args = [
        PYTEST_FLAGS[stream],
        "-m",
        MARKERS[stream],
        "-p",
        "no:cacheprovider",
        f"--junitxml={xml_path}",
        *data["test_files"],
    ]
    if (coverage_file is None) != (coverage_rc is None):
        raise ValueError("coverage-file and coverage-rc must be provided together")
    if coverage_file is None:
        command = [sys.executable, "-m", "pytest", *pytest_args]
    else:
        rc_path = coverage_rc.resolve()
        if not rc_path.is_file():
            raise FileNotFoundError(rc_path)
        coverage_path = coverage_file.resolve()
        if stream == "static":
            rank = int(os.environ["RANK"])
            coverage_path = Path(f"{coverage_path}.rank-{rank}")
        coverage_path.parent.mkdir(parents=True, exist_ok=True)
        os.environ["COVERAGE_FILE"] = str(coverage_path)
        command = [
            sys.executable,
            "-m",
            "coverage",
            "run",
            f"--rcfile={rc_path}",
            "-m",
            "pytest",
            *pytest_args,
        ]
    os.chdir(repo_root)
    os.execv(sys.executable, command)  # noqa: S606 - fixed interpreter and argv


def _coverage_repo_path(repo_root: Path, filename: str) -> str:
    path = Path(filename)
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(repo_root.resolve())
        except ValueError as error:
            raise ValueError(f"coverage path escapes repository: {filename}") from error
    normalized = _normalize_repo_path(path.as_posix())
    if not normalized.startswith(("physicsnemo/", "test/")):
        raise ValueError(f"unexpected coverage path: {normalized}")
    resolved = (repo_root / normalized).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as error:
        raise ValueError(
            f"coverage symlink escapes repository: {normalized}"
        ) from error
    if not resolved.is_file():
        raise ValueError(f"measured file does not exist: {normalized}")
    return normalized


def _data_has_lines(data: Any) -> bool:
    return any(data.lines(filename) for filename in data.measured_files())


def _junit_summary(junit_dir: Path, stream: str, nproc: int) -> JUnitSummary:
    files = sorted(junit_dir.glob(f"multigpu-{stream}-rank-*.xml"))
    expected_ranks = set(range(nproc)) if stream == "static" else {0}
    actual_ranks: set[int] = set()
    totals = {"tests": 0, "skipped": 0, "failures": 0, "errors": 0}
    skip_reasons: list[str] = []
    for path in files:
        match = re.fullmatch(rf"multigpu-{stream}-rank-([0-9]+)\.xml", path.name)
        if match is None:
            raise RuntimeError(f"unexpected JUnit filename: {path.name}")
        rank = int(match.group(1))
        if rank in actual_ranks:
            raise RuntimeError(f"duplicate JUnit rank: {rank}")
        actual_ranks.add(rank)
        # JUnit is generated by pytest in this job, never supplied externally.
        root = ET.parse(path).getroot()  # noqa: S314
        if root.tag == "testsuite":
            suites = [root]
        else:
            suites = list(root.findall("testsuite"))
        if not suites:
            raise RuntimeError(f"JUnit file has no testsuite: {path}")
        rank_tests = 0
        for suite in suites:
            for field_name in totals:
                try:
                    value = int(suite.attrib.get(field_name, "0"))
                except ValueError as error:
                    raise RuntimeError(
                        f"invalid JUnit {field_name} total in {path}"
                    ) from error
                if value < 0:
                    raise RuntimeError(f"negative JUnit {field_name} total in {path}")
                totals[field_name] += value
                if field_name == "tests":
                    rank_tests += value
            for skipped in suite.findall(".//testcase/skipped"):
                reason = skipped.attrib.get("message") or (skipped.text or "")
                skip_reasons.append(reason.strip())
        if rank_tests <= 0:
            raise RuntimeError(f"JUnit rank {rank} contains no tests")
    if actual_ranks != expected_ranks:
        raise RuntimeError(
            f"incomplete JUnit ranks: actual={sorted(actual_ranks)}, "
            f"expected={sorted(expected_ranks)}"
        )
    if totals["failures"] or totals["errors"]:
        raise RuntimeError("cannot publish impact from failing JUnit data")
    if totals["tests"] <= totals["skipped"]:
        raise RuntimeError("cannot publish impact from an all-skipped stream")
    if len(skip_reasons) != totals["skipped"]:
        raise RuntimeError("JUnit skipped totals do not match testcase details")
    unexpected_skips = sorted(
        {reason for reason in skip_reasons if not _is_allowed_skip(stream, reason)}
    )
    if unexpected_skips:
        raise RuntimeError(
            "cannot publish impact with unexpected skips: "
            + "; ".join(unexpected_skips[:5])
        )
    return JUnitSummary(
        file_count=len(files),
        tests=totals["tests"],
        skipped=totals["skipped"],
        failures=totals["failures"],
        errors=totals["errors"],
        skip_reasons=tuple(sorted(set(skip_reasons))),
    )


def _ancestor_conftests(repo_root: Path, test_files: Iterable[str]) -> set[str]:
    conftests: set[str] = set()
    for test_file in test_files:
        parent = PurePosixPath(test_file).parent
        while parent.parts and parent.parts[0] == "test":
            candidate = parent / "conftest.py"
            if (repo_root / candidate).is_file():
                conftests.add(candidate.as_posix())
            if parent == PurePosixPath("test"):
                break
            parent = parent.parent
    return conftests


def build_manifest(
    repo_root: Path,
    stream: str,
    test_list: Path,
    coverage_file: Path,
    coverage_rc: Path,
    junit_dir: Path,
    output: Path,
    source_sha: str,
    nproc: int,
    runner_profile: str,
) -> None:
    """Combine rank/process coverage and publish a validated impact manifest."""

    import coverage

    if not SHA_PATTERN.fullmatch(source_sha):
        raise ValueError("source-sha must contain 40 to 64 hexadecimal characters")
    if nproc < 2:
        raise ValueError("nproc must be at least two")
    if not runner_profile.strip():
        raise ValueError("runner-profile must be non-empty")
    test_data = _load_test_list(test_list, stream)
    junit = _junit_summary(junit_dir, stream, nproc)

    base = coverage_file.resolve()
    valid_shards = 0
    if stream == "static":
        for rank in range(nproc):
            rank_shards = sorted(base.parent.glob(f"{base.name}.rank-{rank}.*"))
            rank_valid = 0
            for shard in rank_shards:
                shard_data = coverage.CoverageData(basename=str(shard))
                shard_data.read()
                if _data_has_lines(shard_data):
                    rank_valid += 1
            if rank_valid == 0:
                raise RuntimeError(f"static rank {rank} has no valid coverage shard")
            valid_shards += rank_valid
    else:
        shards = sorted(base.parent.glob(f"{base.name}.*"))
        for shard in shards:
            shard_data = coverage.CoverageData(basename=str(shard))
            shard_data.read()
            if _data_has_lines(shard_data):
                valid_shards += 1
    minimum_shards = nproc if stream == "static" else nproc + 1
    if valid_shards < minimum_shards:
        raise RuntimeError(
            f"incomplete coverage data: valid shards={valid_shards}, "
            f"required={minimum_shards}"
        )

    cov = coverage.Coverage(
        data_file=str(base),
        config_file=str(coverage_rc.resolve()),
    )
    cov.combine(data_paths=[str(base.parent)], strict=True, keep=True)
    cov.save()
    combined = coverage.CoverageData(basename=str(base))
    combined.read()
    executed_files = sorted(
        {
            _coverage_repo_path(repo_root, filename)
            for filename in combined.measured_files()
            if combined.lines(filename)
        }
    )
    if not any(path.startswith("physicsnemo/") for path in executed_files):
        raise RuntimeError("coverage contains no executed physicsnemo files")
    if not any(path.startswith("test/") for path in executed_files):
        raise RuntimeError("coverage contains no executed test files")

    test_files = list(test_data["test_files"])
    support_files = sorted(_ancestor_conftests(repo_root, test_files))
    files = sorted(set(executed_files).union(test_files, support_files))
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stream": stream,
        "complete": True,
        "source_sha": source_sha.lower(),
        "generated_at": generated_at,
        "coverage_version": coverage.__version__,
        "coverage_shards": valid_shards,
        "nproc": nproc,
        "runner_profile": runner_profile,
        "test_definition_count": test_data["test_definition_count"],
        "test_files": test_files,
        "executed_files": executed_files,
        "support_files": support_files,
        "junit_files": junit.file_count,
        "junit_tests": junit.tests,
        "junit_skipped": junit.skipped,
        "junit_skip_reasons": list(junit.skip_reasons),
        "files": files,
    }
    validate_manifest_data(
        manifest,
        stream,
        now=datetime.now(timezone.utc),
        max_age_hours=1,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def verify_gpus(expected: int) -> None:
    """Fail before launch unless runner visibility matches configured ranks."""

    import torch

    if expected < 2:
        raise ValueError("multi-GPU streams require at least two devices")
    visible = torch.cuda.device_count()
    print(f"torch.cuda.device_count() = {visible}")
    if visible != expected:
        raise RuntimeError(
            f"runner exposes {visible} CUDA device(s), but exactly {expected} are configured"
        )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(path: Path) -> ManifestInput:
    """Load a manifest while preserving fail-open error classification."""

    if not path.is_file():
        return ManifestInput(error_reason="manifest-missing", error_message=str(path))
    try:
        data = _load_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return ManifestInput(
            error_reason="manifest-invalid",
            error_message=f"cannot load {path}: {error}",
        )
    if not isinstance(data, Mapping):
        return ManifestInput(
            error_reason="manifest-invalid",
            error_message=f"manifest is not an object: {path}",
        )
    return ManifestInput(data=data)


def _flatten_file_payload(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping) and "error" in payload:
        raise ValueError(f"file API error: {payload['error']}")
    if not isinstance(payload, list):
        raise ValueError("pull-request files payload must be a list")
    if payload and all(isinstance(page, list) for page in payload):
        records: list[Any] = [record for page in payload for record in page]
    else:
        records = payload
    if not all(isinstance(record, Mapping) for record in records):
        raise ValueError("pull-request files payload contains a non-object")
    return records


def _normalize_changes(payload: Any, expected_count: int) -> list[FileChange]:
    records = _flatten_file_payload(payload)
    if expected_count <= 0 or expected_count >= 3000:
        raise ValueError("changed-file count is empty or reaches the API cap")
    if len(records) != expected_count:
        raise ValueError(
            f"changed-file count mismatch: expected={expected_count}, got={len(records)}"
        )
    changes: list[FileChange] = []
    seen: set[str] = set()
    for record in records:
        filename = _normalize_repo_path(record.get("filename"))
        if filename in seen:
            raise ValueError(f"duplicate changed-file record: {filename}")
        seen.add(filename)
        status = record.get("status")
        if status not in VALID_CHANGE_STATUSES:
            raise ValueError(f"unknown change status for {filename}: {status!r}")
        previous = record.get("previous_filename")
        if previous is not None:
            previous = _normalize_repo_path(previous)
        if status == "renamed" and previous is None:
            raise ValueError(f"renamed file has no previous_filename: {filename}")
        changes.append(FileChange(filename, status, previous))
    return changes


def _git_baseline_changes(
    repo_root: Path, source_sha: str, base_sha: str
) -> list[FileChange]:
    """Return a complete local source-to-base diff or raise to fail open."""

    if source_sha == base_sha:
        return []
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is unavailable for manifest-to-base comparison")
    ancestor = subprocess.run(  # noqa: S603 - fixed git argv; SHAs are validated
        [git, "merge-base", "--is-ancestor", source_sha, base_sha],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if ancestor.returncode != 0:
        message = ancestor.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"manifest source is unavailable or not a PR-base ancestor: {message}"
        )
    diff = subprocess.run(  # noqa: S603 - fixed git argv; SHAs are validated
        [
            git,
            "diff",
            "--name-status",
            "-z",
            "--find-renames",
            source_sha,
            base_sha,
            "--",
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if diff.returncode != 0:
        message = diff.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"cannot inspect manifest-to-base diff: {message}")
    tokens = diff.stdout.split(b"\0")
    if tokens and tokens[-1] == b"":
        tokens.pop()
    changes: list[FileChange] = []
    index = 0
    status_map = {
        "A": "added",
        "C": "copied",
        "D": "removed",
        "M": "modified",
        "R": "renamed",
        "T": "changed",
    }
    try:
        while index < len(tokens):
            raw_status = tokens[index].decode("ascii")
            index += 1
            code = raw_status[:1]
            if code not in status_map:
                raise ValueError(f"unsupported git change status: {raw_status!r}")
            if code in {"C", "R"}:
                previous = _normalize_repo_path(tokens[index].decode("utf-8"))
                filename = _normalize_repo_path(tokens[index + 1].decode("utf-8"))
                index += 2
            else:
                previous = None
                filename = _normalize_repo_path(tokens[index].decode("utf-8"))
                index += 1
            changes.append(FileChange(filename, status_map[code], previous))
    except (IndexError, UnicodeError, ValueError) as error:
        raise RuntimeError(f"cannot parse manifest-to-base diff: {error}") from error
    return changes


def _required_sha(container: Any, field_name: str) -> str:
    if not isinstance(container, Mapping):
        raise ValueError(f"{field_name} is not an object")
    value = container.get("sha")
    if not isinstance(value, str) or not SHA_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name}.sha is invalid")
    return value.lower()


def _pr_metadata(pr_data: Any, expected_number: int) -> PRSnapshot:
    if not isinstance(pr_data, Mapping):
        raise ValueError("pull-request metadata is not an object")
    number = pr_data.get("number")
    if type(number) is not int or number != expected_number:
        raise ValueError("pull-request number does not match mirrored branch")
    changed_files = pr_data.get("changed_files")
    if type(changed_files) is not int:
        raise ValueError("changed_files is not an integer")
    raw_labels = pr_data.get("labels")
    if not isinstance(raw_labels, list):
        raise ValueError("labels is not a list")
    labels: set[str] = set()
    for label in raw_labels:
        name = label.get("name") if isinstance(label, Mapping) else label
        if not isinstance(name, str):
            raise ValueError("label name is not a string")
        labels.add(name)
    return PRSnapshot(
        number=number,
        changed_files=changed_files,
        labels=frozenset(labels),
        head_sha=_required_sha(pr_data.get("head"), "head"),
        base_sha=_required_sha(pr_data.get("base"), "base"),
    )


def _uncertain_result(reason: str) -> dict[str, StreamDecision]:
    decisions = {stream: StreamDecision() for stream in STREAMS}
    for decision in decisions.values():
        decision.trigger(reason)
    return decisions


def _path_matches_prefix(path: str, prefix: str) -> bool:
    return path == prefix.rstrip("/") or path.startswith(prefix)


def _is_runtime_asset(path: str) -> bool:
    """Return whether coverage cannot trace a package/test resource read."""

    return path.startswith(("physicsnemo/", "test/")) and not path.endswith(".py")


def _marker_directories(test_files: Iterable[str]) -> set[str]:
    directories = {PurePosixPath(path).parent.as_posix() for path in test_files}
    return {
        directory
        for directory in directories
        if directory != "test" and len(PurePosixPath(directory).parts) > 1
    }


def _baseline_change_impacts_stream(
    stream: str, manifest: ValidatedManifest, change: FileChange
) -> str | None:
    """Classify a default-branch change since a stream's manifest."""

    for path in change.paths:
        if _is_runtime_asset(path):
            return path
        if path in GLOBAL_TRIGGER_PATHS or any(
            _path_matches_prefix(path, prefix) for prefix in GLOBAL_TRIGGER_PREFIXES
        ):
            return path
        if _path_matches_prefix(path, "physicsnemo/distributed/"):
            return path
        if stream == "static" and _path_matches_prefix(
            path, "physicsnemo/domain_parallel/"
        ):
            return path
        if path in manifest.files:
            return path
        if any(
            _path_matches_prefix(path, f"{directory}/")
            for directory in _marker_directories(manifest.test_files)
        ):
            return path
    if (
        change.status in {"added", "changed", "copied", "renamed"}
        and change.filename.startswith("physicsnemo/")
        and change.filename.endswith(".py")
    ):
        return change.filename
    # The PR checkout can predate its current base, so inspect test changes
    # conservatively: a modified existing file may have gained a marker that
    # is not visible in the checkout's AST inventory.
    if change.filename.startswith("test/") and change.filename.endswith(".py"):
        return change.filename
    return None


def evaluate_gate(
    *,
    repo_root: Path,
    event_name: str,
    ref_name: str,
    dispatch_stream: str,
    pr_data: Any,
    pr_after_data: Any,
    files_payload: Any,
    manifests: Mapping[str, ManifestInput],
    now: datetime,
    max_age_hours: float,
    expected_head_sha: str,
    expected_nprocs: Mapping[str, int],
    expected_runner_profiles: Mapping[str, str],
    test_inventories: Mapping[str, tuple[Sequence[str], int]] | None = None,
    baseline_change_sets: Mapping[str, Sequence[FileChange]] | None = None,
) -> dict[str, StreamDecision]:
    """Compute independent, conservative decisions for both streams."""

    decisions = {stream: StreamDecision() for stream in STREAMS}
    if event_name == "schedule":
        return _uncertain_result("scheduled-run")
    if event_name == "workflow_dispatch":
        if dispatch_stream not in {"both", *STREAMS}:
            return _uncertain_result("manual-dispatch-invalid")
        for stream, decision in decisions.items():
            if dispatch_stream in {"both", stream}:
                decision.trigger("manual-dispatch")
            else:
                decision.finish()
        return decisions
    match = PR_REF_PATTERN.fullmatch(ref_name)
    if event_name != "push" or match is None:
        return _uncertain_result("unsupported-event-or-ref")
    pr_number = int(match.group(1))
    try:
        before = _pr_metadata(pr_data, pr_number)
        after = _pr_metadata(pr_after_data, pr_number)
    except (TypeError, ValueError):
        return _uncertain_result("pr-metadata-unavailable")
    if not SHA_PATTERN.fullmatch(expected_head_sha):
        return _uncertain_result("mirror-head-unavailable")
    if before != after or before.head_sha != expected_head_sha.lower():
        return _uncertain_result("pr-snapshot-raced")
    if FORCE_LABEL in before.labels:
        return _uncertain_result("force-label")
    try:
        changes = _normalize_changes(files_payload, before.changed_files)
    except (TypeError, ValueError):
        return _uncertain_result("changed-files-incomplete")

    if test_inventories is None:
        try:
            inventories = {
                stream: discover_test_files(repo_root, stream) for stream in STREAMS
            }
        except RuntimeError:
            return _uncertain_result("test-inventory-unavailable")
    else:
        inventories = test_inventories

    validated: dict[str, ValidatedManifest] = {}
    for stream in STREAMS:
        manifest_input = manifests[stream]
        if manifest_input.error_reason is not None:
            decisions[stream].trigger(manifest_input.error_reason)
            continue
        try:
            validated[stream] = validate_manifest_data(
                manifest_input.data or {},
                stream,
                now=now,
                max_age_hours=max_age_hours,
                expected_test_files=inventories[stream][0],
                expected_test_definition_count=inventories[stream][1],
                expected_nproc=expected_nprocs[stream],
                expected_runner_profile=expected_runner_profiles[stream],
            )
        except (KeyError, ManifestError) as error:
            if isinstance(error, KeyError):
                decisions[stream].trigger("manifest-config-unavailable")
                continue
            decisions[stream].trigger(error.reason)

    for stream, manifest in validated.items():
        try:
            baseline_changes = (
                list(baseline_change_sets[stream])
                if baseline_change_sets is not None
                else _git_baseline_changes(
                    repo_root, manifest.source_sha, before.base_sha
                )
            )
        except (KeyError, OSError, RuntimeError, subprocess.TimeoutExpired):
            decisions[stream].trigger("manifest-base-unavailable")
            continue
        for change in baseline_changes:
            matched_path = _baseline_change_impacts_stream(stream, manifest, change)
            if matched_path is not None:
                decisions[stream].trigger("manifest-base-drift", matched_path)

    for stream, label in STREAM_FORCE_LABELS.items():
        if label in before.labels:
            decisions[stream].trigger("force-label")

    all_paths = {path for change in changes for path in change.paths}
    for path in sorted(all_paths):
        if _is_runtime_asset(path):
            for decision in decisions.values():
                decision.trigger("runtime-asset", path)
        if path in GLOBAL_TRIGGER_PATHS or any(
            _path_matches_prefix(path, prefix) for prefix in GLOBAL_TRIGGER_PREFIXES
        ):
            for decision in decisions.values():
                decision.trigger("global-path", path)
        if _path_matches_prefix(path, "physicsnemo/distributed/"):
            for decision in decisions.values():
                decision.trigger("stream-prefix", path)
        elif _path_matches_prefix(path, "physicsnemo/domain_parallel/"):
            decisions["static"].trigger("stream-prefix", path)
        for stream, manifest in validated.items():
            if path in manifest.files:
                decisions[stream].trigger("manifest-match", path)

    for change in changes:
        current = repo_root / change.filename
        if (
            change.filename.startswith("test/")
            and change.filename.endswith(".py")
            and change.status != "removed"
        ):
            if not current.is_file():
                for decision in decisions.values():
                    decision.trigger("marker-file-unreadable", change.filename)
            else:
                try:
                    tree = ast.parse(
                        current.read_text(encoding="utf-8"), filename=str(current)
                    )
                except (OSError, SyntaxError):
                    for decision in decisions.values():
                        decision.trigger("marker-file-unreadable", change.filename)
                else:
                    for stream, marker in MARKERS.items():
                        if _marker_present(tree, marker):
                            decisions[stream].trigger("marker-file", change.filename)

        for changed_path in change.paths:
            for stream, manifest in validated.items():
                if any(
                    _path_matches_prefix(changed_path, f"{directory}/")
                    for directory in _marker_directories(manifest.test_files)
                ):
                    decisions[stream].trigger("test-asset", changed_path)

        if (
            change.status in {"added", "changed", "copied", "renamed"}
            and change.filename.startswith("physicsnemo/")
            and change.filename.endswith(".py")
        ):
            for decision in decisions.values():
                decision.trigger("new-production-file", change.filename)

    for decision in decisions.values():
        decision.finish()
    return decisions


def _load_optional_json(path: Path | None) -> Any:
    if path is None or not path.is_file():
        return None
    try:
        return _load_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _decision_payload(decisions: Mapping[str, StreamDecision]) -> dict[str, Any]:
    return {stream: decisions[stream].as_dict() for stream in STREAMS}


def _write_gate_outputs(
    decisions: Mapping[str, StreamDecision],
    github_output: Path | None,
    step_summary: Path | None,
) -> None:
    payload = _decision_payload(decisions)
    compact = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if github_output is not None:
        with github_output.open("a", encoding="utf-8") as output:
            for stream in STREAMS:
                output.write(f"run_{stream}={str(decisions[stream].run).lower()}\n")
            output.write(f"decision_json={compact}\n")
    if step_summary is not None:
        with step_summary.open("a", encoding="utf-8") as summary:
            summary.write("## Multi-GPU PR selection\n\n")
            summary.write("| Stream | Run | Reasons |\n|---|---:|---|\n")
            for stream in STREAMS:
                decision = decisions[stream]
                summary.write(
                    f"| `{stream}` | `{str(decision.run).lower()}` | "
                    f"{', '.join(decision.reasons)} |\n"
                )
            summary.write("\n```json\n")
            summary.write(json.dumps(payload, indent=2, sort_keys=True))
            summary.write("\n```\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _command_list_tests(args: argparse.Namespace) -> int:
    write_test_list(args.repo_root.resolve(), args.stream, args.output)
    return 0


def _command_run_tests(args: argparse.Namespace) -> int:
    exec_tests(
        args.repo_root.resolve(),
        args.stream,
        args.test_list,
        args.junit_dir,
        args.coverage_file,
        args.coverage_rc,
    )
    return 0


def _command_build_manifest(args: argparse.Namespace) -> int:
    build_manifest(
        args.repo_root.resolve(),
        args.stream,
        args.test_list,
        args.coverage_file,
        args.coverage_rc,
        args.junit_dir,
        args.output,
        args.source_sha,
        args.nproc,
        args.runner_profile,
    )
    return 0


def _command_verify_gpus(args: argparse.Namespace) -> int:
    verify_gpus(args.expected)
    return 0


def _command_gate(args: argparse.Namespace) -> int:
    manifests = {
        "dynamic": load_manifest(args.dynamic_manifest),
        "static": load_manifest(args.static_manifest),
    }
    decisions = evaluate_gate(
        repo_root=args.repo_root.resolve(),
        event_name=args.event_name,
        ref_name=args.ref_name,
        dispatch_stream=args.dispatch_stream,
        pr_data=_load_optional_json(args.pr_json),
        pr_after_data=_load_optional_json(args.pr_after_json),
        files_payload=_load_optional_json(args.files_json),
        manifests=manifests,
        now=datetime.now(timezone.utc),
        max_age_hours=args.max_age_hours,
        expected_head_sha=args.expected_head_sha,
        expected_nprocs={
            "dynamic": args.dynamic_nproc,
            "static": args.static_nproc,
        },
        expected_runner_profiles={
            "dynamic": args.dynamic_runner_profile,
            "static": args.static_runner_profile,
        },
    )
    _write_gate_outputs(decisions, args.github_output, args.step_summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_tests = subparsers.add_parser("list-tests")
    list_tests.add_argument("--stream", type=_stream, required=True)
    list_tests.add_argument("--output", type=Path, required=True)
    list_tests.set_defaults(func=_command_list_tests)

    run_tests = subparsers.add_parser("run-tests")
    run_tests.add_argument("--stream", type=_stream, required=True)
    run_tests.add_argument("--test-list", type=Path, required=True)
    run_tests.add_argument("--junit-dir", type=Path, required=True)
    run_tests.add_argument("--coverage-file", type=Path)
    run_tests.add_argument("--coverage-rc", type=Path)
    run_tests.set_defaults(func=_command_run_tests)

    manifest = subparsers.add_parser("build-manifest")
    manifest.add_argument("--stream", type=_stream, required=True)
    manifest.add_argument("--test-list", type=Path, required=True)
    manifest.add_argument("--coverage-file", type=Path, required=True)
    manifest.add_argument("--coverage-rc", type=Path, required=True)
    manifest.add_argument("--junit-dir", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--source-sha", required=True)
    manifest.add_argument("--nproc", type=int, required=True)
    manifest.add_argument("--runner-profile", required=True)
    manifest.set_defaults(func=_command_build_manifest)

    verify = subparsers.add_parser("verify-gpus")
    verify.add_argument("--expected", type=int, required=True)
    verify.set_defaults(func=_command_verify_gpus)

    gate = subparsers.add_parser("gate")
    gate.add_argument("--event-name", required=True)
    gate.add_argument("--ref-name", default="")
    gate.add_argument("--dispatch-stream", default="both")
    gate.add_argument("--pr-json", type=Path)
    gate.add_argument("--pr-after-json", type=Path)
    gate.add_argument("--files-json", type=Path)
    gate.add_argument("--expected-head-sha", default="")
    gate.add_argument("--dynamic-nproc", type=int, default=2)
    gate.add_argument("--static-nproc", type=int, default=2)
    gate.add_argument(
        "--dynamic-runner-profile",
        default="linux-amd64-gpu-h100-latest-2",
    )
    gate.add_argument(
        "--static-runner-profile",
        default="linux-amd64-gpu-h100-latest-2",
    )
    gate.add_argument("--dynamic-manifest", type=Path, required=True)
    gate.add_argument("--static-manifest", type=Path, required=True)
    gate.add_argument("--max-age-hours", type=float, default=72)
    gate.add_argument("--github-output", type=Path)
    gate.add_argument("--step-summary", type=Path)
    gate.set_defaults(func=_command_gate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected multi-GPU CI utility command."""

    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
