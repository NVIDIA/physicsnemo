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

"""Small, dependency-free helpers for the PhysicsNeMo release workflows."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

VERSION_PATH = Path("physicsnemo/__init__.py")
CHANGELOG_PATH = Path("CHANGELOG.md")

_VERSION_ASSIGNMENT_RE = re.compile(
    r'^(?P<prefix>__version__\s*=\s*")(?P<version>[^"]+)(?P<suffix>")$',
    re.MULTILINE,
)
_VERSION_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)\."
    r"(?P<minor>0|[1-9]\d*)\."
    r"(?P<patch>0|[1-9]\d*)"
    r"(?P<prerelease>a0)?$"
)
_CHANGELOG_HEADING_RE = re.compile(
    r"^## \[(?P<version>[^\]]+)\] - (?P<release_date>[^\n]+)$",
    re.MULTILINE,
)
_NEXT_CHANGELOG_TEMPLATE = """## [{version}] - 2026-XX-YY

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security

### Dependencies

"""


class ReleaseError(RuntimeError):
    """Raised when a release operation would violate an expected invariant."""


@dataclass(frozen=True)
class ProjectVersion:
    """A PhysicsNeMo final or alpha-zero development version."""

    raw: str
    major: int
    minor: int
    patch: int
    prerelease: str | None

    @classmethod
    def parse(cls, value: str) -> ProjectVersion:
        """Parse the version formats used by the release process."""
        match = _VERSION_RE.fullmatch(value)
        if match is None:
            raise ReleaseError(
                f"Unsupported version {value!r}; expected X.Y.Z or X.Y.Za0"
            )
        return cls(
            raw=value,
            major=int(match.group("major")),
            minor=int(match.group("minor")),
            patch=int(match.group("patch")),
            prerelease=match.group("prerelease"),
        )

    @property
    def base(self) -> str:
        """Return the final-version portion."""
        return f"{self.major}.{self.minor}.{self.patch}"

    @property
    def release_tuple(self) -> tuple[int, int, int]:
        """Return the numeric release components for ordering."""
        return (self.major, self.minor, self.patch)

    @property
    def is_final(self) -> bool:
        """Return whether the version has no prerelease suffix."""
        return self.prerelease is None

    @property
    def is_alpha_zero(self) -> bool:
        """Return whether the version is the development alpha-zero form."""
        return self.prerelease == "a0"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ReleaseError(f"Required file does not exist: {path}") from exc


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def read_project_version(repo_root: Path) -> ProjectVersion:
    """Read the single package version assignment."""
    version_path = repo_root / VERSION_PATH
    content = _read(version_path)
    matches = list(_VERSION_ASSIGNMENT_RE.finditer(content))
    if len(matches) != 1:
        raise ReleaseError(
            f"Expected exactly one __version__ assignment in {version_path}; "
            f"found {len(matches)}"
        )
    return ProjectVersion.parse(matches[0].group("version"))


def _replace_project_version(repo_root: Path, new_version: ProjectVersion) -> None:
    version_path = repo_root / VERSION_PATH
    content = _read(version_path)
    matches = list(_VERSION_ASSIGNMENT_RE.finditer(content))
    if len(matches) != 1:
        raise ReleaseError(
            f"Expected exactly one __version__ assignment in {version_path}; "
            f"found {len(matches)}"
        )
    match = matches[0]
    replacement = f"{match.group('prefix')}{new_version.raw}{match.group('suffix')}"
    updated = content[: match.start()] + replacement + content[match.end() :]
    _write(version_path, updated)


def _first_changelog_heading(content: str) -> re.Match[str]:
    match = _CHANGELOG_HEADING_RE.search(content)
    if match is None:
        raise ReleaseError("CHANGELOG.md has no release heading")
    return match


def _validate_release_date(value: str) -> None:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ReleaseError(
            f"Invalid release date {value!r}; expected YYYY-MM-DD"
        ) from exc
    if parsed.isoformat() != value:
        raise ReleaseError(f"Invalid release date {value!r}; expected YYYY-MM-DD")


def start_rc(repo_root: Path, release_version: str, release_date: str) -> None:
    """Change a development checkout to the corresponding final RC version."""
    release = ProjectVersion.parse(release_version)
    if not release.is_final:
        raise ReleaseError("The RC release version must be a final X.Y.Z version")
    _validate_release_date(release_date)

    current = read_project_version(repo_root)
    expected_current = f"{release.raw}a0"
    if current.raw != expected_current:
        raise ReleaseError(
            f"Current package version is {current.raw}, but starting "
            f"{release.raw} expects {release.raw}a0"
        )

    changelog_path = repo_root / CHANGELOG_PATH
    changelog = _read(changelog_path)
    heading = _first_changelog_heading(changelog)
    allowed_headings = {release.raw, f"{release.raw}a0"}
    if heading.group("version") not in allowed_headings:
        raise ReleaseError(
            "The first changelog section is "
            f"{heading.group('version')}, but expected {release.raw} "
            f"or {release.raw}a0"
        )

    _replace_project_version(repo_root, release)
    replacement = f"## [{release.raw}] - {release_date}"
    updated_changelog = (
        changelog[: heading.start()] + replacement + changelog[heading.end() :]
    )
    _write(changelog_path, updated_changelog)


def prepare_mergeback(repo_root: Path, next_dev_version: str) -> str:
    """Start the next development cycle on a checkout of the final RC."""
    current = read_project_version(repo_root)
    if not current.is_final:
        raise ReleaseError(
            f"The RC package version must be final before merge-back; "
            f"found {current.raw}"
        )

    next_version = ProjectVersion.parse(next_dev_version)
    if not next_version.is_alpha_zero:
        raise ReleaseError("The next development version must use X.Y.Za0")
    if next_version.release_tuple <= current.release_tuple:
        raise ReleaseError(
            f"Next development version {next_version.raw} must be newer than "
            f"release {current.raw}"
        )

    changelog_path = repo_root / CHANGELOG_PATH
    changelog = _read(changelog_path)
    heading = _first_changelog_heading(changelog)
    if heading.group("version") != current.raw:
        raise ReleaseError(
            "The first changelog section is "
            f"{heading.group('version')}, but the RC package version is "
            f"{current.raw}"
        )
    existing_versions = {
        match.group("version") for match in _CHANGELOG_HEADING_RE.finditer(changelog)
    }
    if next_version.base in existing_versions:
        raise ReleaseError(
            f"CHANGELOG.md already contains a {next_version.base} section"
        )

    _replace_project_version(repo_root, next_version)
    new_section = _NEXT_CHANGELOG_TEMPLATE.format(version=next_version.base)
    updated_changelog = (
        changelog[: heading.start()] + new_section + changelog[heading.start() :]
    )
    _write(changelog_path, updated_changelog)
    return current.raw


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (defaults to the current directory)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("show-version", help="Print the current package version")

    start_parser = subparsers.add_parser(
        "start-rc", help="Update the checkout for an RC branch"
    )
    start_parser.add_argument("--release-version", required=True)
    start_parser.add_argument("--release-date", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare-mergeback",
        help="Update an RC checkout for its disposable merge-back branch",
    )
    prepare_parser.add_argument("--next-dev-version", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the release helper command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        if args.command == "show-version":
            print(read_project_version(repo_root).raw)
        elif args.command == "start-rc":
            start_rc(repo_root, args.release_version, args.release_date)
            print(f"Prepared release version {args.release_version}")
        elif args.command == "prepare-mergeback":
            release_version = prepare_mergeback(repo_root, args.next_dev_version)
            print(f"Prepared {args.next_dev_version} after release {release_version}")
        else:  # pragma: no cover - argparse prevents this branch
            parser.error(f"Unknown command {args.command}")
    except ReleaseError as exc:
        prefix = "::error::" if os.getenv("GITHUB_ACTIONS") == "true" else "error: "
        print(f"{prefix}{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
