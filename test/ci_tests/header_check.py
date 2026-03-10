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


"""A script to check that copyright headers exist.

This script can be run in two modes:
1. With filenames passed as arguments (used by pre-commit)
2. With --all-files to check all files in the repository
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import git

_COPYRIGHT_RE = re.compile(r"Copyright.*NVIDIA.*", re.IGNORECASE)


class HeaderCheckResult(NamedTuple):
    """Result of checking a single file's copyright header."""

    is_problematic: bool
    has_gpl: bool
    error_msg: str | None


def read_header_comments(filepath: Path, *, max_lines: int) -> list[str]:
    """Read comment lines from the top of a file.

    Iterates lazily, stopping at the first non-blank, non-comment line
    or after *max_lines* total lines have been seen, whichever comes first.
    Works correctly for all file types (Python, YAML, Dockerfile, etc.).
    """
    comments: list[str] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            if line.strip() == "":
                continue
            if line.startswith("#"):
                comments.append(line)
            else:
                break
    return comments


def get_git_tracked_files(exts: list[str], exclude_prefixes: list[str]) -> list[Path]:
    """Return git-tracked files matching *exts*, excluding paths that start
    with any of *exclude_prefixes*.

    Uses ``git ls-files`` so that only committed/staged files are returned and
    everything in ``.gitignore`` is automatically skipped.
    """
    repo = git.Repo(search_parent_directories=True)
    root = Path(repo.working_dir)
    pathspecs = [f"*.{ext.lstrip('.')}" if ext.startswith(".") else ext for ext in exts]
    output = repo.git.ls_files("--cached", "--", *pathspecs)
    exclude_tuple = tuple(exclude_prefixes)
    return [
        root / line
        for line in output.splitlines()
        if not line.startswith(exclude_tuple)
    ]


def check_file_header(
    filename: Path,
    pyheader: list[str],
    starting_year: int,
    current_year: int,
) -> HeaderCheckResult:
    """Check a single file for proper copyright header."""
    max_lines = 3 * len(pyheader)
    try:
        data = read_header_comments(filename, max_lines=max_lines)
    except (OSError, UnicodeDecodeError) as e:
        return HeaderCheckResult(True, False, f"Could not read file: {e}")

    if data and "# ignore_header_test" in data[0]:
        return HeaderCheckResult(False, False, None)

    if len(data) < len(pyheader) - 1:
        return HeaderCheckResult(
            True, False, "has less header lines than the copyright template"
        )

    ### Look for NVIDIA copyright line
    found = False
    is_problematic = False
    error_msg: str | None = None

    for i, line in enumerate(data):
        if _COPYRIGHT_RE.search(line):
            found = True
            year_good = False
            for year in range(starting_year, current_year + 1):
                year_line = pyheader[0].format(CURRENT_YEAR=year)
                if year_line in data[i]:
                    year_good = True
                    break
                year_line_aff = year_line.split(".")
                year_line_aff = year_line_aff[0] + " & AFFILIATES." + year_line_aff[1]
                if year_line_aff in data[i]:
                    year_good = True
                    break
            if not year_good:
                is_problematic = True
                error_msg = "had an error with the year"
            break

    if not found:
        is_problematic = True
        error_msg = "did not match the regex: `Copyright.*NVIDIA.*`"

    has_gpl = any("gpl" in line.lower() for line in data)

    return HeaderCheckResult(is_problematic, has_gpl, error_msg)


def main() -> int:
    """Main function to check the copyright headers."""
    parser = argparse.ArgumentParser(description="Check copyright headers in files.")
    parser.add_argument(
        "filenames",
        nargs="*",
        help="Filenames to check (passed by pre-commit).",
    )
    parser.add_argument(
        "-a",
        "--all-files",
        action="store_true",
        help="Check all git-tracked files matching the configured extensions.",
    )
    args = parser.parse_args()

    config_dir = Path(__file__).parent.resolve()
    with open(config_dir / "config.json") as f:
        config = json.loads(f.read())

    current_year = int(datetime.today().year)
    starting_year = 2024

    with open(config_dir / config["copyright_file"], "r", encoding="utf-8") as f:
        pyheader: list[str] = f.read().split("\n")

    ### Determine which files to check
    if args.all_files:
        exts: list[str] = config["include-ext"]
        exclude_prefixes: list[str] = config.get("exclude-dir", [])
        filenames = get_git_tracked_files(exts, exclude_prefixes)
        print("License check config:")
        print(json.dumps(config, sort_keys=True, indent=4))
        print(f"Checking {len(filenames)} git-tracked files")
    elif args.filenames:
        filenames = [Path(f) for f in args.filenames]
    else:
        print("No files to check.")
        return 0

    problematic_files: list[Path] = []
    gpl_files: list[Path] = []

    for filename in filenames:
        result = check_file_header(filename, pyheader, starting_year, current_year)

        if result.is_problematic:
            print(f"{filename} {result.error_msg}")
            problematic_files.append(filename)

        if result.has_gpl:
            gpl_files.append(filename)

    if problematic_files:
        print(
            "header_check.py found the following files that might not have a "
            "copyright header:"
        )
        for _file in problematic_files:
            print(f"  {_file}")

    if gpl_files:
        print(
            "header_check.py found the following files that might have GPL copyright:"
        )
        for _file in gpl_files:
            print(f"  {_file}")

    if problematic_files or gpl_files:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
