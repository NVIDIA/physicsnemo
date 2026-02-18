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


def get_top_comments(_data):
    """
    Get all lines where comments should exist.
    """
    lines_to_extract = []
    for i, line in enumerate(_data):
        # If empty line, skip
        if line in ["", "\n", "", "\r", "\r\n"]:
            continue
        # If it is a comment line, we should get it
        if line.startswith("#"):
            lines_to_extract.append(i)
        # Assume all copyright headers occur before any import or from statements
        # and not enclosed in a comment block
        elif "import" in line:
            break
        elif "from" in line:
            break

    comments = [_data[line] for line in lines_to_extract]

    return comments


def get_all_files(working_path, exts, exclude_dirs):
    """
    Get a list of all files in the directory with specified extensions,
    excluding files in excluded directories.
    """
    all_files = []
    exclude_paths = [Path(p).resolve() for p in exclude_dirs]

    for ext in exts:
        # Handle extensions that start with "." vs filenames like "Dockerfile"
        if ext.startswith("."):
            pattern = f"*{ext}"
        else:
            pattern = ext

        for filepath in working_path.rglob(pattern):
            # Check if file is under any excluded directory
            is_excluded = any(
                exclude_path in filepath.parents or filepath == exclude_path
                for exclude_path in exclude_paths
            )
            if not is_excluded:
                all_files.append(filepath)

    return all_files


def check_file_header(filename, pyheader, pyheader_lines, starting_year, current_year):
    """
    Check a single file for proper copyright header.

    Returns:
        tuple: (is_problematic, has_gpl, error_message)
    """
    try:
        with open(str(filename), "r", encoding="utf-8") as original:
            data = original.readlines()
    except (OSError, UnicodeDecodeError) as e:
        return True, False, f"Could not read file: {e}"

    data = get_top_comments(data)

    # Check for ignore marker
    if data and "# ignore_header_test" in data[0]:
        return False, False, None

    # Check if enough header lines exist
    if len(data) < pyheader_lines - 1:
        return True, False, "has less header lines than the copyright template"

    # Look for NVIDIA copyright line
    found = False
    is_problematic = False
    error_msg = None

    for i, line in enumerate(data):
        if re.search(re.compile("Copyright.*NVIDIA.*", re.IGNORECASE), line):
            found = True
            # Check year
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

    # Check for GPL license
    has_gpl = any("gpl" in line.lower() for line in data)

    return is_problematic, has_gpl, error_msg


def main():
    """
    Main function to check the copyright headers.
    """
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
        help="Check all files in the directory instead of files passed as arguments.",
    )
    args = parser.parse_args()

    with open(Path(__file__).parent.resolve() / Path("config.json")) as f:
        config = json.loads(f.read())

    current_year = int(datetime.today().year)
    starting_year = 2024
    python_header_path = Path(__file__).parent.resolve() / Path(
        config["copyright_file"]
    )

    with open(python_header_path, "r", encoding="utf-8") as original:
        pyheader = original.read().split("\n")
        pyheader_lines = len(pyheader)

    # Determine which files to check
    if args.all_files:
        working_path = Path(__file__).parent.resolve() / Path(config["dir"])
        exts = config["include-ext"]
        exclude_dirs = [
            (Path(__file__).parent / Path(path)).resolve()
            for path in config.get("exclude-dir", [])
        ]
        filenames = get_all_files(working_path.resolve(), exts, exclude_dirs)
        print("License check config:")
        print(json.dumps(config, sort_keys=True, indent=4))
    elif args.filenames:
        # Files passed from pre-commit (already filtered by pre-commit config)
        filenames = [Path(f) for f in args.filenames]
    else:
        # No files to check
        print("No files to check.")
        return 0

    problematic_files = []
    gpl_files = []

    for filename in filenames:
        is_problematic, has_gpl, error_msg = check_file_header(
            filename, pyheader, pyheader_lines, starting_year, current_year
        )

        if is_problematic:
            print(f"{filename} {error_msg}")
            problematic_files.append(filename)

        if has_gpl:
            gpl_files.append(filename)

    if len(problematic_files) > 0:
        print(
            "header_check.py found the following files that might not have a "
            "copyright header:"
        )
        for _file in problematic_files:
            print(f"  {_file}")

    if len(gpl_files) > 0:
        print(
            "header_check.py found the following files that might have GPL copyright:"
        )
        for _file in gpl_files:
            print(f"  {_file}")

    if len(problematic_files) > 0 or len(gpl_files) > 0:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
