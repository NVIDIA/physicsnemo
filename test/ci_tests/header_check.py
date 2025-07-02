# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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


"""A script to check that copyright headers exists"""

import argparse
import fnmatch
import itertools
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


def read_gitignore(gitignore_path):
    """
    Read the .gitignore file and collect patterns to skip
    """
    ignore_patterns = []
    if gitignore_path.exists():
        with gitignore_path.open("r") as file:
            for line in file:
                stripped_line = line.strip()
                if stripped_line and not stripped_line.startswith("#"):
                    ignore_patterns.append(stripped_line)
    return ignore_patterns


def is_ignored(path, working_path, ignore_patterns):
    """
    Check if the path needs to be ignored
    """
    # Get the git root path to stop the search
    git_root_path = Path(__file__) / Path(working_path)
    git_root_path = git_root_path.resolve()

    for pattern in ignore_patterns:
        normalized_pattern = pattern.rstrip("/")

        # Filter paths that are outside git root
        relevant_children = [
            part
            for part in [path] + list(path.parents)
            if git_root_path in part.parents or part == git_root_path
        ]

        # Check the directory itself and each parent directory
        for part in relevant_children:
            # Match directories (patterns ending with '/')
            if pattern.endswith("/") and part.is_dir():
                if fnmatch.fnmatch(part.name, normalized_pattern):
                    return True

            # Match files or directories without a trailing '/'
            if not pattern.endswith("/") and (
                fnmatch.fnmatch(str(part), pattern)
                or fnmatch.fnmatch(part.name, normalized_pattern)
            ):
                return True

    return False


def get_top_comments(_data):
    """
    Get all lines where comments should exist
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


def get_committed_files():
    """
    Get a list of files that are part of the current commit
    """
    git_executable = shutil.which("git")
    if not git_executable:
        raise RuntimeError("git executable not found in PATH")
    result = subprocess.run(
        [git_executable, "diff", "--name-only", "--cached"],  # noqa: S603
        capture_output=True,
        text=True,
    )
    files = result.stdout.splitlines()
    return files


def get_all_files(working_path, exts):
    """
    Get a list of all files in the directory with specified extensions
    """
    all_files = []
    for ext in exts:
        all_files.extend(working_path.rglob(f"*{ext}"))
    return all_files


def main():
    """
    Main function to check the copyright headers
    """
    parser = argparse.ArgumentParser(description="Check copyright headers in files.")
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Check all files in the directory instead of just committed files.",
    )
    args = parser.parse_args()

    with open(Path(__file__).parent.resolve() / Path("config.json")) as f:
        config = json.loads(f.read())
    print("License check config:")
    print(json.dumps(config, sort_keys=True, indent=4))

    current_year = int(datetime.today().year)
    starting_year = 2024
    python_header_path = Path(__file__).parent.resolve() / Path(
        config["copyright_file"]
    )
    working_path = Path(__file__).parent.resolve() / Path(config["dir"])
    exts = config["include-ext"]

    with open(python_header_path, "r", encoding="utf-8") as original:
        pyheader = original.read().split("\n")
        pyheader_lines = len(pyheader)

    # Determine which files to check
    if args.all_files:
        # Build list of files to check, excluding those in exclude-dir
        exclude_paths = [
            (Path(__file__).parent / Path(path)).resolve().rglob("*")
            for path in config.get("exclude-dir", [])
        ]
        all_exclude_paths = itertools.chain.from_iterable(exclude_paths)
        exclude_filenames = [p for p in all_exclude_paths if p.suffix in exts]
        filenames = [p for p in working_path.resolve().rglob("*") if p.suffix in exts]
        filenames = [
            filename for filename in filenames if filename not in exclude_filenames
        ]
    else:
        committed_files = get_committed_files()
        filenames = [Path(f) for f in committed_files if Path(f).suffix in exts]

    problematic_files = []
    gpl_files = []

    ignore_patterns = read_gitignore(working_path / Path(".gitignore"))

    for filename in filenames:
        # Check if the file is ignored in gitignore. # NOTE this can be removed if
        # the files don't need to be tested against gitignore patters.
        if not is_ignored(filename, working_path, ignore_patterns):
            with open(str(filename), "r", encoding="utf-8") as original:
                data = original.readlines()

            data = get_top_comments(data)
            if data and "# ignore_header_test" in data[0]:
                continue
            if len(data) < pyheader_lines - 1:
                print(f"{filename} has less header lines than the copyright template")
                problematic_files.append(filename)
                continue

            found = False
            for i, line in enumerate(data):
                if re.search(re.compile("Copyright.*NVIDIA.*", re.IGNORECASE), line):
                    found = True
                    # Check 1st line manually
                    year_good = False
                    for year in range(starting_year, current_year + 1):
                        year_line = pyheader[0].format(CURRENT_YEAR=year)
                        if year_line in data[i]:
                            year_good = True
                            break
                        year_line_aff = year_line.split(".")
                        year_line_aff = (
                            year_line_aff[0] + " & AFFILIATES." + year_line_aff[1]
                        )
                        if year_line_aff in data[i]:
                            year_good = True
                            break
                    if not year_good:
                        problematic_files.append(filename)
                        print(f"{filename} had an error with the year")
                        break
                    # while "opyright" in data[i]:
                    #    i += 1
                    # for j in range(1, pyheader_lines):
                    #    if pyheader[j] not in data[i + j - 1]:
                    #        problematic_files.append(filename)
                    #        print(f"{filename} missed the line: {pyheader[j]}")
                    #        break
                if found:
                    break
            if not found:
                print(f"{filename} did not match the regex: `Copyright.*NVIDIA.*`")
                problematic_files.append(filename)

            # test if GPL license exists
            for lines in data:
                if "gpl" in lines.lower():
                    gpl_files.append(filename)
                    break

    if len(problematic_files) > 0:
        print(
            "test_header.py found the following files that might not have a copyright header:"
        )
        for _file in problematic_files:
            print(_file)
    if len(gpl_files) > 0:
        print("test_header.py found the following files that might have GPL copyright:")
        for _file in gpl_files:
            print(_file)
    assert len(problematic_files) == 0, "header test failed!"
    assert len(gpl_files) == 0, "found gpl license, header test failed!"

    print("Success: File headers look good!")


if __name__ == "__main__":
    main()
