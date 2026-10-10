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


"""Tests for the interrogate output parser in check_docstring_coverage."""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).with_name("check_docstring_coverage.py")
_spec = importlib.util.spec_from_file_location("check_docstring_coverage", _SCRIPT)
check_docstring_coverage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_docstring_coverage)

REPO_ROOT = Path("/home/runner/work/physicsnemo/physicsnemo")
COVERED_DIR = f"{REPO_ROOT}/physicsnemo/utils/"

DETAIL_TABLE = """\
------------------------------ Detailed Coverage -------------------------------
| Name                                      |                           Status |
|-------------------------------------------|----------------------------------|
| memory.py                                 |                                  |
|   srt2bool (L41)                          |                           MISSED |
|-------------------------------------------|----------------------------------|
"""


def test_parser_reads_a_banner_of_any_width() -> None:
    """Interrogate pads its section banner with "=" out to the terminal width.

    A long checkout path leaves as little as one "=" a side, which is what a
    GitHub Actions checkout does for every directory below the top level, so
    the parser cannot require a minimum run of them.
    """
    for padding in ("=====", "===", "==", "="):
        banner = f"{padding} Coverage for {COVERED_DIR} {padding}"
        parsed = check_docstring_coverage._parse_interrogate_output(
            f"{banner}\n{DETAIL_TABLE}", REPO_ROOT
        )
        assert parsed == ["physicsnemo/utils/memory.py:srt2bool"], padding
