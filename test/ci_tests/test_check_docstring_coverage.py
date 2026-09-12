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

from pathlib import Path

from check_docstring_coverage import _parse_interrogate_output


def test_parse_interrogate_output_single_equals_header():
    """Interrogate 1.7 uses single-equals section headers."""
    repo_root = Path("/repo")
    output = """
= Coverage for /repo/examples/structural_mechanics/crash/ =
| datapipe.py                                            |                     |
| SimSample.to (L70)                                     |              MISSED |
"""
    results = _parse_interrogate_output(output, repo_root)
    assert results == ["examples/structural_mechanics/crash/datapipe.py:SimSample.to"]


def test_parse_interrogate_output_multi_equals_header():
    """Older interrogate versions may use longer equals runs."""
    repo_root = Path("/repo")
    output = """
===== Coverage for /repo/physicsnemo/utils/ =====
| logging.py                                             |                     |
| PythonLogger.info (L10)                                |              MISSED |
"""
    results = _parse_interrogate_output(output, repo_root)
    assert results == ["physicsnemo/utils/logging.py:PythonLogger.info"]
