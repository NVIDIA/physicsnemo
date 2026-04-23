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


"""Check docstring coverage against a baseline of known undocumented items.

Runs ``interrogate`` on specified files and fails if any undocumented items
are found that aren't accounted for in the baseline file. This ensures that
new code has docstrings while grandfathering existing gaps.

Accepts filenames as positional arguments (the interface pre-commit uses).
File discovery, extension filtering, and exclusions are all handled by
pre-commit via the ``types:`` and ``exclude:`` keys in
``.pre-commit-config.yaml``.

Usage::

    # Pre-commit hook (files passed as positional args):
    python test/ci_tests/check_docstring_coverage.py file1.py file2.py

    # Generate/update the baseline from the full repo:
    python test/ci_tests/check_docstring_coverage.py --generate-baseline
"""

import argparse
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASELINE_PATH = Path(__file__).resolve().parent / "interrogate_baseline.txt"

### Interrogate configuration
# These flags correct the broken syntax from the original .pre-commit-config.yaml
# where --ignore-regex received a Python list literal that was compiled as a
# regex character class, silently skipping ~83% of all items.
_IGNORE_REGEX_PATTERNS = [
    "forward",
    "backward",
    "reset_parameters",
    "extra_repr",
    "MetaData",
    "apply_activation",
    "exec_activation",
]
INTERROGATE_ARGS = [
    "--ignore-init-method",
    "--ignore-init-module",
    "--ignore-module",
    "--ignore-private",
    "--ignore-semiprivate",
    "--ignore-magic",
    "--ignore-nested-functions",  # not in original config; added per #1485
    *[arg for p in _IGNORE_REGEX_PATTERNS for arg in ("--ignore-regex", p)],
]

# Must match the ``exclude`` pattern in .pre-commit-config.yaml.
_EXCLUDE_RE = re.compile(r"^(docs/|physicsnemo/experimental/|test/)")


def _parse_interrogate_output(output: str, repo_root: Path) -> list[str]:
    """Parse ``interrogate -vv`` output into ``filepath:item_name`` strings.

    Interrogate groups results by directory. Each group starts with a section
    header (``===== Coverage for /abs/path/ =====``), followed by per-file
    detail blocks containing item-level COVERED/MISSED verdicts.
    """
    results: list[str] = []
    current_dir = ""
    current_file = ""

    for line in output.splitlines():
        ### Section header
        # Interrogate section headers vary by version ("==== ... ====" vs
        # "===== ... ====="), so accept both 4+ and 5+ equals styles.
        m = re.match(r"={4,}\s+Coverage for (.+?)\s*={4,}", line)
        if m:
            current_dir = m.group(1).rstrip("/")
            current_file = ""
            continue

        if not line.startswith("|"):
            continue

        ### MISSED item: ``| ItemName (L123) | MISSED |``
        m = re.match(r"\|\s+(.+?)\s+\(L\d+\)\s+\|\s+MISSED\s+\|", line)
        if m and current_file:
            results.append(f"{current_file}:{m.group(1).strip()}")
            continue

        ### Skip non-file-header lines
        if "COVERED" in line or "MISSED" in line:
            continue
        if re.search(r"\|\s+\d+\s+\|", line):
            continue
        if "---" in line:
            continue

        ### File header: ``| relative/path/file.py |  |``
        m = re.match(r"\|\s+(\S+\.pyi?)\s+\|", line)
        if m:
            full_path = f"{current_dir}/{m.group(1)}"
            current_file = os.path.relpath(full_path, repo_root)
            continue

    ### Sanity check: if interrogate reported MISSED items but we parsed
    ### nothing, the output format may have changed. Fail hard rather than
    ### silently passing with an empty result (which is the exact class of
    ### bug this script exists to prevent).
    if not results and re.search(r"MISSED", output):
        print(
            "FAILED: interrogate output contains MISSED items but the parser "
            "found none. The -vv output format may have changed.",
            file=sys.stderr,
        )
        sys.exit(1)

    return results


def _read_baseline(baseline_path: Path) -> list[str]:
    """Read the baseline file, returning ``filepath:item_name`` entries."""
    if not baseline_path.exists():
        return []
    return [
        line.strip()
        for line in baseline_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _run_interrogate(files: list[str], repo_root: Path) -> list[str]:
    """Run interrogate on *files* and return the MISSED items."""
    result = subprocess.run(  # noqa: S603 — command is hardcoded, not user input
        [sys.executable, "-m", "interrogate", "-vv", "--fail-under=0", "--no-color"]
        + INTERROGATE_ARGS
        + ["--"]
        + files,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    if result.returncode != 0:
        print(f"interrogate exited with code {result.returncode}:", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        sys.exit(result.returncode)

    return _parse_interrogate_output(result.stdout, repo_root)


def _generate_baseline(repo_root: Path, baseline_path: Path) -> int:
    """Scan all non-excluded Python files and write the baseline."""
    files = sorted(
        str(rel)
        for p in repo_root.rglob("*.py")
        if "__pycache__" not in (rel := p.relative_to(repo_root)).parts
        and not any(part.startswith(".") for part in rel.parts)
        and not _EXCLUDE_RE.match(str(rel))
    )

    print(f"Scanning {len(files)} Python files...")
    items = _run_interrogate(files, repo_root)
    items.sort()

    header = (
        "# interrogate_baseline.txt\n"
        "# Known undocumented items (grandfathered). Each line: filepath:item_name\n"
        "# New code must have docstrings. Remove entries as docstrings are backfilled.\n"
        "# Regenerate: uv run python test/ci_tests/check_docstring_coverage.py"
        " --generate-baseline\n"
    )
    baseline_path.write_text(header + "\n".join(items) + "\n")
    print(f"Wrote {len(items)} baseline entries to {baseline_path}")
    return 0


def _check_coverage(files: list[str], repo_root: Path, baseline_path: Path) -> int:
    """Check *files* for new undocumented items not in the baseline."""
    if not files:
        return 0

    current_counts = Counter(_run_interrogate(files, repo_root))
    baseline_counts = Counter(_read_baseline(baseline_path))

    ### Fail on new items not accounted for in the baseline
    new_items = sorted((current_counts - baseline_counts).elements())

    if new_items:
        print("FAILED: New undocumented items (not in baseline):\n")
        for item in new_items:
            filepath, name = item.split(":", 1)
            print(f"  {filepath}: {name}")
        print(f"\nAdd docstrings to the {len(new_items)} item(s) above.")
        rel_baseline = baseline_path.relative_to(repo_root)
        print(f"To grandfather instead, add to {rel_baseline}")
        return 1

    ### Informational: baseline items whose files were checked and are now documented
    checked_files = {os.path.normpath(f) for f in files}
    checked_baseline = Counter(
        {
            k: v
            for k, v in baseline_counts.items()
            if os.path.normpath(k.split(":", 1)[0]) in checked_files
        }
    )
    stale = sorted((checked_baseline - current_counts).elements())

    if stale:
        print(
            "INFO: These baseline items now have docstrings "
            "(consider removing from baseline):"
        )
        for item in stale:
            print(f"  {item}")

    return 0


def main() -> int:
    """Entry point for the docstring coverage check."""
    parser = argparse.ArgumentParser(
        description="Check docstring coverage against a baseline of known gaps.",
    )
    parser.add_argument(
        "--generate-baseline",
        action="store_true",
        help="Scan all non-excluded Python files and write/update the baseline.",
    )
    parser.add_argument(
        "filenames",
        nargs="*",
        help="Files to check (passed by pre-commit).",
    )
    args = parser.parse_args()

    if args.generate_baseline:
        return _generate_baseline(REPO_ROOT, BASELINE_PATH)

    return _check_coverage(args.filenames, REPO_ROOT, BASELINE_PATH)


if __name__ == "__main__":
    sys.exit(main())
