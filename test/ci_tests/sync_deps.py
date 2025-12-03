# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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


#!/usr/bin/env python3
import re
import sys
from pathlib import Path

import tomlkit

PYPROJECT = Path("pyproject.toml")


# ---------------------------------------------
# Requirement helper
# ---------------------------------------------
REQ_RE = re.compile(r"^([A-Za-z0-9_.\-]+)(.*)$")


def parse_req(req: str):
    """Split 'numpy>=1.2' → ('numpy', '>=1.2')."""
    m = REQ_RE.match(req.strip())
    if not m:
        raise ValueError(f"Invalid requirement syntax: {req}")
    name, spec = m.group(1), m.group(2).strip()
    return name.lower(), spec


# ---------------------------------------------
# Group resolution with DFS cycle detection
# ---------------------------------------------
def resolve_group(name, groups, stack, out_list):
    if name in stack:
        raise ValueError(f"Cycle detected: {' -> '.join(stack)} -> {name}")

    stack.append(name)

    for item in groups[name]:
        if isinstance(item, str):
            out_list.append(item)

        elif isinstance(item, dict) and "include-group" in item:
            sub = item["include-group"]
            if sub not in groups:
                raise ValueError(f'Group "{name}" includes missing group "{sub}"')
            resolve_group(sub, groups, stack, out_list)

        else:
            raise ValueError(f"Unsupported dependency-group item: {item}")

    stack.pop()


def resolve_full_group(root, groups):
    flat = []
    resolve_group(root, groups, stack=[], out_list=flat)
    return flat


# ---------------------------------------------
# Deduplication + version mismatch detection
# ---------------------------------------------
def dedupe_and_validate(deps):
    seen = {}
    final = []

    for req in deps:
        name, spec = parse_req(req)

        if name not in seen:
            # first sighting
            seen[name] = spec
            final.append(req)
            continue

        prev_spec = seen[name]

        if spec == prev_spec:
            # exact match -> OK, dedupe silently
            continue

        # mismatch conditions:
        # - one empty / one not
        # - differing version specs
        raise ValueError(
            f"Version mismatch for package '{name}': '{prev_spec}' vs '{spec}'"
        )

    return final


# ---------------------------------------------
# Main sync logic
# ---------------------------------------------
def main():
    text = PYPROJECT.read_text()
    doc = tomlkit.parse(text)

    groups = doc.get("dependency-groups")
    if groups is None:
        sys.exit("No [dependency-groups] table found.")

    DEFAULT_GROUP = "utils"  # choose the group that represents "full install"

    if DEFAULT_GROUP not in groups:
        sys.exit(f'Default group "{DEFAULT_GROUP}" missing in [dependency-groups].')

    # Step 1: flatten includes
    try:
        flat = resolve_full_group(DEFAULT_GROUP, groups)
    except Exception as e:
        sys.exit(f"Dependency group resolution failed: {e}")

    # Step 2: dedupe + v
    try:
        resolved = dedupe_and_validate(flat)
    except Exception as e:
        sys.exit(f"Dependency conflict: {e}")

    # Create TOML array
    arr = tomlkit.array()
    for d in resolved:
        arr.append(d)
    arr.multiline(True)

    project = doc.setdefault("project", {})
    current = list(project.get("dependencies", []))

    if current == resolved:
        print("Dependencies already synchronized.")
        return

    project["dependencies"] = arr
    PYPROJECT.write_text(tomlkit.dumps(doc))
    sys.exit("Updated project.dependencies. Re-run commit.")


if __name__ == "__main__":
    main()
