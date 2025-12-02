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

import importlib.util
import os
import sys
import sysconfig
from pathlib import Path
from typing import Dict, List, Set, Union

import tomllib
from importlinter import Contract, ContractCheck, fields, output
from packaging.requirements import Requirement

Dependency = Union[str, Dict[str, str]]

# For irregular mappings that we don't want to have cause errors:
dep_to_import_name = {
    "warp-lang": "warp",
    "hydra-core": "hydra",
    "GitPython": "git",
}


class ForbiddenImportContract(Contract):
    """
    PhysicsNemo specific contract to prevent external imports
    that are not included in requirements.

    This will, for each sub-package, check the external imports and ensure
    via uv that the list dependencies encompass the entire import graph.
    """

    container = fields.StringField()
    dependency_group = fields.StringField()

    def check(self, graph, verbose):
        output.verbose_print(
            verbose,
            f"Getting import details from {self.container} vs uv group {self.dependency_group}...",
        )

        upstream_modules = graph.find_upstream_modules(self.container, as_package=True)

        # Remove any models that start with "physicsnemo":
        upstream_modules = set(
            module
            for module in upstream_modules
            if not module.startswith("physicsnemo")
        )

        upstream_external_modules = remove_standard_library(upstream_modules)

        # Now, read the tree from pyproject.toml:
        dependency_tree = resolve_dependency_group_no_versions(
            Path("pyproject.toml"), self.dependency_group
        )

        broken_imports = upstream_external_modules - dependency_tree
        violations = {}

        for broken_import in broken_imports:
            violations[broken_import] = graph.find_modules_that_directly_import(
                broken_import
            )
            violations[broken_import] = [
                v for v in violations[broken_import] if self.container in v
            ]

        return ContractCheck(
            kept=len(broken_imports) == 0,
            metadata={
                "broken_imports": list(broken_imports),
                "violations": violations,
            },
        )

    def render_broken_contract(self, check):
        inverted_violations = {}

        output.print_error("Listing broken imports by external package...")
        output.new_line()

        for broken_import in check.metadata["broken_imports"]:
            violating_files = check.metadata["violations"][broken_import]
            for violating_file in violating_files:
                if violating_file not in inverted_violations:
                    inverted_violations[violating_file] = []
                inverted_violations[violating_file].append(broken_import)
            violations = ", ".join(check.metadata["violations"][broken_import])
            output.print_error(
                f"{self.container} is not allowed to import {broken_import} (from {violations})",
                bold=True,
            )
            output.new_line()

        output.print_error("Listing broken imports by internal file...")
        output.new_line()
        for violating_file, violating_imports in inverted_violations.items():
            output.print_error(
                f"{violating_file} is not allowed to import: {', '.join(violating_imports)}",
                bold=True,
            )
            output.new_line()
            output.new_line()


def resolve_dependency_group_no_versions(
    pyproject_path: str | Path, group_name: str
) -> List[str]:
    """
    Open a uv-style pyproject.toml, recursively resolve a dependency group,
    and strip version specifiers from all dependencies.
    """
    pyproject_path = Path(pyproject_path)
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    dep_groups: Dict[str, List[Dependency]] = data.get("dependency-groups", {})

    if group_name not in dep_groups:
        raise KeyError(f"Dependency group '{group_name}' not found")

    def _resolve(group: str, seen: set[str] = None) -> List[str]:
        if seen is None:
            seen = set()
        if group in seen:
            return []
        seen.add(group)
        deps: List[str] = []
        for item in dep_groups.get(group, []):
            if isinstance(item, str):
                # strip version using packaging
                deps.append(Requirement(item).name)
            elif isinstance(item, dict) and "include-group" in item:
                deps.extend(_resolve(item["include-group"], seen))
            else:
                raise ValueError(f"Unknown dependency format: {item}")
        return deps

    # remove duplicates while preserving order
    resolved = _resolve(group_name)

    # Convert dep tree names to what they import as:
    resolved = [dep_to_import_name.get(d, d) for d in resolved]

    seen_ordered = set()
    return set([d for d in resolved if not (d in seen_ordered or seen_ordered.add(d))])


def flatten_deps(tree: Dict) -> Set[str]:
    """Flatten nested dependency dict into a set of package names."""
    packages = set()

    def recurse(d: Dict):
        for name, info in d.items():
            packages.add(name.replace("-", "_"))  # normalize for imports
            recurse(info["dependencies"])

    recurse(tree)
    return packages


def remove_standard_library(packages: Set[str]) -> Set[str]:
    """Remove standard library packages from the set of packages.

    Heuristics:
    - Builtins (sys.builtin_module_names)
    - sys.stdlib_module_names (when available, Python 3.10+)
    - importlib spec origin located within sysconfig stdlib/platstdlib
    - 'built-in' or 'frozen' origins
    """
    builtin_names = set(sys.builtin_module_names)
    stdlib_names = set(getattr(sys, "stdlib_module_names", ()))

    stdlib_dirs = {
        d
        for d in {
            sysconfig.get_path("stdlib"),
            sysconfig.get_path("platstdlib"),
        }
        if d
    }
    stdlib_dirs = {os.path.realpath(d) for d in stdlib_dirs}

    def is_in_stdlib_path(path: str) -> bool:
        if not path:
            return False
        real = os.path.realpath(path)
        for d in stdlib_dirs:
            # Match dir itself or any descendant
            if real == d or real.startswith(d + os.sep):
                return True
        return False

    def is_stdlib(mod_name: str) -> bool:
        # Fast checks
        if mod_name in builtin_names or mod_name in stdlib_names:
            return True

        spec = importlib.util.find_spec(mod_name)
        if spec is None:
            return False

        # Built-in/frozen indicators
        if spec.origin in ("built-in", "frozen"):
            return True

        # Package locations
        if spec.submodule_search_locations:
            for loc in spec.submodule_search_locations:
                if is_in_stdlib_path(loc):
                    return True
            return False

        # Modules
        return is_in_stdlib_path(spec.origin)

    return {p for p in packages if not is_stdlib(p)}
