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

import importlib.util
import os
import sys
import sysconfig
import tomllib
from pathlib import Path
from typing import Dict, List, Set

from importlinter import Contract, ContractCheck, fields, output
from packaging.requirements import Requirement

"""
This is a script meant to be used in import-linter as a pre-commit hook to
prevent unlisted / not-required imports as a bare import in physicsnemo.

It will do the following:
- Scan the entire "container" (configured in .importlinter) for
  imports, using import-linter and grimp.  Automatic.
- Extract all the "upstream" modules: things that go "import ABC"
- From that list, remove all the upstream modules from the standard library.
- Scan pyproject.toml for the requirements listed in `[project.dependencies]`
- From the upstream list, Remove all the modules listed as a requirement.
- From the remaining list,  find all the importers but exclude anything in
  `container.e for e in exclude`.
- Pass if all upstream modules are standard or hard requirements.
- Fail otherwise, and report which modules and from what files.
"""

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
    exclude = fields.ListField(fields.StringField(), required=False)

    def check(self, graph, verbose):
        output.verbose_print(
            verbose,
            "Getting import details from "
            f"{self.container} vs project core dependencies...",
        )

        upstream_modules = graph.find_upstream_modules(self.container, as_package=True)

        # Remove any models that start with "physicsnemo":
        upstream_modules = set(
            module
            for module in upstream_modules
            if not module.startswith(self.container)
        )

        upstream_external_modules = remove_standard_library(upstream_modules)

        # Now, read the tree from pyproject.toml:
        dependency_tree = resolve_core_dependencies(Path("pyproject.toml"))

        # This list hasn't been pruned for excludes:
        broken_imports = upstream_external_modules - dependency_tree
        violations = {}

        unexcluded_broken_imports = set[str]()

        for broken_import in broken_imports:
            local_violations = graph.find_modules_that_directly_import(broken_import)

            # Remove violations that start with any exclusions:
            if self.exclude is not None:
                exclusions = [self.container + "." + ex for ex in self.exclude]
            else:
                exclusions = []
            local_violations = set[str](
                lv
                for lv in local_violations
                if not any(lv.startswith(ex) for ex in exclusions)
            )

            if len(local_violations) > 0:
                unexcluded_broken_imports.add(broken_import)

                violations[broken_import] = local_violations

                violations[broken_import] = [
                    v for v in violations[broken_import] if self.container in v
                ]

        return ContractCheck(
            kept=len(unexcluded_broken_imports) == 0,
            metadata={
                "broken_imports": list(unexcluded_broken_imports),
                "violations": violations,
            },
        )

    def render_broken_contract(self, check):
        inverted_violations = {}

        output.print_error("Listing broken imports by external package...")
        output.new_line()

        n_invalid_imports = 0
        n_file_violations = 0
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
            n_invalid_imports += 1
            output.new_line()

        output.print_error("Listing broken imports by internal file...")
        output.new_line()
        for violating_file, violating_imports in inverted_violations.items():
            output.print_error(
                f"{violating_file} is not allowed to import: {', '.join(violating_imports)}",
                bold=True,
            )
            output.new_line()

        output.print_error(
            f"Found {n_invalid_imports} invalid imports and {n_file_violations} file violations"
        )


def resolve_core_dependencies(pyproject_path: str | Path) -> Set[str]:
    """
    Load and normalize the dependencies declared under ``[project].dependencies``
    so that we can compare external imports against the canonical list of core
    dependencies shipped with the package.
    """
    pyproject_path = Path(pyproject_path)
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    project_table = data.get("project") or {}
    dependency_list: List[str] | None = project_table.get("dependencies")
    if dependency_list is None:
        raise KeyError("Core dependency list not found under [project].dependencies")

    resolved: List[str] = []
    for item in dependency_list:
        requirement = Requirement(item)
        # PyPI names use hyphens; import names use underscores (PEP 503).
        default_import_name = requirement.name.replace("-", "_")
        resolved.append(dep_to_import_name.get(requirement.name, default_import_name))

    seen: Set[str] = set()
    ordered_unique: List[str] = []
    for dep in resolved:
        if dep not in seen:
            ordered_unique.append(dep)
            seen.add(dep)
    return set(ordered_unique)


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
