<!-- markdownlint-disable MD012 MD013 MD024 MD031 MD032 MD033 MD034 MD040 MD046 -->

# EXTERNAL_IMPORTS - Coding Standards

## Overview

This document defines the policies for managing external dependencies within
`physicsnemo`. The objectives are to maintain a predictable dependency surface,
prevent accidental coupling across modules, and ensure that optional
accelerations never compromise default functionality.

**Important:** These requirements are enforced rigorously. Any deviation must be
explicitly justified in code comments and approved during code review.

## Rule Index

| Rule ID | Summary | Apply When |
|---------|---------|------------|
| `EXT-001` | Keep `pyproject.toml` as the single source of truth for dependencies | Declaring or modifying package requirements |
| `EXT-002` | Preserve the dependency hierarchy via PEP 735 groups | Adding dependencies to any `physicsnemo` submodule |
| `EXT-003` | Classify every external import as hard or optional and guard optional ones | Importing third-party packages anywhere in the codebase |
| `EXT-004` | Use the delayed-error pattern for locally necessary optional packages | Implementing features that absolutely require an optional dependency |
| `EXT-005` | Provide guarded accelerated paths alongside a reference implementation | Adding performance-oriented backends that rely on optional packages |

## Source of Truth for Dependencies

The `pyproject.toml` file is the single authoritative record of every supported
dependency for the Python package and the test suite. Example applications may
list additional packages under `examples/**/requirements.txt`, but those
requirements must not leak into the core package.

## Dependency Hierarchy and Groups

`physicsnemo` is structured as an acyclic hierarchy. Lower-level packages (for
example, `physicsnemo.core`) have strictly fewer dependencies than higher-level
packages (such as `physicsnemo.nn`). To enforce this layering, dependencies are
organized via the PEP 735 "dependency group" model in `pyproject.toml`; higher
groups compose all dependencies from lower groups.

## Classification of External Imports

Every import from a third-party package must fall into one of two categories:

1. **Hard dependency.** The package is part of the mandatory dependency group
   of the importing submodule or any lower-level submodule. Typical examples
   include `torch` and `warp`.
2. **Optional dependency.** The package resides in an extras group or optional
   dependency group. Its usage must be guarded so that importing the module
   succeeds even when the package is absent.

Packages such as `cuml`, `torch_geometric`, and `torch_scatter` remain optional
because of their installation complexity; they are surfaced only through extras
groups or per-example requirements.

## Protecting Imports

Two complementary patterns are used to guard optional dependencies.

### Locally Necessary Imports

Certain features cannot be delivered without a specific package (for example,
PyG for GraphCast backends). For such dependencies, follow the delayed-error
pattern:

1. Perform a soft availability check via
   `physicsnemo.core.version_check.check_version_spec`.
2. When the dependency is present, import it with `importlib.import_module`
   inside the guarded block and expose the fully functional implementation.
3. When the dependency is absent, expose the same symbols, but raise an
   informative exception upon instantiation or call. Static methods should be
   treated as free functions for this purpose.

Raised exceptions must explain who is raising the error, which package is
missing, the minimum required version, and where to find installation
instructions.

```python
import importlib
import torch

from physicsnemo.core.version_check import check_version_spec

CUML_AVAILABLE = check_version_spec("cuml", "24.0.0", hard_fail=False)
CUPY_AVAILABLE = check_version_spec("cupy", "13.0.0", hard_fail=False)

if CUML_AVAILABLE and CUPY_AVAILABLE:
    cuml = importlib.import_module("cuml")
    cp = importlib.import_module("cupy")

    def knn_impl(points, queries, k) -> torch.Tensor:
        ...
else:

    def knn_impl(*args, **kwargs) -> torch.Tensor:
        """
        Dummy implementation for when cuML or CuPy is unavailable.
        """

        raise ImportError(
            "physicsnemo.nn.neighbors: cuML>=24.0.0 and CuPy>=13.0.0 are required "
            "for the accelerated kNN backend. Install both packages; see "
            "https://docs.rapids.ai/install for instructions."
        )
```

### Locally Optional Imports

Some dependencies simply provide accelerated code paths. In these situations,
always provide a reference implementation that only relies on core
dependencies, and add accelerated paths behind guarded imports. Two patterns
are acceptable:

1. **Module-level runtime dispatch.** The dependency is a central part of the
   implementation. Provide an entry-point that selects among backends
   (`"auto"` should try accelerated paths first while falling back to the
   reference path). Each backend implementation must live in its own module and
   independently guard its imports. Example: `physicsnemo.nn.neighbors`.
2. **File-level runtime dispatch.** The dependency affects a small portion of
   the implementation. Keep reference and accelerated code in the same module.
   Use `check_version_spec` to pick the execution path automatically or to
   respect a user override that demands the accelerated backend.

In both cases the default behavior must rely exclusively on baseline
dependencies, and accelerated code paths must never raise at import time merely
because an optional dependency is missing.

## Compliance

- **Code review enforcement.** All pull requests must cite the relevant `EXT-00x`
  rules when introducing new dependencies or optional backends. Reviewers block
  changes that bypass `pyproject.toml`, break the dependency hierarchy, or ship
  unguarded imports; deviations require explicit justification.
- **Import-linter enforcement.** `test/ci_tests/prevent_untracked_imports.py`
  and `.importlinter` translate these rules into automated checks. Import Linter
  fails CI when modules violate declared contracts (for example, high-level
  packages importing from disallowed lower layers or pulling in unapproved
  third-party modules). Keep dependency declarations synchronized so these
  automated guards remain authoritative.
