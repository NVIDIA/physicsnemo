<!-- markdownlint-disable MD012 MD013 MD024 MD031 MD032 MD033 MD034 MD040 MD046 -->
<!-- MD012: Multiple consecutive blank lines -->
<!-- MD013: Line length -->
<!-- MD024: Multiple headings with the same content -->
<!-- MD031: Fenced code blocks should be surrounded by blank lines -->
<!-- MD032: Lists should be surrounded by blank lines -->
<!-- MD033: Inline HTML -->
<!-- MD034: Bare URL used -->
<!-- MD040: Fenced code blocks should have a language specified -->
<!-- MD046: Code block style -->

# FUNCTIONAL_APIS - Coding Standards

## Overview

This document defines the conventions for functional APIs in PhysicsNeMo. These
rules are designed to ensure consistency, maintainability, and high code
quality across all functional implementations.

**Important:** These rules are enforced as strictly as possible. Deviations
from these standards should only be made when absolutely necessary and must be
documented with clear justification in code comments and approved during code
review.

## Document Organization

This document is structured in two main sections:

1. **Rule Index**: A quick-reference table listing all rules with their IDs,
   one-line summaries, and the context in which they apply. Use this section
   to quickly identify relevant rules when implementing or reviewing code.

2. **Detailed Rules**: Comprehensive descriptions of each rule, including:
   - Clear descriptions of what the rule requires
   - Rationale explaining why the rule exists
   - Examples demonstrating correct implementation
   - Anti-patterns showing common mistakes to avoid

## How to Use This Document

- **When adding a new functional**: Review rules FNC-000 through FNC-006.
- **When reviewing code**: Use the Rule Index to quickly verify compliance.
- **When refactoring**: Ensure refactored code maintains or improves compliance.
- **For AI agents that generate code**: Each rule has a unique ID and structured
  sections (Description, Rationale, Example, Anti-pattern) that can be extracted
  and used as context. When generating code based on a rule, explicitly quote
  the rule ID and the relevant extract being used as context.
- **For AI agents that review code**: Explicitly identify which rules are
  violated, why, and quote the rule ID and relevant extract being used as
  context.

## Rule Index

| Rule ID | Summary | Apply When |
|---------|---------|------------|
| [`FNC-000`](#fnc-000-functionals-must-use-functionspec) | Functionals must use FunctionSpec | Creating new functional APIs |
| [`FNC-001`](#fnc-001-functional-location-and-public-api) | Functional location and public API | Organizing or exporting functionals |
| [`FNC-002`](#fnc-002-file-layout-for-functionals) | File layout for functionals | Adding or refactoring functional files |
| [`FNC-003`](#fnc-003-registration-and-dispatch-rules) | Registration and dispatch rules | Registering implementations |
| [`FNC-004`](#fnc-004-optional-dependency-handling) | Optional dependency handling | Using optional backends |
| [`FNC-005`](#fnc-005-benchmarking-hooks) | Benchmarking hooks | Implementing `make_inputs`/`compare` |
| [`FNC-006`](#fnc-006-testing-functionals) | Testing functionals | Adding functional tests |

---

## Detailed Rules

### FNC-000: Functionals must use FunctionSpec

**Description:**

All functionals must be implemented with `FunctionSpec`, even if only a single
implementation exists. This ensures the operation participates in validation
and benchmarking via `make_inputs` and `compare`.

**Rationale:**

`FunctionSpec` provides a consistent structure for backend registration,
selection, benchmarking and verification across the codebase.

**Example:**

```python
import importlib
import torch

from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.core.version_check import check_version_spec

WARP_AVAILABLE = check_version_spec("warp", "0.6.0", hard_fail=False)

if WARP_AVAILABLE:
    wp = importlib.import_module("warp")
    wp.init()
    wp.config.quiet = True

    @wp.kernel
    def _identity_kernel(
        x: wp.array(dtype=wp.float32),
        y: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        y[i] = x[i]

    @torch.library.custom_op("physicsnemo::identity_warp", mutates_args=())
    def identity_impl(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        device, stream = FunctionSpec.warp_launch_context(x)
        wp_x = wp.from_torch(x, dtype=wp.float32, return_ctype=True)
        wp_y = wp.from_torch(out, dtype=wp.float32, return_ctype=True)
        with wp.ScopedStream(stream):
            wp.launch(
                kernel=_identity_kernel,
                dim=x.numel(),
                inputs=[wp_x, wp_y],
                device=device,
                stream=stream,
            )
        return out

    @identity_impl.register_fake
    def identity_impl_fake(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)
else:

    def identity_impl(*args, **kwargs) -> torch.Tensor:
        raise ImportError(
            "warp>=0.6.0 is required for the Warp identity implementation"
        )

def identity_torch(x: torch.Tensor) -> torch.Tensor:
    return x.clone()

class Identity(FunctionSpec):
    """Identity function with Warp and PyTorch backends."""

    @FunctionSpec.register(
        name="warp",
        required_imports=("warp>=0.6.0",),
        rank=0,
    )
    def warp_forward(x: torch.Tensor) -> torch.Tensor:
        return identity_impl(x)

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(x: torch.Tensor) -> torch.Tensor:
        return identity_torch(x)

    @classmethod
    def make_inputs(cls, device: torch.device | str = "cpu"):
        device = torch.device(device)
        for size in (1024, 4096):
            yield (torch.randn(size, device=device),)

    @classmethod
    def compare(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        torch.testing.assert_close(output, reference)

identity = Identity.make_function("identity")

x = torch.arange(8, device="cuda")
y = identity(x)
```

**Anti-pattern:**

```python
def my_op(x):
    return x
```

---

### FNC-001: Functional location and public API

**Description:**

Functionals live under `physicsnemo/nn/functional` and must be re-exported from
`physicsnemo/nn/functional/__init__.py`.

**Rationale:**

Keeping functionals in a single location makes them easy to discover and keeps
the public API consistent.

**Example:**

```python
# physicsnemo/nn/functional/__init__.py
from .knn import knn
__all__ = ["knn"]
```

**Anti-pattern:**

```python
# Function defined in a random model module and not exported.
```

---

### FNC-002: File layout for functionals

**Description:**

- Single-file functionals go in `physicsnemo/nn/functional/<name>.py`.
- When implementations get too large for a single file, use
  `physicsnemo/nn/functional/<name>/`.
  - Keep each backend in its own module (e.g., `_torch_impl.py`).
  - Keep shared helpers in `utils.py`.

**Rationale:**

Separating backend-specific code keeps optional dependencies isolated and makes
maintenance easier.

**Example:**

```text
physicsnemo/nn/functional/knn/
    __init__.py
    knn.py
    _torch_impl.py
    _cuml_impl.py
    _scipy_impl.py
    utils.py
```

**Anti-pattern:**

```text
physicsnemo/nn/functional/knn.py  # all backends mixed in one file
```

---

### FNC-003: Registration and dispatch rules

**Description:**

Use `@FunctionSpec.register` inside the class body for every implementation.
`rank` selects the default implementation (lower is preferred). Exactly one
implementation should be marked `baseline=True`. Baseline implementations are
usually the straight PyTorch backend.

**Rationale:**

Consistent registration and rank-based dispatch keep functional selection
predictable and debuggable.

**Example:**

```python
class MyOp(FunctionSpec):
    @FunctionSpec.register(name="warp", rank=0)
    def warp_forward(x):
        return x

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(x):
        return x
```

**Anti-pattern:**

```python
def warp_forward(x):
    return x
```

---

### FNC-004: Optional dependency handling

**Description:**

Backend modules must guard optional imports and expose a stub that raises a
clear `ImportError` when called if the dependency is missing. Do not raise at
import time.

**Rationale:**

Optional backends should not prevent importing the package or unrelated
functionals.

**Example:**

```python
if has_dep:
    def knn_impl(...):
        ...
else:
    def knn_impl(*args, **kwargs):
        raise ImportError("missing dependency")
```

**Anti-pattern:**

```python
import missing_dep  # raises at import time
```

---

### FNC-005: Benchmarking hooks

**Description:**

Implement `make_inputs` and `compare` for every functional. `make_inputs` should
yield representative inputs; `compare` should validate output consistency.

**Rationale:**

This enables automated benchmarking and correctness testing across backends.

**Example:**

```python
@classmethod
def make_inputs(cls, device="cpu"):
    yield (torch.randn(1024, device=device),)
```

**Anti-pattern:**

```python
@classmethod
def make_inputs(cls, device="cpu"):
    pass
```

---

### FNC-006: Testing functionals

**Description:**

Add tests under `test/nn/functional/` to validate selection, optional
dependencies, and output correctness.

**Rationale:**

Functional APIs are public entry points and need coverage for both the API and
backend behavior.

**Example:**

```python
def test_knn_cpu():
    indices, distances = knn(points, queries, k=4)
```

**Anti-pattern:**

```python
# No tests for a new functional.
```
