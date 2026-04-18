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
| [`FNC-005`](#fnc-005-benchmarking-hooks) | Benchmarking hooks | Implementing `make_inputs_forward`/`make_inputs_backward`/`compare_forward` |
| [`FNC-006`](#fnc-006-testing-functionals) | Testing functionals | Adding functional tests |
| [`FNC-007`](#fnc-007-benchmark-registry) | Benchmark registry | Adding a functional to ASV |
| [`FNC-008`](#fnc-008-warp-integration-must-use-torch-custom-ops) | Warp integration must use torch custom ops | Adding/refactoring Warp-backed functionals |

---

## Detailed Rules

### FNC-000: Functionals must use FunctionSpec

**Description:**

All functionals must be implemented with `FunctionSpec`, even if only a single
implementation exists. This ensures the operation participates in validation
and benchmarking through input generators and `compare_forward` (and
`compare_backward` where needed).

**Rationale:**

`FunctionSpec` provides a consistent structure for backend registration,
selection, benchmarking and verification across the codebase.

**Example:**

```python
import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

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
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        device = torch.device(device)
        yield ("small", (torch.randn(1024, device=device),), {})
        yield ("medium", (torch.randn(4096, device=device),), {})
        yield ("large", (torch.randn(16384, device=device),), {})

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        device = torch.device(device)
        yield (
            "small-bwd",
            (torch.randn(1024, device=device, requires_grad=True),),
            {},
        )
        yield (
            "medium-bwd",
            (torch.randn(4096, device=device, requires_grad=True),),
            {},
        )
        yield (
            "large-bwd",
            (torch.randn(16384, device=device, requires_grad=True),),
            {},
        )

    @classmethod
    def compare_forward(
        cls, output: torch.Tensor, reference: torch.Tensor
    ) -> None:
        torch.testing.assert_close(output, reference)

    @classmethod
    def compare_backward(
        cls, output: torch.Tensor, reference: torch.Tensor
    ) -> None:
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
  - For complex Warp backends, prefer a dedicated `_warp_impl/` package with:
    - `op.py` for torch custom-op registration and validation
    - `launch_forward.py` for forward launch dispatch
    - `launch_backward.py` for backward launch dispatch
    - `_kernels/` with one kernel per file
    - `utils.py` for shared Warp constants/functions
  - Keep `launch_forward.py` and `launch_backward.py` as the only launch
    surfaces; avoid extra launch helper modules unless there is a strong reason.

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

```text
physicsnemo/nn/functional/interpolation/grid_to_point_interpolation/
    grid_to_point_interpolation.py
    _torch_impl.py
    _warp_impl/
        __init__.py
        op.py
        launch_forward.py
        launch_backward.py
        _kernels/
            forward_3d_stride2.py
            backward_3d_stride2.py
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

Implement `make_inputs_forward` for every functional so it can be benchmarked.
Implement `compare_forward` when a functional has multiple implementations and
needs cross-backend parity checks in tests.

Implement `make_inputs_backward` only for functionals with a meaningful
backward pass (for example differentiable functionals). Implement
`compare_backward` when a functional has backward support and multiple
implementations that need backward parity checks.

Input generators should yield labeled inputs ordered from smaller to larger
cases. Labels do not have to be exactly "small/medium/large", and you can
provide more than three cases. Compare hooks should validate output
consistency where implemented. Labels are used for benchmark plots and
summaries.

**Rationale:**

This enables automated benchmarking, labeling, and correctness testing across
backends.

**Example:**

```python
@classmethod
def make_inputs_forward(cls, device="cpu"):
    yield ("small", (torch.randn(1024, device=device),), {})
    yield ("medium", (torch.randn(4096, device=device),), {})
    yield ("large", (torch.randn(16384, device=device),), {})

@classmethod
def make_inputs_backward(cls, device="cpu"):
    x = torch.randn(4096, device=device, requires_grad=True)
    yield ("medium", (x,), {})

@classmethod
def compare_forward(cls, output, reference):
    torch.testing.assert_close(output, reference)

@classmethod
def compare_backward(cls, output, reference):
    torch.testing.assert_close(output, reference)
```

**Anti-pattern:**

```python
@classmethod
def make_inputs_forward(cls, device="cpu"):
    pass
```

---

### FNC-006: Testing functionals

**Description:**

Add tests under `test/nn/functional/` to validate selection, optional
dependencies, and output correctness.

Use a consistent test layout when possible. This is **highly recommended** for
readability and review speed, but it is **not strictly required** when a
functional needs a different shape.

Baseline spec-contract tests (expected for every functional):

1. Backend and reference correctness:
   - `test_<functional_name>_<implementation_name>`
2. Dispatch behavior (only when custom dispatch behavior exists):
   - `test_<functional_name>_dispatch_*`
3. Benchmark-input contract:
   - `test_<functional_name>_make_inputs_forward`
   - `test_<functional_name>_make_inputs_backward` (only when backward is meaningful)
4. Validation/deprecation path coverage:
   - `test_<functional_name>_error_handling` (when validation branches exist)

Cross-backend parity tests and compare-hook tests
(required only when multiple implementations exist):

1. Forward parity:
   - `test_<functional_name>_backend_forward_parity`
   - `test_<functional_name>_compare_forward_contract`
2. Backward parity:
   - `test_<functional_name>_backend_backward_parity` (only for differentiable ops)
   - `test_<functional_name>_compare_backward_contract` (only when backward is meaningful)

Where possible, keep all backend parity checks in one functional test file and
use the functional's `compare_forward`/`compare_backward` hooks for consistency.
For single-implementation functionals, `compare_forward`/`compare_backward`
overrides and compare-hook contract tests are not required.

**Rationale:**

Functional APIs are public entry points and need coverage for both the API and
backend behavior.

**Example:**

```python
def test_grid_to_point_interpolation_torch():
    ...

def test_grid_to_point_interpolation_warp():
    ...

def test_grid_to_point_interpolation_backend_forward_parity():
    ...

def test_grid_to_point_interpolation_backend_backward_parity():
    ...

def test_grid_to_point_interpolation_error_handling():
    ...
```

**Anti-pattern:**

```python
# No tests for a new functional.
```

---

### FNC-007: Benchmark registry

**Description:**

Functionals that should be benchmarked must be added to
`benchmarks/physicsnemo/nn/functional/registry.py`. Only add a functional once
its input generators (`make_inputs_forward`, and optionally
`make_inputs_backward`) yield labeled inputs.

**Rationale:**

Centralizing the benchmark list keeps ASV configuration minimal and ensures
every benchmarked functional provides the inputs and labels needed for
consistent plotting across small-to-large cases.

**Example:**

```python
from physicsnemo.nn.functional.knn.knn import KNN
from physicsnemo.nn.functional.radius_search.radius_search import RadiusSearch

FUNCTIONAL_SPECS = (KNN, RadiusSearch)
```

**Anti-pattern:**

```python
# Adding a functional before input generators are implemented.
FUNCTIONAL_SPECS = (MyFunctionalWithoutInputs,)
```

---

### FNC-008: Warp integration must use torch custom ops

**Description:**

Warp-backed functionals in `physicsnemo/nn/functional/**` must be integrated
into PyTorch using `torch.library.custom_op`, `register_fake`, and (when
backward is supported) `register_autograd`. Do not use
`torch.autograd.Function` wrappers for Warp-backed functionals.

If a functional has no meaningful backward path, `register_autograd` is not
required; otherwise, the custom op must register a backward implementation.

**Rationale:**

`torch.library.custom_op` provides a consistent integration path for eager,
`torch.compile`, fake tensor propagation, and runtime dispatch behavior.
Avoiding per-functional `torch.autograd.Function` wrappers also keeps backend
integration uniform across functionals.

**Example:**

```python
@torch.library.custom_op("physicsnemo::my_warp_op", mutates_args=())
def my_warp_op_impl(x: torch.Tensor) -> torch.Tensor:
    ...
    return y

@my_warp_op_impl.register_fake
def _my_warp_op_impl_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)

def setup_my_warp_op_context(ctx, inputs, output):
    ...

def backward_my_warp_op(ctx, grad_output):
    ...
    return grad_x

my_warp_op_impl.register_autograd(
    backward_my_warp_op,
    setup_context=setup_my_warp_op_context,
)
```

**Anti-pattern:**

```python
class _MyWarpAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ...
```
