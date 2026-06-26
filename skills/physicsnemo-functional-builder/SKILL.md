---
name: physicsnemo-functional-builder
description: Official NVIDIA-authored workflow for adding a new functional op (or a new optimized backend for an existing op) to physicsnemo.nn.functional. Scaffolds a FunctionSpec with multi-backend dispatch (a torch reference plus optional Warp, cuML, or SciPy backends), wires re-exports, writes cross-backend equivalence tests, and runs the local CI gates (ruff, interrogate, pytest). Use when a contributor wants to add an op or a Warp/cuML/SciPy backend to physicsnemo.nn.functional. Do NOT use for complete models or reusable nn.Module layers (use physicsnemo-model-builder), datapipes, losses or metrics, training-recipe or example authoring, environment setup, or deciding which existing op to use.
license: Apache-2.0
metadata:
  author: NVIDIA <agent-skills@nvidia.com>
  tags:
    - physicsnemo
    - functional
    - kernels
    - contributing
    - scaffolding
---

# PhysicsNeMo Functional Builder

Drive a contributor from "I have an op (a kNN, an SDF query, a sampler, an
interpolation, a geometry kernel)" — or "I have a faster backend for an
existing op" — to a standards-compliant, tested, CI-green addition to
`physicsnemo.nn.functional`. **You do the mechanical work**: the per-op package
layout, the `FunctionSpec` shell, backend registration and dispatch, the
`torch.library.custom_op` wrapping for accelerated backends, re-exports,
cross-backend tests, and the gates. **The contributor brings the math** — the
actual algorithm and any hand-written Warp/cuML kernel. Keep that division
explicit: never invent their algorithm; scaffold everything around it.

The audience is a researcher fluent in PyTorch but new to PhysicsNeMo, so
**explain the "why"** at each step (name the rule, give the reason) rather than
silently emitting files.

This skill is standalone — it does not depend on any other PhysicsNeMo skill.

## Core principle

1. **`physicsnemo/core/function_spec.py` and `CODING_STANDARDS/` are ground
   truth — read them, don't paraphrase from memory.** The dispatch/registration
   machinery lives in `FunctionSpec` (`physicsnemo/core/function_spec.py`); the
   tensor-annotation / docstring / import rules live in `CODING_STANDARDS/`
   (`MOD-***`, `EXT-***`). Open the real class and the cited rule before relying
   on them, and reference them by name when you justify a decision. The exact
   `FunctionSpec` method names (`register`, `dispatch`, `make_function`,
   `make_inputs_forward`, `compare_forward`, `warp_launch_context`) may evolve —
   confirm against the source.
2. **Study a live exemplar before scaffolding.** The house pattern is consistent
   and best learned by reading one op end-to-end:
   `physicsnemo/nn/functional/geometry/farthest_point_sampling/` (Warp + torch)
   and `physicsnemo/nn/functional/neighbors/knn/` (cuML + SciPy + torch). Mirror
   their structure for the new op.
3. **Verify every path before you cite it.** Glob/Read the live repo; a path
   recalled from memory or pattern-matched from a neighbor is disproof — drop it.

## Scope

In scope: a **new functional op** in `physicsnemo/nn/functional/<category>/`, and
**adding a backend** (Warp, cuML, SciPy) to an existing op. Both center on
`FunctionSpec`.

Out of scope — stop and redirect: complete models / reusable `nn.Module` layers
(→ `physicsnemo-model-builder`), datapipes (`physicsnemo/datapipes/`), losses &
metrics (`physicsnemo/metrics/`), training recipes (`examples/`), and "which op
should I use" (a usage question, not an authoring one).

## Key facts that differ from models/layers

State these early — contributors coming from the model side get them wrong:

- **Functionals live in the STABLE tree**, `physicsnemo/nn/functional/...` —
  **not** `experimental/`. (Contrast `MOD-002a`, which sends new *models/layers*
  to `experimental/`. There is no `experimental/nn/functional`.) See
  `references/placement.md`.
- **There is no `Module`, no parameters, no serialization, no `ModelMetaData`,
  no checkpoint round-trip.** A functional is a stateless op. The testing story
  is **cross-backend equivalence**, not `validate_checkpoint`.
- **Accelerated backends must be wrapped in `torch.library.custom_op`** (plus a
  `register_fake`), so they compose with `torch.compile`/autograd — even an
  inline Warp kernel.

## Workflow

Run in order. Confirm the consequential choices (new-op vs new-backend, category,
which backends); scaffold the rest.

### 1. Intake & classify

Ask only what you can't infer (≤4 questions). Resolve:

- **Task:** a brand-new op, or a new backend for an existing op?
- **Identity:** op name (snake_case) and `FunctionSpec` class (PascalCase); the
  signature (inputs/outputs + tensor shapes via jaxtyping); the category
  (`geometry`, `neighbors`, `interpolation`, …).
- **Backends:** which ones to provide. **Always a torch reference (`baseline`)**;
  then optionally Warp (CUDA kernels), cuML (CUDA, optional dep), SciPy (CPU,
  optional dep). (`references/backends.md`.)

### 2. Place it (and say why)

Per-op package under `physicsnemo/nn/functional/<category>/<op_name>/`:
`<op_name>.py` (the `FunctionSpec` subclass + `dispatch`), `_torch_impl.py`,
`_warp_impl.py` + `kernels.py` (if Warp), `_cuml_impl.py` / `_scipy_impl.py` (if
those deps), `utils.py` (shared validation), `__init__.py`. Re-export up the
chain: op `__init__` → category `__init__` → `physicsnemo/nn/functional/__init__.py`.
Say the rule: **stable tree, not experimental** (`references/placement.md`).

### 3. Scaffold the `FunctionSpec` + dispatch

From `references/dispatch.md` and the skeletons in `references/scaffolds.md`:

- Subclass `FunctionSpec`; register each backend with
  `@FunctionSpec.register(name=..., required_imports=(...), rank=..., baseline=...)`
  (lower `rank` = preferred; **exactly one** `baseline=True`, the torch ref).
- Implement `dispatch()` for backend selection: explicit `implementation=`
  override → availability check; else auto-select (fast backend on CUDA, CPU
  backend on CPU) with a one-time fallback warning.
- Expose the public op via `OpClass.make_function("op_name")`.
- Add `make_inputs_forward` (benchmark inputs) and `compare_forward`
  (tie-aware equivalence) hooks.
- jaxtyping on every tensor arg (`MOD-006`); NumPy `r"""` docstrings
  (`Parameters`/`Returns`/`Raises`); shape validation in `utils.py` guarded by
  `if not torch.compiler.is_compiling():` (`MOD-005`); upward-only imports
  (`EXT-***`).

### 4. Backends (torch always; Warp/cuML/SciPy as chosen)

From `references/backends.md`:

- **torch** reference impl in `_torch_impl.py` — the `baseline`, always present,
  device-agnostic, the equivalence oracle.
- **Warp:** pure `@wp.kernel`s in `kernels.py` (no torch import); `_warp_impl.py`
  wraps the launch in `@torch.library.custom_op(...)` + `@<op>.register_fake`,
  converts with `wp.from_torch(..., return_ctype=True)`, and uses
  `FunctionSpec.warp_launch_context(tensor)` for device/stream. Warp kernels are
  typically CUDA-only — raise clearly on CPU.
- **cuML / SciPy:** gate the whole impl on
  `check_version_spec(pkg, ver, hard_fail=False)`; inside, wrap with
  `torch.library.custom_op` + `register_fake`, move data zero-copy via DLPack
  (cuML) or numpy (SciPy), and **check `tensor.device.type` inside the impl**.
  When the dep is missing, register a stub that raises a clear `ImportError`.

### 5. Cross-backend tests

From `references/testing.md` (mirror source path under `test/nn/functional/...`):

- A **known-answer** test on a deterministic input (backend-independent truth).
- **Per-backend** parametrization with `pytest.skip` for device (Warp/cuML are
  CUDA-only) and for missing optional deps (`check_version_spec`).
- A **backend-parity** test via `OpClass.compare_forward(...)` — and remember the
  classic trap: for neighbor ops compare **distances, not indices** (equal-distance
  ties order differently across backends); sort before comparing.
- `torch.library.opcheck(...)` for each `custom_op` backend.

### 6. Gates

From the repo root, run and iterate to green (explain each):

```
make lint          # ruff format --check + ruff check
make interrogate   # docstring coverage
make pytest        # or: pytest test/nn/functional/<category>/... -q
```

Unlike `experimental/`, `nn/functional/` is **not** lint/interrogate-exempt —
the new op must pass ruff and docstring coverage.

### 7. Finish & review

- Add a one-line `CHANGELOG.md` entry and SPDX Apache-2.0 headers to new files;
  remind the contributor commits need `-s` (sign-off).
- Do an independent **code-review pass over the diff** before opening the PR —
  re-check it against `FunctionSpec`, the standards (`MOD-***`/`EXT-***`),
  correctness, and backend parity, ideally with fresh eyes (a separate review
  session/agent). If the host agent offers a built-in code-review command (for
  example Claude Code's `/code-review`), use it; otherwise review the diff
  directly. Then open the PR — CODEOWNERS review + CI re-run the gates.

## Common gotchas

Surface the relevant traps inline as you scaffold (full catalogue:
`references/lessons.md`):

- **Stable tree, not `experimental/`** — the opposite of the models/layers rule.
- **Backend ties differ:** compare neighbor outputs by *distances* (sorted), not
  indices; use `compare_forward` to encode the tie-aware comparison.
- **Device checks belong inside the impl**, not only in `dispatch` (e.g. cuML
  raises on CPU tensors, Warp on CPU tensors).
- **`custom_op` + `register_fake` are mandatory** for Warp/cuML backends, or
  `torch.compile`/`opcheck` break.
- **Optional deps are gated by `check_version_spec(..., hard_fail=False)`** with a
  stub `ImportError` fallback — never a bare top-level `import cuml`.
- **Exactly one `baseline=True`** (the torch reference); it's the benchmark and
  equivalence oracle.

## Related resources

- `references/placement.md` — where functionals go (stable tree), per-op package
  layout, re-exports, and what's *not* a functional.
- `references/dispatch.md` — `FunctionSpec`: registration, `dispatch`,
  `make_function`, benchmark/compare hooks.
- `references/backends.md` — Warp (kernels + `custom_op` wrap), cuML & SciPy
  (optional-dep gating, DLPack), and the torch reference.
- `references/testing.md` — cross-backend equivalence, device/dep skips, `opcheck`.
- `references/scaffolds.md` — copy-paste skeletons for the package, each backend,
  and the test module.
- `references/lessons.md` — gotchas distilled from real functional PRs.
- `physicsnemo/core/function_spec.py`, `CODING_STANDARDS/` — the authoritative
  source; read before relying on them.
