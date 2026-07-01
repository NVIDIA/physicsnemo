---
name: physicsnemo-model-builder
description: Official NVIDIA-authored workflow for adding a new model or reusable layer to PhysicsNeMo, or integrating an existing PyTorch model. Scaffolds a standards-compliant physicsnemo.Module (or a Module.from_torch wrapper for an external nn.Module), places it correctly, wires exports, writes tests against the house test helpers, and runs the local CI gates (ruff, interrogate, pytest). Use when a contributor wants to add or port a model or layer into the physicsnemo package. Do NOT use for datapipes, nn.functional ops/backends such as FunctionSpec, losses or metrics, training-recipe or example authoring, environment/installation setup, or merely deciding which existing model fits a task (use physicsnemo-discover for that).
license: Apache-2.0
metadata:
  author: NVIDIA <agent-skills@nvidia.com>
  tags:
    - physicsnemo
    - models
    - contributing
    - scaffolding
    - integration
---

# PhysicsNeMo Model Builder

Drive a contributor from "I have a model (or layer, or an existing PyTorch
module)" to a standards-compliant, tested, CI-green addition to the
`physicsnemo` package. **You do the mechanical work** — placement, the
`physicsnemo.Module` shell, serialization wiring, docstrings, type
annotations, validation, exports, tests, and gates. **The contributor brings
the architecture** — the novel `forward` math. Keep that division explicit:
never invent their model; scaffold everything around it.

The audience is a researcher fluent in PyTorch but new to PhysicsNeMo, so
**explain the "why"** at each step (name the rule, give the reason) rather
than silently emitting files.

## When NOT to use this skill

Stop and **redirect — do not activate** — when the request is to *pick*, *use*,
or *configure* something that already exists, or targets another surface:

- **"Which existing model should I use / which fits my data?"** — selection and
  discovery, not authoring → `physicsnemo-discover`.
- **Datapipes** (`physicsnemo/datapipes/`), **losses or metrics**
  (`physicsnemo/metrics/`), **functional ops / backends**
  (`physicsnemo/nn/functional/`, `FunctionSpec`), **training recipes / examples**
  (`examples/`).

This skill is only for **authoring or porting** a model or reusable layer into
the `physicsnemo` package.

## Core principle

1. **The written standards are ground truth — read them, don't paraphrase from
   memory.** The authoritative rules live in `CODING_STANDARDS/` at the repo
   root: `MODELS_IMPLEMENTATION.md` (rules `MOD-***`) and
   `EXTERNAL_IMPORTS.md` (rules `EXT-***`). Open the cited rule before relying
   on it; reference it by ID when you justify a decision. They evolve — a rule
   recalled from memory may be stale.
2. **Reuse before you build — discover, don't reinvent.** Half of a clean
   integration is *not* writing code that already exists. Before scaffolding a
   `forward`, enumerate what `physicsnemo.nn` already provides and tell the
   contributor what to import (`references/reuse_map.md`).
3. **Verify every path before you cite it.** Glob/Read the live repo; a path
   recalled from memory or pattern-matched from a neighbor is disproof — drop
   it.

## Repo root resolution

Resolve the PhysicsNeMo repo root **first** (see `CONTRIBUTING.md §Repo root
resolution`); all `CODING_STANDARDS/…` and `physicsnemo/…` paths are rooted
there, and scaffolded files are written under it. **If no local clone is on the
path** (e.g. headless against the skills repo in an eval), shallow-clone the
canonical repo once and anchor to it — read its existing tree read-only for
standards / reuse / path verification, write new files under it:
`DEST="${TMPDIR:-/tmp}/physicsnemo-src"; [ -d "$DEST/physicsnemo" ] || git clone --depth 1 https://github.com/NVIDIA/physicsnemo "$DEST"`.
Use that URL verbatim; never interpolate one from user input.

## Scope

In scope: **complete models** (`physicsnemo/experimental/models/`), **reusable
layers** (`physicsnemo/nn/module/`), and **wrapping an existing PyTorch
`nn.Module`** via `Module.from_torch`.

Out of scope — stop and redirect: datapipes (`physicsnemo/datapipes/`),
functional ops / custom backends (`physicsnemo/nn/functional/`, `FunctionSpec`),
losses & metrics (`physicsnemo/metrics/`), training recipes (`examples/`), and
"which model should I use" (→ `physicsnemo-discover`).

## Workflow

Run in order, **after resolving the repo root** (§Repo root resolution).
Confirm the consequential choices with the contributor (artifact type,
placement, external-wrap vs from-scratch); scaffold the rest.

**Default to action.** When the repository is available, *create the actual
files* — `__init__.py`, `<name>.py`, and the test module — with a placeholder
`forward` that raises `NotImplementedError` (marked `# TODO: contributor's
forward math`), rather than only describing them in prose. Ask only the
questions that genuinely block placement; then produce files and iterate. The
skill's value is the working scaffold on disk, not a description of one.

### 1. Intake & classify

Ask only what you can't infer (≤4 questions). Resolve:

- **Artifact type:** complete *model*, reusable *layer*, or *wrap* an existing
  PyTorch module? (Decision tree + rationale: `references/placement.md`.)
- **Identity:** class name (PascalCase), one-line purpose, the forward
  inputs/outputs and their tensor shapes, heavy deps.
- **For wrap:** the import path of their `nn.Module`, and whether its
  `__init__` args are JSON-serializable — this picks the serialization path
  (`references/serialization.md`).

### 2. Place it (and say why)

- New **model** → `physicsnemo/experimental/models/<name>/` (`MOD-002a`: new
  models start in `experimental`). Layout: `__init__.py` (re-exports) +
  `<name>.py`. New **layer** → `physicsnemo/nn/module/<name>.py`, re-exported
  from both `physicsnemo/nn/module/__init__.py` and `physicsnemo/nn/__init__.py`
  (`MOD-000a`). Tests mirror the source path under `test/`.
- State the rule ID and the reason (experimental = API may change; layers are
  shared building blocks).

### 3. Reuse audit

Before writing `forward`, enumerate existing primitives the contributor would
otherwise reinvent — attention bases, embeddings, `Mlp`, neighbor ops
(`knn`, `radius_search`), the TE-aware `LayerNorm`. Use the live search
patterns in `references/reuse_map.md`; verify each path before citing it. Say
explicitly "import `X` from `physicsnemo.nn` instead of writing your own." Keep
genuinely novel, model-specific pieces local to the model.

### 4. Scaffold the shell

Generate from the skeletons in `references/scaffolds.md`, adapting to the
contributor's shapes; explain what each enforced piece is for.

- **New model / layer:** subclass **`physicsnemo.Module`** (not
  `torch.nn.Module` — `MOD-001`); a `ModelMetaData`; a **constructor taking
  JSON-serializable config** (no splatted `**kwargs` — `MOD-010`; no
  string-based class selection — `MOD-009`); a `forward` with jaxtyping on
  every tensor arg (`MOD-006`), `if not torch.compiler.is_compiling():` shape
  validation (`MOD-005`), and NumPy `r"""` docstrings with
  `Parameters`/`Forward`/`Outputs` sections and `:math:` shapes (`MOD-003`).
  Imports upward-only (`EXT-***`).
- **Wrap external:** `Module.from_torch(TheirModule, meta=...)`. **The
  serialization gotcha lives here** — `physicsnemo.Module` save/from_checkpoint
  requires `__init__` args to be JSON-serializable; nested `nn.Module` args
  must each be converted via `Module.from_torch`. Walk them through
  `references/serialization.md`, then prove it with the round-trip test below.

### 5. Tests

Generate the test module from `references/scaffolds.md`: class-per-public-class,
the `device` fixture, parametrized constructor/attribute checks (≥2 configs —
`MOD-008a`), `validate_forward_accuracy` for non-regression (`MOD-008b`), and
`validate_checkpoint` for the save/load round-trip (`MOD-008c`). These helpers
are **mandatory and come from `test.common`** — write the import explicitly in
the generated test and name them in your summary; don't hand-roll what they
provide:

```python
from test.common import validate_forward_accuracy, validate_checkpoint
```

### 6. Gates

From the repo root, run and iterate to green (explain each):

```
make lint          # ruff format --check + ruff check
make interrogate   # docstring coverage
make pytest        # or: pytest test/<mirrored/path> -q
```

`physicsnemo/experimental/` is exempt from ruff/interrogate, but **not** from
runtime contracts — the serialization round-trip test must still pass there.

### 7. Finish & review

- Add a one-line `CHANGELOG.md` entry and SPDX Apache-2.0 headers to new files;
  remind the contributor commits need `-s` (sign-off).
- Do an independent **code-review pass over the diff** before opening the PR —
  re-check it against the standards (`MOD-***`/`EXT-***`), correctness, and the
  reuse audit, ideally with fresh eyes (a separate review session/agent). If the
  host agent offers a built-in code-review command (for example Claude Code's
  `/code-review`), use it; otherwise review the diff directly. Then open the PR
  — CODEOWNERS review + CI re-run the gates.

### 8. Definition of done

Confirm each before declaring success; fix any miss before finishing:

- [ ] Repo root resolved; every cited path verified to exist (no memory/guesses).
- [ ] Placed right: model → `experimental/models/` (`MOD-002a`); layer →
  `nn/module/` + both `__init__` re-exports (`MOD-000a`).
- [ ] Subclasses `physicsnemo.Module` with a `ModelMetaData` (`MOD-001`);
  `__init__` is JSON-serializable — no splatted `**kwargs` (`MOD-010`), no
  string class selection (`MOD-009`).
- [ ] `forward` has jaxtyping on every tensor arg (`MOD-006`) +
  `is_compiling()`-guarded shape validation (`MOD-005`); NumPy `r"""` docstrings
  with `:math:` shapes (`MOD-003`).
- [ ] Reuse audit done — nothing reimplemented that `physicsnemo.nn` provides.
- [ ] Tests use `test.common`: `validate_forward_accuracy` (`MOD-008b`) +
  `validate_checkpoint` (`MOD-008c`), ≥2 constructor configs (`MOD-008a`).
- [ ] Gates green (`make lint`, `make interrogate`, `make pytest`);
  `CHANGELOG.md` entry + SPDX headers added.

## Common gotchas

Surface the relevant traps inline as you scaffold (full catalogue:
`references/lessons.md`):

- **`Module` serialization** is a common external-integration failure: raw
  `nn.Module` submodule args break `from_checkpoint` (`references/serialization.md`).
- The **TE-aware `LayerNorm`** runs only on CUDA when Transformer Engine is
  present; tests must skip the CPU case under TE.
- **`experimental/` skips lint, not runtime contracts.**
- Promote a model-specific layer to `physicsnemo.nn` only when a **second**
  consumer appears — keep it local until then.

## Related resources

- `references/placement.md` — artifact decision tree and where each kind goes.
- `references/reuse_map.md` — live search patterns for existing primitives.
- `references/serialization.md` — `physicsnemo.Module`, JSON args, `from_torch`.
- `references/scaffolds.md` — model / layer / external-wrap / test skeletons.
- `references/lessons.md` — gotchas distilled from real integrations.
- `CODING_STANDARDS/MODELS_IMPLEMENTATION.md`, `EXTERNAL_IMPORTS.md` — the
  authoritative rules; read the cited rule before relying on it.
