# Placement — what am I adding, and where does it go?

Resolve two questions before writing anything: **what kind of artifact** is it,
and **where in the tree** does it live. Both are governed by
`CODING_STANDARDS/MODELS_IMPLEMENTATION.md` — read the cited rule.

## Decision tree

```
Is it a complete, trainable architecture (composes layers; users instantiate it)?
├─ YES → MODEL
│        new code → physicsnemo/experimental/models/<name>/   (MOD-002a)
│        (graduates to physicsnemo/models/<name>/ later, after API review)
│
└─ NO → Is it a reusable building block (a layer/block other models would compose)?
        ├─ YES → LAYER → physicsnemo/nn/module/<name>.py        (MOD-000a)
        │        re-export from nn/module/__init__.py AND nn/__init__.py
        │
        └─ NO → is it just one example's helper? → examples/<example>/...  (out of scope here)

Separately: do you ALREADY have a trained PyTorch nn.Module to bring in?
└─ YES → EXTERNAL WRAP → Module.from_torch(...)  (see references/serialization.md)
         placed as a MODEL or LAYER per the same tree above.
```

## Rules that decide placement

- **`MOD-000a` — reusable layers live in `physicsnemo/nn/module/`** and are
  re-exported from `physicsnemo/nn/__init__.py` so users do
  `from physicsnemo.nn import MyLayer`. A layer placed under
  `physicsnemo/models/` is the anti-pattern.
- **`MOD-000b` — complete models live in `physicsnemo/models/`** (re-exported
  from `physicsnemo/models/__init__.py`).
- **`MOD-002a` — new models AND new layers start in
  `physicsnemo/experimental/`** (`experimental/models/`, `experimental/nn/…`).
  `experimental/` means "API may change"; it is **exempt from ruff/interrogate
  lint**, but **not** from runtime contracts (it must still import, run, and —
  if it's a `Module` — serialize). Graduation to the stable tree requires
  stability + API review.

So in practice, a brand-new contribution almost always starts under
`experimental/`. Tell the contributor this explicitly and why (it lets the API
settle without a major-version commitment).

## File layout for a model

```
physicsnemo/experimental/models/<name>/
  __init__.py        # re-exports the public class(es) only
  <name>.py          # the architecture (Module subclass + ModelMetaData)
  <name>_utils.py    # optional: model-specific helpers (keep them local)
```

Tests mirror the source path:
```
test/experimental/models/<name>/test_<name>.py
# or, for a layer:
test/nn/module/test_<name>.py
```

## Model vs. layer — the litmus test

- A **layer** has a generic tensor-in / tensor-out signature, no training
  recipe, and would plausibly be reused by ≥1 other model. It does **not** own
  a `ModelMetaData` describing AMP/ONNX/etc. capabilities.
- A **model** is the thing a user trains and checkpoints; it owns a
  `ModelMetaData` and is the unit `Module.from_checkpoint` reconstructs.

If unsure, start it as a layer local to your model dir; promote it to
`physicsnemo/nn/module/` only when a **second** consumer actually appears
(don't pre-generalize — see `references/lessons.md`).

## Don't put these here (redirect)

- A **loss or metric** → `physicsnemo/metrics/` (or
  `physicsnemo/experimental/metrics/`).
- A **functional op / custom CUDA-Warp/cuML backend** →
  `physicsnemo/nn/functional/` via a `FunctionSpec`.
- A **datapipe** → `physicsnemo/datapipes/`.
- A **training recipe** → `examples/`.

These are out of scope for this skill; hand the contributor off accordingly.
