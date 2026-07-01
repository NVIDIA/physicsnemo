# Common gotchas

General traps that bite contributors adding models or layers to PhysicsNeMo.
Surface the relevant one *inline* as you scaffold — they're cheap to avoid up
front and expensive to discover in review or after a checkpoint won't load.
(These are guidance; the authoritative rules are in `CODING_STANDARDS/`.)

## 1. `Module` serialization breaks on raw `nn.Module` constructor args

A common external-integration failure: passing a **raw `torch.nn.Module`** as a
constructor argument makes `save()`/`from_checkpoint()` raise a `TypeError` —
even though `forward` works fine, so it's easy to miss until a user tries to
persist the model. Use constructor-from-config, or convert injected submodules
with `Module.from_torch`, and prove it with `validate_checkpoint`. Detail:
`references/serialization.md`.

## 2. `experimental/` skips lint, not runtime contracts

`physicsnemo/experimental/` is excluded from ruff and interrogate (incomplete
docstrings / missing tests are tolerated there), but it is **not** exempt from
behavior: imports must work, `forward` must run, and a `physicsnemo.Module`
must still serialize. "It's experimental" does not excuse a broken
`from_checkpoint`.

## 3. Use `test.common`, don't hand-roll test machinery

`test.common` provides `validate_forward_accuracy` (auto-managed reference
`.pth` for non-regression — `MOD-008b`) and `validate_checkpoint` (save/load
round-trip — `MOD-008c`). Hand-rolling committed golden files + ad-hoc
comparisons is non-idiomatic and brittle; the helpers are the house pattern
(see `test/nn/module/test_mlp_layers.py`).

## 4. The TE-aware `LayerNorm` is CUDA-only when Transformer Engine is present

If you reuse `physicsnemo/nn/module/layer_norm.py::LayerNorm` (recommended over
`torch.nn.LayerNorm` for the faster backward), note it resolves **once at
import** to Transformer Engine's LayerNorm when TE + CUDA are available, and
**TE's LayerNorm cannot run on CPU tensors**. Consequences:
- The model runs on a CPU-only box (torch fallback) and on GPU (TE), but on a
  TE-enabled GPU box it cannot run on CPU.
- The `device` fixture in `test/` parametrizes **both** cpu and cuda when a GPU
  is present, so tests must **skip the cpu case when TE is the active backend**
  (the repo's own `test/nn/module/test_layer_norm.py` does this;
  `PHYSICSNEMO_FORCE_TE=0` forces the torch path for CPU work).

## 5. jaxtyping single-token shape strings trip `F821`

`Float[Tensor, "n dim"]` (multi-token) is fine, but a single-token annotation
like `Int[Tensor, "n"]` makes ruff flag `F821` (undefined name). Add
`# noqa: F821` on that line — the established convention. Annotate **all**
tensor args, including optional ones, per `MOD-006`.

## 6. Promote a layer to `physicsnemo.nn` only on the second consumer

A layer used by exactly one model is not yet a reusable primitive — keep it
local to the model directory. Promote it to `physicsnemo/nn/module/` when a
**second** model actually needs it. Generalizing a single-consumer layer
prematurely freezes an API for one user and is its own anti-pattern.
