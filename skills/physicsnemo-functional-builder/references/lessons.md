# Common gotchas (functional ops)

Distilled from real `physicsnemo.nn.functional` PRs. Surface the relevant one
inline as you scaffold.

- **Stable tree, not `experimental/`.** New functionals go directly into
  `physicsnemo/nn/functional/...`. This is the *opposite* of the models/layers
  rule (`MOD-002a`), and the mistake a contributor coming from the model side
  makes first. There is no `experimental/nn/functional`.

- **Compare neighbor outputs by distance, not index.** Equal-distance ties are
  ordered differently across cuML / SciPy / torch. Parity tests that compare
  indices are spuriously red. Compare **sorted distances**; for samplers compare
  **sorted indices** (set-equality). Put this in `compare_forward` so every test
  inherits it.

- **`custom_op` + `register_fake` are mandatory** for Warp/cuML/SciPy backends.
  Without the `custom_op` wrapper the op won't compose with autograd/`torch.compile`;
  without `register_fake` shape inference (and `opcheck`) fails.

- **Device check belongs inside the impl, not only in `dispatch`.** A cuML/Warp
  impl must raise on a CPU tensor (and SciPy on CUDA) even when called directly
  via `implementation="..."`. `dispatch` picks by device; the impl enforces it.

- **Optional deps are gated, never bare-imported.** Use
  `check_version_spec(pkg, ver, hard_fail=False)` and an `if available: … else:
  stub-raising-ImportError` block. A top-level `import cuml` breaks import on
  every CPU-only install.

- **Cast unsupported dtypes before the backend.** Warp and cuML/cuPy generally
  want fp32 — upcast bf16 (and often fp16) on the way in, and cast results back
  to the caller's dtype on the way out.

- **Warp needs contiguous, zero-copy tensors.** `wp.from_torch(t,
  return_ctype=True)` requires `.contiguous()`; launch under
  `wp.ScopedStream(stream)` from `FunctionSpec.warp_launch_context(...)` so it
  honors the active CUDA stream. `wp.init()` once at module load.

- **Exactly one `baseline=True`** — the torch reference. It's the benchmark
  reference and the equivalence oracle, and it guarantees `dispatch` always has
  an available fallback.

- **Dynamic-shape backends don't `torch.compile` cleanly.** Ops whose output size
  depends on data at runtime (e.g. radius search with an unbounded neighbor
  count) are compile-incompatible — document it and keep a bounded/`max_*` path
  for the compiled case.

- **`nn/functional/` is linted.** Unlike `experimental/`, it is **not** exempt
  from ruff/interrogate — the op needs jaxtyping, full NumPy docstrings
  (`Parameters`/`Returns`/`Raises`), and clean formatting to pass CI.

- **Validate under `if not torch.compiler.is_compiling():`** (`MOD-005`). Keep the
  shape/precondition checks in `utils.py` and guard them so they're skipped under
  compilation rather than tracing into the graph.

- **Read `core/function_spec.py` for the live API.** `register` /
  `dispatch` / `make_function` / `make_inputs_forward` / `compare_forward` /
  `warp_launch_context` are the current names — confirm against source rather than
  trusting a skeleton verbatim.
