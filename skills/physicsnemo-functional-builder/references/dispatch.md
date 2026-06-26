# Dispatch — `FunctionSpec`, registration, backend selection

Every functional is a subclass of **`FunctionSpec`**
(`physicsnemo/core/function_spec.py`). It is the single mechanism for: declaring
backends, choosing one at call time, and providing benchmark/equivalence hooks.
**Read `core/function_spec.py` before scaffolding** — the method names below are
the current convention but may evolve; treat the source as truth.

## The shape of a functional

```python
class FarthestPointSampling(FunctionSpec):
    """One-line contract + Parameters/Returns/Raises (NumPy docstring)."""

    # Backends. Lower rank = preferred. EXACTLY ONE baseline=True (the oracle).
    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(points, num_samples, random_start=False): ...

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(points, num_samples, random_start=False): ...

    @classmethod
    def dispatch(cls, points, num_samples, random_start=False, implementation=None):
        ...  # backend selection — see below

    @classmethod
    def make_inputs_forward(cls, device="cpu"):
        ...  # yields (label, args_tuple, kwargs_dict) for benchmarking

    @classmethod
    def compare_forward(cls, output, reference):
        ...  # tie-aware equivalence (see references/testing.md)

# Public callable the rest of physicsnemo imports:
farthest_point_sampling = FarthestPointSampling.make_function("farthest_point_sampling")
```

## `@FunctionSpec.register(...)`

- `name`: backend id used by `implementation="..."` and in tests
  (`"warp"`, `"torch"`, `"cuml"`, `"scipy"`).
- `required_imports`: version specs gating availability, e.g.
  `("cuml>=26.2.0", "cupy>=13.6.0")`. The registry marks the backend
  unavailable (rather than crashing) when an import is missing.
- `rank`: integer preference; the lowest-rank *available* backend wins
  auto-selection.
- `baseline=True`: marks the reference impl (the torch one). Exactly one. It is
  the benchmark reference and the equivalence oracle.

The registered functions take the **op's arguments directly** (no `self`).

## `dispatch()` — the selection recipe

Two paths: explicit override, then auto-select. The canonical shape:

```python
@classmethod
def dispatch(cls, points, num_samples, random_start=False, implementation=None):
    impls = cls._get_impls()
    cls._check_impl(implementation, impls)            # validate the name

    if implementation is not None:                    # explicit override
        impl = impls[implementation]
        if not impl.available:
            raise ImportError(f"Implementation '{implementation}' is not available...")
        return impl.func(points, num_samples, random_start)

    # auto-select: fast backend on CUDA, fall back to the CPU/torch reference
    warp_impl = impls.get("warp")
    if points.is_cuda and warp_impl is not None and warp_impl.available:
        return warp_impl.func(points, num_samples, random_start)
    return impls["torch"].func(points, num_samples, random_start)
```

For an op with cuML (CUDA) + SciPy (CPU) + torch, prefer by device and warn once
on fallback:

```python
preferred_name = "cuml" if points.is_cuda else "scipy"
preferred = impls.get(preferred_name)
impl = preferred if (preferred is not None and preferred.available) else None
if impl is None:
    impl = impls["torch"]
    cls._warn_fallback(preferred, impl)               # one-time fallback warning
return impl.func(points, queries, k)
```

Rules of thumb:
- The auto path must always reach an **available** backend — the torch baseline
  guarantees that.
- Put the *device-appropriateness* decision here (CUDA→Warp/cuML, CPU→torch/SciPy),
  but keep the *hard device check* inside each impl too (a cuML impl must reject a
  CPU tensor itself — see `references/backends.md`).

## `make_function(...)`

`OpClass.make_function("op_name")` returns the public callable that runs
`dispatch`. Bind it at module scope and re-export it (see
`references/placement.md`). Users call `op_name(...)` and may pass
`implementation="warp"` to force a backend (mainly for tests/benchmarks).

## Benchmark + equivalence hooks

- `make_inputs_forward(cls, device)` — a generator yielding
  `(label, args_tuple, kwargs_dict)` covering representative sizes; powers the
  benchmark harness. Keep cases small but realistic.
- `compare_forward(cls, output, reference)` — how two backends' outputs are
  judged equal. Encode tie-invariance here (sort indices; for neighbor ops
  compare sorted **distances**, not indices). Optional `make_inputs_backward` /
  `compare_backward` exist for differentiable ops.

## Cross-cutting (applies to every backend signature)

- **jaxtyping** on tensor args/returns (`MOD-006`):
  `Float[torch.Tensor, "*batch num_points dim"]`, `Int[...]`.
- **NumPy `r"""` docstrings** with `Parameters` / `Returns` / `Raises`.
- **Shape/precondition validation** in `utils.py`, guarded by
  `if not torch.compiler.is_compiling():` (`MOD-005`) so it's skipped under
  compilation.
- **Upward-only imports** (`EXT-***`): functional code may import from
  `physicsnemo.core` (e.g. `FunctionSpec`, `check_version_spec`) but must not
  reach into `models/`.
