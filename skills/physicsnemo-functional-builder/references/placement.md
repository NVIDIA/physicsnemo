# Placement — where a functional lives

Resolve two questions before writing: **is this actually a functional**, and
**where in `nn/functional/` does it go**.

## Is it a functional? (litmus test)

A `physicsnemo.nn.functional` op is a **stateless tensor-in / tensor-out
operation** — no learnable parameters, no `nn.Module`, no checkpoint. It usually
has (or could have) more than one implementation (a torch reference plus an
accelerated backend). Examples: nearest neighbors, radius search, signed-distance
queries, farthest-point sampling, mesh voxelization, interpolation.

If it has parameters / state / a `forward` users compose into a model → it's a
**layer or model**, not a functional → redirect to `physicsnemo-model-builder`.
If it's a loss/metric → `physicsnemo/metrics/`. If it's data loading →
`physicsnemo/datapipes/`.

## Where it goes — the STABLE tree (not experimental)

```
physicsnemo/nn/functional/<category>/<op_name>/
  __init__.py          # re-export the FunctionSpec subclass + the public op
  <op_name>.py         # FunctionSpec subclass + dispatch + make_function(...)
  _torch_impl.py       # torch reference impl (the baseline)
  kernels.py           # pure-Warp @wp.kernel definitions (if Warp backend)
  _warp_impl.py        # Warp impl wrapped in torch.library.custom_op (if Warp)
  _cuml_impl.py        # cuML backend, dep-gated (if provided)
  _scipy_impl.py       # SciPy backend, dep-gated (if provided)
  utils.py             # shared input validation / small helpers
```

> **Key difference from models/layers:** new functionals go straight into the
> **stable** `physicsnemo/nn/functional/` tree — there is **no
> `experimental/nn/functional`**. (`MOD-002a` routes new *models/layers* to
> `experimental/`; it does **not** apply here.) Confirm by checking that existing
> ops like `farthest_point_sampling` and `knn` live under `nn/functional/`, not
> `experimental/`.

A small op can be a **single file** (`<category>/<op_name>.py`) holding the
`FunctionSpec` + inline `custom_op` + kernel (see `geometry/sdf.py`). Prefer the
package layout once there's more than one backend or a kernel file.

## Choosing the category

Pick the existing category that fits; don't invent one without reason:

```bash
ls physicsnemo/nn/functional/                 # geometry, neighbors, interpolation, ...
ls physicsnemo/nn/functional/<category>/      # see sibling ops for the pattern
```

- neighbor / search ops → `neighbors/`
- geometry, SDF, sampling, voxelization → `geometry/`
- resampling / gather-scatter interpolation → `interpolation/`

## Re-exports (wire all the way up)

After creating the op, export it at each level (verify the exact `__init__`
contents in the live repo first):

1. **op `__init__.py`** → `from .<op_name> import <OpClass>, <op_name>`
2. **category `__init__.py`** (`<category>/__init__.py`) → re-export `<op_name>`
   (and the class) and add to `__all__`.
3. **`physicsnemo/nn/functional/__init__.py`** → re-export from the category so
   users can `from physicsnemo.nn.functional import <op_name>`.

Check whether siblings are also surfaced on `physicsnemo.nn` (`nn/__init__.py`);
match the prevailing convention rather than guessing.

## Tests mirror the source path

```
test/nn/functional/<category>/test_<op_name>.py
```

## Don't put these here (redirect)

- A parameterized building block / `nn.Module` → `physicsnemo/nn/module/`
  (→ `physicsnemo-model-builder`).
- A loss or metric → `physicsnemo/metrics/`.
- A datapipe / transform → `physicsnemo/datapipes/`.
- "Which existing op should I use" → a usage question, not an authoring task.
