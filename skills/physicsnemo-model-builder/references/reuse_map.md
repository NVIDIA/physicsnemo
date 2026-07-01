# Reuse map — find it before you build it

The biggest lever on a clean integration is **not** writing primitives that
already exist. Before scaffolding a `forward`, audit `physicsnemo.nn` for the
pieces the architecture needs and tell the contributor what to import.

**Discover, don't remember.** Class names and paths rot as the repo evolves, so
this file gives *search patterns*, not a frozen list. Run the searches against
the live repo every time, and **verify each path with `ls`/Read before citing
it** (a path pattern-matched from a neighbor is disproof — drop it).

## How to audit (general loop)

1. Name the primitives the architecture needs (e.g., "multi-head attention",
   "positional embedding", "MLP", "nearest-neighbor gather", "layer norm").
2. For each, search the two surfaces:
   - **Modules (layers):** `physicsnemo/nn/module/` — `__init__.py` shows what's
     exported; the files show what exists.
   - **Functionals (ops):** `physicsnemo/nn/functional/` — knn, radius search,
     geometry/SDF, etc.
3. Prefer importing over reimplementing. Keep only genuinely novel,
   model-specific pieces local.

```bash
# what does physicsnemo.nn export?
sed -n '1,200p' physicsnemo/nn/__init__.py
# enumerate layer modules and their public classes
ls physicsnemo/nn/module/
grep -rnE "^class [A-Z]" physicsnemo/nn/module/*.py
# enumerate functional ops
ls physicsnemo/nn/functional/ ; grep -rnE "^def |^class " physicsnemo/nn/functional/*/*.py
```

## Search patterns by category

| Need | Search | Likely already there |
|---|---|---|
| Attention (mesh/point/grid) | `ls physicsnemo/nn/module/*attention*.py` ; `grep -rn "class .*Attention" physicsnemo/nn/module/` | physics-attention base + subclasses; Earth/UNet attention |
| MLP / fully-connected | `grep -rn "class Mlp\|class .*FCLayer\|FullyConnected" physicsnemo/nn/module/` | `Mlp` (configurable, TE-aware), FC layers |
| Positional / Fourier embeddings | `grep -rn "class .*Embedding\|fourier" physicsnemo/nn/module/embedding_layers.py physicsnemo/nn/module/fourier_layers.py` | Fourier / sinusoidal / positional embeddings |
| Normalization | `grep -rn "LayerNorm\|GroupNorm\|RunningNorm" physicsnemo/nn/module/` | **TE-aware `LayerNorm`** (`physicsnemo/nn/module/layer_norm.py`) — use this, not `torch.nn.LayerNorm`, to get Transformer-Engine acceleration |
| SIREN / activations | `grep -rn "class Siren\|get_activation\|ACT2FN" physicsnemo/nn/module/` | `SirenLayer`, activation registry |
| Nearest neighbors / radius | `grep -rn "def knn\|radius_search\|class KNN" physicsnemo/nn/functional/neighbors/` | `knn`, `radius_search` (multi-backend: torch/scipy/cuML) |
| Geometry / SDF / sampling | `ls physicsnemo/nn/functional/geometry/` | SDF ops, point sampling |
| Pooling | `grep -rn "Pooling" physicsnemo/nn/module/pooling.py` | mean / attention pooling |

## Important specifics

- **Normalization:** prefer `physicsnemo.nn.module.layer_norm.LayerNorm` (the
  TE-aware one) over `torch.nn.LayerNorm`. It transparently uses Transformer
  Engine on CUDA (faster backward) and falls back to torch on CPU — but see the
  CPU/TE caveat in `references/lessons.md`.
- **Neighbor ops are functionals with backends:** `physicsnemo.nn.functional.knn`
  auto-dispatches (cuML on CUDA, scipy/torch otherwise). Don't hand-roll a
  distance-matrix kNN.
- **Don't reach across modules the wrong way (`EXT-***`):** imports go
  upward-only, `core → nn → models`. A layer in `nn/` must **not** import from
  `models/` or `experimental/models/`. Read `EXTERNAL_IMPORTS.md`.

## When NOT to reuse

If a primitive is genuinely novel to this architecture (a bespoke attention
variant, a custom tokenizer), keep it **local to the model directory**. Promote
it to `physicsnemo/nn/module/` only when a **second** model wants it — premature
generalization of a single-consumer layer is its own anti-pattern
(`references/lessons.md`).
